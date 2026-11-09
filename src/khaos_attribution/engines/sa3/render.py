"""Stable Audio 3's render slot. Stability's code needs its own Python environment (its own torch, Flash Attention
on CUDA, MLX on Apple silicon), so the renderer never imports it: it runs one worker per render in the engine's
environment, hands it the request as JSON and reads the result back. Both runtimes render an adapter on the base
checkpoint it trained on. Same seed, same bytes, on one runtime; the two runtimes size the latent differently. A
rework is planned here (sa3/rework.py) and spliced into the source here."""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from khaos_attribution.engines import worker_process as W
from khaos_attribution.engines.worker_process import stop_tree  # noqa: F401 — sa3.train and the room still import it here
from khaos_attribution.engines.render import RenderCancelled, RenderFailed, RenderRequest, RenderResult
from khaos_attribution.engines.sa3 import DESCRIPTOR
from khaos_attribution.engines.sa3 import rework as RW

log = logging.getLogger(__name__)

ROOT_ENV = "SA3_PATH"
MLX_SUBDIR = Path("optimized") / "mlx"
WORKER = Path(__file__).with_name("worker.py")


def find_root(repo_dir=None) -> Path:
    """Stability's checkout: the caller's, else SA3_PATH, else a sibling of the working directory."""
    if repo_dir:
        return Path(repo_dir).resolve()
    env = os.environ.get(ROOT_ENV, "").strip()
    return Path(env).resolve() if env else (Path.cwd().parent / "stable-audio-3").resolve()


def runtime_for(device: str) -> str:
    """"mlx" on Apple silicon, "torch" elsewhere: the two runtimes the descriptor declares."""
    if device == "mps" or (device == "auto" and sys.platform == "darwin"):
        return "mlx"
    return "torch"


PYTHON_ENV = {"torch": "SA3_PYTHON", "mlx": "SA3_MLX_PYTHON"}
WINDOWS = os.name == "nt"


def venv_python(root: Path, runtime: str) -> Path:
    """The interpreter provisioning makes for a runtime, in the layout this platform's venvs have."""
    home = root / MLX_SUBDIR if runtime == "mlx" else root
    return home / ".venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")


def python_for(root: Path, runtime: str) -> Path:
    """The engine's own interpreter for a runtime: SA3_PYTHON / SA3_MLX_PYTHON when set (an environment made
    elsewhere, such as the trial's), else the one provisioning makes; RenderFailed when neither is there."""
    env = os.environ.get(PYTHON_ENV[runtime], "").strip()
    if env:
        if not Path(env).is_file():
            raise RenderFailed(f"{PYTHON_ENV[runtime]} names no interpreter at {env}")
        return Path(env)
    venv = venv_python(root, runtime)
    if not venv.is_file():
        raise RenderFailed(f"Stable Audio 3 is not provisioned at {root} for {runtime}: no interpreter at {venv}")
    return venv


class Renderer:
    """The surface the rooms drive; every render is one worker process in the engine's environment."""

    def __init__(self, repo_dir=None, checkpoint_dir=None, device: str = "auto", base_checkpoint=None,
                 base_resident: bool = False, default_adapter_scale: float | None = None):
        self.root = find_root(repo_dir)
        self.checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        self.device = device
        self.runtime = runtime_for(device)
        self.default_adapter_scale = float(DESCRIPTOR.adapter_strength if default_adapter_scale is None else default_adapter_scale)
        self.loaded: str | None = None
        self.adapter: tuple | None = None
        self.descriptor = DESCRIPTOR

    def load(self, variant: str | None = None) -> None:
        """Nothing stays resident between renders; load checks the environment is there and the variant is known."""
        variant = variant or DESCRIPTOR.default_variant
        if variant not in DESCRIPTOR.variants:
            raise RenderFailed(f"Stable Audio 3 has no variant {variant!r}")
        if not self.root.is_dir():
            raise RenderFailed(f"Stable Audio 3 not found at {self.root}")
        python_for(self.root, self.runtime)
        self.loaded = variant

    def resolve_variant(self, requested: str | None) -> tuple:
        wanted = requested or DESCRIPTOR.default_variant
        if wanted in DESCRIPTOR.variants:
            return wanted, None
        return DESCRIPTOR.default_variant, f"variant {wanted!r} is not available on this engine; rendered on {DESCRIPTOR.default_variant}"

    def apply_adapter(self, path, scale: float | None, variant: str | None = None) -> tuple:
        scale = self.default_adapter_scale if scale is None else float(scale)
        self.adapter = (str(Path(path).resolve()), scale)
        return scale, "lora_strength"

    def forget_adapter(self) -> None:
        self.adapter = None

    def release_base(self) -> None:
        return None

    def yield_memory(self) -> None:
        return None

    def worker_request(self, req: RenderRequest, out_path: Path, **rework) -> dict:
        """The worker's JSON: Khaos's request in the engine's words, the descriptor's row filling what is unset; a
        rework passes its planned source, regions, noise and seed, which replace the request's own."""
        variant, _ = self.resolve_variant(req.variant)
        row = DESCRIPTOR.render_settings(variant, steps=req.steps, guidance_scale=req.guidance_scale,
                                         apg_scale=req.extra.get("apg_scale"))
        lo, hi = DESCRIPTOR.bound("duration_seconds")
        adapter_path, scale = self.adapter or (None, None)
        seconds = rework.pop("seconds", req.duration_seconds)
        return {"runtime": self.runtime, "model": DESCRIPTOR.variant_config(variant), "variant": variant,
                "prompt": req.prompt, "negative_prompt": req.negative_prompt,
                "seconds": min(max(float(seconds), lo), hi), "seed": req.seed,
                "steps": int(row["steps"]), "cfg_scale": float(row["guidance_scale"]), "apg_scale": float(row["apg_scale"]),
                "adapter": adapter_path, "adapter_scale": scale, "out": str(out_path),
                "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
                "mode": "text2music", "source_audio": None, "regions": [], "noise_level": None, **rework}

    def generate(self, req: RenderRequest, adapter_path=None, out_path=None, on_progress=None, should_stop=None,
                 release_base: bool = False) -> RenderResult:
        if self.loaded is None:
            self.load(req.variant)
        if adapter_path is not None:
            self.apply_adapter(adapter_path, req.adapter_scale, req.variant)
        if req.lyrics and not req.instrumental:
            raise RenderFailed("Stable Audio 3 cannot sing: a sung request must not reach this engine")
        out = Path(out_path or Path.cwd() / "sa3_render.wav").resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if should_stop is not None and should_stop():
            raise RenderCancelled()
        mode = req.mode or "text2music"
        if mode == "text2music":
            request = self.worker_request(req, out)
            reply = self._render(request, on_progress, should_stop, 0, 1)
            if not out.is_file():
                raise RenderFailed(f"the Stable Audio 3 worker wrote no audio at {out}")
            return self._result(req, request, reply, out, {})
        if mode not in REWORKS:
            raise RenderFailed(f"Stable Audio 3 has no rework {mode!r}")
        if req.source_audio is None or not Path(req.source_audio).is_file():
            raise RenderFailed(f"a Stable Audio 3 {mode} needs the source audio, and none was found at {req.source_audio}")
        work = Path(tempfile.mkdtemp(prefix="khaos-sa3-rework-"))
        try:
            return REWORKS[mode](self, req, out, work, on_progress, should_stop)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _render(self, request: dict, on_progress, should_stop, done: int, total: int) -> dict:
        if on_progress is not None:
            on_progress(done, total)
        reply = run_worker(self.root, self.runtime, request, should_stop=should_stop)
        if on_progress is not None:
            on_progress(done + 1, total)
        return reply

    def _result(self, req: RenderRequest, request: dict, reply: dict, out: Path, facts: dict) -> RenderResult:
        seed = int(facts.pop("seed", reply["seed"]))
        params = {"caption": req.prompt, "duration": request["seconds"], "seed": seed, "khaos_seed": seed,
                  "variant": request["variant"], "steps": request["steps"], "guidance_scale": request["cfg_scale"],
                  "apg_scale": request["apg_scale"], "runtime": self.runtime, "sample_rate": int(reply.get("sample_rate") or 0),
                  "instrumental": True, "lyrics": None, "model": request["model"], **facts}
        return RenderResult(audio_path=out, seed=seed, variant=request["variant"], params=params,
                            applied_scale=request["adapter_scale"] if self.adapter else None,
                            scale_applied_via="lora_strength" if self.adapter else None, dropped=list(reply.get("dropped") or []))

    # ── the reworks ──────────────────────────────────────────────────────────
    def _limits(self) -> tuple:
        """``(sample_rate, frame, window)`` in samples; at the window the worker's seconds clamp to the bound itself."""
        sr = DESCRIPTOR.sample_rate
        frame = int(DESCRIPTOR.rework["frame_samples"])
        return sr, frame, int(DESCRIPTOR.bound("duration_seconds")[1] * sr)

    def _readable(self, path: Path, work: Path) -> Path:
        """The source as both runtimes read it natively, 16-bit WAV at the engine's rate: the file itself when it
        already is, else a copy; another rate is left to the MLX port's ffmpeg, which must then be installed."""
        sf = RW._soundfile()
        info = sf.info(str(path))
        if info.format == "WAV" and info.subtype == WORKER_INPUT and info.samplerate == DESCRIPTOR.sample_rate:
            return path
        if info.samplerate == DESCRIPTOR.sample_rate:
            audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
            copy = work / "source.wav"
            RW.write_audio(copy, audio, info.samplerate, WORKER_INPUT)
            return copy
        if self.runtime == "mlx" and shutil.which("ffmpeg") is None:
            raise RenderFailed(f"the source is at {info.samplerate} Hz; on this Mac Stable Audio 3 reads another "
                               f"rate through ffmpeg, which is not installed (brew install ffmpeg)")
        return path

    @staticmethod
    def _seconds(samples: int, sr: int) -> float:
        # A quarter sample over: torch truncates seconds * rate, MLX rounds it; both then give exactly ``samples``.
        return (samples + 0.25) / sr

    @staticmethod
    def _seed(seed) -> int:
        return int(seed) % (2 ** 31) if seed is not None else secrets.randbelow(2 ** 31 - 1)

    def _whole(self, req: RenderRequest, out: Path, work: Path, on_progress, should_stop) -> RenderResult:
        """Another take and Take it somewhere new: the whole source as init audio, noised to the strength's level."""
        mode, sr, (_, _, window) = req.mode, DESCRIPTOR.sample_rate, self._limits()
        info = RW._soundfile().info(str(req.source_audio))
        n = int(round(info.frames * sr / info.samplerate))
        if n > window:
            raise RenderFailed(f"this piece is {n / sr:.0f} s; Stable Audio 3 takes a whole piece in at most "
                               f"{window / sr:.0f} s, so it can fix a part of it or make it longer but not "
                               f"{'take it again' if mode == 'retake' else 'take it somewhere new'}")
        level = RW.noise_level(mode, req, DESCRIPTOR.rework)
        seed = self._seed(req.retake_seed if mode == "retake" else req.seed)
        request = self.worker_request(req, out, seconds=self._seconds(n, sr), mode=mode, seed=seed,
                                      source_audio=str(self._readable(Path(req.source_audio).resolve(), work)),
                                      noise_level=level)
        reply = self._render(request, on_progress, should_stop, 0, 1)
        request["seconds"] = n / sr
        facts = {"seed": seed, "mode": mode, "noise_level": level, "source_duration": n / sr,
                 **({"retake_variance": req.retake_variance} if mode == "retake" else {"cover_strength": req.cover_strength})}
        return self._result(req, request, reply, out, facts)

    def _repaint(self, req: RenderRequest, out: Path, work: Path, on_progress, should_stop) -> RenderResult:
        """Fix a section: every part masked in one pass, then each spliced into the source with a crossfade."""
        sr, frame, window = self._limits()
        table = DESCRIPTOR.rework
        source, src_sr, subtype, step = RW.read_audio(req.source_audio)
        if src_sr != sr:
            raise RenderFailed(f"the source is at {src_sr} Hz; a Stable Audio 3 fix splices at {sr} Hz")
        n = len(source)
        asked = req.repaint_regions or [(req.repaint_start or 0.0, req.repaint_end)]
        regions = RW.plan_regions(asked, n, sr, frame)
        crossfade = int(round(float(table["crossfade_seconds"]) * sr))
        offset, length = RW.repaint_window(regions, n, window, frame, crossfade, sr)
        given = self._readable(Path(req.source_audio).resolve(), work)
        if offset or length < n:
            given = work / "window.wav"
            RW.write_audio(given, source[offset:offset + length], sr, WORKER_INPUT)
        level = RW.noise_level("repaint", req, table)
        seed = self._seed(req.seed)
        regenerated = work / "regenerated.wav"
        request = self.worker_request(req, regenerated, seconds=self._seconds(length, sr), mode="repaint", seed=seed,
                                      source_audio=str(given), noise_level=level,
                                      regions=[[(s - offset) / sr, (e - offset) / sr] for s, e in regions])
        reply = self._render(request, on_progress, should_stop, 0, 1)
        if not regenerated.is_file():
            raise RenderFailed(f"the Stable Audio 3 worker wrote no audio at {regenerated}")
        spliced = RW.splice(source, RW.read_as(regenerated, source), regions, crossfade, offset, step)
        RW.write_audio(out, spliced, sr, subtype)
        request["seconds"] = n / sr
        facts = {"seed": seed, "mode": "repaint", "noise_level": level, "repaint_mode": req.repaint_mode or "balanced",
                 "source_duration": n / sr, "regions": [[s / sr, e / sr] for s, e in regions],
                 "requested_regions": [[float(s or 0.0), None if e is None else float(e)] for s, e in asked],
                 "crossfade_seconds": crossfade / sr, "window": [offset / sr, (offset + length) / sr]}
        return self._result(req, request, reply, out, facts)

    def _extend(self, req: RenderRequest, out: Path, work: Path, on_progress, should_stop) -> RenderResult:
        """Make it longer: chunks that each fit the window, every one continuing from the end (or the start) of
        what is there so far, joined with a crossfade before (or after) the seam."""
        sr, frame, window = self._limits()
        table = DESCRIPTOR.rework
        current, src_sr, subtype, step = RW.read_audio(req.source_audio)
        if src_sr != sr:
            raise RenderFailed(f"the source is at {src_sr} Hz; a Stable Audio 3 extend joins at {sr} Hz")
        n = len(current)
        head = bool(req.extend_head and req.extend_head > 0)
        asked = int(round(float(req.duration_seconds) * sr)) - n
        if head:
            # A head names its own length; the room's requested total still caps it.
            wanted = int(round(float(req.extend_head) * sr))
            add = min(wanted, asked) if asked > 0 else wanted
        else:
            add = asked
        if add <= 0:
            raise RenderFailed("nothing to add: the requested length is not longer than the piece")
        context = int(round(float(table["extend_context_seconds"]) * sr))
        chunk = table["extend_chunk_seconds"]
        crossfade = int(round(float(table["crossfade_seconds"]) * sr))
        plans = RW.extend_plan(add, n, window, context, int(round(chunk * sr)) if chunk else None, frame, head=head)
        base = self._seed(req.seed)
        chunks, request, reply = [], None, {}
        for i, (ctx, new) in enumerate(plans):
            if should_stop is not None and should_stop():
                raise RenderCancelled()
            seed = (base + i) % (2 ** 31)
            given, made = work / f"context{i}.wav", work / f"chunk{i}.wav"
            if head:
                pad = -(-new // frame) * frame
                buffer = current[:ctx]
                RW.write_audio(given, np.concatenate([np.zeros((pad, current.shape[1]), current.dtype), buffer]),
                               sr, WORKER_INPUT)
                length, regions = pad + ctx, [[0.0, pad / sr]]
            else:
                RW.write_audio(given, current[len(current) - ctx:], sr, WORKER_INPUT)
                length, regions = ctx + new, [[ctx / sr, (ctx + new) / sr]]
            request = self.worker_request(req, made, seconds=self._seconds(length, sr), mode="extend", seed=seed,
                                          source_audio=str(given), noise_level=1.0, regions=regions)
            reply = self._render(request, on_progress, should_stop, i, len(plans))
            if not made.is_file():
                raise RenderFailed(f"the Stable Audio 3 worker wrote no audio at {made}")
            # The worker's seconds never fall under the engine's floor, so a short chunk comes back long.
            generated = RW.read_as(made, current)[:length]
            current = (RW.join_head(current, generated, pad, new, crossfade, step) if head
                       else RW.join_tail(current, generated, ctx, crossfade, step))
            chunks.append({"seed": seed, "context_seconds": ctx / sr, "new_seconds": new / sr})
        RW.write_audio(out, current, sr, subtype)
        request["seconds"] = len(current) / sr
        facts = {"seed": base, "mode": "extend", "noise_level": 1.0, "source_duration": n / sr,
                 "extend_side": "head" if head else "tail", "added_seconds": add / sr, "crossfade_seconds": crossfade / sr,
                 "chunks": chunks}
        return self._result(req, request, reply, out, facts)


# What the worker is handed is read natively by both runtimes; the output keeps the source's own format.
WORKER_INPUT = "PCM_16"
REWORKS = {"retake": Renderer._whole, "audio2audio": Renderer._whole, "repaint": Renderer._repaint,
           "extend": Renderer._extend}


def run_worker(root: Path, runtime: str, request: dict, *, should_stop=None, timeout: float | None = None) -> dict:
    """One worker process through the contract's shared loop; a stop request kills it and nothing is kept."""
    from khaos_attribution.engines import contract_guard  # noqa: PLC0415
    python = python_for(root, runtime)
    contract_guard.require(python, engine="Stable Audio 3", engine_id=DESCRIPTOR.id, error=RenderFailed)
    env = {**os.environ, "PYTHONPATH": str(root), "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
           "WANDB_MODE": "disabled"}
    try:
        return W.run(python, WORKER, request, cwd=root, env=env, label="Stable Audio 3", should_stop=should_stop,
                     timeout=timeout, required="seed")
    except W.WorkerCancelled:
        raise RenderCancelled() from None
    except W.WorkerFailed as exc:
        raise RenderFailed(str(exc)) from None


def adapter_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
