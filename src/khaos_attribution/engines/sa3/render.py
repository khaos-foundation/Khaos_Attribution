"""Stable Audio 3's render slot. Stability's code needs its own Python environment (its own torch, Flash Attention
on CUDA, MLX on Apple silicon), so the renderer never imports it: it runs one worker per render in the engine's
environment, hands it the request as JSON and reads the result back. Same seed, same bytes, on either runtime."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

from khaos_attribution.engines import worker_process as W
from khaos_attribution.engines.worker_process import stop_tree  # noqa: F401 — sa3.train and the room still import it here
from khaos_attribution.engines.render import RenderCancelled, RenderFailed, RenderRequest, RenderResult
from khaos_attribution.engines.sa3 import DESCRIPTOR

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

    def worker_request(self, req: RenderRequest, out_path: Path) -> dict:
        """The worker's JSON: Khaos's request in the engine's words, the descriptor's row filling what is unset."""
        variant, _ = self.resolve_variant(req.variant)
        row = DESCRIPTOR.render_settings(variant, steps=req.steps, guidance_scale=req.guidance_scale,
                                         apg_scale=req.extra.get("apg_scale"))
        lo, hi = DESCRIPTOR.bound("duration_seconds")
        adapter_path, scale = self.adapter or (None, None)
        return {"runtime": self.runtime, "model": DESCRIPTOR.variant_config(variant), "variant": variant,
                "prompt": req.prompt, "negative_prompt": req.negative_prompt,
                "seconds": min(max(float(req.duration_seconds), lo), hi), "seed": req.seed,
                "steps": int(row["steps"]), "cfg_scale": float(row["guidance_scale"]), "apg_scale": float(row["apg_scale"]),
                "adapter": adapter_path, "adapter_scale": scale, "out": str(out_path),
                "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
                "mode": req.mode, "source_audio": str(req.source_audio) if req.source_audio else None,
                "repaint_start": req.repaint_start, "repaint_end": req.repaint_end, "cover_strength": req.cover_strength}

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
        request = self.worker_request(req, out)
        if should_stop is not None and should_stop():
            raise RenderCancelled()
        if on_progress is not None:
            on_progress(0, 1)
        reply = run_worker(self.root, self.runtime, request, should_stop=should_stop)
        if on_progress is not None:
            on_progress(1, 1)
        if not out.is_file():
            raise RenderFailed(f"the Stable Audio 3 worker wrote no audio at {out}")
        params = {"caption": req.prompt, "duration": request["seconds"], "seed": int(reply["seed"]),
                  "variant": request["variant"], "steps": request["steps"], "guidance_scale": request["cfg_scale"],
                  "apg_scale": request["apg_scale"], "runtime": self.runtime, "sample_rate": int(reply.get("sample_rate") or 0),
                  "instrumental": True, "lyrics": None, "model": request["model"]}
        return RenderResult(audio_path=out, seed=int(reply["seed"]), variant=request["variant"], params=params,
                            applied_scale=request["adapter_scale"] if self.adapter else None,
                            scale_applied_via="lora_strength" if self.adapter else None, dropped=list(reply.get("dropped") or []))


def run_worker(root: Path, runtime: str, request: dict, *, should_stop=None, timeout: float | None = None) -> dict:
    """One worker process through the contract's shared loop; a stop request kills it and nothing is kept."""
    python = python_for(root, runtime)
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
