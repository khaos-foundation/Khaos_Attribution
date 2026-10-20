"""mimetic-a0's render slot. Its code needs its own environment (stable-audio-tools at Mick's pin and its torch),
so the renderer never imports it: one worker per render in the engine's environment, the request as JSON."""

from __future__ import annotations

import os
from pathlib import Path

from khaos_attribution.engines import worker_process as W
from khaos_attribution.engines.mimetic import DESCRIPTOR
from khaos_attribution.engines.render import RenderCancelled, RenderFailed, RenderRequest, RenderResult

ROOT_ENV = "MIMETIC_PATH"
PYTHON_ENV = "MIMETIC_PYTHON"
WORKER = Path(__file__).with_name("worker.py")
WINDOWS = os.name == "nt"
LABEL = "mimetic-a0"


def find_root(repo_dir=None) -> Path:
    """Mick's checkout: the caller's, else MIMETIC_PATH, else a sibling of the working directory."""
    if repo_dir:
        return Path(repo_dir).resolve()
    env = os.environ.get(ROOT_ENV, "").strip()
    return Path(env).resolve() if env else (Path.cwd().parent / "mimetic-a0").resolve()


def venv_python(root: Path) -> Path:
    return root / ".venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")


def python_for(root: Path) -> Path:
    """MIMETIC_PYTHON when set (an environment made elsewhere), else the one provisioning makes."""
    env = os.environ.get(PYTHON_ENV, "").strip()
    if env:
        if not Path(env).is_file():
            raise RenderFailed(f"{PYTHON_ENV} names no interpreter at {env}")
        return Path(env)
    venv = venv_python(root)
    if not venv.is_file():
        raise RenderFailed(f"mimetic-a0 is not provisioned at {root}: no interpreter at {venv}")
    return venv


def worker_env(root: Path) -> dict:
    return {**os.environ, "PYTHONPATH": str(root), "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
            "WANDB_MODE": "disabled", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}


class Renderer:
    """The surface the rooms drive; every render is one worker process in the engine's environment."""

    def __init__(self, repo_dir=None, checkpoint_dir=None, device: str = "auto", base_checkpoint=None,
                 base_resident: bool = False, default_adapter_scale: float = 0.85):
        self.root = find_root(repo_dir)
        self.checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        self.device = device
        self.default_adapter_scale = float(default_adapter_scale)
        self.loaded: str | None = None
        self.adapter: tuple | None = None
        self.descriptor = DESCRIPTOR

    def load(self, variant: str | None = None) -> None:
        variant = variant or DESCRIPTOR.default_variant
        if variant not in DESCRIPTOR.variants:
            raise RenderFailed(f"mimetic-a0 has no variant {variant!r}")
        if not self.root.is_dir():
            raise RenderFailed(f"mimetic-a0 not found at {self.root}")
        python_for(self.root)
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
        variant, _ = self.resolve_variant(req.variant)
        row = DESCRIPTOR.render_settings(variant, steps=req.steps, guidance_scale=req.guidance_scale,
                                         apg_scale=req.extra.get("apg_scale"))
        lo, hi = DESCRIPTOR.bound("duration_seconds")
        adapter_path, scale = self.adapter or (None, None)
        return {"device": None if self.device == "auto" else self.device, "variant": variant,
                "prompt": req.prompt, "negative_prompt": req.negative_prompt,
                "seconds": min(max(float(req.duration_seconds), lo), hi), "seed": req.seed,
                "steps": int(row["steps"]), "cfg_scale": float(row["guidance_scale"]), "apg_scale": float(row["apg_scale"]),
                "adapter": adapter_path, "adapter_scale": scale, "out": str(out_path), "mode": req.mode}

    def generate(self, req: RenderRequest, adapter_path=None, out_path=None, on_progress=None, should_stop=None,
                 release_base: bool = False) -> RenderResult:
        if self.loaded is None:
            self.load(req.variant)
        if adapter_path is not None:
            self.apply_adapter(adapter_path, req.adapter_scale, req.variant)
        if req.lyrics and not req.instrumental:
            raise RenderFailed("mimetic-a0 cannot sing: a sung request must not reach this engine")
        out = Path(out_path or Path.cwd() / "mimetic_render.wav").resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        request = self.worker_request(req, out)
        if should_stop is not None and should_stop():
            raise RenderCancelled()
        if on_progress is not None:
            on_progress(0, 1)
        reply = run_worker(self.root, request, should_stop=should_stop)
        if on_progress is not None:
            on_progress(1, 1)
        if not out.is_file():
            raise RenderFailed(f"the mimetic-a0 worker wrote no audio at {out}")
        params = {"caption": req.prompt, "duration": request["seconds"], "seed": int(reply["seed"]),
                  "variant": request["variant"], "steps": request["steps"], "guidance_scale": request["cfg_scale"],
                  "apg_scale": request["apg_scale"], "runtime": "torch", "device": reply.get("device"),
                  "sample_rate": int(reply.get("sample_rate") or 0), "instrumental": True, "lyrics": None,
                  "model": DESCRIPTOR.variant_config(request["variant"])}
        return RenderResult(audio_path=out, seed=int(reply["seed"]), variant=request["variant"], params=params,
                            applied_scale=request["adapter_scale"] if self.adapter else None,
                            scale_applied_via="lora_strength" if self.adapter else None, dropped=list(reply.get("dropped") or []))


def run_worker(root: Path, request: dict, *, should_stop=None, timeout: float | None = None) -> dict:
    try:
        return W.run(python_for(root), WORKER, request, cwd=root, env=worker_env(root), label=LABEL,
                     should_stop=should_stop, timeout=timeout, required="seed")
    except W.WorkerCancelled as exc:
        raise RenderCancelled() from exc
    except W.WorkerFailed as exc:
        raise RenderFailed(str(exc)) from exc
