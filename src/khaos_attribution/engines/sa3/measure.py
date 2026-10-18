"""Stable Audio 3's measure slot, not built yet: the descriptor declares no platform measures on this engine, and
every call here says so rather than guessing a number."""

from __future__ import annotations

from typing import Any

from khaos_attribution.engines.sa3 import DESCRIPTOR

SYSTEM_RESERVE_GB = 6.0
LOW_MEMORY_OVERRIDE_ENV = "KHAOS_INFLUENCE_LOW_MEMORY"
NOT_BUILT = f"{DESCRIPTOR.name} has no influence measure yet; outputs made on it carry no measured split"

__all__ = ["Handle", "LOW_MEMORY_OVERRIDE_ENV", "NOT_BUILT", "SYSTEM_RESERVE_GB", "enable_checkpointing", "flush_accelerator",
           "index_rows", "memory_plan", "output_feature", "output_loss", "paired_loss", "parameter_count", "scorer_need_gb"]


_SCORER = DESCRIPTOR.memory.get("scorer") or {}
# Measured once on khaos on a 56 s clip (the longest the Workshop cuts): the worker's whole-model fp32 peak. Not fitted
# by length; a longer clip would need another measurement.
SCORER_BASE_GB = float(_SCORER.get("base_gb") or 0.0)
SCORER_PROCESS_GB = float(_SCORER.get("process_gb") or 0.0)


def _scorer(device_type: str) -> dict:
    """The worker's measured peak on a device type: CUDA's under "scorer", any other under "scorer_<device>" (mps:
    the MLX worker on the Studio on a 56 s clip, its MLX peak plus the rest of the process, memory being shared)."""
    if device_type == "cuda":
        return _SCORER
    return DESCRIPTOR.memory.get(f"scorer_{device_type}") or {}


def scorer_need_gb(duration_sec: float, *, checkpointed: bool = False, device_type: str = "cuda") -> float:
    """What the feature worker needs on the device for any clip up to the longest the Workshop cuts."""
    s = _scorer(device_type)
    if not s.get("base_gb"):
        raise NotImplementedError(f"{DESCRIPTOR.name}'s measure has not been measured on {device_type}")
    return float(s["base_gb"]) + float(s.get("process_gb") or 0.0)


def memory_plan(device_type: str, duration_sec: float, *, total_gb: float | None = None, env: dict | None = None) -> dict:
    """The worker runs in its own process on the device, so the plan is one number against the device's memory;
    nothing is checkpointed and nothing of the engine is resident to yield."""
    import os as _os  # noqa: PLC0415
    from khaos_attribution.engines.acestep.measure import machine_memory_gb  # noqa: PLC0415
    if not _scorer(device_type).get("base_gb"):
        return {"fits": None, "checkpointed": False, "need_gb": None, "total_gb": total_gb,
                "refusal": f"{DESCRIPTOR.name}'s features have not been measured on {device_type}; this machine scores on {device_type}"}
    total = machine_memory_gb(device_type) if total_gb is None else total_gb
    need = scorer_need_gb(duration_sec, device_type=device_type)
    plan = {"fits": "resident", "checkpointed": False, "need_gb": round(need, 1), "total_gb": total, "refusal": None}
    if total is None or need + SYSTEM_RESERVE_GB <= total or (env if env is not None else _os.environ).get(LOW_MEMORY_OVERRIDE_ENV) == "1":
        return plan
    return {**plan, "fits": None, "refusal": (f"measuring an output on {DESCRIPTOR.name} needs about {need + SYSTEM_RESERVE_GB:.0f} GB on the "
                                              f"device, and this one has {total:.0f} GB")}


def enable_checkpointing(module: Any) -> int:
    return 0


def flush_accelerator() -> None:
    return None


# ── the features, by a worker in the engine's environment ─────────────────────────────────────────────────────────

import json  # noqa: E402
import subprocess  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

MEASURE_WORKER = Path(__file__).with_name("measure_worker.py")
MEASURE_WORKER_MLX = Path(__file__).with_name("measure_worker_mlx.py")


@dataclass
class Handle:
    """What open_module gives for Stable Audio 3: not a module, but where the engine and the adapter are. The
    worker loads them for each call; device_type and dtype are what the contract's readings record."""
    root: Path
    python: Path
    adapter: Path
    device_type: str = "cuda"
    dtype: str = "torch.float32"
    model: str = DESCRIPTOR.variant_config("base") or "medium-base"
    runtime: str = "torch"
    base_npz: Path | None = None

    @property
    def device(self) -> str:
        """What an index records as the device it was built on: the worker's device type."""
        return self.device_type


def _env(handle: Handle) -> dict:
    import os  # noqa: PLC0415
    return {**os.environ, "PYTHONPATH": str(handle.root), "SA3_ROOT": str(handle.root), "WANDB_MODE": "disabled",
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")}


_REPLY_KEYS = {"parameters": ("n_params", "n_tensors", "signature"), "index": ("rows", "matrix"), "feature": ("feature",),
               "loss": ("with", "without")}


ROW_PREFIX = "khaos-row "


def _ask(handle: Handle, request: dict, on_progress=None) -> dict:
    """The worker's reply; its ``khaos-row`` lines reach ``on_progress`` as they come, so a long index shows its
    progress while the worker runs."""
    import tempfile  # noqa: PLC0415
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    worker = MEASURE_WORKER_MLX if handle.runtime == "mlx" else MEASURE_WORKER
    lines = []
    # stderr to a file, never a pipe: a worker that writes much to it would block while stdout is being read.
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as err:
        try:
            proc = subprocess.Popen([str(handle.python), str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=err, cwd=str(handle.root), text=True, errors="replace", env=_env(handle))
        except OSError as exc:
            raise GradientUnavailable(f"the Stable Audio 3 measure worker could not start: {exc}") from exc
        proc.stdin.write(json.dumps(request))
        proc.stdin.close()
        for line in proc.stdout:
            if line.startswith(ROW_PREFIX):
                if on_progress is not None:
                    try:
                        on_progress(json.loads(line[len(ROW_PREFIX):]))
                    except ValueError:
                        pass
                continue
            lines.append(line)
        returncode = proc.wait()
        err.seek(0)
        stderr = err.read()
    last = "".join(lines).strip().splitlines()[-1:] or [""]
    try:
        reply = json.loads(last[0])
    except ValueError:
        reply = None
    if not isinstance(reply, dict) or reply.get("error") or returncode != 0:
        why = (reply or {}).get("error") if isinstance(reply, dict) else "\n".join(stderr.splitlines()[-6:])
        raise GradientUnavailable(f"the Stable Audio 3 measure worker failed: {why}")
    missing = [k for k in _REPLY_KEYS.get(request.get("command"), ()) if k not in reply]
    if missing:
        raise GradientUnavailable(f"the Stable Audio 3 measure worker's reply lacks {', '.join(missing)}")
    return reply



def _base(handle: Handle, index_meta: dict, files, command: str) -> dict:
    return {"command": command, "model": handle.model, "adapter": str(Path(handle.adapter).resolve()),
            "runtime": handle.runtime, "base_npz": str(handle.base_npz) if handle.base_npz else None,
            "files": [str(Path(f).resolve()) for f in files], "timesteps": [float(t) for t in index_meta["timesteps"]],
            "noise_seed": int(index_meta["noise_seed"]), "target": str(index_meta.get("target") or "vnorm"),
            "index_meta": {k: index_meta[k] for k in ("projection",) if k in index_meta}}


def parameter_count(handle: Handle) -> tuple:
    r = _ask(handle, {**_base(handle, {"timesteps": [0.5], "noise_seed": 0}, [], "parameters")})
    return int(r["n_params"]), int(r["n_tensors"]), str(r["signature"])


def index_rows(handle: Handle, files, *, projection, timesteps, noise_seed: int, target: str = "vnorm",
               timestep_norm: bool = False, on_row=None) -> tuple:
    """(matrix, rows) of an index through the worker: the matrix lands beside the files as a temp .npy."""
    import numpy as np  # noqa: PLC0415
    if timestep_norm:
        raise NotImplementedError("per-timestep normalisation is not built on the Stable Audio 3 worker")
    out = Path(files[0]).resolve().parent / ".khaos_index_rows.tmp.npy" if files else Path(".khaos_index_rows.tmp.npy")
    seen = set()

    def progress(row: dict) -> None:
        if on_row is not None and isinstance(row.get("row"), int) and 0 <= row["row"] < len(files):
            seen.add(row["row"])
            on_row(row["row"], Path(files[row["row"]]), row.get("gradient_norm"))

    r = _ask(handle, {**_base(handle, {"timesteps": timesteps, "noise_seed": noise_seed, "target": target}, files, "index"),
                      "projection": {"dim": int(projection.dim), "seed": int(projection.seed)}, "out": str(out)},
             on_progress=progress)
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    try:
        matrix = np.load(str(r["matrix"]))
    except (OSError, ValueError) as exc:
        raise GradientUnavailable(f"the Stable Audio 3 measure worker left no readable matrix: {exc}") from exc
    Path(r["matrix"]).unlink(missing_ok=True)
    if on_row is not None:
        for row in r["rows"]:
            if row["row"] not in seen:
                on_row(row["row"], Path(files[row["row"]]), row["gradient_norm"])
    return matrix.astype(np.float32), r["rows"]


def output_feature(handle: Handle, index_meta: dict, files):
    import numpy as np  # noqa: PLC0415
    r = _ask(handle, _base(handle, index_meta, files, "feature"))
    return np.asarray(r["feature"], dtype=np.float32)


def paired_loss(handle: Handle, index_meta: dict, files) -> dict:
    r = _ask(handle, _base(handle, index_meta, files, "loss"))
    return {"with": float(r["with"]), "without": float(r["without"])}


def measured_peak_gb(handle: Handle, index_meta: dict, files) -> dict:
    """The worker's peak device memory for a feature and for a paired loss on these files: what the descriptor's
    memory block is measured from."""
    f = _ask(handle, _base(handle, index_meta, files, "feature"))
    return {"feature_peak_gb": f.get("peak_gb"), "n": len(files)}


def output_loss(handle: Handle, index_meta: dict, files) -> float:
    return paired_loss(handle, index_meta, files)["with"]
