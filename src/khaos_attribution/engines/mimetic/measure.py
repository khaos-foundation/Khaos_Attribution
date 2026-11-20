"""mimetic-a0's measure slot: features through a worker in the engine's own environment, on torch on every device,
at full precision as the adapters train."""

from __future__ import annotations

from typing import Any

from khaos_attribution.engines.memory import (LOW_MEMORY_OVERRIDE_ENV, SYSTEM_RESERVE_GB,  # noqa: F401
                                             process_memory_plan, process_scorer_need_gb)
from khaos_attribution.engines.mimetic import DESCRIPTOR

__all__ = ["Handle", "LOW_MEMORY_OVERRIDE_ENV", "ROW_PREFIX", "SYSTEM_RESERVE_GB", "expected_dtype", "enable_checkpointing", "flush_accelerator",
           "index_rows", "memory_plan", "output_feature", "output_loss", "paired_loss", "parameter_count", "scorer_need_gb"]


def expected_dtype(device_type: str) -> str:
    """The precision this measure takes features at: float32 on every device, as the adapters train."""
    return "fp32"


def scorer_need_gb(duration_sec: float, *, checkpointed: bool = False, device_type: str = "cuda") -> float:
    """What the feature worker needs on the device for any clip up to the longest the Workshop cuts."""
    return process_scorer_need_gb(DESCRIPTOR, device_type)


def memory_plan(device_type: str, duration_sec: float, *, total_gb: float | None = None, env: dict | None = None) -> dict:
    """One number, the worker's measured peak, against the device's memory: nothing is resident to yield."""
    return process_memory_plan(DESCRIPTOR, device_type, total_gb=total_gb, env=env)


def enable_checkpointing(module: Any) -> int:
    return 0


def flush_accelerator() -> None:
    return None


# ── the features, by a worker in the engine's environment ─────────────────────────────────────────────────────────

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

from khaos_attribution.engines.worker_process import ROW_PREFIX  # noqa: E402, F401 — the workers read it here

MEASURE_WORKER = Path(__file__).with_name("measure_worker.py")


@dataclass
class Handle:
    """What open_module gives for mimetic-a0: where the engine and the adapter are; the worker loads them per call."""
    root: Path
    python: Path
    adapter: Path
    device_type: str = "cuda"
    dtype: str = "float32"
    runtime: str = "torch"

    @property
    def device(self) -> str:
        """What an index records as the device it was built on: the worker's device type."""
        return self.device_type


def _env(handle: Handle) -> dict:
    from khaos_attribution.engines.mimetic.render import worker_env  # noqa: PLC0415
    return worker_env(handle.root)


_REPLY_KEYS = {"parameters": ("n_params", "n_tensors", "signature"), "index": ("rows", "matrix"), "feature": ("feature",),
               "loss": ("with", "without")}


def _ask(handle: Handle, request: dict, on_progress=None) -> dict:
    """The worker's reply; its ``khaos-row`` lines reach ``on_progress`` as they come, so a long index shows its
    progress while the worker runs."""
    from khaos_attribution.engines import contract_guard  # noqa: PLC0415
    from khaos_attribution.engines import worker_process as W  # noqa: PLC0415
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    contract_guard.require(handle.python, engine=DESCRIPTOR.name, engine_id=DESCRIPTOR.id, error=GradientUnavailable)
    try:
        return W.stream(handle.python, MEASURE_WORKER, request, cwd=handle.root, env=_env(handle), label="mimetic-a0 measure",
                        on_row=on_progress, required=_REPLY_KEYS.get(request.get("command"), ()))
    except W.WorkerFailed as exc:
        raise GradientUnavailable(str(exc)) from exc


def _base(handle: Handle, index_meta: dict, files, command: str) -> dict:
    return {"command": command, "adapter": str(Path(handle.adapter).resolve()),
            "device": None if handle.device_type in (None, "auto") else handle.device_type,
            "files": [str(Path(f).resolve()) for f in files], "timesteps": [float(t) for t in index_meta["timesteps"]],
            "noise_seed": int(index_meta["noise_seed"]), "target": str(index_meta.get("target") or "vnorm"),
            "index_meta": {k: index_meta[k] for k in ("projection",) if k in index_meta}}


def parameter_count(handle: Handle) -> tuple:
    r = _ask(handle, {**_base(handle, {"timesteps": [0.5], "noise_seed": 0}, [], "parameters")})
    return int(r["n_params"]), int(r["n_tensors"]), str(r["signature"])


def index_rows(handle: Handle, files, *, projection, timesteps, noise_seed: int, target: str = "vnorm",
               timestep_norm: bool = False, param_filter: str | None = None, latent_transform: dict | None = None,
               latent_pool: int | None = None, noise_draws: int | None = None, noise_shared: bool = False,
               param_groups: str | None = None, groups_out=None, on_row=None) -> tuple:
    """(matrix, rows) of an index through the worker: the matrix lands beside the files in a file of this call's own."""
    import numpy as np  # noqa: PLC0415
    from khaos_attribution import gradient as _G  # noqa: PLC0415
    from khaos_attribution.engines import worker_process as W  # noqa: PLC0415
    if param_groups or groups_out:
        raise _G.GradientUnavailable("block sketches are not built on the mimetic-a0 worker")
    if _G.research_int(latent_pool, "latent_pool") > 1 or _G.research_int(noise_draws, "noise_draws") > 1:
        raise _G.GradientUnavailable("latent pooling and noise draws are not built on the mimetic-a0 worker")
    if noise_shared:
        raise _G.GradientUnavailable("shared noise is not built on the mimetic-a0 worker")
    if latent_transform:
        from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
        raise GradientUnavailable("a latent transform is not built on the mimetic-a0 worker")
    if timestep_norm:
        from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
        raise GradientUnavailable("per-timestep normalisation is not built on the mimetic-a0 worker")
    if param_filter:
        from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
        raise GradientUnavailable("a parameter filter is not built on the mimetic-a0 worker")
    seen = set()

    def progress(row: dict) -> None:
        if on_row is not None and isinstance(row.get("row"), int) and 0 <= row["row"] < len(files):
            seen.add(row["row"])
            on_row(row["row"], Path(files[row["row"]]), row.get("gradient_norm"))

    from khaos_attribution.gradient import GradientUnavailable, dtype_tag  # noqa: PLC0415
    folder = Path(files[0]).resolve().parent if files else Path.cwd()
    with W.scratch_file(folder, ".khaos_index_rows.", ".npy") as out:
        r = _ask(handle, {**_base(handle, {"timesteps": timesteps, "noise_seed": noise_seed, "target": target}, files, "index"),
                          "projection": {"dim": int(projection.dim), "seed": int(projection.seed)}, "out": str(out)},
                 on_progress=progress)
        if r.get("compute_dtype") and r["compute_dtype"] != dtype_tag(handle.dtype):
            raise GradientUnavailable(f"the worker ran at {r['compute_dtype']}, and the index would record "
                                      f"{dtype_tag(handle.dtype)}; they must be one precision")
        try:
            matrix = np.load(str(r["matrix"]))
        except (OSError, ValueError) as exc:
            raise GradientUnavailable(f"the mimetic-a0 measure worker left no readable matrix: {exc}") from exc
    if on_row is not None:
        for row in r["rows"]:
            if row["row"] not in seen:
                on_row(row["row"], Path(files[row["row"]]), row["gradient_norm"])
    return matrix.astype(np.float32), r["rows"]


def _index_refusal(index_meta: dict) -> str | None:
    """Why an index cannot be read against this measure's features, or None: it must have been built by this
    feature version, since the noise and the parameter order are part of what a row means."""
    built = index_meta.get("feature_version")
    if built != DESCRIPTOR.feature_version:
        return (f"the influence index was built by {DESCRIPTOR.name}'s measure {built or 'before it was versioned'}, "
                f"and this one is {DESCRIPTOR.feature_version}; rebuild the index")
    return None


def output_feature(handle: Handle, index_meta: dict, files, *, groups_out=None):
    import numpy as np  # noqa: PLC0415
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    if groups_out or index_meta.get("param_groups"):
        raise GradientUnavailable("block sketches are not built on the mimetic-a0 worker; rebuild the index without them")
    if index_meta.get("latent_transform"):
        raise GradientUnavailable("a latent transform is not built on the mimetic-a0 worker; rebuild the index without it")
    from khaos_attribution.gradient import research_int  # noqa: PLC0415
    if research_int(index_meta.get("latent_pool"), "latent_pool") > 1 or research_int(index_meta.get("noise_draws"), "noise_draws") > 1:
        raise GradientUnavailable("latent pooling and noise draws are not built on the mimetic-a0 worker; rebuild the index without them")
    why = _index_refusal(index_meta)
    if why:
        raise GradientUnavailable(why)
    r = _ask(handle, _base(handle, index_meta, files, "feature"))
    built_on = (index_meta.get("projection") or {}).get("parameter_signature")
    if built_on and r.get("signature") and r["signature"] != built_on:
        raise GradientUnavailable("this adapter's parameters are not laid out as the index's were; rebuild the index")
    return np.asarray(r["feature"], dtype=np.float32)


def paired_loss(handle: Handle, index_meta: dict, files) -> dict:
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    # Abstention reads the index's thresholds against these losses, so the version gate comes first here too.
    why = _index_refusal(index_meta)
    if why:
        raise GradientUnavailable(why)
    r = _ask(handle, _base(handle, index_meta, files, "loss"))
    return {"with": float(r["with"]), "without": float(r["without"])}


def output_loss(handle: Handle, index_meta: dict, files) -> float:
    return paired_loss(handle, index_meta, files)["with"]
