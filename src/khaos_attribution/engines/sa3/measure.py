"""Stable Audio 3's measure slot: features through a worker in the engine's own environment, torch (CUDA) or MLX
(a Mac). The descriptor's measure flag names the influence index, which no platform declares on this engine yet;
group removal reads paired_loss and output_losses here whatever the flag says."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from khaos_attribution.engines.memory import (LOW_MEMORY_OVERRIDE_ENV, SYSTEM_RESERVE_GB,  # noqa: F401
                                             process_memory_plan, process_scorer_need_gb)
from khaos_attribution.engines import process_engine as PE
from khaos_attribution.engines.sa3 import DESCRIPTOR
from khaos_attribution.engines.worker_process import ROW_PREFIX  # noqa: F401 — the workers read it here

log = logging.getLogger(__name__)

__all__ = ["Handle", "LOW_MEMORY_OVERRIDE_ENV", "ROW_PREFIX", "SYSTEM_RESERVE_GB", "expected_dtype", "enable_checkpointing", "flush_accelerator",
           "index_rows", "latent_stats", "memory_plan", "output_feature", "output_loss", "output_losses", "paired_loss", "parameter_count", "scorer_need_gb",
           "tid_factors", "tid_invert", "tid_kernels"]


def expected_dtype(device_type: str) -> str:
    """The precision this measure takes features at on a device type: the base as each runtime's trainer holds it,
    bfloat16 on CUDA and float16 on a Mac (the MLX port), full precision elsewhere."""
    return {"cuda": "bf16", "mps": "fp16"}.get(device_type, "fp32")


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
    from khaos_attribution.engines.worker_process import engine_env  # noqa: PLC0415
    return engine_env(handle.root, SA3_ROOT=str(handle.root))


_REPLY_KEYS = {**PE.REPLY_KEYS, "latent_stats": ("transform",),
               "tid_factors": ("tid_version", "modules", "live_tensors", "dropped_tensors"),
               "tid_invert": ("tid_version", "inverses", "factors"),
               "tid_kernels": ("kernels", "draws_done", "grid", "live_tensors", "dropped_tensors", "normaliser_zero", "seconds")}


def _ask(handle: Handle, request: dict, on_progress=None) -> dict:
    """The worker's reply; its ``khaos-row`` lines reach ``on_progress`` as they come, so a long index shows its
    progress while the worker runs."""
    worker = MEASURE_WORKER_MLX if handle.runtime == "mlx" else MEASURE_WORKER
    return PE.measure_reply(handle.python, worker, request, cwd=handle.root, env=_env(handle), label="Stable Audio 3",
                            descriptor=DESCRIPTOR, engine="Stable Audio 3", reply_keys=_REPLY_KEYS, on_progress=on_progress)


def _base(handle: Handle, index_meta: dict, files, command: str) -> dict:
    return {"command": command, "model": handle.model, "adapter": str(Path(handle.adapter).resolve()),
            "runtime": handle.runtime, "base_npz": str(handle.base_npz) if handle.base_npz else None,
            "files": [str(Path(f).resolve()) for f in files], "timesteps": [float(t) for t in index_meta["timesteps"]],
            "noise_seed": int(index_meta["noise_seed"]), "target": str(index_meta.get("target") or "vnorm"),
            # Research settings travel with the index, so an output's feature is taken as the rows were.
            "timestep_norm": bool(index_meta.get("timestep_norm", False)),
            "param_filter": index_meta.get("param_filter") or None,
            "latent_transform": index_meta.get("latent_transform") or None,
            "latent_pool": index_meta.get("latent_pool") or None,
            "noise_draws": index_meta.get("noise_draws") or None,
            "noise_shared": bool(index_meta.get("noise_shared", False)),
            "param_groups": index_meta.get("param_groups") or None,
            "index_meta": {k: index_meta[k] for k in ("projection",) if k in index_meta}}


def parameter_count(handle: Handle) -> tuple:
    r = _ask(handle, {**_base(handle, {"timesteps": [0.5], "noise_seed": 0}, [], "parameters")})
    return int(r["n_params"]), int(r["n_tensors"]), str(r["signature"])


def index_rows(handle: Handle, files, *, projection, timesteps, noise_seed: int, target: str = "vnorm",
               timestep_norm: bool = False, param_filter: str | None = None, latent_transform: dict | None = None,
               latent_pool: int | None = None, noise_draws: int | None = None, noise_shared: bool = False,
               param_groups: str | None = None, groups_out=None, on_row=None) -> tuple:
    """(matrix, rows) of an index through the worker: the matrix lands beside the files in a file of this call's own;
    with ``param_groups`` each row's block sketches land at ``groups_out``, whole or not at all, their names beside it."""
    import numpy as np  # noqa: PLC0415
    from khaos_attribution.engines import worker_process as W  # noqa: PLC0415
    from khaos_attribution.gradient import GradientUnavailable, dtype_tag  # noqa: PLC0415
    if param_groups and not groups_out:
        raise GradientUnavailable("an index with block sketches needs groups_out, where they are written")
    if groups_out and not param_groups:
        raise GradientUnavailable("groups_out needs param_groups, the pattern that names the blocks")
    seen = set()

    def progress(row: dict) -> None:
        if on_row is not None and isinstance(row.get("row"), int) and 0 <= row["row"] < len(files):
            seen.add(row["row"])
            on_row(row["row"], Path(files[row["row"]]), row.get("gradient_norm"))

    folder = Path(files[0]).resolve().parent if files else Path.cwd()
    groups = Path(groups_out).resolve() if groups_out else None
    try:
        with W.scratch_file(folder, ".khaos_index_rows.", ".npy") as out:
            r = _ask(handle, {**_base(handle, {"timesteps": timesteps, "noise_seed": noise_seed, "target": target,
                                               "timestep_norm": timestep_norm, "param_filter": param_filter,
                                               "latent_transform": latent_transform, "latent_pool": latent_pool,
                                               "noise_draws": noise_draws, "noise_shared": noise_shared,
                                               "param_groups": param_groups}, files, "index"),
                              "projection": {"dim": int(projection.dim), "seed": int(projection.seed)}, "out": str(out),
                              "groups_out": str(groups) if groups else None},
                     on_progress=progress)
            if r.get("compute_dtype") and r["compute_dtype"] != dtype_tag(handle.dtype):
                raise GradientUnavailable(f"the worker ran at {r['compute_dtype']}, and the index would record "
                                          f"{dtype_tag(handle.dtype)}; they must be one precision")
            try:
                matrix = np.load(str(r["matrix"]))
            except (OSError, ValueError) as exc:
                raise GradientUnavailable(f"the Stable Audio 3 measure worker left no readable matrix: {exc}") from exc
    finally:
        # A worker stopped part way leaves its sketches beside their name, never at it.
        if groups:
            W.part_of(groups).unlink(missing_ok=True)
    if param_groups:
        if not Path(groups_out).is_file() or not r.get("group_labels"):
            raise GradientUnavailable("the Stable Audio 3 measure worker wrote no block sketches")
        Path(groups_out).with_suffix(".labels.json").write_text(json.dumps(r["group_labels"]) + "\n", encoding="utf-8")
    if on_row is not None:
        for row in r["rows"]:
            if row["row"] not in seen:
                on_row(row["row"], Path(files[row["row"]]), row["gradient_norm"])
    return matrix.astype(np.float32), r["rows"]


def latent_stats(handle: Handle, files, *, kind: str, variance: float = 0.9) -> dict:
    """A research latent transform measured once over ``files`` (the training pairs), for the index to carry."""
    from khaos_attribution.gradient import LATENT_TRANSFORMS, GradientUnavailable  # noqa: PLC0415
    if kind not in LATENT_TRANSFORMS:
        raise GradientUnavailable(f"latent transform must be one of {LATENT_TRANSFORMS}, not {kind!r}")
    if handle.runtime == "mlx":
        raise GradientUnavailable("a latent transform is not built on the MLX worker")
    r = _ask(handle, {**_base(handle, {"timesteps": [0.5], "noise_seed": 0}, files, "latent_stats"),
                      "latent_kind": kind, "latent_variance": float(variance)})
    return dict(r["transform"])


def _index_refusal(index_meta: dict) -> str | None:
    return PE.index_refusal(DESCRIPTOR, index_meta)


def output_feature(handle: Handle, index_meta: dict, files, *, groups_out=None):
    """The output's projected feature; with ``groups_out`` its block sketches are written there too."""
    import numpy as np  # noqa: PLC0415
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    why = _index_refusal(index_meta)
    if why:
        raise GradientUnavailable(why)
    if groups_out and not index_meta.get("param_groups"):
        raise GradientUnavailable("this index was built without block sketches; an output's cannot be read against it")
    from khaos_attribution.engines import worker_process as W  # noqa: PLC0415
    request = _base(handle, index_meta, files, "feature")
    if groups_out:
        request["groups_out"] = str(Path(groups_out).resolve())
    try:
        r = _ask(handle, request)
    finally:
        if groups_out:
            W.part_of(Path(groups_out).resolve()).unlink(missing_ok=True)
    if groups_out and not Path(groups_out).is_file():
        raise GradientUnavailable("the Stable Audio 3 measure worker wrote no block sketches for this output")
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


# ── TID, the research reading of khaos_attribution.tid, built on the CUDA worker only ─────────────────────────────

def _tid_request(handle: Handle, command: str, **fields) -> dict:
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    if handle.runtime == "mlx":
        raise GradientUnavailable("TID is built on the CUDA worker only; this handle runs on MLX")
    return {"command": command, "model": handle.model, "adapter": str(Path(handle.adapter).resolve()),
            "runtime": handle.runtime, **fields}


def _tid_dtype(handle: Handle, reply: dict) -> dict:
    from khaos_attribution.gradient import GradientUnavailable, dtype_tag  # noqa: PLC0415
    if reply.get("compute_dtype") and reply["compute_dtype"] != dtype_tag(handle.dtype):
        raise GradientUnavailable(f"the worker ran at {reply['compute_dtype']}, and this handle records "
                                  f"{dtype_tag(handle.dtype)}; they must be one precision")
    return reply


def tid_factors(handle: Handle, files, *, grid, noise_seed: int, target: str, out_dir, factor_groups: int = 1,
                modules: str | None = None, on_row=None) -> dict:
    """TID's Kronecker factors of the adapter over ``files`` at the aligned draws of ``grid``, written under
    ``out_dir``; the reply is the factors record, ``factors.json``."""
    request = _tid_request(handle, "tid_factors", files=[str(Path(f).resolve()) for f in files],
                           grid=[float(t) for t in grid], noise_seed=int(noise_seed), target=str(target),
                           out_dir=str(Path(out_dir).resolve()), factor_groups=int(factor_groups), modules=modules)
    return _tid_dtype(handle, _ask(handle, request, on_progress=on_row))


def tid_invert(handle: Handle, factors_dir, *, alpha: float = 0.1) -> dict:
    """The damped inverse of every live factor under ``factors_dir``; the reply is the spectra summary,
    ``inverses.json``."""
    return _ask(handle, _tid_request(handle, "tid_invert", factors_dir=str(Path(factors_dir).resolve()), alpha=float(alpha)))


def tid_kernels(handle: Handle, train_files, query_files, query_groups, *, grid, noise_seed: int, target: str,
                factors_dir, out, train_chunk: int = 1, keep_layer_kernels: bool = False, resume: bool = True,
                on_row=None) -> dict:
    """TID's per-draw kernels written to ``out`` (.npy, float32, draws × query files × training files, in grid,
    query and training order); ``query_groups`` names each query file's output. The reply names what was written."""
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    if len(query_groups) != len(query_files):
        raise GradientUnavailable(f"{len(query_groups)} query groups for {len(query_files)} query files")
    if not str(out).endswith(".npy"):
        raise GradientUnavailable(f"the kernels are written as .npy, not to {out}")
    request = _tid_request(handle, "tid_kernels", train_files=[str(Path(f).resolve()) for f in train_files],
                           query_files=[str(Path(f).resolve()) for f in query_files], query_groups=list(query_groups),
                           grid=[float(t) for t in grid], noise_seed=int(noise_seed), target=str(target),
                           factors_dir=str(Path(factors_dir).resolve()), out=str(Path(out).resolve()),
                           train_chunk=int(train_chunk), keep_layer_kernels=bool(keep_layer_kernels), resume=bool(resume))
    return _tid_dtype(handle, _ask(handle, request, on_progress=on_row))


# ── group removal: one piece's loss under several adapters ───────────────────────────────────────────────────────

def output_losses(api: Any, clips_dir, adapters: dict, *, seed: int, n_buckets: int | None = None,
                  checkpoint_root=None) -> dict:
    """``{name: mean loss}`` of the clips under ``clips_dir`` (one piece) scored under each named adapter, by the
    held-out worker its trainer uses, at one noise seed so every adapter sees the same draws."""
    from khaos_attribution.engines import contract_guard  # noqa: PLC0415
    from khaos_attribution.engines.sa3 import train as T  # noqa: PLC0415
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    contract_guard.require(api.python, engine="Stable Audio 3", engine_id=DESCRIPTOR.id, error=GradientUnavailable)
    timesteps = T.bucket_timesteps(n_buckets or T.DEFAULT_BUCKETS)
    if DESCRIPTOR.runtime.get("heldout_in_one_process") and api.runtime == "torch" and len(adapters) > 1 \
            and all(adapters.values()):
        together = _losses_in_one_process(api, clips_dir, adapters, seed=seed, timesteps=timesteps,
                                          checkpoint_root=checkpoint_root)
        if together is not None:
            return together
    out = {}
    for name, weights in adapters.items():
        try:
            out[name] = float(T.heldout_summary(api, clips_dir, weights, seed=seed, timesteps=timesteps,
                                                checkpoint_root=checkpoint_root)["mean"])
        except (RuntimeError, ValueError) as exc:
            raise GradientUnavailable(str(exc)) from exc
    return out


def _losses_in_one_process(api: Any, clips_dir, adapters: dict, *, seed: int, timesteps: list, checkpoint_root) -> dict | None:
    """``{name: mean loss}`` from one held-out worker that loads the base once and switches each adapter on in turn;
    None when the worker cannot vouch that a switched-off adapter leaves the base untouched, so each gets its own."""
    from khaos_attribution.engines import worker_process as W  # noqa: PLC0415
    from khaos_attribution.engines.heldout import summarise  # noqa: PLC0415
    from khaos_attribution.engines.sa3 import train as T  # noqa: PLC0415
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    names = list(adapters)
    request = {**T.heldout_request(api, clips_dir, None, seed=seed, timesteps=timesteps, checkpoint_root=checkpoint_root),
               "adapters": [str(Path(adapters[n]).resolve()) for n in names]}
    try:
        reply = W.run(api.python, T.HELDOUT_WORKER, request, cwd=api.root, env=T._env(api),
                      label="Stable Audio 3 held-out", timeout=None, required="per_adapter")
    except W.WorkerFailed as exc:
        # Every adapter loaded at once takes more memory than one: a worker that failed so is tried one at a time.
        log.warning("the one-process held-out worker failed (%s); each adapter gets its own", exc.why)
        return None
    if reply.get("fallback"):
        return None
    per_adapter = reply["per_adapter"]
    try:
        return {n: float(summarise(per_adapter[i], list(timesteps))["mean"]) for i, n in enumerate(names)}
    except (KeyError, IndexError, ValueError) as exc:
        raise GradientUnavailable(f"the Stable Audio 3 held-out worker scored the adapters incompletely: {exc}") from exc
