"""Stable Audio 3's measure slot: features through a worker in the engine's own environment, torch (CUDA) or MLX
(a Mac). The descriptor declares no platform measure yet, so no room serves a measured split on this engine."""

from __future__ import annotations

from typing import Any

from khaos_attribution.engines.sa3 import DESCRIPTOR

SYSTEM_RESERVE_GB = 6.0
LOW_MEMORY_OVERRIDE_ENV = "KHAOS_INFLUENCE_LOW_MEMORY"

__all__ = ["Handle", "LOW_MEMORY_OVERRIDE_ENV", "ROW_PREFIX", "SYSTEM_RESERVE_GB", "expected_dtype", "enable_checkpointing", "flush_accelerator",
           "index_rows", "latent_stats", "memory_plan", "output_feature", "output_loss", "paired_loss", "parameter_count", "scorer_need_gb",
           "tid_factors", "tid_invert", "tid_kernels"]


_SCORER = DESCRIPTOR.memory.get("scorer") or {}


def expected_dtype(device_type: str) -> str:
    """The precision this measure takes features at on a device type: the base as each runtime's trainer holds it,
    bfloat16 on CUDA and float16 on a Mac (the MLX port), full precision elsewhere."""
    return {"cuda": "bf16", "mps": "fp16"}.get(device_type, "fp32")


def _scorer(device_type: str) -> dict:
    """The worker's measured peak on a device type, each on a 56 s clip (the longest the Workshop cuts): CUDA's under
    "scorer", any other under "scorer_<device>" (mps: the MLX peak plus the rest of the process, memory being shared)."""
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
               "loss": ("with", "without"), "latent_stats": ("transform",),
               "tid_factors": ("tid_version", "modules", "live_tensors", "dropped_tensors"),
               "tid_invert": ("tid_version", "inverses", "factors"),
               "tid_kernels": ("kernels", "draws_done", "grid", "live_tensors", "dropped_tensors", "normaliser_zero", "seconds")}


ROW_PREFIX = "khaos-row "


def _ask(handle: Handle, request: dict, on_progress=None) -> dict:
    """The worker's reply; its ``khaos-row`` lines reach ``on_progress`` as they come, so a long index shows its
    progress while the worker runs."""
    import tempfile  # noqa: PLC0415
    from khaos_attribution.engines import contract_guard  # noqa: PLC0415
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    contract_guard.require(handle.python, engine="Stable Audio 3", engine_id=DESCRIPTOR.id, error=GradientUnavailable)
    worker = MEASURE_WORKER_MLX if handle.runtime == "mlx" else MEASURE_WORKER
    lines = []
    # stderr to a file, never a pipe: a worker that writes much to it would block while stdout is being read.
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as err:
        try:
            proc = subprocess.Popen([str(handle.python), str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=err, cwd=str(handle.root), text=True, errors="replace", env=_env(handle))
        except OSError as exc:
            raise GradientUnavailable(f"the Stable Audio 3 measure worker could not start: {exc}") from exc
        try:
            try:
                proc.stdin.write(json.dumps(request))
                proc.stdin.close()
            except BrokenPipeError:
                pass
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
        finally:
            # A worker left running would hold the device's memory after its caller has gone.
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            for pipe in (proc.stdin, proc.stdout):
                if pipe is not None and not pipe.closed:
                    pipe.close()
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
    """(matrix, rows) of an index through the worker: the matrix lands beside the files as a temp .npy; with
    ``param_groups`` each row's block sketches land at ``groups_out`` and their names beside it."""
    import json  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    if param_groups and not groups_out:
        raise GradientUnavailable("an index with block sketches needs groups_out, where they are written")
    if groups_out and not param_groups:
        raise GradientUnavailable("groups_out needs param_groups, the pattern that names the blocks")
    out = Path(files[0]).resolve().parent / ".khaos_index_rows.tmp.npy" if files else Path(".khaos_index_rows.tmp.npy")
    seen = set()

    def progress(row: dict) -> None:
        if on_row is not None and isinstance(row.get("row"), int) and 0 <= row["row"] < len(files):
            seen.add(row["row"])
            on_row(row["row"], Path(files[row["row"]]), row.get("gradient_norm"))

    r = _ask(handle, {**_base(handle, {"timesteps": timesteps, "noise_seed": noise_seed, "target": target,
                                       "timestep_norm": timestep_norm, "param_filter": param_filter,
                                       "latent_transform": latent_transform, "latent_pool": latent_pool,
                                       "noise_draws": noise_draws, "noise_shared": noise_shared,
                                       "param_groups": param_groups}, files, "index"),
                      "projection": {"dim": int(projection.dim), "seed": int(projection.seed)}, "out": str(out),
                      "groups_out": str(Path(groups_out).resolve()) if groups_out else None},
             on_progress=progress)
    from khaos_attribution.gradient import GradientUnavailable, dtype_tag  # noqa: PLC0415
    if r.get("compute_dtype") and r["compute_dtype"] != dtype_tag(handle.dtype):
        Path(r["matrix"]).unlink(missing_ok=True)
        raise GradientUnavailable(f"the worker ran at {r['compute_dtype']}, and the index would record "
                                  f"{dtype_tag(handle.dtype)}; they must be one precision")
    try:
        matrix = np.load(str(r["matrix"]))
    except (OSError, ValueError) as exc:
        raise GradientUnavailable(f"the Stable Audio 3 measure worker left no readable matrix: {exc}") from exc
    Path(r["matrix"]).unlink(missing_ok=True)
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
    """Why an index cannot be read against this measure's features, or None: it must have been built by this
    feature version, since the noise and the parameter order are part of what a row means."""
    built = index_meta.get("feature_version")
    if built != DESCRIPTOR.feature_version:
        return (f"the influence index was built by {DESCRIPTOR.name}'s measure {built or 'before it was versioned'}, "
                f"and this one is {DESCRIPTOR.feature_version}; rebuild the index")
    return None


def output_feature(handle: Handle, index_meta: dict, files, *, groups_out=None):
    """The output's projected feature; with ``groups_out`` its block sketches are written there too."""
    import numpy as np  # noqa: PLC0415
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    why = _index_refusal(index_meta)
    if why:
        raise GradientUnavailable(why)
    if groups_out and not index_meta.get("param_groups"):
        raise GradientUnavailable("this index was built without block sketches; an output's cannot be read against it")
    request = _base(handle, index_meta, files, "feature")
    if groups_out:
        request["groups_out"] = str(Path(groups_out).resolve())
    r = _ask(handle, request)
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
