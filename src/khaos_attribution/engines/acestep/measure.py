"""What measuring an output's influence on ACE-Step costs, and how the gradient is taken in less memory: the
engine's footprint and the scorer's peak from the descriptor, the plan that picks the cheapest way this machine can
score, and gradient checkpointing on the decoder's layers. Both rooms read these; the Engine Room pins them equal."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from khaos_attribution.engines.acestep import DESCRIPTOR
from khaos_attribution.engines.acestep.render import flush_accelerator

_MEMORY = DESCRIPTOR.memory
_SCORER = _MEMORY["scorer"]
# The scorer's peak in bfloat16: it grows with the output's length plain, barely when checkpointed.
SCORER_BASE_GB = float(_SCORER["base_gb"])
SCORER_GB_PER_SEC = float(_SCORER["gb_per_sec"])
SCORER_CHECKPOINTED_GB = float(_SCORER["checkpointed_gb"])
SCORER_CHECKPOINTED_GB_PER_SEC = float(_SCORER["checkpointed_gb_per_sec"])
SCORER_PROCESS_GB = float(_SCORER["process_gb"])
ENGINE_RESIDENT_GB = float(_MEMORY["resident_gb"])
ENGINE_YIELDED_GB = float(_MEMORY["yielded_gb"])
# What the machine keeps for itself whatever is scored, and the override that lets a small machine try anyway.
SYSTEM_RESERVE_GB = 6.0
LOW_MEMORY_OVERRIDE_ENV = "KHAOS_INFLUENCE_LOW_MEMORY"

__all__ = ["ENGINE_RESIDENT_GB", "ENGINE_YIELDED_GB", "LOW_MEMORY_OVERRIDE_ENV", "SCORER_BASE_GB",
           "SCORER_CHECKPOINTED_GB", "SCORER_CHECKPOINTED_GB_PER_SEC", "SCORER_GB_PER_SEC", "SCORER_PROCESS_GB",
           "SYSTEM_RESERVE_GB", "enable_checkpointing", "flush_accelerator", "index_rows", "machine_memory_gb",
           "memory_plan", "output_feature", "output_loss", "paired_loss", "parameter_count", "physical_memory_gb",
           "scorer_need_gb"]


def physical_memory_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        return None


def machine_memory_gb(device_type: str) -> float | None:
    """The memory the scorer shares: the card's on cuda, the machine's elsewhere."""
    if device_type == "cuda":
        try:
            import torch  # noqa: PLC0415
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        except Exception:  # noqa: BLE001
            return None
    return physical_memory_gb()


def scorer_need_gb(duration_sec: float, *, checkpointed: bool = False) -> float:
    """What scoring an output of this length takes at its peak."""
    d = max(float(duration_sec), 0.0)
    if checkpointed:
        return SCORER_CHECKPOINTED_GB + SCORER_CHECKPOINTED_GB_PER_SEC * d + SCORER_PROCESS_GB
    return SCORER_BASE_GB + SCORER_GB_PER_SEC * d + SCORER_PROCESS_GB


def memory_plan(device_type: str, duration_sec: float, *, total_gb: float | None = None,
                env: dict | None = None) -> dict:
    """The cheapest way this machine scores an output of this length: beside the engine, then checkpointed, then
    with the engine's models released too; ``fits`` None refuses."""
    total = machine_memory_gb(device_type) if total_gb is None else total_gb
    plain, saved = scorer_need_gb(duration_sec), scorer_need_gb(duration_sec, checkpointed=True)
    plan = {"fits": "resident", "checkpointed": False, "need_gb": round(plain, 1), "total_gb": total, "refusal": None}
    if total is None or plain + ENGINE_RESIDENT_GB + SYSTEM_RESERVE_GB <= total:
        return plan
    plan = {**plan, "checkpointed": True, "need_gb": round(saved, 1)}
    if saved + ENGINE_RESIDENT_GB + SYSTEM_RESERVE_GB <= total:
        return plan
    least = saved + ENGINE_YIELDED_GB + SYSTEM_RESERVE_GB
    if least <= total or (env if env is not None else os.environ).get(LOW_MEMORY_OVERRIDE_ENV) == "1":
        return {**plan, "fits": "yielded"}
    return {**plan, "fits": None,
            "refusal": (f"measuring a {float(duration_sec):.0f} s output needs about {least:.0f} GB with everything else "
                        f"set aside, and this machine has {total:.0f} GB")}


def enable_checkpointing(module: Any) -> int:
    """Gradient checkpointing on the decoder's layers, returning how many took it: the same gradient for less
    memory. Their contents stay in eval and the decoder's cache is off, so the recomputation is the forward pass
    again."""
    import torch  # noqa: PLC0415
    from functools import partial  # noqa: PLC0415
    from torch.utils.checkpoint import checkpoint  # noqa: PLC0415

    decoder = module.model.decoder
    config = getattr(decoder, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    device_type, dtype = str(getattr(module, "device_type", "cpu")), getattr(module, "dtype", torch.float32)
    kwargs: dict = {"use_reentrant": False}
    if device_type in ("cuda", "xpu", "mps") and dtype != torch.float32:
        # The contract takes the gradient outside its autocast block, so the recomputation needs the block restored.
        kwargs["context_fn"] = lambda: (torch.autocast(device_type=device_type, dtype=dtype),
                                        torch.autocast(device_type=device_type, dtype=dtype))
    n = 0
    for layer in decoder.modules():
        if "GradientCheckpointingLayer" not in [c.__name__ for c in type(layer).__mro__]:
            continue
        cls = type(layer)
        if not getattr(cls, "_khaos_checkpointed", False):
            # The layer's own gate reads `training`; its children follow eval() as before.
            layer.__dict__.pop("training", None)
            layer.__class__ = type(cls.__name__ + "Checkpointed", (cls,), {
                "training": property(lambda self: True, lambda self, value: None), "_khaos_checkpointed": True})
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = partial(checkpoint, **kwargs)
        n += 1
    return n


# ── the features: ACE-Step's live module through the contract's gradient maths ───────────────────────────────────

def index_rows(handle: Any, files, *, projection, timesteps, noise_seed: int, target: str = "vnorm",
               timestep_norm: bool = False, param_filter: str | None = None, latent_transform: dict | None = None,
               on_row=None) -> tuple:
    """(matrix, rows) of an index: one L2-normalised projected gradient per pair file, in file order, with each
    row's pre-normalisation norm kept so a zero-gradient pair stays visible."""
    import numpy as np  # noqa: PLC0415
    from khaos_attribution import gradient as G  # noqa: PLC0415
    if param_filter:
        raise G.GradientUnavailable("a parameter filter is not built on ACE-Step's measure")
    if latent_transform:
        raise G.GradientUnavailable("a latent transform is not built on ACE-Step's measure")
    params = G.adapter_parameters(handle)
    rows, vectors = [], []
    for i, path in enumerate(files):
        path = Path(path)
        flat = G.pair_gradient(handle, G.load_sample(path), timesteps=timesteps, target=target,
                               noise_seed=G.pair_noise_seed(noise_seed, path.name), params=params,
                               timestep_norm=timestep_norm)
        projected = G.project(flat, projection)
        norm = float(np.linalg.norm(projected))
        vectors.append(G.l2_normalise(projected))
        rows.append({"row": i, "pair_id": path.stem, "track_id": G.track_of_pair(path.stem), "gradient_norm": round(norm, 8)})
        if on_row is not None:
            on_row(i, path, norm)
    matrix = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, projection.dim), np.float32)
    return matrix, rows


def output_feature(handle: Any, index_meta: dict, files) -> Any:
    from khaos_attribution import gradient as G  # noqa: PLC0415
    return G.output_feature(handle, index_meta, list(files))


def output_loss(handle: Any, index_meta: dict, files) -> float:
    from khaos_attribution import gradient as G  # noqa: PLC0415
    return G.output_loss(handle, index_meta, list(files))


def paired_loss(handle: Any, index_meta: dict, files) -> dict:
    from khaos_attribution import gradient as G  # noqa: PLC0415
    return G.paired_loss(handle, index_meta, list(files))


def parameter_count(handle: Any) -> tuple:
    """(n_params, n_tensors, signature) of the adapter the handle holds."""
    from khaos_attribution import gradient as G  # noqa: PLC0415
    params = G.adapter_parameters(handle)
    return int(sum(int(p.numel()) for p in params)), len(params), G.parameter_signature(params)
