"""The mock engine's measure slot: everything fits, nothing is checkpointed, no accelerator to flush, and features
drawn from each file's name, so the rooms' tests can drive a whole measurement without a model."""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

SCORER_BASE_GB = 0.1
SYSTEM_RESERVE_GB = 0.0
LOW_MEMORY_OVERRIDE_ENV = "KHAOS_INFLUENCE_LOW_MEMORY"
# The adapter the mock pretends to hold: this many parameters in one tensor.
N_PARAMS = 64

__all__ = ["LOW_MEMORY_OVERRIDE_ENV", "N_PARAMS", "SCORER_BASE_GB", "SYSTEM_RESERVE_GB", "enable_checkpointing",
           "flush_accelerator", "index_rows", "memory_plan", "output_feature", "output_loss", "paired_loss",
           "parameter_count", "scorer_need_gb"]


def scorer_need_gb(duration_sec: float, *, checkpointed: bool = False) -> float:
    return SCORER_BASE_GB


def memory_plan(device_type: str, duration_sec: float, *, total_gb: float | None = None, env: dict | None = None) -> dict:
    return {"fits": "resident", "checkpointed": False, "need_gb": SCORER_BASE_GB, "total_gb": total_gb, "refusal": None}


def enable_checkpointing(module: Any) -> int:
    return 0


def flush_accelerator() -> None:
    return None


def parameter_count(handle: Any) -> tuple:
    return N_PARAMS, 1, "mock"


def _feature(path, dim: int, noise_seed: int):
    import numpy as np  # noqa: PLC0415
    seed = (zlib.crc32(Path(path).name.encode("utf-8")) + int(noise_seed)) % (2 ** 32)
    return np.random.default_rng(seed).standard_normal(int(dim)).astype(np.float32) + 0.1


def index_rows(handle: Any, files, *, projection, timesteps, noise_seed: int, target: str = "vnorm",
               timestep_norm: bool = False, param_filter: str | None = None, latent_transform: dict | None = None,
               latent_pool: int | None = None, noise_draws: int | None = None, noise_shared: bool = False,
               param_groups: str | None = None, groups_out=None, on_row=None) -> tuple:
    """(matrix, rows) of an index in the contract's shapes: one unit row per file, the same for the same name and seed."""
    import numpy as np  # noqa: PLC0415
    from khaos_attribution import gradient as G  # noqa: PLC0415
    if param_groups or groups_out:
        raise G.GradientUnavailable("block sketches are not built on the mock's measure")
    rows, vectors = [], []
    for i, path in enumerate(files):
        v = _feature(path, projection.dim, noise_seed)
        norm = float(np.linalg.norm(v))
        vectors.append(v / norm)
        rows.append({"row": i, "pair_id": Path(path).stem, "track_id": G.track_of_pair(Path(path).stem),
                     "gradient_norm": round(norm, 8)})
        if on_row is not None:
            on_row(i, Path(path), norm)
    matrix = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, int(projection.dim)), np.float32)
    return matrix, rows


def output_feature(handle: Any, index_meta: dict, files, *, groups_out=None):
    import numpy as np  # noqa: PLC0415
    from khaos_attribution import gradient as G  # noqa: PLC0415
    if groups_out:
        raise G.GradientUnavailable("block sketches are not built on the mock's measure")
    dim = int(index_meta["projection"]["dim"])
    return np.mean([_feature(f, dim, index_meta["noise_seed"]) for f in files], axis=0).astype(np.float32)


def paired_loss(handle: Any, index_meta: dict, files) -> dict:
    """The output's loss with the adapter and without it: lower with, as a trained adapter's is."""
    spread = sum(zlib.crc32(Path(f).name.encode("utf-8")) % 100 for f in files) / (100.0 * max(1, len(files)))
    return {"with": 0.4 + 0.1 * spread, "without": 0.6 + 0.1 * spread}


def output_loss(handle: Any, index_meta: dict, files) -> float:
    return paired_loss(handle, index_meta, files)["with"]
