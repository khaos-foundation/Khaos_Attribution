"""The held-out record every engine's evaluator writes: the timestep grid, the per-pair noise seed, the bucket keys
and the summary of per-pair bucket losses into pooled buckets, a mean, the high-noise band and per-track rows. The
forward that takes the losses stays with each engine."""

from __future__ import annotations

from pathlib import Path

from khaos_attribution import gradient
from khaos_attribution.gradient import pair_noise_seed
from khaos_attribution.identifiers import track_of_segment

__all__ = ["DEFAULT_BUCKETS", "HIGH_NOISE_T", "bucket_key", "bucket_timesteps", "pair_noise_seed", "summarise",
           "summary_of"]

# Buckets at or above this timestep are the high-noise band, the one that turns first when an adapter memorises.
HIGH_NOISE_T = 0.8
DEFAULT_BUCKETS = 10


def bucket_timesteps(n_buckets: int = DEFAULT_BUCKETS) -> list:
    """Bucket centres on (0, 1), ``(i + 0.5) / n``: no bucket at pure data or pure noise."""
    return gradient.bucket_timesteps(n_buckets)


def bucket_key(t: float) -> str:
    """A bucket's key, two fixed decimals, so rows from different evaluations line up as columns."""
    return f"{t:.2f}"


def summary_of(sums: dict, per_track: dict, n: int, timesteps) -> dict:
    """The record from pooled bucket sums over ``n`` pairs and per-track ``{"sums", "n_pairs"}`` rows."""
    buckets = {k: v / n for k, v in sums.items()}
    high = [buckets[bucket_key(t)] for t in timesteps if t >= HIGH_NOISE_T]
    by_track = {}
    for track, row in sorted(per_track.items()):
        tb = {k: v / row["n_pairs"] for k, v in row["sums"].items()}
        th = [tb[bucket_key(t)] for t in timesteps if t >= HIGH_NOISE_T]
        by_track[track] = {"mean": sum(tb.values()) / len(tb), "high_noise": (sum(th) / len(th)) if th else None,
                           "n_pairs": row["n_pairs"]}
    return {"buckets": buckets, "mean": sum(buckets.values()) / len(buckets),
            "high_noise": (sum(high) / len(high)) if high else None, "n_pairs": n,
            "timesteps": [float(t) for t in timesteps], "by_track": by_track}


def summarise(per_sample: dict, timesteps: list) -> dict:
    """The record from per-sample bucket losses ``{name: {bucket_key: loss}}``, as a worker returns them."""
    keys = [bucket_key(t) for t in timesteps]
    if not per_sample:
        raise ValueError("the held-out worker scored no sample")
    sums = {k: 0.0 for k in keys}
    per_track: dict = {}
    for name, row in per_sample.items():
        tr = per_track.setdefault(track_of_segment(Path(name).stem), {"sums": {k: 0.0 for k in keys}, "n_pairs": 0})
        tr["n_pairs"] += 1
        for k in keys:
            sums[k] += float(row[k])
            tr["sums"][k] += float(row[k])
    return summary_of(sums, per_track, len(per_sample), timesteps)
