"""The blend: exposure prior × acoustic similarity, with ranges.

Pure functions over plain data; the one implementation both apps use. The
method the estimate's `method` block points at: (1) exposure prior — each
track's share of training segments; (2) similarity — top-k mean cosine per
track, softmaxed at a temperature equal to the score spread, or declared
uninformative below SPREAD_FLOOR; (3) posterior ∝ exposure × similarity;
(4) ranges — the blend with the temperature halved and doubled; (5) money —
Σ track influence × party share, with unattributed influence reported and
never renormalised away.
"""

from __future__ import annotations

import math

import numpy as np

ESTIMATOR_VERSION = "0.6.0"
# The version moves whenever the document's contents or the handling of a
# collapsed signal change, so two estimates stamped alike are alike.
TOP_K = 3            # segments per track that speak for it
SPREAD_FLOOR = 0.005  # cosine std-dev below which similarity is noise


def largest_remainder_pcts(shares: dict[str, float]) -> dict[str, float]:
    """Fractions → percentages at 4dp that sum to exactly 100.

    Naive per-entry rounding drifts past the validator's sum tolerance at
    around 200 tracks; largest-remainder keeps the sum exact at any N.
    """
    total = sum(shares.values())
    if not (math.isfinite(total) and total > 0):
        raise ValueError(f"shares that add to {total} have no percentages")
    scaled = {t: v / total * 100 for t, v in shares.items()}
    floored = {t: math.floor(v * 10_000) / 10_000 for t, v in scaled.items()}
    residue = round((100 - sum(floored.values())) * 10_000)
    # An equal remainder goes by name, so the result never depends on the order the shares were listed in.
    by_remainder = sorted(shares, key=lambda t: (-(scaled[t] - floored[t]), str(t)))
    out = dict(floored)
    for t in by_remainder[:residue]:
        out[t] = round(out[t] + 0.0001, 4)
    return {t: round(v, 4) for t, v in out.items()}


def exposure_shares(segments_per_track: dict[str, int]) -> dict[str, float]:
    """Fraction of training exposure per track (sums to 1)."""
    total = sum(segments_per_track.values())
    if total <= 0:
        return {}
    return {t: n / total for t, n in segments_per_track.items() if n > 0}


def similarity_scores(output_embedding: np.ndarray,
                      embeddings: np.ndarray,
                      row_track_ids: list[str],
                      tracks: set[str]) -> dict[str, float]:
    """Top-k mean cosine similarity per track.

    Embeddings are unit vectors on both sides, so cosine is a dot product.
    """
    sims = embeddings @ output_embedding.astype(np.float32)
    by_track: dict[str, list[float]] = {}
    for sim, track_id in zip(sims, row_track_ids):
        if track_id in tracks:
            by_track.setdefault(track_id, []).append(float(sim))
    scores = {}
    for track_id, values in by_track.items():
        values.sort(reverse=True)
        k = min(TOP_K, len(values))
        scores[track_id] = sum(values[:k]) / k
    return scores


def softmax_weights(scores: dict[str, float],
                    temperature: float) -> dict[str, float]:
    """Scores → weights at an explicit temperature (the sweep uses this)."""
    values = np.array(list(scores.values()), dtype=np.float64)
    logits = (values - values.max()) / temperature
    weights = np.exp(logits)
    weights /= weights.sum()
    return dict(zip(scores.keys(), (float(w) for w in weights)))


def similarity_weights(scores: dict[str, float]
                       ) -> tuple[dict[str, float] | None, float | None]:
    """Softmax the scores into weights, or (None, None) when uninformative.

    Temperature = the score spread itself: a catalogue whose tracks genuinely
    differ gets sharp weights, a tight cluster gets flat ones, and below the
    floor there is no signal to sharpen.
    """
    if len(scores) < 2:
        return None, None
    values = np.array(list(scores.values()), dtype=np.float64)
    spread = float(values.std())
    if spread < SPREAD_FLOOR:
        return None, None
    temperature = spread
    return softmax_weights(scores, temperature), temperature


# The default factor-2 robustness band, used when no catalogue calibration exists.
TEMPERATURE_SWEEP = (0.5, 1.0, 2.0)


def temperature_sweep_ranges(exposure: dict[str, float],
                             scores: dict[str, float],
                             temperature: float,
                             factors: tuple[float, ...] = TEMPERATURE_SWEEP
                             ) -> dict[str, tuple[float, float]]:
    """Per-track [min, max] of the blend across the temperature sweep.

    `factors` must include 1.0 so the interval contains the point estimate
    by construction and validator containment can never fail.
    """
    if 1.0 not in factors:
        factors = tuple(sorted({*factors, 1.0}))
    members = [blend_shares(exposure, softmax_weights(scores, temperature * f))
               for f in factors]
    return disagreement_ranges(members)


def blend_shares(exposure: dict[str, float],
                 sim_weights: dict[str, float] | None) -> dict[str, float]:
    """Posterior ∝ exposure × similarity; the prior alone when no signal."""
    if not sim_weights:
        return dict(exposure)
    product = {t: exposure[t] * sim_weights.get(t, 0.0) for t in exposure}
    total = sum(product.values())
    if total <= 0:
        return dict(exposure)
    return {t: v / total for t, v in product.items()}


def disagreement_ranges(per_method: list[dict[str, float]]
                        ) -> dict[str, tuple[float, float]]:
    """Per-track [min, max] across every method that produced a share."""
    tracks = set().union(*(m.keys() for m in per_method if m))
    out = {}
    for t in tracks:
        seen = [m[t] for m in per_method if m and t in m]
        out[t] = (min(seen), max(seen))
    return out


def money_splits(blended: dict[str, float],
                 ranges: dict[str, tuple[float, float]],
                 rights_by_track: dict[str, dict]) -> dict:
    """Writer and publisher shares of the output, with ranges, in percent."""
    writers: dict[str, dict] = {}
    publishers: dict[str, dict] = {}
    masters: dict[str, dict] = {}
    unattributed = 0.0

    def _accumulate(pool: dict, name: str, fraction: float,
                    share: float, lo: float, hi: float) -> None:
        entry = pool.setdefault(name, {"pct": 0.0, "lo": 0.0, "hi": 0.0})
        entry["pct"] += share * fraction / 100.0
        entry["lo"] += lo * fraction / 100.0
        entry["hi"] += hi * fraction / 100.0

    for track_id, share in blended.items():
        rights = rights_by_track.get(track_id)
        if rights is None:
            unattributed += share
            continue
        lo, hi = ranges.get(track_id, (share, share))
        for w in rights["writers"]:
            _accumulate(writers, w["name"], w["share_pct"], share, lo, hi)
        for p in rights["publishers"]:
            _accumulate(publishers, p["name"], p["share_pct"], share, lo, hi)
        for m in rights.get("masters") or []:
            _accumulate(masters, m["name"], m["share_pct"], share, lo, hi)

    def _pool(pool: dict, pcts: dict | None = None) -> list[dict]:
        # The hi bound sums independent per-track maxima and can pass 100%
        # for a party with shares in every track; nobody owns more than the whole.
        return [{"name": name,
                 "share_pct": pcts[name] if pcts is not None else round(e["pct"] * 100, 4),
                 "share_range_pct": [round(min(e["lo"], 1.0) * 100, 4),
                                     round(min(e["hi"], 1.0) * 100, 4)]}
                for name, e in sorted(pool.items(),
                                      key=lambda kv: -kv[1]["pct"])]

    # The writers and the unattributed remainder are what the money check adds up, so they are rounded together.
    money = _apportion_pcts({**{("writer", n): e["pct"] * 100 for n, e in writers.items()},
                             ("unattributed",): unattributed * 100})
    return {"writers": _pool(writers, {n: money[("writer", n)] for n in writers}),
            "publishers": _pool(publishers), "masters": _pool(masters),
            "unattributed_pct": money[("unattributed",)]}


def _apportion_pcts(pcts: dict) -> dict:
    """Percentages at four places that add up to their exact sum's own rounding: each is floored and the units left
    go to the largest remainders, equal ones by name, so none moves from its own rounding by more than its
    last digit."""
    units = {k: v * 10_000 for k, v in pcts.items()}
    if not all(math.isfinite(u) for u in units.values()):
        return {k: round(v, 4) for k, v in pcts.items()}   # left for the validator to refuse in words
    floored = {k: math.floor(u) for k, u in units.items()}
    left = round(sum(units.values())) - sum(floored.values())
    for k in sorted(units, key=lambda k: (floored[k] - units[k], str(k)))[:max(left, 0)]:
        floored[k] += 1
    return {k: n / 10_000 for k, n in floored.items()}
