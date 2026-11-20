"""The blend estimator's arithmetic: pure functions, checkable with floats."""

from __future__ import annotations

import numpy as np
import pytest

from khaos_attribution import blend


def test_exposure_follows_training_segments():
    shares = blend.exposure_shares({"a": 30, "b": 10})
    assert shares == {"a": 0.75, "b": 0.25}


def test_exposure_ignores_trackless_counts():
    assert blend.exposure_shares({"a": 0, "b": 10}) == {"b": 1.0}
    assert blend.exposure_shares({}) == {}


def test_similarity_scores_are_topk_not_max_or_mean():
    """One lucky segment must not speak for a track (max would), and a long
    track's quiet passages must not dilute it (mean would)."""
    output = np.array([1.0, 0.0], dtype=np.float32)
    embeddings = np.array([
        [1.0, 0.0],    # a: sims 1.0, 0.8, 0.6, 0.0 -> top3 mean 0.8
        [0.8, 0.6],
        [0.6, 0.8],
        [0.0, 1.0],
        [0.5, 0.866],  # b: single segment 0.5
    ], dtype=np.float32)
    rows = ["a", "a", "a", "a", "b"]
    scores = blend.similarity_scores(output, embeddings, rows, {"a", "b"})
    assert scores["a"] == pytest.approx(0.8, abs=1e-6)
    assert scores["b"] == pytest.approx(0.5, abs=1e-3)


def test_similarity_ignores_tracks_outside_the_run():
    """The store holds every track ever embedded; only the adapter's
    training tracks may influence its outputs."""
    output = np.array([1.0, 0.0], dtype=np.float32)
    embeddings = np.eye(2, dtype=np.float32)
    scores = blend.similarity_scores(output, embeddings, ["in", "out"], {"in"})
    assert set(scores) == {"in"}


def test_tight_clusters_are_declared_uninformative():
    """A 0.001 spread is noise, not track-level signal."""
    weights, temperature = blend.similarity_weights(
        {"a": 0.5001, "b": 0.5000, "c": 0.4999})
    assert weights is None and temperature is None


def test_single_track_has_no_similarity_signal():
    assert blend.similarity_weights({"only": 0.9}) == (None, None)


def test_informative_similarity_sharpens_toward_the_closer_track():
    weights, temperature = blend.similarity_weights({"near": 0.6, "far": 0.3})
    assert weights["near"] > weights["far"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert temperature == pytest.approx(0.15)


def test_blend_is_prior_times_likelihood():
    exposure = {"a": 0.8, "b": 0.2}
    sim = {"a": 0.3, "b": 0.7}
    blended = blend.blend_shares(exposure, sim)
    assert blended["a"] == pytest.approx(0.24 / 0.38)
    assert sum(blended.values()) == pytest.approx(1.0)


def test_blend_without_signal_is_the_prior():
    exposure = {"a": 0.8, "b": 0.2}
    assert blend.blend_shares(exposure, None) == exposure


def test_ranges_span_every_method_including_the_blend():
    """The product of prior and likelihood can sharpen past both — the range
    must cover the blend too, or the point lands outside its own interval."""
    exposure = {"a": 0.8, "b": 0.2}
    sim = {"a": 0.7, "b": 0.3}
    blended = blend.blend_shares(exposure, sim)   # a ≈ 0.903 — outside [0.7, 0.8]
    ranges = blend.disagreement_ranges([exposure, sim, blended])
    lo, hi = ranges["a"]
    assert lo <= blended["a"] <= hi
    assert lo == 0.7 and hi == pytest.approx(blended["a"])


def test_money_follows_influence_times_track_shares():
    blended = {"t1": 0.6, "t2": 0.4}
    ranges = {"t1": (0.5, 0.7), "t2": (0.3, 0.5)}
    rights = {
        "t1": {"writers": [{"name": "Ava", "share_pct": 50},
                           {"name": "Ben", "share_pct": 50}],
               "publishers": [{"name": "Pub Ltd", "share_pct": 100}]},
        "t2": {"writers": [{"name": "Ava", "share_pct": 100}],
               "publishers": []},
    }
    splits = blend.money_splits(blended, ranges, rights)
    by_name = {w["name"]: w for w in splits["writers"]}
    # Ava: 50% of t1 (0.6) + 100% of t2 (0.4) = 70%
    assert by_name["Ava"]["share_pct"] == pytest.approx(70.0)
    assert by_name["Ben"]["share_pct"] == pytest.approx(30.0)
    assert by_name["Ava"]["share_range_pct"][0] <= 70.0 <= by_name["Ava"]["share_range_pct"][1]
    assert splits["publishers"][0]["share_pct"] == pytest.approx(60.0)
    assert splits["unattributed_pct"] == 0.0


def test_missing_rights_become_unattributed_not_redistributed():
    """Unknown ownership is reported, never absorbed into the known writers."""
    blended = {"known": 0.7, "unknown": 0.3}
    ranges = {t: (v, v) for t, v in blended.items()}
    rights = {"known": {"writers": [{"name": "Ava", "share_pct": 100}],
                        "publishers": []}}
    splits = blend.money_splits(blended, ranges, rights)
    assert splits["unattributed_pct"] == pytest.approx(30.0)
    assert splits["writers"][0]["share_pct"] == pytest.approx(70.0), (
        "Ava's share must stay 70 — the unknown 30 is not hers to inherit")


def test_party_ranges_never_exceed_the_whole_output():
    """Summed per-track maxima can pass 100%; nobody owns more than the whole."""
    blended = {"a": 0.6, "b": 0.4}
    ranges = {"a": (0.3, 0.9), "b": (0.2, 0.8)}     # hi sum = 1.7
    rights = {t: {"writers": [{"name": "Solo", "share_pct": 100}],
                  "publishers": []} for t in blended}
    splits = blend.money_splits(blended, ranges, rights)
    (writer,) = splits["writers"]
    assert writer["share_range_pct"][1] == 100.0
    assert writer["share_range_pct"][0] <= writer["share_pct"] <= writer["share_range_pct"][1]


def test_largest_remainder_survives_a_251_track_catalogue():
    """Naive 4dp rounding drifts past the validator's sum tolerance at 201+
    tracks; largest-remainder sums to exactly 100 at any size."""
    shares = {f"t{i}": 20 / 5022 for i in range(250)}
    shares["big"] = 22 / 5022
    total = sum(shares.values())
    shares = {t: v / total for t, v in shares.items()}
    pcts = blend.largest_remainder_pcts(shares)
    assert round(sum(pcts.values()), 4) == 100.0
    assert all(abs(pcts[t] - shares[t] * 100) <= 0.0002 for t in shares)


def test_a_torn_bundle_is_refused_not_misattributed():
    """When the index and the embedding array disagree the store is torn;
    zip-truncation would silently drop tracks from similarity."""
    from khaos_attribution.estimator import build_estimate
    with pytest.raises(ValueError, match="torn"):
        build_estimate(
            generation_id="g", artist_id="a", adapter_version="r",
            run_tracks={"a", "b", "c"},
            segment_counts={"a": 1, "b": 1, "c": 1},
            n_training_pairs=3,
            embeddings=np.eye(2, dtype=np.float32),
            row_track_ids=["a", "b", "c"],
            output_embedding=np.array([1.0, 0.0], dtype=np.float32),
            rights={}, fallback_titles={},
            embedding_model=None, embedding_version=None,
            exposure_basis="test")


def test_ranges_are_temperature_sensitivity_not_ingredient_spread():
    """The range must contain the point (factor 1.0 is in the sweep) and be
    narrower than the raw exposure-to-similarity interval."""
    exposure = {"a": 0.8, "b": 0.2}
    scores = {"a": 0.30, "b": 0.60}
    weights, temperature = blend.similarity_weights(scores)
    blended = blend.blend_shares(exposure, weights)
    ranges = blend.temperature_sweep_ranges(exposure, scores, temperature)
    for track in exposure:
        lo, hi = ranges[track]
        assert lo <= blended[track] <= hi
    old_lo = min(exposure["a"], weights["a"], blended["a"])
    old_hi = max(exposure["a"], weights["a"], blended["a"])
    lo, hi = ranges["a"]
    assert (hi - lo) < (old_hi - old_lo), (
        "the sweep interval should be tighter than the ingredient spread")



def test_the_temperature_sweep_is_what_makes_the_range_a_range():
    """The default sweep must reach both sides of 1.0 and move the shares;
    a single-factor sweep must give the degenerate interval."""
    assert min(blend.TEMPERATURE_SWEEP) < 1.0 < max(blend.TEMPERATURE_SWEEP)
    exposure = {"a": 0.5, "b": 0.3, "c": 0.2}
    scores = {"a": 0.30, "b": 0.60, "c": 0.45}
    _, temperature = blend.similarity_weights(scores)
    swept = blend.temperature_sweep_ranges(exposure, scores, temperature)
    assert all(hi > lo for lo, hi in swept.values()), "the sweep moved nothing"
    single = blend.temperature_sweep_ranges(exposure, scores, temperature, factors=(1.0,))
    assert all(hi == lo for lo, hi in single.values())
    # Halving the temperature sharpens: the best-scoring track's high bound
    # is its share at the sharp end, and the worst's low bound at the same end.
    sharp = blend.blend_shares(exposure, blend.softmax_weights(scores, temperature * 0.5))
    assert swept["b"][1] == pytest.approx(sharp["b"])
    assert swept["a"][0] == pytest.approx(sharp["a"])


def test_largest_remainder_refuses_shares_that_add_to_nothing():
    """A zero or not-a-number total has no percentages; it is refused, never spread as 0.0001 each."""
    import math
    import pytest
    for shares in ({"a": 0.0, "b": 0.0}, {"a": math.nan, "b": 1.0}):
        with pytest.raises(ValueError, match="add to"):
            blend.largest_remainder_pcts(shares)


def test_the_money_is_rounded_together_and_no_share_moves_past_its_last_digit():
    """Writers and the unattributed remainder add up to their exact sum's own rounding, each within 0.0001 of its own."""
    import random
    from decimal import Decimal
    rng = random.Random(3)
    for _ in range(500):
        tracks = [f"t{i}" for i in range(rng.randint(1, 30))]
        weights = [rng.random() + 0.001 for _ in tracks]
        blended = {t: w / sum(weights) for t, w in zip(tracks, weights)}
        rights = {t: {"writers": [{"name": f"w{rng.randint(0, 9)}{j}", "share_pct": s}
                                  for j, s in enumerate(rng.choice([[100.0], [50.0, 50.0], [33.3333, 33.3333, 33.3334],
                                                                    [14.2857] * 6 + [14.2858], [60.0, 39.9901]]))],
                      "publishers": []} for t in tracks if rng.random() < 0.8}
        splits = blend.money_splits(blended, {}, rights)
        exact = {}
        for t, share in blended.items():
            for w in (rights.get(t) or {}).get("writers", []):
                exact[w["name"]] = exact.get(w["name"], 0.0) + share * w["share_pct"]
        exact_unattributed = sum(s for t, s in blended.items() if t not in rights) * 100
        units = [round(Decimal(repr(w["share_pct"])) * 10_000) for w in splits["writers"]]
        units.append(round(Decimal(repr(splits["unattributed_pct"])) * 10_000))
        assert sum(units) == round((sum(exact.values()) + exact_unattributed) * 10_000)
        for w in splits["writers"]:
            assert abs(w["share_pct"] - exact[w["name"]]) <= 0.0001 + 1e-9, w
        assert abs(splits["unattributed_pct"] - exact_unattributed) <= 0.0001 + 1e-9


def test_equal_remainders_go_by_name_whatever_order_the_shares_came_in():
    """Two documents from the same shares listed in another order carry the same numbers."""
    import itertools
    shares = {"c": 1 / 3, "a": 1 / 3, "b": 1 / 3}
    seen = {tuple(sorted(blend.largest_remainder_pcts(dict(order)).items()))
            for order in itertools.permutations(shares.items())}
    assert seen == {(("a", 33.3334), ("b", 33.3333), ("c", 33.3333))}
    rights = {t: {"writers": [{"name": w, "share_pct": 100.0}], "publishers": []} for t, w in (("t1", "Zed"), ("t2", "Amy"), ("t3", "Kit"))}
    blended = {"t1": 1 / 3, "t2": 1 / 3, "t3": 1 / 3}
    money = {tuple((w["name"], w["share_pct"]) for w in sorted(blend.money_splits(dict(order), {}, rights)["writers"],
                                                                key=lambda w: w["name"]))
             for order in itertools.permutations(blended.items())}
    assert money == {(("Amy", 33.3334), ("Kit", 33.3333), ("Zed", 33.3333))}
