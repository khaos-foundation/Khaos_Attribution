"""Group removal (W): groups by writer split pattern, the positive-rise split, its money, its record and its schema."""

import copy

import pytest

from khaos_attribution import group_removal as G
from khaos_attribution.validation import AttributionValidationError, validate_attribution_estimate


def _r(*writers, publishers=(), masters=()):
    return {"writers": [{"name": n, "share_pct": p} for n, p in writers],
            "publishers": [{"name": n, "share_pct": p} for n, p in publishers],
            "masters": [{"name": n, "share_pct": p} for n, p in masters]}


RIGHTS = {"a": _r(("X", 100)), "b": _r(("X", 50), ("Y", 50)), "c": _r(("X", 100.0)), "d": _r(("Y", 100)),
          "e": _r(("Y", 50), ("X", 50))}


def test_tracks_with_the_same_writer_split_share_a_group_whatever_the_order_or_number_format():
    """Writer order and 100 against 100.0 do not split a group; the partition covers every track once."""
    groups = G.split_groups(list(RIGHTS), RIGHTS)
    assert sorted(map(tuple, groups)) == [("a", "c"), ("b", "e"), ("d",)]
    assert sorted(t for g in groups for t in g) == sorted(RIGHTS)


def test_the_partition_hash_ignores_writer_names_and_group_order():
    """A rename that keeps the partition keeps the hash; moving a track between groups changes it."""
    groups = G.split_groups(list(RIGHTS), RIGHTS)
    renamed = {t: _r(*((("Z" if w["name"] == "X" else w["name"]), w["share_pct"]) for w in r["writers"]))
               for t, r in RIGHTS.items()}
    assert G.partition_sha256(G.split_groups(list(renamed), renamed)) == G.partition_sha256(groups)
    assert G.partition_sha256(list(reversed(groups))) == G.partition_sha256(groups)
    moved = {**RIGHTS, "c": _r(("Y", 100))}
    assert G.partition_sha256(G.split_groups(list(moved), moved)) != G.partition_sha256(groups)


def test_a_groups_share_is_its_positive_rise_over_the_served_adapter_normalised():
    """Rises 0.3, 0.1 and a fall give 0.75, 0.25, 0; matching the research reader's positive split."""
    assert G.group_shares(1.0, [1.3, 1.1, 0.9]) == pytest.approx([0.75, 0.25, 0.0])


def test_no_positive_rise_is_not_attributable():
    """A piece no group's removal raises carries no split."""
    with pytest.raises(G.NoRise):
        G.group_shares(1.0, [1.0, 0.8])


def test_money_follows_the_groups_writers_and_spreads_publishers_by_the_training_share():
    """Writers are exact per group; a group's publisher money inside it follows the training-share rule."""
    rights = {"a": _r(("X", 100), publishers=(("P", 100),)), "c": _r(("X", 100), publishers=(("Q", 100),)),
              "d": _r(("Y", 100), publishers=(("P", 100),))}
    groups = G.split_groups(list(rights), rights)
    shares = [0.6, 0.4] if groups[0] == ["a", "c"] else [0.4, 0.6]
    m = G.money(shares, groups, rights, {"a": 3.0, "c": 1.0, "d": 5.0})
    writers = {w["name"]: w["share_pct"] for w in m["writers"]}
    pubs = {p["name"]: p["share_pct"] for p in m["publishers"]}
    assert writers == pytest.approx({"X": 60.0, "Y": 40.0})
    assert pubs == pytest.approx({"P": 45.0 + 40.0, "Q": 15.0})
    even = G.track_shares([1.0], [["a", "c"]], None)
    assert even == pytest.approx({"a": 0.5, "c": 0.5})


def _record(groups):
    return G.build_record(groups=groups, served_sha256="s" * 64, served_epoch=400,
                          adapters_sha256={i: "a" * 64 for i in G.group_ids(groups)}, noise_seed=42, device="cuda",
                          compute_dtype="bf16")


def test_a_record_reads_back_its_groups_and_refuses_a_broken_one():
    """The record's groups come back in order; a changed group, a missing adapter or another schema refuse."""
    groups = G.split_groups(list(RIGHTS), RIGHTS)
    rec = _record(groups)
    assert G.read_record(rec) == groups
    bad = copy.deepcopy(rec)
    bad["groups"]["g1"] = bad["groups"]["g1"] + ["z"]
    with pytest.raises(G.RecordRefused, match="partition"):
        G.read_record(bad)
    bad = copy.deepcopy(rec)
    bad["adapters_sha256"].pop("g2")
    with pytest.raises(G.RecordRefused, match="one adapter per group"):
        G.read_record(bad)
    with pytest.raises(G.RecordRefused, match="schema"):
        G.read_record({**rec, "schema": "khaos.group_removal/9"})
    with pytest.raises(G.RecordRefused):
        G.build_record(groups=groups, served_sha256="s", served_epoch=1, adapters_sha256={"g1": "a"}, noise_seed=1,
                       device="cuda", compute_dtype="bf16")


def test_a_record_goes_stale_with_the_served_adapter_or_the_partition_not_the_names():
    """A new served checkpoint or a regrouping makes it stale; renaming a writer does not."""
    groups = G.split_groups(list(RIGHTS), RIGHTS)
    rec = _record(groups)
    assert G.stale_because(rec, served_sha256="s" * 64, rights_by_track=RIGHTS) is None
    assert "served adapter" in G.stale_because(rec, served_sha256="t" * 64, rights_by_track=RIGHTS)
    renamed = {**RIGHTS, "d": _r(("W", 100))}
    assert G.stale_because(rec, served_sha256="s" * 64, rights_by_track=renamed) is None
    moved = {**RIGHTS, "c": _r(("Y", 100))}
    assert "differently" in G.stale_because(rec, served_sha256="s" * 64, rights_by_track=moved)


def _estimate(**over):
    doc = {"schema_version": G.ESTIMATE_SCHEMA_VERSION, "generation_id": "gen", "artist_id": "artist",
           "adapter_version": "v1", "created_at": "2026-11-16T00:00:00Z",
           "method": {"estimator_version": "0", "exposure_basis": "none", "similarity_informative": False,
                      "kind": G.METHOD_KIND},
           "groups": [{"group_id": "g1", "track_ids": ["a", "c"], "share_pct": 60.0},
                      {"group_id": "g2", "track_ids": ["d"], "share_pct": 40.0}],
           "splits": {"writers": [{"name": "X", "share_pct": 60.0, "share_range_pct": [60.0, 60.0]},
                                  {"name": "Y", "share_pct": 40.0, "share_range_pct": [40.0, 40.0]}],
                      "publishers": [], "masters": [], "unattributed_pct": 0.0},
           "caveats": []}
    doc.update(over)
    return doc


def test_a_group_removal_estimate_validates_without_per_track_rows_and_only_at_its_own_version():
    """Group rows stand in for influence rows; they must sum to 100 and the record must say 1.2.0."""
    validate_attribution_estimate(_estimate())
    with pytest.raises(AttributionValidationError):
        validate_attribution_estimate(_estimate(schema_version="1.1.0"))
    with pytest.raises(AttributionValidationError, match="group shares"):
        validate_attribution_estimate(_estimate(groups=[{"group_id": "g1", "track_ids": ["a"], "share_pct": 90.0}]))
    no_groups = _estimate()
    no_groups.pop("groups")
    with pytest.raises(AttributionValidationError):
        validate_attribution_estimate(no_groups)


def test_another_kind_still_needs_its_per_track_rows():
    """Dropping influence from the required list is for group removal only."""
    doc = _estimate(schema_version="1.1.0")
    doc["method"] = {**doc["method"], "kind": "dtrak"}
    doc.pop("groups")
    with pytest.raises(AttributionValidationError):
        validate_attribution_estimate(doc)


def test_sa3_scores_one_piece_under_each_adapter_at_one_noise_seed(monkeypatch):
    """Every adapter is scored over the same clips with the same seed and timesteps; the mean loss comes back."""
    from khaos_attribution.engines.sa3 import measure as M
    from khaos_attribution.engines.sa3 import train as T
    calls = []

    def fake(api, clips_dir, adapter, *, seed, timesteps, checkpoint_root):
        calls.append((str(clips_dir), str(adapter), seed, tuple(timesteps), checkpoint_root))
        return {"mean": {"full": 1.0, "g1": 1.25}[str(adapter)]}

    import types

    from khaos_attribution.engines import contract_guard
    monkeypatch.setattr(contract_guard, "refusal", lambda *a, **k: None)
    monkeypatch.setattr(T, "heldout_summary", fake)
    out = M.output_losses(types.SimpleNamespace(python="/x/python"), "clips", {"full": "full", "g1": "g1"}, seed=42,
                          checkpoint_root="ck")
    assert out == {"full": 1.0, "g1": 1.25}
    assert {(c[0], c[2], c[3], c[4]) for c in calls} == {("clips", 42, tuple(T.bucket_timesteps(T.DEFAULT_BUCKETS)), "ck")}


def test_the_split_follows_the_records_groups_when_a_rename_would_reorder_them():
    """Losses are keyed by the record's group ids; renaming a writer reorders split_groups but moves no money."""
    rights = {"a": _r(("Alice", 100)), "b": _r(("Bob", 100))}
    groups = G.split_groups(list(rights), rights)
    assert groups == [["a"], ["b"]]
    rec = _record(groups)
    losses = {"served": 1.0, "g1": 1.3, "g2": 1.1}
    renamed = {"a": _r(("Zed", 100)), "b": _r(("Bob", 100))}
    assert G.split_groups(list(renamed), renamed) == [["b"], ["a"]]
    out = G.split(rec, losses, renamed)
    assert [g["track_ids"] for g in out["groups"]] == [["a"], ["b"]]
    writers = {w["name"]: w["share_pct"] for w in out["money"]["writers"]}
    assert writers == pytest.approx({"Zed": 75.0, "Bob": 25.0})
    with pytest.raises(G.RecordRefused, match="g2"):
        G.split(rec, {"served": 1.0, "g1": 1.3}, rights)


def test_a_record_with_odd_ids_or_shared_tracks_is_refused_not_crashed():
    """A key that is not gN, or a track in two groups, is a refusal."""
    rec = _record([["a"], ["b"]])
    odd = {**rec, "groups": {"g": ["a"], "g2": ["b"]}}
    with pytest.raises(G.RecordRefused):
        G.read_record(odd)
    shared = {**rec, "groups": {"g1": ["a"], "g2": ["a"]}}
    shared["partition_sha256"] = G.partition_sha256([["a"], ["a"]])
    with pytest.raises(G.RecordRefused, match="share a track"):
        G.read_record(shared)


def test_the_adapter_files_must_be_the_ones_the_record_pins():
    """A group adapter whose weights hash differs from the pin is named."""
    rec = _record([["a"], ["b"]])
    assert G.adapter_refusal(rec, {"g1": "a" * 64, "g2": "a" * 64}) is None
    assert "g2" in G.adapter_refusal(rec, {"g1": "a" * 64, "g2": "b" * 64})


def test_group_rows_belong_to_group_removal_alone_and_once_each():
    """Another kind may not carry group rows; a group id twice is refused."""
    doc = _estimate(schema_version="1.1.0")
    doc["method"] = {**doc["method"], "kind": "dtrak"}
    doc["influence"] = [{"track_id": "a", "exposure_share_pct": 100.0, "blended_share_pct": 100.0,
                         "influence_share_pct": 100.0, "share_range_pct": [100.0, 100.0]}]
    with pytest.raises(AttributionValidationError):
        validate_attribution_estimate(doc)
    twice = _estimate(groups=[{"group_id": "g1", "track_ids": ["a"], "share_pct": 50.0},
                              {"group_id": "g1", "track_ids": ["d"], "share_pct": 50.0}])
    with pytest.raises(AttributionValidationError, match="twice"):
        validate_attribution_estimate(twice)


def test_scoring_refuses_an_engine_environment_on_another_contract(monkeypatch):
    """The loss loop is guarded like every other measure call, and a worker failure is GradientUnavailable."""
    import types

    from khaos_attribution.engines import contract_guard
    from khaos_attribution.engines.sa3 import measure as M
    from khaos_attribution.engines.sa3 import train as T
    from khaos_attribution.gradient import GradientUnavailable
    api = types.SimpleNamespace(python="/x/python")
    monkeypatch.setattr(contract_guard, "refusal", lambda *a, **k: "holds contract 0.1.0")
    with pytest.raises(GradientUnavailable, match="0.1.0"):
        M.output_losses(api, "clips", {"full": "f"}, seed=42)
    monkeypatch.setattr(contract_guard, "refusal", lambda *a, **k: None)

    def boom(*a, **k):
        raise RuntimeError("the Stable Audio 3 held-out worker failed: oom")

    monkeypatch.setattr(T, "heldout_summary", boom)
    with pytest.raises(GradientUnavailable, match="oom"):
        M.output_losses(api, "clips", {"full": "f"}, seed=42)
