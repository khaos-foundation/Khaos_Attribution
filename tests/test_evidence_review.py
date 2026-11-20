"""A published run's evidence review: what the Workshop says of tracks cleared on the importer's word, and the key an
operator's approval binds to."""
import copy

import pytest

from khaos_attribution import AttributionValidationError, evidence_review_key, validate_evidence_review

TRACK = {"track_id": "t1", "title": "Harbour", "state": "evidence_given",
         "matches": [{"title": "Harbour", "artists": ["Example Artist"], "releases": ["Night Lines"]}],
         "names": {"ours": ["Example Artist"], "theirs": ["Example Artist"], "matched": True},
         "evidence": {"kind": "distributor", "says": "released through my distributor", "note": "Example Dist, ref 12", "at_utc": "2026-11-20T10:00:00Z"},
         "workshop_review": {"by": "Chris", "at_utc": "2026-11-20T11:00:00Z"}}
GOOD = {"schema_version": "1.0.0", "artist_id": "art", "run_id": "run1", "written_at_utc": "2026-11-20T12:00:00Z", "tracks": [TRACK]}


def test_a_review_validates_and_one_with_no_tracks_does_too():
    assert validate_evidence_review(copy.deepcopy(GOOD))
    assert validate_evidence_review({**copy.deepcopy(GOOD), "tracks": []})


@pytest.mark.parametrize("change", [
    lambda d: d["tracks"].append(copy.deepcopy(d["tracks"][0])),
    lambda d: d["tracks"][0].__setitem__("state", "pending_evidence"),
    lambda d: d["tracks"][0].pop("workshop_review"),
    lambda d: d["tracks"][0]["workshop_review"].__setitem__("by", ""),
    lambda d: d["tracks"][0]["names"].__setitem__("matched", "yes"),
    lambda d: d.__setitem__("extra", 1),
])
def test_a_review_that_breaks_a_rule_is_refused(change):
    bad = copy.deepcopy(GOOD)
    change(bad)
    with pytest.raises(AttributionValidationError):
        validate_evidence_review(bad)


def test_the_key_moves_with_the_evidence_and_the_matches_and_not_with_the_names_or_the_review():
    k = evidence_review_key(TRACK)
    assert len(k) == 64 and k == evidence_review_key(copy.deepcopy(TRACK))
    for change in (lambda t: t["evidence"].__setitem__("note", "other"), lambda t: t["evidence"].__setitem__("at_utc", "2026-11-21T10:00:00Z"),
                   lambda t: t["matches"][0].__setitem__("title", "Other"), lambda t: t.__setitem__("state", "not_a_match")):
        moved = copy.deepcopy(TRACK)
        change(moved)
        assert evidence_review_key(moved) != k
    for change in (lambda t: t["names"].__setitem__("ours", ["Someone"]), lambda t: t["workshop_review"].__setitem__("by", "Other"),
                   lambda t: t.__setitem__("title", "Renamed")):
        same = copy.deepcopy(TRACK)
        change(same)
        assert evidence_review_key(same) == k


def test_the_key_of_a_recorded_review_is_the_hash_it_always_had():
    """An approval binds to this key, so moving its serialisation to canonical.compact changes no byte of it."""
    assert evidence_review_key(TRACK) == "55b8f8a4995d42bfa92d972db88fe8c2ef2907e1c3d8c5a3e1223f1b4d2777b8"
