"""The track-level rights contract and the attribution estimate, including
the money maths the schema language cannot express."""

import copy

import pytest

from khaos_attribution import (
    ATTRIBUTION_ESTIMATE_SCHEMA_VERSION,
    TRACK_RIGHTS_SCHEMA_VERSION,
    AttributionValidationError,
    validate_attribution_estimate,
    validate_track_rights,
)

GOOD_RIGHTS = {
    "schema_version": "1.0.0",
    "track_id": "trk0001",
    "title": "Glasshouse",
    "isrc": {"status": "assigned", "value": "GBAYE2500123"},
    "iswc": {"status": "not_yet_looked_up"},
    "writers": [
        {"name": "Ava Example", "role": "composer_lyricist", "share_pct": 60},
        {"name": "Ben Sample", "role": "composer", "ipi": "123456789",
         "share_pct": 40},
    ],
    "publishers": [
        {"name": "Example Music Ltd", "share_pct": 100},
    ],
    "source": "operator-entered",
    "verified_date": "2026-08-01",
}


def test_every_listed_source_validates_and_no_other_does():
    from khaos_attribution import load_schema
    listed = load_schema("track_rights.schema.json")["properties"]["source"]["enum"]
    assert listed == ["operator-entered", "society-confirmed", "distributor", "musicbrainz", "ddex_ern", "other"]
    for source in listed:
        assert validate_track_rights({**copy.deepcopy(GOOD_RIGHTS), "source": source}) is not None
    with pytest.raises(AttributionValidationError):
        validate_track_rights({**copy.deepcopy(GOOD_RIGHTS), "source": "confirmed"})


def test_good_rights_record_passes():
    assert validate_track_rights(copy.deepcopy(GOOD_RIGHTS)) is not None
    assert GOOD_RIGHTS["schema_version"] == TRACK_RIGHTS_SCHEMA_VERSION


def test_writer_shares_must_sum_to_100():
    record = copy.deepcopy(GOOD_RIGHTS)
    record["writers"][0]["share_pct"] = 55            # 55 + 40 = 95
    with pytest.raises(AttributionValidationError, match="sum to 95"):
        validate_track_rights(record)


def test_publishers_may_be_empty_but_not_partial():
    """An empty publisher pool is legitimate; a partial one is an error."""
    record = copy.deepcopy(GOOD_RIGHTS)
    record["publishers"] = []
    validate_track_rights(record)

    record["publishers"] = [{"name": "Example Music Ltd", "share_pct": 50}]
    with pytest.raises(AttributionValidationError, match="publishers"):
        validate_track_rights(record)


def test_writers_may_not_be_empty():
    record = copy.deepcopy(GOOD_RIGHTS)
    record["writers"] = []
    with pytest.raises(AttributionValidationError):
        validate_track_rights(record)


def test_isrc_format_is_enforced():
    record = copy.deepcopy(GOOD_RIGHTS)
    record["isrc"] = {"status": "assigned", "value": "not-an-isrc"}
    with pytest.raises(AttributionValidationError):
        validate_track_rights(record)


def test_iswc_format_is_enforced():
    record = copy.deepcopy(GOOD_RIGHTS)
    record["iswc"] = {"status": "assigned", "value": "T-12345-X"}
    with pytest.raises(AttributionValidationError):
        validate_track_rights(record)


def test_assigned_code_requires_a_value():
    """Either the code is known, or the status says why not."""
    record = copy.deepcopy(GOOD_RIGHTS)
    record["isrc"] = {"status": "assigned"}
    with pytest.raises(AttributionValidationError):
        validate_track_rights(record)


def test_absence_is_explicit_not_omitted():
    """A record without the isrc/iswc fields must not validate."""
    record = copy.deepcopy(GOOD_RIGHTS)
    del record["iswc"]
    with pytest.raises(AttributionValidationError):
        validate_track_rights(record)


# ── the estimate ─────────────────────────────────────────────────────────────

GOOD_ESTIMATE = {
    "schema_version": "1.1.0",
    "generation_id": "abc123",
    "artist_id": "artist-0001",
    "adapter_version": "run_20260810",
    "created_at": "2026-08-17T12:00:00Z",
    "method": {
        "estimator_version": "0.1.0",
        "exposure_basis": "training segments per track",
        "embedding_model": "laion/larger_clap_music",
        "embedding_version": "0.2.0",
        "similarity_informative": True,
        "temperature": 0.05,
    },
    "influence": [
        {"track_id": "trk0001", "title": "Glasshouse",
         "exposure_share_pct": 50.0, "similarity_share_pct": 70.0,
         "blended_share_pct": 60.0, "share_range_pct": [50.0, 70.0]},
        {"track_id": "trk0002", "title": "Northern Line",
         "exposure_share_pct": 50.0, "similarity_share_pct": 30.0,
         "blended_share_pct": 40.0, "share_range_pct": [30.0, 50.0]},
    ],
    "splits": {
        "writers": [
            {"name": "Ava Example", "share_pct": 100.0,
             "share_range_pct": [100.0, 100.0]},
        ],
        "publishers": [],
        "unattributed_pct": 0.0,
    },
    "caveats": ["This is an estimate, not a legal statement of ownership."],
}


def test_good_estimate_passes():
    assert validate_attribution_estimate(copy.deepcopy(GOOD_ESTIMATE)) is not None
    assert GOOD_ESTIMATE["schema_version"] == ATTRIBUTION_ESTIMATE_SCHEMA_VERSION


def test_influence_must_account_for_the_whole_output():
    record = copy.deepcopy(GOOD_ESTIMATE)
    record["influence"][0]["blended_share_pct"] = 30.0     # 30 + 40 = 70
    record["influence"][0]["share_range_pct"] = [20.0, 70.0]
    with pytest.raises(AttributionValidationError, match="sums to 70"):
        validate_attribution_estimate(record)


def test_a_point_outside_its_own_range_is_rejected():
    """A blended share outside its own range is a claim no method made."""
    record = copy.deepcopy(GOOD_ESTIMATE)
    record["influence"][0]["share_range_pct"] = [10.0, 20.0]
    with pytest.raises(AttributionValidationError, match="outside its own range"):
        validate_attribution_estimate(record)


def test_similarity_share_may_be_null_but_not_missing():
    """An uninformative similarity share is null, not absent."""
    record = copy.deepcopy(GOOD_ESTIMATE)
    for entry in record["influence"]:
        entry["similarity_share_pct"] = None
        entry["blended_share_pct"] = entry["exposure_share_pct"]
        entry["share_range_pct"] = [entry["exposure_share_pct"]] * 2
    record["method"]["similarity_informative"] = False
    record["method"]["temperature"] = None
    validate_attribution_estimate(record)

    del record["influence"][0]["similarity_share_pct"]
    with pytest.raises(AttributionValidationError):
        validate_attribution_estimate(record)


def test_the_method_block_is_required():
    record = copy.deepcopy(GOOD_ESTIMATE)
    del record["method"]
    with pytest.raises(AttributionValidationError):
        validate_attribution_estimate(record)


def test_code_errors_speak_english_not_regex():
    """Code errors are reported in words, never as the schema regex."""
    record = copy.deepcopy(GOOD_RIGHTS)
    record["iswc"] = {"status": "assigned", "value": ""}
    with pytest.raises(AttributionValidationError) as exc:
        validate_track_rights(record)
    assert "carries no code" in str(exc.value)
    assert "does not match" not in str(exc.value)

    record["iswc"] = {"status": "assigned", "value": "T-12345-X"}
    with pytest.raises(AttributionValidationError) as exc:
        validate_track_rights(record)
    assert "T-123456789-1" in str(exc.value), "the message must show the shape"

    record["iswc"] = {"status": "not_yet_looked_up", "value": "T-123456789-1"}
    with pytest.raises(AttributionValidationError) as exc:
        validate_track_rights(record)
    assert "status says" in str(exc.value)


def test_empty_code_string_under_any_status_speaks_english():
    """An empty code string under a non-assigned status must not leak the regex."""
    record = copy.deepcopy(GOOD_RIGHTS)
    record["iswc"] = {"status": "none_assigned", "value": ""}
    with pytest.raises(AttributionValidationError) as exc:
        validate_track_rights(record)
    assert "does not match" not in str(exc.value)
    assert "empty code" in str(exc.value)


def test_masters_pool_is_optional_but_must_add_up():
    """The masters pool is optional, but a partial pool is an error."""
    record = copy.deepcopy(GOOD_RIGHTS)
    validate_track_rights(record)                     # no masters: fine
    record["masters"] = [{"name": "Example Records", "share_pct": 100}]
    validate_track_rights(record)
    record["masters"] = [{"name": "Example Records", "share_pct": 60}]
    with pytest.raises(AttributionValidationError, match="masters"):
        validate_track_rights(record)


def test_performers_are_credits_not_money():
    record = copy.deepcopy(GOOD_RIGHTS)
    record["performers"] = [{"name": "Ava Example", "role": "vocals"},
                            {"name": "Ben Sample", "role": "drums"}]
    validate_track_rights(record)
    record["performers"] = [{"name": "Ava Example", "role": "vocals",
                             "share_pct": 50}]
    with pytest.raises(AttributionValidationError):
        validate_track_rights(record)                 # shares are refused


def test_model_card_structured_consent_is_optional_and_validated():
    from khaos_attribution import validate_model_card
    card = {
        "schema_version": "1.0.0", "artist_name": "Ava Example",
        "artist_id": "artist-0001",
        "consent_statement": "Ava consents to training on the listed tracks.",
        "training_catalogue": [{"title": "Glasshouse", "duration": 214.5}],
        "training_date": "2026-07-01", "adapter_version": "0.3.1",
        "base_model": "khaos-audio-base-1", "base_model_licence": "MIT",
    }
    validate_model_card(copy.deepcopy(card))          # old cards stay valid
    card.update({
        "credit_line": "Ava Example",
        "phonogram_notice": "℗ 2026 Ava Example",
        "copyright_notice": "© 2026 Ava Example",
        "consent": {"party": "Ava Example", "capacity": "artist",
                    "scope": {"train": True, "generate": "operator_only",
                              "commercial_outputs": False},
                    "date": "2026-08-19", "text_version": "1"},
    })
    validate_model_card(copy.deepcopy(card))
    card["consent"]["capacity"] = "someone"
    with pytest.raises(AttributionValidationError):
        validate_model_card(card)
