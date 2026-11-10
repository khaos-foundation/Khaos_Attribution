import copy

import pytest

from khaos_attribution import (
    MODEL_CARD_SCHEMA_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    AttributionValidationError,
    load_schema,
    validate_model_card,
    validate_provenance_record,
)

GOOD_PROVENANCE_RECORD = {
    "schema_version": "1.0.0",
    "artist_name": "Ava Example",
    "artist_id": "artist-0001",
    "adapter_version": "0.3.1",
    "adapter_hash": "a3f5c9d871be4402913377aa5cc0e1b2",
    "prompt": "A slow ambient piece with tape hiss and distant piano.",
    "timestamp": "2026-08-10T12:00:00Z",
    "base_model": "khaos-audio-base-1",
    "base_model_licence": "CC-BY-NC-4.0",
    "watermark_id": None,
}

GOOD_MODEL_CARD = {
    "schema_version": "1.0.0",
    "artist_name": "Ava Example",
    "artist_id": "artist-0001",
    "consent_statement": "I, Ava Example, consent to an adapter being trained on the listed tracks.",
    "training_catalogue": [
        {"title": "Glasshouse", "duration": 214.5},
        {"title": "Northern Line", "duration": 187.0},
    ],
    "training_date": "2026-07-01",
    "adapter_version": "0.3.1",
    "base_model": "khaos-audio-base-1",
    "base_model_licence": "CC-BY-NC-4.0",
}


def test_good_provenance_record_passes():
    validate_provenance_record(GOOD_PROVENANCE_RECORD)


def test_provenance_record_with_integer_watermark_passes():
    record = copy.deepcopy(GOOD_PROVENANCE_RECORD)
    record["watermark_id"] = 65535
    validate_provenance_record(record)


def test_bad_provenance_record_rejected():
    record = copy.deepcopy(GOOD_PROVENANCE_RECORD)
    del record["adapter_hash"]
    record["watermark_id"] = 70000
    with pytest.raises(AttributionValidationError) as excinfo:
        validate_provenance_record(record)
    message = str(excinfo.value)
    assert "adapter_hash" in message
    assert "watermark_id" in message


def test_derived_output_lineage_fields_pass():
    record = copy.deepcopy(GOOD_PROVENANCE_RECORD)
    record["generation_mode"] = "retake"
    record["source_generation_id"] = "a1b2c3d4e5f60718"
    validate_provenance_record(record)


def test_fresh_generation_lineage_defaults_pass():
    record = copy.deepcopy(GOOD_PROVENANCE_RECORD)
    record["generation_mode"] = "text2music"
    record["source_generation_id"] = None
    validate_provenance_record(record)


# A literal v0.1.0 record: no lineage or watermark keys. Frozen deliberately;
# it guards backwards compatibility.
V010_PROVENANCE_RECORD = {
    "schema_version": "1.0.0",
    "artist_name": "Ava Example",
    "artist_id": "artist-0001",
    "adapter_version": "0.1.0",
    "adapter_hash": "a3f5c9d871be4402913377aa5cc0e1b2",
    "prompt": "A slow ambient piece with tape hiss and distant piano.",
    "timestamp": "2026-01-15T12:00:00Z",
    "base_model": "khaos-audio-base-1",
    "base_model_licence": "CC-BY-NC-4.0",
}


def test_v010_record_without_lineage_fields_still_valid():
    # Records written before v0.2.0 must keep validating unchanged.
    validate_provenance_record(copy.deepcopy(V010_PROVENANCE_RECORD))


def test_unknown_generation_mode_rejected():
    record = copy.deepcopy(GOOD_PROVENANCE_RECORD)
    record["generation_mode"] = "remix"
    with pytest.raises(AttributionValidationError) as excinfo:
        validate_provenance_record(record)
    assert "generation_mode" in str(excinfo.value) or "remix" in str(excinfo.value)


def test_good_model_card_passes():
    validate_model_card(GOOD_MODEL_CARD)


def test_bad_model_card_rejected():
    card = copy.deepcopy(GOOD_MODEL_CARD)
    del card["consent_statement"]
    card["training_catalogue"] = [{"title": "Glasshouse"}]
    with pytest.raises(AttributionValidationError) as excinfo:
        validate_model_card(card)
    message = str(excinfo.value)
    assert "consent_statement" in message
    assert "duration" in message


def test_wrong_provenance_schema_version_rejected():
    record = copy.deepcopy(GOOD_PROVENANCE_RECORD)
    record["schema_version"] = "9.9.9"
    with pytest.raises(AttributionValidationError) as excinfo:
        validate_provenance_record(record)
    assert "schema_version" in str(excinfo.value)


def test_wrong_model_card_schema_version_rejected():
    card = copy.deepcopy(GOOD_MODEL_CARD)
    card["schema_version"] = "9.9.9"
    with pytest.raises(AttributionValidationError) as excinfo:
        validate_model_card(card)
    assert "schema_version" in str(excinfo.value)


def test_exported_constants_match_schema_const():
    provenance_schema = load_schema("provenance_record.schema.json")
    card_schema = load_schema("model_card.schema.json")
    assert (
        PROVENANCE_SCHEMA_VERSION
        == provenance_schema["properties"]["schema_version"]["const"]
    )
    # The card schema accepts both versions; writers stamp the newest.
    accepted = card_schema["properties"]["schema_version"]["enum"]
    assert MODEL_CARD_SCHEMA_VERSION == max(accepted)
    assert "1.0.0" in accepted


def test_catalogue_vocal_flag_optional_and_boolean():
    # 'vocal' is optional but must be a boolean when present.
    card = copy.deepcopy(GOOD_MODEL_CARD)
    card["training_catalogue"] = [
        {"title": "Sung one", "duration": 201.5, "vocal": True},
        {"title": "Instrumental one", "duration": 187.0, "vocal": False},
        {"title": "Pre-0.4.1 entry", "duration": 90.0},
    ]
    validate_model_card(card)

    card["training_catalogue"][0]["vocal"] = "yes"
    with pytest.raises(AttributionValidationError):
        validate_model_card(card)


def test_tombstone_validates_and_refuses_garbage():
    from khaos_attribution import AttributionValidationError, validate_tombstone
    good = {"schema_version": "1.0.0", "generation_id": "g1",
            "artist_id": "art", "content_sha256": "0" * 64,
            "deleted_at_utc": "2026-08-20T12:00:00Z",
            "watermark_id": 1782, "run_id": "run_x",
            "fingerprint_version": "v1", "fingerprints": [[1, 0], [2, 150]]}
    validate_tombstone(good)
    import pytest
    with pytest.raises(AttributionValidationError):
        validate_tombstone({**good, "content_sha256": "short"})
    with pytest.raises(AttributionValidationError):
        validate_tombstone({k: v for k, v in good.items()
                            if k != "deleted_at_utc"})


def test_model_card_accepts_watermark_id_and_both_versions():
    # 1.1.0 adds the optional field; 1.0.0 cards must keep validating.
    base = copy.deepcopy(GOOD_MODEL_CARD)
    validate_model_card(base)                       # 1.0.0, no watermark_id
    card = copy.deepcopy(GOOD_MODEL_CARD)
    card["schema_version"] = "1.1.0"
    card["watermark_id"] = 52464
    validate_model_card(card)
    bad = copy.deepcopy(GOOD_MODEL_CARD)
    bad["watermark_id"] = 99999999
    with pytest.raises(AttributionValidationError):
        validate_model_card(bad)


def test_a_document_whose_money_contradicts_its_influence_is_refused():
    """Writers plus unattributed must be the whole; on measured influence the shown share is the measured one."""
    import copy
    from khaos_attribution import validate_attribution_estimate, AttributionValidationError
    from khaos_attribution.influence import InfluenceIndex  # noqa: F401
    base = {"schema_version": "1.1.0", "generation_id": "g", "artist_id": "a", "adapter_version": "r",
            "created_at": "2026-10-07T00:00:00+00:00",
            "method": {"kind": "dtrak", "estimator_version": "0.1.0", "exposure_basis": "x", "similarity_informative": False},
            "influence": [{"track_id": "t", "title": "T", "exposure_share_pct": 100.0, "similarity_share_pct": None,
                           "influence_share_pct": 100.0, "blended_share_pct": 100.0, "share_range_pct": [96.7, 100.0]}],
            "splits": {"writers": [{"name": "W", "share_pct": 100.0, "share_range_pct": [96.7, 100.0]}],
                       "publishers": [], "masters": [], "unattributed_pct": 0.0},
            "caveats": []}
    assert validate_attribution_estimate(copy.deepcopy(base))
    lying = copy.deepcopy(base); lying["splits"]["unattributed_pct"] = 100.0
    with pytest.raises(AttributionValidationError, match="plus unattributed"):
        validate_attribution_estimate(lying)
    lying = copy.deepcopy(base); lying["influence"][0]["influence_share_pct"] = 40.0
    with pytest.raises(AttributionValidationError, match="not the share the money follows"):
        validate_attribution_estimate(lying)


def test_a_rights_record_with_a_nan_share_is_refused():
    from khaos_attribution import validate_track_rights, AttributionValidationError
    rec = {"schema_version": "1.0.0", "track_id": "t", "title": "T", "isrc": {"status": "not_yet_looked_up"},
           "iswc": {"status": "not_yet_looked_up"}, "writers": [{"name": "W", "role": "composer", "share_pct": float("nan")}],
           "publishers": [], "source": "operator-entered", "verified_date": "2026-08-01"}
    with pytest.raises(AttributionValidationError, match="not a finite number"):
        validate_track_rights(rec)


def test_a_record_from_a_newer_contract_says_so_in_plain_words():
    """A room one version behind names the cause, not the validator's 'additional properties are not allowed'."""
    from khaos_attribution import WrittenByNewerContract, validate_attribution_estimate, validate_tombstone, validate_track_rights
    card = dict(GOOD_MODEL_CARD, schema_version="1.2.0", a_field_of_the_newer_contract=1)
    with pytest.raises(WrittenByNewerContract) as newer:
        validate_model_card(card)
    said = str(newer.value)
    assert "written by a newer contract" in said and "schema_version 1.2.0" in said and "reads up to 1.1.0" in said
    assert "Additional properties" not in said and isinstance(newer.value, AttributionValidationError)
    with pytest.raises(WrittenByNewerContract, match="reads up to 1.0.0"):
        validate_provenance_record(dict(GOOD_PROVENANCE_RECORD, schema_version="2.0.0"))
    for validate in (validate_track_rights, validate_attribution_estimate, validate_tombstone):
        with pytest.raises(WrittenByNewerContract, match="newer contract"):
            validate({"schema_version": "7.0.0"})


def test_an_older_or_malformed_version_is_not_called_newer():
    from khaos_attribution import WrittenByNewerContract
    for version in ("0.9.0", "one", None, 3):
        with pytest.raises(AttributionValidationError) as bad:
            validate_provenance_record(dict(GOOD_PROVENANCE_RECORD, schema_version=version))
        assert not isinstance(bad.value, WrittenByNewerContract) and "schema_version" in str(bad.value)
    validate_model_card(dict(GOOD_MODEL_CARD, schema_version="1.0.0"))


def test_an_unknown_field_under_a_known_version_is_named_in_plain_words():
    with pytest.raises(AttributionValidationError) as bad:
        validate_provenance_record(dict(GOOD_PROVENANCE_RECORD, prompt_languge="en", another=1))
    said = str(bad.value)
    assert "fields this contract does not know: another, prompt_languge" in said and "newer contract" in said
    assert "Additional properties" not in said
    card = copy.deepcopy(GOOD_MODEL_CARD)
    card["training_catalogue"][0]["tempo"] = 120
    with pytest.raises(AttributionValidationError, match="at training_catalogue/0: fields this contract does not know: tempo"):
        validate_model_card(card)
