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


def _rights(*shares):
    return {"schema_version": "1.0.0", "track_id": "t", "title": "T", "isrc": {"status": "not_yet_looked_up"},
            "iswc": {"status": "not_yet_looked_up"},
            "writers": [{"name": f"W{i}", "role": "composer", "share_pct": s} for i, s in enumerate(shares)],
            "publishers": [], "source": "operator-entered", "verified_date": "2026-08-01"}


def _main_refuses(total) -> bool:
    """The comparison every stored record was first validated under, as it stands on main."""
    return abs(total - 100.0) > 0.01


def _verdict(validate, record) -> bool:
    try:
        return bool(validate(record))
    except AttributionValidationError:
        return False


@pytest.mark.parametrize("shares", [(33.33, 33.33, 33.33), (50.0, 49.99), (50.0, 50.01), (60.0, 40.01), (99.99,),
                                    (50.0, 49.98), (50.0, 50.02), (50.0, 49.99004), (33.33, 33.33, 33.34),
                                    (50.0, 49.9901), (33.3333, 33.3333, 33.3333)])
def test_a_rights_sum_is_judged_exactly_as_it_always_was(shares):
    """Stored rights are validated again when read, so a sum is accepted or refused exactly as main's comparison did."""
    from khaos_attribution import validate_track_rights
    assert _verdict(validate_track_rights, _rights(*shares)) is not _main_refuses(sum(shares)), shares


def test_every_generated_rights_record_gets_mains_verdict():
    """Shares with up to eight decimals, near and at the boundary: the verdict is main's for every one."""
    import random
    from khaos_attribution import validate_track_rights
    rng = random.Random(7)
    for _ in range(5000):
        n = rng.randint(1, 5)
        shares = [round(rng.uniform(0, 100), rng.randint(0, 8)) for _ in range(n - 1)]
        shares.append(round(100.0 - sum(shares) + rng.choice([-0.01, -0.00995, -0.005, 0.0, 0.005, 0.00995, 0.01,
                                                               0.0001, -0.0001, 0.01000001, -0.01000001]),
                            rng.randint(2, 8)))
        if not all(0 <= s <= 100 for s in shares):
            continue   # the schema's own bounds, apart from the sum
        assert _verdict(validate_track_rights, _rights(*shares)) is not _main_refuses(sum(shares)), shares


def test_a_share_held_as_a_numpy_number_is_a_number():
    """The schema takes numpy's floats as numbers, so the sum check does too."""
    np = pytest.importorskip("numpy")
    from khaos_attribution import validate_track_rights
    assert validate_track_rights(_rights(np.float32(50.0), np.float64(50.0)))
    with pytest.raises(AttributionValidationError, match="sum to"):
        validate_track_rights(_rights(np.float32(50.0), np.float32(49.98)))
    with pytest.raises(AttributionValidationError):
        validate_track_rights(_rights(True, 99.0))


def _estimate(writers, unattributed):
    return {"schema_version": "1.1.0", "generation_id": "g", "artist_id": "a", "adapter_version": "r",
            "created_at": "2026-10-07T00:00:00+00:00",
            "method": {"kind": "dtrak", "estimator_version": "0.1.0", "exposure_basis": "x", "similarity_informative": False},
            "influence": [{"track_id": "t", "title": "T", "exposure_share_pct": 100.0, "similarity_share_pct": None,
                           "influence_share_pct": 100.0, "blended_share_pct": 100.0, "share_range_pct": [90.0, 100.0]}],
            "splits": {"writers": [{"name": n, "share_pct": s, "share_range_pct": [0.0, 100.0]} for n, s in writers],
                       "publishers": [], "masters": [], "unattributed_pct": unattributed},
            "caveats": []}


WRITER_SPLITS = [
    [100.0], [50.0, 50.0], [33.3333, 33.3333, 33.3334], [14.2857] * 6 + [14.2858], [0.5] * 200,
    [50.0, 49.9901], [50.0, 50.0099], [33.33, 33.33, 33.3399],
]


def test_an_estimate_made_from_valid_rights_always_validates():
    """The estimator rounds the writers and the unattributed remainder together, so a document built from valid
    rights (many writers, or a sum just inside the hundredth) passes the unwidened money check."""
    import random
    from khaos_attribution import blend, validate_attribution_estimate, validate_track_rights
    for split in WRITER_SPLITS:
        validate_track_rights(_rights(*split))
    rng = random.Random(11)
    for _ in range(999):
        tracks = [f"t{i}" for i in range(rng.randint(1, 12))]
        weights = [rng.random() + 0.01 for _ in tracks]
        blended = {t: w / sum(weights) for t, w in zip(tracks, weights)}
        rights = {t: {"writers": [{"name": f"{t}w{j}", "share_pct": s} for j, s in enumerate(rng.choice(WRITER_SPLITS))],
                      "publishers": []}
                  for t in tracks if rng.random() < 0.85}
        money = blend.money_splits(blended, {}, rights)
        doc = _estimate([(w["name"], w["share_pct"]) for w in money["writers"]], money["unattributed_pct"])
        assert validate_attribution_estimate(doc), (len(tracks), len(money["writers"]))


@pytest.mark.parametrize("writers, unattributed", [([("W", 99.9)], 0.0), ([("W", 100.0)], 0.01),
                                                   ([("A", 50.0), ("B", 50.01)], 0.0), ([("A", 49.99)], 50.0),
                                                   ([("A", 33.33), ("B", 33.33), ("C", 33.33)], 0.0),
                                                   ([("A", 50.0)], 49.9901), ([("A", 60.0), ("B", 39.99004)], 0.0)])
def test_the_money_is_judged_exactly_as_it_always_was(writers, unattributed):
    """The money check is not widened: writers and the unattributed remainder get main's verdict."""
    from khaos_attribution import validate_attribution_estimate
    total = sum(s for _, s in writers) + unattributed
    assert _verdict(validate_attribution_estimate, _estimate(writers, unattributed)) is not _main_refuses(total)


def test_a_total_that_is_not_a_number_is_refused_everywhere_it_is_checked():
    from khaos_attribution import validate_attribution_estimate
    doc = _estimate([("W", 100.0)], 0.0)
    doc["influence"][0]["blended_share_pct"] = doc["influence"][0]["influence_share_pct"] = float("nan")
    with pytest.raises(AttributionValidationError):
        validate_attribution_estimate(doc)


def test_a_group_removal_estimates_groups_are_finite_shares_of_the_whole():
    from khaos_attribution import validate_attribution_estimate
    from khaos_attribution.validation import _check_groups
    with pytest.raises(AttributionValidationError, match="not a finite share"):
        _check_groups([{"group_id": "g1", "share_pct": float("inf")}, {"group_id": "g2", "share_pct": 0.0}])
    with pytest.raises(AttributionValidationError, match="between 0 and 100"):
        _check_groups([{"group_id": "g1", "share_pct": 120.0}, {"group_id": "g2", "share_pct": -20.0}])
    _check_groups([{"group_id": "g1", "share_pct": 60.0}, {"group_id": "g2", "share_pct": 40.0}])
    assert validate_attribution_estimate


def test_a_record_that_is_not_an_object_is_refused_in_words():
    from khaos_attribution import validate_track_rights
    for record in ([], "rights", None):
        with pytest.raises(AttributionValidationError):
            validate_track_rights(record)


def test_a_schemas_validator_is_built_once():
    from khaos_attribution import validation as V
    V._validator.cache_clear()
    for _ in range(3):
        V.validate_provenance_record(copy.deepcopy(GOOD_PROVENANCE_RECORD))
    assert V._validator.cache_info().misses == 1
