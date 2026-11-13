"""The base model identifier: its shape, the legacy reading of records from before it, and every schema and
record check that carries it."""
import pytest

from khaos_attribution import base_model as B
from khaos_attribution import registry as R
from khaos_attribution import validate_model_card, validate_provenance_record
from khaos_attribution.credentials import RECORD_FIELDS


def test_the_id_is_engine_at_variant_and_nothing_else():
    assert B.is_base_model_id("acestep-1.5@turbo") and B.is_base_model_id("stable-audio-3-medium@base")
    for bad in ("ace-step-1.5 (turbo)", "acestep-1.5", "@turbo", "ACESTEP@turbo", "", None, 3):
        assert not B.is_base_model_id(bad), bad
    assert B.split_base_model_id("acestep-1.5@base") == ("acestep-1.5", "base") and B.engine_of("x@y") == "x"
    with pytest.raises(ValueError):
        B.split_base_model_id("acestep-1.5")
    assert B.make_base_model_id("acestep-1.5", "turbo") == "acestep-1.5@turbo"


def test_a_record_from_before_the_field_reads_as_acestep_or_not_at_all():
    """Only ACE-Step 1.5 ever ran: its name is read with the variant the manifest or the text gives; anything else
    is None, for a migration to report rather than guess."""
    assert B.legacy_base_model_id("ace-step-1.5 (turbo)") == "acestep-1.5@turbo"
    assert B.legacy_base_model_id("ACE-Step 1.5 (acestep-v15-base)") == "acestep-1.5@base"
    assert B.legacy_base_model_id("ace-step-1.5", "base") == "acestep-1.5@base"
    assert B.legacy_base_model_id("ace-step-1.5 (turbo)", "base") == "acestep-1.5@base", "the manifest's variant wins"
    assert B.legacy_base_model_id("ace-step-1.5") is None, "no variant anywhere: not guessed"
    assert B.legacy_base_model_id("mock-engine 1.0", "turbo") is None and B.legacy_base_model_id(None) is None
    assert B.base_model_id_of({"base_model_id": "stable-audio-3-medium@base", "base_model": "ace-step-1.5 (turbo)"}) == "stable-audio-3-medium@base"
    assert B.base_model_id_of({"base_model": "ace-step-1.5 (turbo)"}, {"model_variant": "turbo"}) == "acestep-1.5@turbo"
    assert B.base_model_id_of({"base_model": "ace-step-1.5"}, {"model_variant": "base"}) == "acestep-1.5@base"


def test_the_card_the_record_and_the_register_record_carry_the_id_in_one_shape(tmp_path):
    card = {"schema_version": "1.0.0", "artist_name": "A", "artist_id": "a", "consent_statement": "yes",
            "training_catalogue": [{"title": "One", "duration": 10.0}], "training_date": "2026-10-16",
            "adapter_version": "run_1", "base_model": "ace-step-1.5 (turbo)", "base_model_licence": "MIT"}
    validate_model_card(card)
    validate_model_card({**card, "base_model_id": "acestep-1.5@turbo", "engine_commit": None})
    with pytest.raises(Exception):
        validate_model_card({**card, "base_model_id": "ace-step-1.5 (turbo)"})
    record = {"schema_version": "1.0.0", "artist_id": "a", "artist_name": "A", "adapter_version": "run_1",
              "adapter_hash": "ab" * 16, "prompt": "p", "timestamp": "2026-10-16T00:00:00Z", "base_model": "ace-step-1.5 (turbo)",
              "base_model_licence": "MIT"}
    validate_provenance_record(record)
    validate_provenance_record({**record, "base_model_id": "acestep-1.5@turbo", "engine_commit": "abc"})
    with pytest.raises(Exception):
        validate_provenance_record({**record, "base_model_id": "nope"})
    assert "base_model_id" in RECORD_FIELDS and "engine_commit" in RECORD_FIELDS, "the signed record carries the id"


def test_the_register_record_check_reads_the_id_when_present():
    from tests.test_registry import _record
    good = _record()
    assert R.record_problems(good) == []
    with_id = {**good, "model": {**good["model"], "base_model_id": "acestep-1.5@turbo"}}
    assert R.record_problems(with_id) == []
    wrong = {**good, "model": {**good["model"], "base_model_id": "ace-step-1.5 (turbo)"}}
    assert any("base_model_id" in p for p in R.record_problems(wrong))
