"""A dtrak estimate stamped 1.3.0 carries its method's declared shape: the error every share travels with, whether it
was calibrated, and what it was measured over; one stamped earlier is read as it always was."""
import copy

import pytest

pytest.importorskip("numpy")

from khaos_attribution import DTRAK_ESTIMATE_SCHEMA_VERSION, AttributionValidationError, validate_attribution_estimate  # noqa: E402
from tests.test_influence import _doc  # noqa: E402


def test_a_dtrak_document_at_the_declared_version_needs_its_methods_fields(tmp_path):
    doc = copy.deepcopy(_doc(tmp_path))
    doc["schema_version"] = DTRAK_ESTIMATE_SCHEMA_VERSION
    validate_attribution_estimate(doc)
    for field in ("measured_error_pp", "calibrated", "index_sha256", "n_pairs"):
        bare = copy.deepcopy(doc)
        del bare["method"][field]
        with pytest.raises(AttributionValidationError):
            validate_attribution_estimate(bare)
    wrong = copy.deepcopy(doc)
    wrong["method"]["calibrated"] = "yes"
    with pytest.raises(AttributionValidationError):
        validate_attribution_estimate(wrong)


def test_an_earlier_dtrak_document_is_read_as_it_was_written(tmp_path):
    doc = copy.deepcopy(_doc(tmp_path))
    assert doc["schema_version"] == "1.1.0"
    del doc["method"]["measured_error_pp"]
    validate_attribution_estimate(doc)


def test_what_the_writer_puts_in_a_calibrated_or_abstaining_document_fits_the_shape(tmp_path):
    from tests.test_influence import _calibration
    for i, over in enumerate(({"abstention": {"checked": False, "accepted": None, "reason": "no thresholds"}},
                              {"calibration": _calibration()})):
        (tmp_path / str(i)).mkdir()
        doc = copy.deepcopy(_doc(tmp_path / str(i), **over))
        assert set(over) <= set(doc["method"])
        doc["schema_version"] = DTRAK_ESTIMATE_SCHEMA_VERSION
        validate_attribution_estimate(doc)
