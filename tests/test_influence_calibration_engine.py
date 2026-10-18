"""An index's calibration is read only when it was measured on the index's own engine, whatever wrote it there."""
import numpy as np
import pytest

from khaos_attribution import influence as I

BLOCK = {"factors": {"t1": 0.5}, "fitted_on": {"record": "r.json"}}


def test_a_calibration_is_read_only_on_its_own_engine():
    assert I.calibration_of({"calibration": BLOCK})["factors"] == {"t1": 0.5}, "no engine named on either side is ACE-Step's"
    assert I.calibration_of({"calibration": BLOCK, "base_model_id": "acestep-1.5@turbo"}) is not None
    assert I.calibration_of({"calibration": BLOCK, "base_model_id": "stable-audio-3@base"}) is None
    sa3 = {**BLOCK, "fitted_on": {"engine": "stable-audio-3"}}
    assert I.calibration_of({"calibration": sa3, "base_model_id": "stable-audio-3@base"}) is not None
    assert I.calibration_of({"calibration": sa3, "base_model_id": "acestep-1.5@base"}) is None
    assert I.calibration_of({}) is None and I.calibration_of(None) is None
    assert I.calibration_of({"calibration": {"factors": {}}}) is None


def test_the_card_records_a_calibration_from_another_engine_as_refused_in_words():
    """The block stays on the document as written, so a current document matches its index; the split is the
    kernel's and the reason is said."""
    block = {"schema": I.CALIBRATION_SCHEMA, "artist_id": "a", "factors": {"t1": 2.0, "t2": 0.5}, "fit_outputs": 80,
             "measured_error_pp": 2.5, "validation": {"money_on_the_right_tracks": 0.88}, "fitted_on": {"record": "r"}}
    read = I.read_calibration(block, index_tracks=["t1", "t2"], artist_id="a", engine="stable-audio-3")
    assert not read["applied"] and "measured on acestep-1.5" in read["reason"]
    assert I.read_calibration(block, index_tracks=["t1", "t2"], artist_id="a", engine="acestep-1.5")["applied"]
    assert I.index_engine({"engine": "stable-audio-3", "base_model_id": "acestep-1.5@turbo"}) == "stable-audio-3"
    assert I.index_engine({}) == "acestep-1.5" and I.index_engine({"base_model_id": "nonsense"}) is None
