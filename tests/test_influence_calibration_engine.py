"""An index's calibration is read only when it was measured on the index's own engine, whatever wrote it there."""
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
