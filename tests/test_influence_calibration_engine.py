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


def test_read_calibration_refuses_another_engines_block_in_words():
    """A block measured on another engine is not applied, and the reason says which engines; the index names its
    own engine first, then by its base model id."""
    block = {"schema": I.CALIBRATION_SCHEMA, "artist_id": "a", "factors": {"t1": 2.0, "t2": 0.5}, "fit_outputs": 80,
             "measured_error_pp": 2.5, "validation": {"money_on_the_right_tracks": 0.88}, "fitted_on": {"record": "r"}}
    read = I.read_calibration(block, index_tracks=["t1", "t2"], artist_id="a", on="stable-audio-3")
    assert not read["applied"] and "measured on acestep-1.5" in read["reason"]
    assert I.read_calibration(block, index_tracks=["t1", "t2"], artist_id="a", on="acestep-1.5")["applied"]
    assert I.index_engine({"engine": "stable-audio-3", "base_model_id": "acestep-1.5@turbo"}) == "stable-audio-3"
    assert I.index_engine({}) == "acestep-1.5" and I.index_engine({"base_model_id": "nonsense"}) is None


def test_a_calibration_belongs_to_one_instrument_an_engine_on_one_runtime():
    """A Mac's factors never apply to a CUDA index of the same engine, nor the other way round."""
    mac = {**BLOCK, "fitted_on": {"engine": "stable-audio-3", "runtime": "mlx"}}
    cuda = {**BLOCK, "fitted_on": {"engine": "stable-audio-3"}}
    mac_index = {"engine": "stable-audio-3", "runtime": "mlx"}
    cuda_index = {"engine": "stable-audio-3", "runtime": "torch"}
    assert I.calibration_of({**mac_index, "calibration": mac}) is not None
    assert I.calibration_of({**cuda_index, "calibration": mac}) is None
    assert I.calibration_of({**mac_index, "calibration": cuda}) is None
    assert I.calibration_of({**cuda_index, "calibration": cuda}) is not None, "torch is the engine's plain name"
    assert I.index_instrument({}, "stable-audio-3@base") == "stable-audio-3", "an index without an engine field reads its base model id"
    assert I.index_instrument({"engine": "acestep-1.5"}, "stable-audio-3@base") == "acestep-1.5", "its own field wins"
