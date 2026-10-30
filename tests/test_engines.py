"""The socket: every known engine ships a descriptor the schema accepts, read without importing its code; the
ACE-Step descriptor says what both rooms used to carry as constants; the pins helper reads a checkout and nothing
else."""
import subprocess

import pytest

from khaos_attribution import engines as E
from khaos_attribution.engines import pins
from khaos_attribution.engines.descriptor import DescriptorError, validate_descriptor


def test_every_known_engine_has_a_valid_descriptor_read_as_data():
    for engine_id in E.KNOWN_ENGINES:
        d = E.descriptor(engine_id)
        assert d.id == engine_id and d.default_variant in d.variants
    assert set(E.descriptors()) == set(E.KNOWN_ENGINES)
    with pytest.raises(E.EngineUnavailable):
        E.descriptor("no-such-engine")


def test_the_acestep_descriptor_carries_the_facts_the_rooms_pinned_as_constants():
    d = E.descriptor("acestep-1.5")
    assert d.commit == "ca1e85fe9430179831e6bc6be790c332190a3866" and pins.is_sha(d.commit)
    assert d.variant_config("turbo") == "acestep-v15-turbo" and d.variant_config("base") == "acestep-v15-base"
    assert d.render_defaults("turbo") == {"steps": 20, "shift": 3.0, "guidance_scale": 7.0, "cfg_interval_start": 0.0,
                                          "cfg_interval_end": 1.0, "dcw_enabled": False}
    assert d.adapter_strength == 0.75
    assert d.render_defaults("base")["steps"] == 32 and d.render_defaults("base")["dcw_enabled"] is False
    assert d.render_settings("turbo", steps=0, shift=None) == d.render_defaults("turbo")
    assert d.render_settings("base", steps=20, dcw_enabled=True)["steps"] == 20
    assert d.render_settings("nowhere")["steps"] == 20, "an unknown variant resolves on the default's row"
    with pytest.raises(TypeError):
        d.render_settings("turbo", colour="blue")
    assert d.step_clamp("turbo") is None and d.step_clamp("base") is None, "the renderer lifts the engine's clamp"
    assert d.bound("duration_seconds") == (5, 600) and d.bound("takes") == (1, 16)
    assert list(d.vocal_languages) == ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"] and d.default_vocal_language == "en"
    assert d.sings and d.conditioning["lyrics"] and "repaint" in d.rework_modes
    assert d.supports("mps", "train") and not d.supports("cpu", "train")
    assert [w["repo"] for w in d.weights] == ["ACE-Step/Ace-Step1.5", "ACE-Step/acestep-v15-base"]
    assert d.adapter["type"] == "lokr" and d.licence["name"] == "MIT" and d.sample_rate == 48000


def test_a_descriptor_that_says_too_little_or_contradicts_itself_is_refused():
    good = E.descriptor("mock").doc
    with pytest.raises(DescriptorError, match="default_variant"):
        validate_descriptor({**good, "default_variant": "nope"})
    with pytest.raises(DescriptorError, match="lacks render settings"):
        validate_descriptor({**good, "variants": {"turbo": {"render": {"steps": 8}}}})
    with pytest.raises(DescriptorError):
        validate_descriptor({k: v for k, v in good.items() if k != "licence"})


def test_the_engine_object_carries_its_descriptor_and_an_unknown_id_is_refused():
    assert E.engine("mock").descriptor.id == "mock"
    assert E.engine("acestep-1.5").descriptor.id == "acestep-1.5", "the package imports without ACE-Step installed"
    with pytest.raises(E.EngineUnavailable):
        E.engine("nope")


def test_the_pins_helper_reads_a_checkout_and_never_guesses(tmp_path, monkeypatch):
    assert pins.checkout_commit(None) is None and pins.checkout_commit(tmp_path) is None
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    sha = pins.checkout_commit(tmp_path)
    assert pins.is_sha(sha)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere"))
    assert "GIT_DIR" not in pins.git_env() and pins.checkout_commit(tmp_path) == sha, "an inherited GIT_DIR never redirects the question"


def test_capabilities_say_what_a_form_may_offer_and_explain_each_control_it_must_hide():
    """Both rooms hide a control the engine lacks and say why in the contract's words."""
    ace = E.capabilities("acestep-1.5")
    assert ace["sings"] and ace["musical"] and ace["absent"] == {} and ace["bounds"]["duration"] == [5, 600]
    assert ace["variants"]["turbo"]["step_clamp"] is None and ace["default_variant"] == "turbo"
    sa3 = E.capabilities("stable-audio-3")
    assert not sa3["sings"] and not sa3["musical"] and sa3["bounds"]["duration"] == [5, 120]
    assert sa3["absent"]["lyrics"].startswith("Stable Audio 3 Medium does not sing")
    assert sa3["absent"]["musical"] == ("Stable Audio 3 Medium takes no tempo, key or time signature; "
                                        "describe the feel in the prompt instead.")
    assert sa3["absent"]["rework.extend"] == "Stable Audio 3 Medium cannot make a piece longer."
    assert "rework.repaint" not in sa3["absent"] and sa3["rework_modes"] == ["retake", "repaint", "audio2audio"]
    assert sa3["vocal_languages"] == {} and sa3["sample_rate"] == 44100
    assert E.capabilities("mock")["absent"] == {}
    with pytest.raises(E.EngineUnavailable):
        E.capabilities("no-such-engine")
