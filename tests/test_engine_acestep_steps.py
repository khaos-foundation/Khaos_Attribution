"""The renderer lifts the engine's turbo step clamp: the count asked for is the count rendered, by keyword or by
position, once per handler class; the strength default is the descriptor's."""
import sys
import types
from pathlib import Path


class _Host:
    """The engine's mixin as it clamps: a turbo model never renders more than 8 steps."""
    is_turbo = True

    def _normalize_service_generate_inputs(self, captions, lyrics, keys, metas, vocal_languages, repainting_start,
                                           repainting_end, instructions, audio_code_hints, infer_steps, seed,
                                           return_intermediate=False):
        if self.is_turbo and infer_steps > 8:
            infer_steps = 8
        return {"captions": [captions], "infer_steps": infer_steps, "seed": seed}


def _render(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float32="fp32", bfloat16="bf16", float16="fp16"))
    from khaos_attribution.engines.acestep import render as R
    return R


def test_the_requested_steps_survive_the_engines_clamp_by_keyword_and_by_position(monkeypatch):
    R = _render(monkeypatch)

    class Handler(_Host):
        pass

    R.lift_turbo_step_clamp(Handler)
    h = Handler()
    by_keyword = h._normalize_service_generate_inputs(captions="c", lyrics="", keys=None, metas=None, vocal_languages=None,
                                                      repainting_start=None, repainting_end=None, instructions=None,
                                                      audio_code_hints=None, infer_steps=20, seed=7)
    by_position = h._normalize_service_generate_inputs("c", "", None, None, None, None, None, None, None, 32, 7)
    assert by_keyword["infer_steps"] == 20 and by_position["infer_steps"] == 32
    assert by_keyword["captions"] == ["c"] and by_keyword["seed"] == 7, "the rest of the normalisation stands"
    small = h._normalize_service_generate_inputs("c", "", None, None, None, None, None, None, None, 8, 7)
    assert small["infer_steps"] == 8


def test_the_lift_installs_once(monkeypatch):
    R = _render(monkeypatch)

    class Handler(_Host):
        pass

    R.lift_turbo_step_clamp(Handler)
    first = Handler._normalize_service_generate_inputs
    R.lift_turbo_step_clamp(Handler)
    assert Handler._normalize_service_generate_inputs is first


def test_the_renderers_strength_default_is_the_descriptors(monkeypatch, tmp_path):
    R = _render(monkeypatch)
    from khaos_attribution.engines.acestep import DESCRIPTOR
    assert R.AceStepRenderer(tmp_path).default_adapter_scale == DESCRIPTOR.adapter_strength == 0.75
    assert R.AceStepRenderer(tmp_path, default_adapter_scale=0.6).default_adapter_scale == 0.6


def test_a_seam_the_engine_no_longer_has_is_said_never_skipped_silently(monkeypatch, caplog):
    """A revision without the step normalisation, or whose normalisation takes no infer_steps, is warned about."""
    R = _render(monkeypatch)

    class NoSeam:
        pass

    class Renamed:
        def _normalize_service_generate_inputs(self, captions, steps):
            return {"steps": steps}
    with caplog.at_level("WARNING", logger=R.log.name):
        R.lift_turbo_step_clamp(NoSeam)
        R.lift_turbo_step_clamp(Renamed)
    said = [r.getMessage() for r in caplog.records]
    assert any("_normalize_service_generate_inputs" in m for m in said)
    assert any("infer_steps" in m for m in said), said


def test_a_codec_rule_the_engine_no_longer_has_is_said(monkeypatch, caplog):
    """A handler without _get_vae_dtype is warned about, since its codec may not read the fp32 rule."""
    R = _render(monkeypatch)

    class NoRule:
        pass
    with caplog.at_level("WARNING", logger=R.log.name):
        R.force_codec_fp32(NoRule)
    assert any("_get_vae_dtype" in r.getMessage() for r in caplog.records)


class _Handler:
    """Enough of AceStepHandler for a load: a base folder whose checkpoint may declare turbo."""
    turbo = False

    def initialize_service(self, **kwargs):
        return "ok", True

    def is_turbo_model(self):
        return self.turbo

    def _get_vae_dtype(self, device=None):
        return None


def _renderer(monkeypatch, tmp_path, turbo: bool, with_base: bool = True):
    R = _render(monkeypatch)
    handler_cls = type("AceStepHandler", (_Handler,), {"turbo": turbo})
    monkeypatch.setitem(sys.modules, "acestep", types.ModuleType("acestep"))
    monkeypatch.setitem(sys.modules, "acestep.handler", types.SimpleNamespace(AceStepHandler=handler_cls))
    monkeypatch.setattr(R.AceStepRenderer, "_resolved_device", lambda self: "cpu")
    hub = types.ModuleType("huggingface_hub")
    hub.constants = types.SimpleNamespace(HF_HUB_OFFLINE=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    base = tmp_path / "checkpoints" / "acestep-v15-base"
    if with_base:
        base.mkdir(parents=True, exist_ok=True)
        (base / "config.json").write_text("{}")
        (base / "model.safetensors").write_bytes(b"w")
    return R, R.AceStepRenderer(repo_dir=tmp_path, checkpoint_dir=tmp_path / "checkpoints", device="cpu")


def test_a_turbo_checkpoint_in_the_base_folder_is_refused_however_base_is_loaded(monkeypatch, tmp_path):
    """load("base") (the Workshop's path) and a base render after load() (the Space's) refuse it alike."""
    import pytest
    R, renderer = _renderer(monkeypatch, tmp_path, turbo=True)
    with pytest.raises(RuntimeError, match="declares is_turbo"):
        renderer.load("base")
    R, renderer = _renderer(monkeypatch, tmp_path, turbo=True)
    renderer.load()
    with pytest.raises(RuntimeError, match="declares is_turbo"):
        renderer._handler_for("base")
    R, renderer = _renderer(monkeypatch, tmp_path, turbo=False)
    renderer.load("base")
    assert "base" in renderer._handlers


def test_a_missing_base_is_refused_however_base_is_loaded(monkeypatch, tmp_path):
    """load("base") on a checkout without the base weights is refused with the same sentence as a later render."""
    import pytest
    R, renderer = _renderer(monkeypatch, tmp_path, turbo=False, with_base=False)
    with pytest.raises(RuntimeError, match="Base checkpoint not installed"):
        renderer.load("base")


def test_an_extended_heads_padded_copy_leaves_nothing_in_the_output_folder(monkeypatch, tmp_path):
    """The padded source a head extend writes, and its folder, are gone once the render's temps are removed."""
    import numpy as np
    import soundfile as sf
    R = _render(monkeypatch)
    from khaos_attribution.engines.render import RenderRequest
    src = tmp_path / "src.wav"
    sf.write(str(src), np.zeros((4800, 2), dtype="float32"), 48000)
    out = tmp_path / "out"
    out.mkdir()
    renderer = R.AceStepRenderer(tmp_path)
    kwargs, temps = renderer.engine_kwargs(RenderRequest(prompt="p", duration_seconds=10.0, mode="extend",
                                                         source_audio=src, extend_head=1.0), "turbo", out)
    assert Path(kwargs["src_audio"]).is_file()
    R.remove_temps(temps)
    assert list(out.iterdir()) == [], "the padded copy and its folder go after the render"
