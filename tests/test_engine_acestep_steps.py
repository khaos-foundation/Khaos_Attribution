"""The renderer lifts the engine's turbo step clamp: the count asked for is the count rendered, by keyword or by
position, once per handler class; the strength default is the descriptor's."""
import sys
import types


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
