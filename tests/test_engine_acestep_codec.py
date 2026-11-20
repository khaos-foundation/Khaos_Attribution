"""The codec runs in full precision: the renderer installs fp32 as the engine's codec dtype rule when it builds a
handler, preprocessing loads the codec at the codec precision of the call (fp32, or an older run's bf16 so a query
matches its index) while the text encoder keeps the precision it was asked for, and the training door installs it."""
import sys
import types

import pytest


def _fake_torch(monkeypatch):
    torch = types.SimpleNamespace(float32="fp32", bfloat16="bf16", float16="fp16")
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _fake_training_v2(monkeypatch, asked):
    """ACE-Step's training_v2 as the door imports it, recording what the codec loader is asked for."""
    def module(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    def load_vae(checkpoint_dir, device="cpu", precision="bf16"):
        asked.append(("vae", device, precision))
        return "vae"

    def load_text_encoder(checkpoint_dir, device="cpu", precision="bf16"):
        asked.append(("text", device, precision))
        return "tok", "enc"

    def select_dtype(device_type, requested_precision="auto"):
        return requested_precision

    acestep = module("acestep")
    package = module("acestep.training_v2")
    package.model_loader = module("acestep.training_v2.model_loader", load_vae=load_vae,
                                  load_text_encoder=load_text_encoder, load_decoder_for_training=lambda *a, **k: None)
    package.configs = module("acestep.training_v2.configs", LoKRConfigV2=object, LoRAConfigV2=object, TrainingConfigV2=object)
    package.fixed_lora_module = module("acestep.training_v2.fixed_lora_module", FixedLoRAModule=object,
                                       _select_compute_dtype=select_dtype, _select_fabric_precision=select_dtype)
    package.trainer_fixed = module("acestep.training_v2.trainer_fixed", FixedLoRATrainer=object)
    package.trainer_helpers = module("acestep.training_v2.trainer_helpers", save_final=lambda *a, **k: None)

    def preprocess_audio_files(audio_dir=None, output_dir=None, checkpoint_dir=None, variant=None, dataset_json=None,
                               device=None, precision=None):
        from acestep.training_v2.model_loader import load_text_encoder, load_vae  # noqa: PLC0415
        load_vae(checkpoint_dir, device, precision)
        load_text_encoder(checkpoint_dir, device, precision)
        return {"processed": 1, "failed": 0}

    package.preprocess = module("acestep.training_v2.preprocess", preprocess_audio_files=preprocess_audio_files)
    acestep.training_v2 = package
    return package


def test_the_renderer_forces_the_codec_to_fp32_on_every_device_once(monkeypatch):
    torch = _fake_torch(monkeypatch)
    from khaos_attribution.engines.acestep import render as R

    class Handler:
        device = "cuda:0"

        def _get_vae_dtype(self, device=None):
            return torch.bfloat16 if device != "mps" else torch.float16

    R.force_codec_fp32(Handler)
    h = Handler()
    assert (h._get_vae_dtype(), h._get_vae_dtype("mps"), h._get_vae_dtype("cpu")) == ("fp32", "fp32", "fp32")
    installed = Handler._get_vae_dtype
    R.force_codec_fp32(Handler)
    assert Handler._get_vae_dtype is installed


def test_loading_the_serving_weights_installs_the_fp32_rule_before_the_handler_is_built(monkeypatch, tmp_path):
    torch = _fake_torch(monkeypatch)
    from khaos_attribution.engines.acestep import render as R

    seen = {}

    class AceStepHandler:
        def __init__(self):
            seen["rule_at_build"] = self._get_vae_dtype("cuda:0")

        def _get_vae_dtype(self, device=None):
            return torch.bfloat16

        def initialize_service(self, **kwargs):
            return "ok", True

        def is_turbo_model(self):
            return True

    monkeypatch.setitem(sys.modules, "acestep", types.ModuleType("acestep"))
    monkeypatch.setitem(sys.modules, "acestep.handler", types.SimpleNamespace(AceStepHandler=AceStepHandler))
    monkeypatch.setattr(R.AceStepRenderer, "_resolved_device", lambda self: "cpu")
    renderer = R.AceStepRenderer(repo_dir=tmp_path, checkpoint_dir=tmp_path, device="cpu")
    renderer.load()
    assert seen["rule_at_build"] == torch.float32
    assert renderer.handler._get_vae_dtype("mps") == torch.float32


@pytest.mark.parametrize("version, expected", [
    (None, "bf16"), ("", "bf16"), ("0.45.7", "bf16"), ("0.45.8", "fp32"), ("0.45.10", "fp32"), ("0.46.0", "fp32"),
    ("1.0.0", "fp32"), ("garbage", "bf16"), ("0.45", "bf16"),
])
def test_the_codec_precision_of_a_run_follows_the_contract_that_made_its_latents(version, expected):
    from khaos_attribution.engines.acestep import train as T

    assert T.codec_precision_for(version) == expected


def test_preprocessing_loads_the_codec_at_the_call_s_codec_precision_and_the_text_encoder_as_asked(monkeypatch):
    _fake_torch(monkeypatch)
    asked = []
    package = _fake_training_v2(monkeypatch, asked)
    from khaos_attribution.engines.acestep import train as T

    T.patch_vae_precision()
    wrapped = package.model_loader.load_vae
    T.patch_vae_precision()
    assert package.model_loader.load_vae is wrapped, "a second patch does not wrap the wrapper"
    api = T.TrainingApi(preprocess=package.preprocess.preprocess_audio_files, Trainer=None, TrainingConfig=None,
                        LoRAConfig=None, LoKRConfig=None, load_decoder=None, save_final=None)
    T.preprocess_pairs(api, out_dir="o", checkpoint_root="c", variant="turbo", dataset_json=None, device="cuda",
                       precision="bf16")
    T.preprocess_pairs(api, out_dir="o", checkpoint_root="c", variant="turbo", dataset_json=None, device="cuda",
                       precision="bf16", codec_precision="bf16")
    assert asked == [("vae", "cuda", "fp32"), ("text", "cuda", "bf16"),
                     ("vae", "cuda", "bf16"), ("text", "cuda", "bf16")]
    assert T._codec_precision == "fp32", "the call's codec precision does not outlive the call"


def test_the_training_door_installs_the_codec_rule(monkeypatch, tmp_path):
    _fake_torch(monkeypatch)
    asked = []
    package = _fake_training_v2(monkeypatch, asked)
    from khaos_attribution.engines.acestep import train as T

    api = T.training_api(tmp_path)
    assert getattr(package.model_loader, "_khaos_vae_fp32", False), "the door patched the codec loader"
    package.model_loader.load_vae("c", "mps", "bf16")
    assert asked == [("vae", "mps", "fp32")]
    assert api.preprocess is package.preprocess.preprocess_audio_files


@pytest.mark.parametrize("version, expected", [("0.45.8rc1", "fp32"), ("0.45.7.dev3", "bf16"), ("v0.46.0", "fp32")])
def test_a_version_with_a_suffix_is_read_by_its_numbers(version, expected):
    from khaos_attribution.engines.acestep import train as T
    assert T.codec_precision_for(version) == expected


def test_a_revision_whose_precision_selectors_take_no_requested_precision_is_refused(monkeypatch):
    """The pinned trainer honours the run's precision; one that cannot is refused rather than patched by guesswork."""
    _fake_torch(monkeypatch)
    asked = []
    package = _fake_training_v2(monkeypatch, asked)
    package.fixed_lora_module._select_compute_dtype = lambda device_type: "fp16"
    from khaos_attribution.engines.acestep import train as T
    monkeypatch.setattr(package.fixed_lora_module, "_khaos_mps_precision_patched", False, raising=False)
    with pytest.raises(T.TrainingUnavailable, match="requested_precision"):
        T.patch_mps_precision()


def test_a_constructor_taking_any_keyword_takes_them_all():
    from khaos_attribution.engines.acestep import train as T

    def anything(**kw):
        return kw
    assert T.accepted_kwargs(anything, {"a": 1, "b": 2}) == {"a": 1, "b": 2}


def _api(training_fields, adapter_fields):
    from khaos_attribution.engines.acestep import train as T

    def cls(fields):
        def build(**kw):
            return kw
        import inspect
        build.__signature__ = inspect.Signature([inspect.Parameter(f, inspect.Parameter.KEYWORD_ONLY, default=None)
                                                 for f in fields])
        return build
    return T.TrainingApi(preprocess=None, Trainer=lambda model, adapter, train: (adapter, train),
                         TrainingConfig=cls(training_fields), LoRAConfig=cls(adapter_fields),
                         LoKRConfig=cls(adapter_fields), load_decoder=lambda *a: "model", save_final=None)


RECIPE = {"adapter_type": "lokr", "model_variant": "turbo", "adapter_rank": 8, "adapter_alpha": 8, "factor": 4,
          "weight_decompose": False, "attention_type": "both", "target_modules": None, "learning_rate": 1e-4,
          "batch_size": 1, "max_epochs": 2, "gradient_checkpointing": False, "num_workers": 0, "pin_memory": False,
          "persistent_workers": False, "prefetch_factor": 2, "checkpoint_every_n_epochs": 1, "lr_schedule": "constant",
          "gradient_accumulation_steps": 1}


def test_a_training_field_the_revision_does_not_take_refuses_the_run_never_trains_without_it():
    """The run's record keeps the recipe's value, so a field the trainer would not take stops the run."""
    from khaos_attribution.engines.acestep import train as T
    training = list(T.training_config_kwargs(RECIPE, checkpoint_root="c", tensor_dir="t", output_dir="o", device="cpu",
                                             precision="bf16", seed=1))
    adapter = ["linear_dim", "linear_alpha", "factor", "weight_decompose", "attention_type", "target_modules"]
    kw = dict(checkpoint_root="c", tensor_dir="t", output_dir="o", device="cpu", precision="bf16", seed=1)
    assert T.open_session(_api(training, adapter), RECIPE, **kw).trainer[1]["weight_decay"] == 0.01
    with pytest.raises(T.TrainingUnavailable, match="weight_decay"):
        T.open_session(_api([f for f in training if f != "weight_decay"], adapter), RECIPE, **kw)
    with pytest.raises(T.TrainingUnavailable, match="factor"):
        T.open_session(_api(training, [f for f in adapter if f != "factor"]), RECIPE, **kw)
