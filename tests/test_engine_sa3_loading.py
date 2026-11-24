"""Stable Audio 3's torch model is built from Stability's config with the text encoder read from the open base
repository: the config names the gated distilled repository, whose encoder files are the same bytes."""
import glob
import json
import os

import pytest

from khaos_attribution.engines.sa3 import loading as L
from khaos_attribution.engines.sa3 import DESCRIPTOR


def test_every_conditioner_naming_the_gated_repository_is_pointed_at_the_open_one():
    cfg = {"model": {"conditioning": {"configs": [
        {"id": "prompt", "type": "t5gemma", "config": {"repo_id": L.GATED_ENCODER_REPO, "subfolder": "t5gemma-b-b-ul2"}},
        {"id": "seconds_total", "type": "number", "config": {"min_val": 0}}]}}, "svd_bases_path": "svd_bases.pt"}
    out = L.with_open_encoder(cfg)
    assert out["model"]["conditioning"]["configs"][0]["config"] == {"repo_id": L.OPEN_ENCODER_REPO, "subfolder": "t5gemma-b-b-ul2"}
    assert out["model"]["conditioning"]["configs"][1] == cfg["model"]["conditioning"]["configs"][1]
    assert cfg["model"]["conditioning"]["configs"][0]["config"]["repo_id"] == L.GATED_ENCODER_REPO, "the input is not changed"


def test_the_real_base_config_in_the_cache_names_the_gated_copy_and_is_rewritten():
    found = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--stabilityai--stable-audio-3-medium-base/snapshots/*/model_config.json"))
    if not found:
        pytest.skip("the base model's config is not in this machine's cache")
    cfg = json.load(open(found[0]))
    gated = f'"repo_id": "{L.GATED_ENCODER_REPO}"'
    assert json.dumps(cfg).count(gated) == 1, "Stability's config names the gated repository once"
    assert gated not in json.dumps(L.with_open_encoder(cfg))


def test_the_descriptor_asks_for_no_gated_set():
    assert not [w for w in DESCRIPTOR.weights if w.get("gated")]
    torch_set = [w for w in DESCRIPTOR.weights if w.get("runtime") == "torch"]
    assert [w["repo"] for w in torch_set] == [L.OPEN_ENCODER_REPO]
    assert "t5gemma-b-b-ul2/*" in torch_set[0]["patterns"] and "svd_bases.pt" not in " ".join(torch_set[0]["patterns"])


def _fake_torch(monkeypatch):
    import sys
    import types
    calls = []
    fake = types.SimpleNamespace(bfloat16="bf16", float32="fp32",
                                 autocast=lambda **kw: calls.append(kw) or kw,
                                 set_float32_matmul_precision=lambda p: calls.append({"matmul": p}),
                                 backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(allow_tf32=False),
                                                                cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(
                                                                    allow_fp16_reduced_precision_reduction=False))))
    monkeypatch.setitem(sys.modules, "torch", fake)
    return calls


class _Model:
    """Stands in for Stability's model: knows its device and dtype, and records each device it was moved to with the
    dtype it held while it moved."""

    def __init__(self):
        self.device, self.dtype, self.moves = "cpu", "fp32", []

    def to(self, *args, device=None, dtype=None):
        for a in args:
            if a in ("fp16", "bf16", "fp32"):
                dtype = a
            else:
                device = a
        if dtype is not None:
            self.dtype = dtype
        if device is not None and device != self.device:
            self.device = device
            self.moves.append((device, self.dtype))
        return self

    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self


def _fake_stability(monkeypatch, cuda: bool):
    """Stability's loader as it is at the engine's pin (loading_utils.load_diffusion_cond: build, move to the device,
    then cast to half), and its wrapper, over a fake torch whose CUDA is there or not."""
    import sys
    import types
    calls = _fake_torch(monkeypatch)
    torch = sys.modules["torch"]
    torch.float16 = "fp16"
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    torch.backends.mps = types.SimpleNamespace(is_available=lambda: not cuda)
    built = []

    def load_diffusion_cond(model_config, ckpt_path, device="cuda", model_half=False):
        model = _Model()
        built.append(model)
        model.to(device).eval().requires_grad_(False)
        if model_half:
            model.to(torch.float16)
        return model

    utils = types.ModuleType("stable_audio_3.loading_utils")
    utils.load_diffusion_cond = load_diffusion_cond
    wrapper = types.ModuleType("stable_audio_3.model")
    wrapper.StableAudioModel = lambda model, cfg, device, half: types.SimpleNamespace(
        model=model, model_config=cfg, device=device, model_half=half)
    monkeypatch.setitem(sys.modules, "stable_audio_3.loading_utils", utils)
    monkeypatch.setitem(sys.modules, "stable_audio_3.model", wrapper)
    monkeypatch.setattr(L, "resolved_config", lambda name: ({"model_type": "diffusion_cond"}, "/x/model.safetensors"))
    return built, calls


def test_a_render_model_is_halved_before_it_goes_on_the_card(monkeypatch):
    """A render's model reaches the card already in half precision, never in full precision first."""
    built, _ = _fake_stability(monkeypatch, cuda=True)
    wrapped = L.load_model("medium-base")
    assert built[0].moves == [("cuda", "fp16")], "the only move to the card carries the half-precision model"
    assert (wrapped.device, wrapped.model_half, wrapped.model.dtype) == ("cuda", True, "fp16")


def test_off_cuda_a_render_model_moves_at_full_precision_as_before(monkeypatch):
    built, _ = _fake_stability(monkeypatch, cuda=False)
    wrapped = L.load_model("medium-base")
    assert built[0].moves == [("mps", "fp32")] and wrapped.model_half is False


def test_a_measurement_on_cuda_holds_the_base_at_the_trainers_bfloat16_before_it_goes_on_the_card(monkeypatch):
    """The held-out and influence scorers' model reaches the card already in bfloat16."""
    import sys
    built, calls = _fake_stability(monkeypatch, cuda=True)
    wrapped = L.load_measured("medium-base")
    assert built[0].moves == [("cuda", "bf16")], "the only move to the card carries the bfloat16 model"
    assert wrapped.model_half is False
    torch = sys.modules["torch"]
    assert {"matmul": "high"} in calls and torch.backends.cudnn.allow_tf32
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, "the trainer's settings, not the wrapper's"


def test_off_cuda_a_measurement_keeps_the_base_at_full_precision(monkeypatch):
    built, calls = _fake_stability(monkeypatch, cuda=False)
    L.load_measured("medium-base")
    assert built[0].moves == [("mps", "fp32")] and {"matmul": "high"} not in calls


def test_the_forward_runs_under_bfloat16_autocast_on_cuda_only(monkeypatch):
    calls = _fake_torch(monkeypatch)
    L.measure_autocast("cuda:0")
    L.measure_autocast("mps")
    assert calls[0] == {"device_type": "cuda", "dtype": "bf16", "enabled": True}
    assert calls[1]["enabled"] is False


def test_the_seconds_conditioner_is_left_out_only_when_the_adapter_file_holds_no_weights_for_it():
    """An MLX adapter trained without the seconds conditioner measures on CUDA with exactly the file's parameters."""
    dit_only = ["model.transformer.layers.0.attn.to_q.lora_A", "model.transformer.layers.0.attn.to_q.lora_B"]
    assert L.extra_exclude(dit_only) == ["seconds_total"]
    with_seconds = dit_only + ["conditioners.seconds_total.embedder.embedding.1.lora_A"]
    assert L.extra_exclude(with_seconds) is None, "an adapter that trained the conditioner keeps it"


def test_the_adapter_loads_through_stabilitys_loader_with_that_exclude(monkeypatch):
    import sys
    import types
    seen = {}
    loader = types.ModuleType("stable_audio_3.models.lora.loader")
    loader.load_and_apply_loras = lambda model, paths, model_type, svd_bases_path=None, exclude=None: seen.update(
        model=model, paths=paths, model_type=model_type, svd=svd_bases_path, exclude=exclude)
    utils = types.ModuleType("stable_audio_3.models.lora.utils")
    utils.load_lora_checkpoint = lambda path: ({"model.blocks.0.lora_A": 0, "model.blocks.0.lora_B": 0}, {})
    monkeypatch.setitem(sys.modules, "stable_audio_3.models.lora.loader", loader)
    monkeypatch.setitem(sys.modules, "stable_audio_3.models.lora.utils", utils)
    wrapped = types.SimpleNamespace(model="dit", model_config={"model_type": "diffusion_cond", "svd_bases_path": "b.pt"})
    state, _cfg = L.load_adapter(wrapped, "/x/adapter_model.safetensors")
    assert state == {"model.blocks.0.lora_A": 0, "model.blocks.0.lora_B": 0}, "the file is read once, for the caller too"
    assert seen == {"model": "dit", "paths": ["/x/adapter_model.safetensors"], "model_type": "diffusion_cond",
                    "svd": "b.pt", "exclude": ["seconds_total"]}


def test_the_measure_worker_loads_its_adapter_through_load_adapter():
    """The torch measure worker never adapts a module the adapter file holds no weights for."""
    import inspect
    from khaos_attribution.engines.sa3 import measure_worker as W
    src = inspect.getsource(W._load)
    assert "load_adapter(wrapped" in src and "wrapped.load_lora" not in src
