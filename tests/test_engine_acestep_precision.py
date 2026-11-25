"""The DiT renders in bf16 on CUDA whatever the device string, the text encoder loads in the run's precision and sits
on the accelerator only while it encodes, the codec stays fp32, and a card under SMALL_CARD_GB renders with ACE-Step's
offload and the decode on the processor."""
import contextlib
import os
import sys
import types

import pytest


def _fake_torch(monkeypatch, total_mib=98304):
    props = types.SimpleNamespace(total_memory=int(total_mib * 2**20))
    cuda = types.SimpleNamespace(is_available=lambda: True, get_device_properties=lambda index=0: props,
                                 empty_cache=lambda: None)
    torch = types.SimpleNamespace(float32="fp32", bfloat16="bf16", float16="fp16", cuda=cuda,
                                  backends=types.SimpleNamespace(mps=None),
                                  device=lambda s: types.SimpleNamespace(index=int(s.split(":")[1]) if ":" in s else None))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _render(monkeypatch, total_mib=98304):
    torch = _fake_torch(monkeypatch, total_mib)
    from khaos_attribution.engines.acestep import render as R
    return R, torch


class _Engine:
    """ACE-Step's handler as far as these seams go: it assigns its own dtype inline, as initialize_service does."""
    offload_to_cpu = False

    def __init__(self):
        self.moves = []
        self.loaded = []

    def initialize_service(self, device=None, offload_to_cpu=False, offload_dit_to_cpu=False, config_path=None,
                           **kwargs):
        self.device, self.offload_to_cpu, self.offload_dit_to_cpu = device, offload_to_cpu, offload_dit_to_cpu
        self.config_path = str(config_path)
        self.dtype = "fp32" if device != "cuda" else "bf16"
        self.text_encoder = "te"
        return "ok", True

    def is_turbo_model(self):
        return "base" not in getattr(self, "config_path", "")

    def _get_vae_dtype(self, device=None):
        return "bf16"

    def _load_text_encoder_and_tokenizer(self, *, checkpoint_dir, device):
        self.loaded.append(("engine", device))
        return "path"

    @contextlib.contextmanager
    def _load_model_context(self, model_name):
        self.moves.append(("engine", model_name))
        yield

    def _recursive_to_device(self, model, device, dtype=None):
        self.moves.append((model, device, dtype))


def _loaded(monkeypatch, tmp_path, device="cuda", total_mib=98304):
    R, torch = _render(monkeypatch, total_mib)
    handler_cls = type("AceStepHandler", (_Engine,), {})
    monkeypatch.setitem(sys.modules, "acestep", types.ModuleType("acestep"))
    monkeypatch.setitem(sys.modules, "acestep.handler", types.SimpleNamespace(AceStepHandler=handler_cls))
    monkeypatch.setattr(R.AceStepRenderer, "_resolved_device", lambda self: device)
    renderer = R.AceStepRenderer(repo_dir=tmp_path, checkpoint_dir=tmp_path, device=device)
    renderer.load()
    return R, torch, renderer


@pytest.mark.parametrize("device, expected", [("cuda", "bf16"), ("cuda:0", "bf16"), ("cuda:1", "bf16"),
                                              ("mps", "bf16"), ("cpu", "fp32")])
def test_the_dit_reads_bf16_on_cuda_and_mps_whatever_the_device_string(monkeypatch, device, expected):
    R, torch = _render(monkeypatch)

    class Handler:
        pass
    R.force_dit_bf16(Handler)
    h = Handler()
    h.device = device
    h.dtype = torch.float32
    assert h.dtype == expected
    installed = Handler.dtype
    R.force_dit_bf16(Handler)
    assert Handler.dtype is installed


def test_loading_on_cuda_renders_the_dit_in_bf16_and_the_codec_in_fp32(monkeypatch, tmp_path):
    """The DiT is bf16 on the indexed device string the renderer hands the engine, the codec fp32."""
    R, torch, renderer = _loaded(monkeypatch, tmp_path)
    handler = renderer.handler
    assert handler.device == "cuda:0"
    assert handler.dtype == torch.bfloat16
    assert handler._get_vae_dtype("cuda:0") == torch.float32


def test_the_text_encoder_loads_in_the_run_precision_and_off_the_card(monkeypatch, tmp_path):
    R, torch, renderer = _loaded(monkeypatch, tmp_path)
    asked = []

    class AutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            asked.append(kwargs)
            return types.SimpleNamespace(eval=lambda: None, path=path)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            return "tok"
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModel=AutoModel, AutoTokenizer=AutoTokenizer))
    (tmp_path / "Qwen3-Embedding-0.6B").mkdir()
    handler = renderer.handler
    handler._load_text_encoder_and_tokenizer(checkpoint_dir=str(tmp_path), device="cuda:0")
    assert asked == [{"torch_dtype": torch.bfloat16}]
    assert handler.text_tokenizer == "tok" and handler.text_encoder.path.endswith("Qwen3-Embedding-0.6B")
    assert handler.loaded == [], "the engine's own loader, which casts on the card, is not used"


def test_under_the_engines_own_offload_its_text_encoder_loader_stands(monkeypatch, tmp_path):
    R, torch, renderer = _loaded(monkeypatch, tmp_path)
    handler = renderer.handler
    handler.offload_to_cpu = True
    handler._load_text_encoder_and_tokenizer(checkpoint_dir=str(tmp_path), device="cuda:0")
    assert handler.loaded == [("engine", "cuda:0")]


def test_the_text_encoder_is_on_the_card_only_while_it_encodes(monkeypatch, tmp_path):
    R, torch, renderer = _loaded(monkeypatch, tmp_path)
    handler = renderer.handler
    with handler._load_model_context("text_encoder"):
        assert handler.moves == [("te", "cuda:0", torch.bfloat16)]
    assert handler.moves[-1] == ("te", "cpu", None)
    handler.moves.clear()
    with pytest.raises(RuntimeError):
        with handler._load_model_context("text_encoder"):
            raise RuntimeError("encode failed")
    assert handler.moves[-1] == ("te", "cpu", None), "an encode that raised still leaves the card"
    handler.moves.clear()

    def full(model, device, dtype=None):
        handler.moves.append((model, device, dtype))
        if device != "cpu":
            raise RuntimeError("out of memory")
    handler._recursive_to_device = full
    with pytest.raises(RuntimeError):
        with handler._load_model_context("text_encoder"):
            pass
    assert handler.moves[-1] == ("te", "cpu", None), "a move onto the card that ran out still returns the encoder"
    del handler._recursive_to_device
    handler.moves.clear()
    with handler._load_model_context("vae"):
        pass
    assert handler.moves == [("engine", "vae")]
    handler.offload_to_cpu = True
    handler.moves.clear()
    with handler._load_model_context("text_encoder"):
        pass
    assert handler.moves == [("engine", "text_encoder")], "the engine's own offload moves it when it is on"


def test_the_load_and_the_move_install_together_or_not_at_all(monkeypatch, caplog):
    """The load and the move install together: only together do they leave the audio unchanged."""
    R, torch = _render(monkeypatch)

    class LoaderOnly:
        def _load_text_encoder_and_tokenizer(self, *, checkpoint_dir, device):
            return "engine"
    original = LoaderOnly._load_text_encoder_and_tokenizer
    with caplog.at_level("WARNING", logger=R.log.name):
        R.text_encoder_off_card(LoaderOnly)
    assert LoaderOnly._load_text_encoder_and_tokenizer is original
    assert any("_load_model_context" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("total_mib, small", [(6144, True), (8192, True), (11264, True), (12282, False),
                                              (12288, False), (16376, False), (98304, False)])
def test_a_card_under_twelve_nominal_gb_is_a_small_card(monkeypatch, total_mib, small):
    R, torch = _render(monkeypatch, total_mib)
    assert R.small_card("cuda") is small
    assert R.small_card("mps") is False and R.small_card("cpu") is False


def test_a_small_card_loads_with_the_engines_offload(monkeypatch, tmp_path):
    """A small card takes the DiT off the card between its uses too, so a rework's fp32 encode has room."""
    R, torch, renderer = _loaded(monkeypatch, tmp_path, total_mib=8192)
    assert (renderer.handler.offload_to_cpu, renderer.handler.offload_dit_to_cpu) == (True, True)
    R, torch, renderer = _loaded(monkeypatch, tmp_path, total_mib=12288)
    assert (renderer.handler.offload_to_cpu, renderer.handler.offload_dit_to_cpu) == (False, False)


@pytest.mark.parametrize("total_mib, during", [(8192, "1"), (16384, None)])
def test_a_small_card_decodes_on_the_processor_for_the_render_only(monkeypatch, tmp_path, total_mib, during):
    R, torch, renderer = _loaded(monkeypatch, tmp_path, total_mib=total_mib)
    monkeypatch.delenv(R.VAE_ON_CPU_ENV, raising=False)
    seen = {}
    out = tmp_path / "out"

    def generate_music(handler, llm, params, config, save_dir=None):
        seen["env"] = os.environ.get(R.VAE_ON_CPU_ENV)
        produced = out / "take.wav"
        produced.parent.mkdir(parents=True, exist_ok=True)
        produced.write_bytes(b"x")
        return types.SimpleNamespace(success=True, audios=[{"path": str(produced), "params": {}}])

    class Params:
        def __init__(self, **kw):
            self.kw = kw
    inference = types.SimpleNamespace(GenerationConfig=lambda **kw: kw, GenerationParams=Params,
                                      generate_music=generate_music)
    monkeypatch.setitem(sys.modules, "acestep.inference", inference)
    from khaos_attribution.engines.render import RenderRequest
    result = renderer.generate(RenderRequest(prompt="p", duration_seconds=10), None, out / "piece.wav")
    assert seen["env"] == during
    assert R.VAE_ON_CPU_ENV not in os.environ
    assert result.params["khaos_dit_dtype"] == "bf16"
    assert result.params["khaos_small_card"] is (during == "1")

    def fails(*a, **k):
        assert os.environ.get(R.VAE_ON_CPU_ENV) == during
        raise RuntimeError("out of memory")
    inference.generate_music = fails
    with pytest.raises(RuntimeError):
        renderer.generate(RenderRequest(prompt="p", duration_seconds=10), None, out / "piece2.wav")
    assert R.VAE_ON_CPU_ENV not in os.environ, "a render that raised restores the environment"


def _fake_inference(monkeypatch, out):
    def generate_music(handler, llm, params, config, save_dir=None):
        produced = out / "take.wav"
        produced.parent.mkdir(parents=True, exist_ok=True)
        produced.write_bytes(b"x")
        return types.SimpleNamespace(success=True, audios=[{"path": str(produced), "params": {}}])

    class Params:
        def __init__(self, **kw):
            self.kw = kw
    monkeypatch.setitem(sys.modules, "acestep.inference", types.SimpleNamespace(
        GenerationConfig=lambda **kw: kw, GenerationParams=Params, generate_music=generate_music))


def test_a_base_render_that_releases_its_weights_still_records_its_precision(monkeypatch, tmp_path):
    R, torch, renderer = _loaded(monkeypatch, tmp_path)
    base = tmp_path / "acestep-v15-base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"w")
    hub = types.ModuleType("huggingface_hub")
    hub.constants = types.SimpleNamespace(HF_HUB_OFFLINE=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    _fake_inference(monkeypatch, tmp_path / "out")
    from khaos_attribution.engines.render import RenderRequest
    result = renderer.generate(RenderRequest(prompt="p", duration_seconds=10, variant="base"), None,
                               tmp_path / "out" / "piece.wav", release_base=True)
    assert result.params["khaos_dit_dtype"] == "bf16"
    assert "base" not in renderer._handlers


def test_an_operators_own_cpu_decode_setting_is_kept(monkeypatch, tmp_path):
    R, torch, renderer = _loaded(monkeypatch, tmp_path, total_mib=8192)
    monkeypatch.setenv(R.VAE_ON_CPU_ENV, "0")
    with R.cpu_decode(True):
        assert os.environ[R.VAE_ON_CPU_ENV] == "1"
    assert os.environ[R.VAE_ON_CPU_ENV] == "0"


def test_the_prompt_embedder_puts_the_encoder_on_the_card_for_its_call(monkeypatch, tmp_path):
    R, torch, renderer = _loaded(monkeypatch, tmp_path)
    handler = renderer.handler
    handler.text_tokenizer = "tok"
    calls = []
    import khaos_attribution.prompt_expansion as PE
    monkeypatch.setattr(PE, "embed_texts", lambda model, tok, texts: calls.append(list(handler.moves)) or "emb")
    assert renderer.text_embedder()(["a"]) == "emb"
    assert calls == [[("te", "cuda:0", torch.bfloat16)]]
    assert handler.moves[-1] == ("te", "cpu", None)


def test_the_prompt_embedder_runs_on_an_engine_without_the_offload_context(monkeypatch, tmp_path):
    R, torch = _render(monkeypatch)
    renderer = R.AceStepRenderer(repo_dir=tmp_path, checkpoint_dir=tmp_path, device="cpu")
    renderer.handler = types.SimpleNamespace(text_encoder="te", text_tokenizer="tok")
    import khaos_attribution.prompt_expansion as PE
    monkeypatch.setattr(PE, "embed_texts", lambda model, tok, texts: (model, tok, texts))
    assert renderer.text_embedder()(["a"]) == ("te", "tok", ["a"])


def test_the_real_engine_holds_bf16_on_cuda_after_assigning_float32():
    """Against ACE-Step's own handler class, when torch and the checkout are importable."""
    torch = pytest.importorskip("torch")
    handler_mod = pytest.importorskip("acestep.handler")
    from khaos_attribution.engines.acestep import render as R
    R.force_dit_bf16(handler_mod.AceStepHandler)
    h = handler_mod.AceStepHandler.__new__(handler_mod.AceStepHandler)
    h.device = "cuda:0"
    h.dtype = torch.float32
    assert h.dtype == torch.bfloat16
    h.device = "cpu"
    assert h.dtype == torch.float32
