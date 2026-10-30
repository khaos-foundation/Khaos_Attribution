"""The codec runs in full precision: the renderer replaces the engine's codec dtype rule on every device, and
preprocessing loads the codec in fp32 while the text encoder keeps the precision it was asked for."""
import sys
import types


def test_the_renderer_forces_the_codec_to_fp32_on_every_device_once(monkeypatch):
    torch = types.SimpleNamespace(float32="fp32", bfloat16="bf16", float16="fp16")
    monkeypatch.setitem(sys.modules, "torch", torch)
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


def test_preprocessing_loads_the_codec_in_fp32_and_the_text_encoder_as_asked(monkeypatch):
    asked = []
    loader = types.ModuleType("acestep.training_v2.model_loader")

    def load_vae(checkpoint_dir, device="cpu", precision="bf16"):
        asked.append(("vae", device, precision))
        return "vae"

    def load_text_encoder(checkpoint_dir, device="cpu", precision="bf16"):
        asked.append(("text", device, precision))
        return "tok", "enc"

    loader.load_vae, loader.load_text_encoder = load_vae, load_text_encoder
    package = types.ModuleType("acestep.training_v2")
    package.model_loader = loader
    monkeypatch.setitem(sys.modules, "acestep", types.ModuleType("acestep"))
    monkeypatch.setitem(sys.modules, "acestep.training_v2", package)
    monkeypatch.setitem(sys.modules, "acestep.training_v2.model_loader", loader)
    from khaos_attribution.engines.acestep import train as T

    T.patch_vae_precision()
    T.patch_vae_precision()
    assert loader.load_vae("ckpt", "mps", "bf16") == "vae"
    loader.load_text_encoder("ckpt", "mps", "bf16")
    assert asked == [("vae", "mps", "fp32"), ("text", "mps", "bf16")]
