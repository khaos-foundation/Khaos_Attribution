"""mimetic-a0 loaded inside its own environment: Mick's config and weights at the descriptor's pinned revision,
built by stock stable-audio-tools, held at full precision on every device (training, rendering and measuring agree)."""

from __future__ import annotations

import json

from khaos_attribution.engines.mimetic import DESCRIPTOR


def weights_set(name: str) -> dict:
    return next(w for w in DESCRIPTOR.weights if w["name"] == name)


def pick_device(name: str | None = None) -> str:
    import torch  # noqa: PLC0415
    if name and name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_files() -> tuple:
    """(config path, weights path) from the Hugging Face cache, at the pinned revision; offline when the rooms are."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    w = weights_set("mimetic-a0")
    return (hf_hub_download(w["repo"], "model_config.json", revision=w.get("revision")),
            hf_hub_download(w["repo"], "model.safetensors", revision=w.get("revision")))


def apg_on_cpu(project):
    """stable-audio-tools' APG projection casts to float64, which Metal lacks: it runs on the CPU (small latents)."""
    def wrapped(v0, v1, padding_mask=None):
        if padding_mask is not None:
            padding_mask = padding_mask.cpu()
        parallel, orthogonal = project(v0.cpu(), v1.cpu(), padding_mask=padding_mask)
        return parallel.to(v0.device), orthogonal.to(v0.device)
    return wrapped


def load_model(device: str | None = None):
    """-> (model, config, device): the frozen base, float32, in eval mode."""
    from stable_audio_tools.models.factory import create_model_from_config  # noqa: PLC0415
    from stable_audio_tools.models.utils import load_ckpt_state_dict  # noqa: PLC0415
    device = pick_device(device)
    config_path, weights_path = model_files()
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    model = create_model_from_config(config)
    model.load_state_dict(load_ckpt_state_dict(str(weights_path)))
    model = model.to(device).eval().requires_grad_(False)
    if device == "mps":
        for module in model.modules():
            if hasattr(module, "apg_project"):
                module.apg_project = apg_on_cpu(module.apg_project)
    return model, config, device


def apply_adapter(model, config, path, strength: float = 1.0) -> None:
    """The adapter attached as stable-audio-tools attaches it, at a strength; strength 0 is the base exactly."""
    from stable_audio_tools.models.lora import set_lora_strength  # noqa: PLC0415
    from stable_audio_tools.models.lora.loader import load_and_apply_loras  # noqa: PLC0415
    load_and_apply_loras(model, [str(path)], config["model_type"])
    set_lora_strength(model, float(strength))
