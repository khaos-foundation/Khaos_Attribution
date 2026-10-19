"""How Khaos loads Stable Audio 3's torch model: Stability's own construction from its config, with the text
encoder read from Stability's open base repository. The config names the encoder inside the gated distilled
repository; the base repository carries the same files, byte for byte, so no account is needed to run the engine."""

from __future__ import annotations

import json

GATED_ENCODER_REPO = "stabilityai/stable-audio-3-medium"
OPEN_ENCODER_REPO = "stabilityai/stable-audio-3-medium-base"


def with_open_encoder(model_config: dict) -> dict:
    """The config with every conditioner that names the gated repository pointed at the open one."""
    def walk(o):
        if isinstance(o, dict):
            if o.get("repo_id") == GATED_ENCODER_REPO:
                o = {**o, "repo_id": OPEN_ENCODER_REPO}
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o
    return walk(model_config)


def resolved_config(model_name: str) -> tuple:
    """(config dict, checkpoint path) as Stability resolves them from the cache, the encoder pointed at the open copy."""
    from stable_audio_3.model_configs import all_models  # noqa: PLC0415
    if model_name not in all_models:
        raise ValueError(f"unknown model {model_name!r}; Stability knows {list(all_models)}")
    local_config, local_ckpt = all_models[model_name].resolve()
    with open(local_config) as f:
        return with_open_encoder(json.load(f)), local_ckpt


def load_model(model_name: str, device=None, model_half: bool = True):
    """Stability's StableAudioModel.from_pretrained, step for step, on the rewritten config."""
    import torch  # noqa: PLC0415
    from stable_audio_3.loading_utils import load_diffusion_cond  # noqa: PLC0415
    from stable_audio_3.model import StableAudioModel  # noqa: PLC0415
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if not torch.cuda.is_available():
        model_half = False
    model_config, local_ckpt = resolved_config(model_name)
    model = load_diffusion_cond(model_config, local_ckpt, device=device, model_half=model_half)
    model.use_lora = False
    model.lora_names = []
    return StableAudioModel(model, model_config, device, model_half)


def load_measured(model_name: str):
    """The base as the trainer holds it, for a measurement: bfloat16 on CUDA (an adapter loaded after stays float32),
    full precision elsewhere."""
    import torch  # noqa: PLC0415
    wrapped = load_model(model_name, model_half=False)
    if str(wrapped.device).startswith("cuda"):
        wrapped.model.to(dtype=torch.bfloat16)
        # Stability's inference wrapper turns these off process-wide; its trainer runs with them as below.
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    return wrapped


def measure_autocast(device):
    """The forward's autocast as the trainer's bf16-mixed runs it on CUDA; none elsewhere."""
    import torch  # noqa: PLC0415
    on = str(device).startswith("cuda")
    return torch.autocast(device_type="cuda" if on else "cpu", dtype=torch.bfloat16, enabled=on)


def load_for_training(model_name: str, device):
    """Stability's trainer's load_model, step for step, on the rewritten config."""
    import torch  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415
    from stable_audio_3.loading_utils import copy_state_dict  # noqa: PLC0415
    from stable_audio_3.model_configs import base_models  # noqa: PLC0415
    from stable_audio_3.factory import create_diffusion_cond_from_config  # noqa: PLC0415
    if model_name not in base_models:
        raise ValueError(f"LoRA training requires a base model; Stability knows {list(base_models)}")
    model_config, local_ckpt = resolved_config(model_name)
    model = create_diffusion_cond_from_config(model_config)
    copy_state_dict(model, load_file(local_ckpt))
    model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    if model.pretransform is not None:
        model.pretransform.enable_grad = False
    return model, model_config
