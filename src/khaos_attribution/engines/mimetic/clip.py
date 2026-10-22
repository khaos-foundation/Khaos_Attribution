"""One clip as mimetic-a0's trainer sees it, for the held-out and measure workers: the first window of the file
(padded to the model's window with a padding mask, as stable-audio-tools' dataset pads), the conditioning its
dataset writes (seconds_start 0, seconds_total the file's whole seconds), latents encoded as its trainer encodes
them, and the seeded noise. Run in the engine's environment."""

from __future__ import annotations

import math
from pathlib import Path


def caption_of(wav: Path) -> str:
    txt = Path(wav).with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else ""


def prepare(model, config: dict, wav: Path, device: str, noise_seed: int, *, with_meta: bool = False):
    """-> (latents, conditioning, padding mask at latent rate, noise), every tensor on ``device``; with_meta adds the
    conditioner's input, for a caller that takes the conditioning again (the adapter reaches the conditioner too)."""
    import soundfile as sf  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from stable_audio_tools.training.utils import resize_padding_mask  # noqa: PLC0415

    from khaos_attribution import gradient as G  # noqa: PLC0415

    rate, window = int(config["sample_rate"]), int(config["sample_size"])
    audio, file_rate = sf.read(str(wav), dtype="float32", always_2d=True)
    if file_rate != rate:
        raise RuntimeError(f"{Path(wav).name} is at {file_rate} Hz, the model wants {rate}")
    x = torch.from_numpy(audio.T.copy())
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    x = x[:2]
    n = x.shape[-1]
    if n == 0:
        raise RuntimeError(f"{Path(wav).name} holds no audio")
    take = min(n, window)
    padded = torch.zeros(2, window)
    padded[:, :take] = x[:, :take].clamp(-1, 1)
    mask = torch.zeros(window, dtype=torch.bool)
    mask[:take] = True
    batch = padded.unsqueeze(0).to(device)
    with torch.no_grad():
        # The trainer encodes under CUDA's default autocast, which a Mac ignores.
        with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
            latents = model.pretransform.encode(batch)
        latents = latents.float()
        meta = [{"prompt": caption_of(wav), "seconds_start": 0, "seconds_total": math.ceil(n / rate)}]
        cond = model.conditioner(meta, device)
    cond = detached(cond)
    latent_mask = resize_padding_mask(mask.unsqueeze(0).to(device), latents.shape[-1]).to(torch.bool)
    noise = torch.from_numpy(G.pair_noise(noise_seed, tuple(latents.shape))).to(device)
    if with_meta:
        return latents, cond, latent_mask, noise, meta
    return latents, cond, latent_mask, noise


def detached(value):
    if isinstance(value, dict):
        return {k: detached(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detached(v) for v in value)
    return value.detach() if hasattr(value, "detach") else value


def forward(model, latents, cond, mask, noise, t: float):
    """The rectified-flow forward at one timestep: the trainer's noising, no CFG dropout, padding masked."""
    import torch  # noqa: PLC0415
    tt = torch.full((latents.shape[0],), float(t), device=latents.device, dtype=torch.float32)
    noised = latents * (1.0 - t) + noise * t
    return model(noised, tt, cond=cond, cfg_dropout_prob=0.0, padding_mask=mask).float()


def velocity_mse(out, latents, noise, mask):
    """The trainer's loss: the velocity's squared error over the clip's real frames."""
    target = noise - latents
    return ((out - target) ** 2)[mask.unsqueeze(1).expand_as(out)].mean()
