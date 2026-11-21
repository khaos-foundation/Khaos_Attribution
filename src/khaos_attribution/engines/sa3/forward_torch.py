"""The trainer's forward on Stable Audio 3's torch runtime, in one place for every worker that takes it: a clip
through the autoencoder, its conditioning with the generation mask, its padding mask, and the model's prediction at
a timestep with the randomness removed. Each worker keeps its own noise rule; this module draws none."""

from __future__ import annotations

from pathlib import Path


def encode(model, torch, wav: Path, device) -> tuple:
    """(latents, sample count, valid frame count) of one clip through the engine's autoencoder."""
    import soundfile as sf  # noqa: PLC0415
    sr = int(model.sample_rate)
    ratio = int(model.pretransform.downsampling_ratio)
    audio, file_sr = sf.read(str(wav), dtype="float32", always_2d=True)
    if file_sr != sr:
        raise RuntimeError(f"{wav.name} is at {file_sr} Hz, the model wants {sr}")
    x = torch.from_numpy(audio.T).unsqueeze(0).to(device)
    n = x.shape[-1]
    padded = ((n + ratio - 1) // ratio) * ratio
    if padded != n:
        x = torch.nn.functional.pad(x, (0, padded - n))
    with torch.no_grad():
        with torch.autocast(device_type=device if isinstance(device, str) else device.type,
                            enabled=str(device).startswith("cuda")):
            latents = model.pretransform.encode(x)
    latents = latents.float().detach()
    return latents, n, max(1, min(int(round(n / ratio)), latents.shape[-1]))


def caption(wav: Path) -> str:
    """The clip's caption, from the text file beside it, else nothing."""
    txt = wav.with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else ""


def condition(model, torch, latents, words: str, seconds: float, device) -> dict:
    """The conditioning the trainer gives one clip, detached, with the generation mask: a clip is generated whole,
    so the inpaint mask is all zeros and the masked input empty."""
    from khaos_attribution.engines.sa3.loading import measure_autocast  # noqa: PLC0415
    with torch.no_grad(), measure_autocast(device):
        cond = model.conditioner([{"prompt": words, "seconds_total": seconds}], device)
    cond = detached(cond, torch)
    cond["inpaint_mask"] = [torch.zeros(1, 1, latents.shape[-1], dtype=torch.bool, device=latents.device)]
    cond["inpaint_masked_input"] = [torch.zeros_like(latents)]
    return cond


def padding_mask(torch, latents, valid: int):
    """True on the clip's own frames, as the trainer masks its loss."""
    mask = torch.zeros(1, latents.shape[-1], dtype=torch.bool, device=latents.device)
    mask[:, :valid] = True
    return mask


def detached(value, torch):
    if isinstance(value, dict):
        return {k: detached(v, torch) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detached(v, torch) for v in value)
    if hasattr(value, "detach"):
        return value.detach()
    return value


def forward(model, torch, latents, cond, mask, noise, t: float, *, checkpointing: bool = True,
            input_dtype=None):
    """The model's prediction at ``t`` from the noised latents, cast to ``input_dtype`` when given, as float32."""
    from khaos_attribution.engines.sa3.loading import measure_autocast  # noqa: PLC0415
    tt = torch.full((1,), t, device=latents.device, dtype=torch.float32)
    noised = latents * (1.0 - t) + noise * t
    if input_dtype is not None:
        noised = noised.to(input_dtype)
    # Checkpointed blocks run their forward twice, so a forward hook would see each layer twice.
    extra = {} if checkpointing else {"use_checkpointing": False}
    with measure_autocast(latents.device):
        return model(noised, tt, cond=cond, cfg_dropout_prob=0.0, padding_mask=mask, **extra).float()
