"""The held-out loss on Stable Audio 3, inside the engine's environment: the request as JSON on stdin (clips or
latents, the adapter, the timesteps, the seed), the reply as one JSON line of per-sample bucket losses. The loss is
the trainer's own rectified-flow velocity MSE with the randomness fixed: t a bucket centre, noise seeded per clip,
no CFG dropout, no inpainting, the padding mask as the trainer masks it."""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path


def pair_noise_seed(seed: int, name: str) -> int:
    return (int(seed) * 1_000_003 + zlib.crc32(name.encode("utf-8"))) % (2 ** 31 - 1)


def heldout_torch(req: dict) -> dict:
    import soundfile as sf  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from khaos_attribution.engines.sa3.loading import load_measured, measure_autocast  # noqa: PLC0415

    wrapped = load_measured(req["model"])
    if req.get("adapter"):
        wrapped.load_lora([str(req["adapter"])])
        wrapped.set_lora_strength(1.0)
    model = wrapped.model
    model.eval()
    device = wrapped.device
    sr = int(model.sample_rate)
    ratio = int(model.pretransform.downsampling_ratio)
    timesteps = [float(t) for t in req["timesteps"]]
    per_sample = {}
    clips = sorted(Path(req["clips_dir"]).glob("*.wav"))
    with torch.no_grad():
        for wav in clips:
            audio, file_sr = sf.read(str(wav), dtype="float32", always_2d=True)
            if file_sr != sr:
                raise RuntimeError(f"{wav.name} is at {file_sr} Hz, the model wants {sr}")
            x = torch.from_numpy(audio.T).unsqueeze(0).to(device)
            n = x.shape[-1]
            padded = ((n + ratio - 1) // ratio) * ratio
            if padded != n:
                x = torch.nn.functional.pad(x, (0, padded - n))
            caption = wav.with_suffix(".txt").read_text(encoding="utf-8").strip() if wav.with_suffix(".txt").exists() else ""
            seconds = n / sr
            with torch.autocast(device_type=device if isinstance(device, str) else device.type, enabled=str(device).startswith("cuda")):
                latents = model.pretransform.encode(x)
            latents = latents.float()
            with measure_autocast(device):
                cond = model.conditioner([{"prompt": caption, "seconds_total": seconds}], device)
            # The trainer always conditions on an inpaint mask (0 = generate, 1 = keep); a held-out clip is
            # generated whole, so the mask is all zeros and the masked input empty.
            cond["inpaint_mask"] = [torch.zeros(1, 1, latents.shape[-1], dtype=torch.bool, device=latents.device)]
            cond["inpaint_masked_input"] = [torch.zeros_like(latents)]
            valid = int(round(n / ratio))
            mask = torch.zeros(1, latents.shape[-1], dtype=torch.bool, device=latents.device)
            mask[:, :max(1, min(valid, latents.shape[-1]))] = True
            gen = torch.Generator(device="cpu").manual_seed(pair_noise_seed(int(req["seed"]), wav.name))
            noise = torch.randn(latents.shape, generator=gen, dtype=torch.float32).to(latents.device)
            target = noise - latents
            row = {}
            for t in timesteps:
                tt = torch.full((1,), t, device=latents.device, dtype=torch.float32)
                noised = latents * (1.0 - t) + noise * t
                with measure_autocast(device):
                    out = model(noised.to(model.dtype if hasattr(model, "dtype") else noised.dtype), tt, cond=cond,
                                cfg_dropout_prob=0.0, padding_mask=mask).float()
                mse = ((out - target) ** 2).mean(dim=1)
                row[f"{t:.2f}"] = float(mse[mask].mean())
            per_sample[wav.name] = row
    return {"per_sample": per_sample, "runtime": "torch"}


def heldout_mlx(req: dict) -> dict:
    """The MLX trainer's own loss on its pre-encoded latents, with the model, the conditioner and the adapter loaded
    the way lora_train_mlx.py loads them in main(); t a bucket centre, noise seeded per clip, no CFG dropout."""
    import importlib  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    mlx_dir = Path.cwd() / "optimized" / "mlx"
    sys.path.insert(0, str(mlx_dir))
    sys.path.insert(0, str(mlx_dir / "scripts"))
    import mlx.core as mx  # noqa: PLC0415
    from lora_train_mlx import (DIT_CHOICES, T5GEMMA_NPZ_REL, T5Gemma, TrainBundle,  # noqa: PLC0415
                                TrainableSecondsEmbedder, build_conditioning, ensure_local, inject_from_lora_config,
                                load_conditioner_from_npz, load_lora_checkpoint, load_trainable_lora_state,
                                rectified_flow_loss, underfit_lora_config)

    latents_dir = Path(req["latents_dir"])
    items = []
    for npy in sorted(latents_dir.glob("*.npy")):
        meta = json.loads(npy.with_suffix(".json").read_text(encoding="utf-8"))
        items.append((npy, meta, np.load(str(npy)).astype(np.float32)))
    if not items:
        raise RuntimeError(f"no pre-encoded latents under {latents_dir}")
    t_lat = max(arr.shape[-1] for _n, _m, arr in items)
    base_path = req["base_npz"]
    mod = importlib.import_module(DIT_CHOICES["medium"]["loader"])
    dit_model = mod.load_dit(base_path, T_lat=t_lat, dtype=mx.float16, compile_=False)
    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    padding_emb, secs_embedder = load_conditioner_from_npz(base_path, prefix="cond.")
    secs_module = None
    # Before the first checkpoint there is no adapter: the base model alone, as the torch path scores it.
    if req.get("adapter"):
        sd, ckpt_cfg = load_lora_checkpoint(str(req["adapter"]))
        lora_config = underfit_lora_config()
        lora_config.update({k: ckpt_cfg[k] for k in ("adapter_type", "rank", "alpha", "include", "exclude")
                            if (ckpt_cfg or {}).get(k) is not None})
        inject_from_lora_config(dit_model, lora_config, checkpoint_prefix="model.")
        if any(str(k).startswith("conditioners.seconds_total.") for k in sd):
            secs_module = TrainableSecondsEmbedder(secs_embedder.W, secs_embedder.b)
            inject_from_lora_config(secs_module, lora_config, checkpoint_prefix="conditioners.seconds_total.")
    bundle = TrainBundle(dit_model, secs_module)
    if req.get("adapter"):
        load_trainable_lora_state(bundle, sd)
    # The generation mask, as the renderer and Khaos's patched trainer condition.
    channels = items[0][2].shape[0]
    lac_const = mx.array(np.zeros((1, 1 + channels, t_lat), np.float32).transpose(0, 2, 1)).astype(mx.float16)
    timesteps = [float(t) for t in req["timesteps"]]
    per_sample = {}
    for npy, meta, arr in items:
        t_here = arr.shape[-1]
        if t_here != t_lat:
            pad = np.zeros((arr.shape[0], t_lat - t_here), np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        clean = mx.array(arr[None]).astype(mx.float16)
        mask_np = np.zeros((1, t_lat), bool)
        mask_np[0, :len(meta.get("padding_mask") or [])] = np.asarray(meta.get("padding_mask") or [], bool)[:t_lat]
        loss_mask = mx.array(mask_np)
        prompt = str(meta.get("prompt") or meta.get("text") or "")
        prompt_cond = build_conditioning(t5, padding_emb, [prompt])
        seconds = float(meta.get("seconds_total") or 0.0)
        if bundle.secs is not None:
            sec_tok = bundle.secs(mx.array([seconds], dtype=mx.float32))
        else:
            sec_tok = secs_embedder([seconds])
        sec_tok = mx.stop_gradient(sec_tok.astype(mx.float32))
        cross16 = mx.concatenate([prompt_cond, sec_tok], axis=1).astype(mx.float16)
        global16 = sec_tok[:, 0, :].astype(mx.float16)
        rng = np.random.default_rng(pair_noise_seed(int(req["seed"]), npy.stem + ".wav"))
        noise = mx.array(rng.standard_normal(arr[None].shape).astype(np.float32)).astype(mx.float16)

        def model_fn(noised, t):
            return bundle.dit(noised, t.astype(noised.dtype), cross16, global16, local_add_cond=lac_const)
        row = {}
        for t in timesteps:
            loss = rectified_flow_loss(model_fn, clean, mx.array([t], dtype=mx.float32), noise=noise, loss_mask=loss_mask)
            mx.eval(loss)
            row[f"{t:.2f}"] = float(loss)
        per_sample[npy.stem + ".wav"] = row
    return {"per_sample": per_sample, "runtime": "mlx"}


def main() -> int:
    req = json.loads(sys.stdin.read())
    try:
        reply = heldout_mlx(req) if req.get("runtime") == "mlx" else heldout_torch(req)
    except Exception as exc:  # noqa: BLE001 — the reply carries it
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("\n" + json.dumps(reply) + "\n")
    sys.stdout.flush()
    return 0 if not reply.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
