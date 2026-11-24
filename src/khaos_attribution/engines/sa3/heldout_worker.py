"""The held-out loss on Stable Audio 3, inside the engine's environment: the request as JSON on stdin (clips or
latents, the adapter, the timesteps, the seed), the reply as one JSON line of per-sample bucket losses. The loss is
the trainer's own rectified-flow velocity MSE with the randomness fixed: t a bucket centre, noise seeded per clip,
no CFG dropout, no inpainting, the padding mask as the trainer masks it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __name__ in ("__main__", "__mp_main__"):  # the room's contract, only through engines/launch.py
    import pathlib
    import runpy
    runpy.run_path(str(pathlib.Path(__file__).resolve().parents[1] / "launch.py"), run_name="khaos_contract_check")


def pair_noise_seed(seed: int, name: str) -> int:
    """The contract's per-clip noise seed, read once the launcher has put the room's contract in place."""
    from khaos_attribution.gradient import pair_noise_seed as seed_of  # noqa: PLC0415
    return seed_of(seed, name)


def _load_adapters(wrapped, paths: list, torch) -> None:
    from stable_audio_3.models.lora.utils import name_is_lora  # noqa: PLC0415
    wrapped.load_lora([str(p) for p in paths])
    wrapped.set_lora_strength(1.0)
    if any(p.dtype != torch.float32 for n, p in wrapped.model.named_parameters() if name_is_lora(n)):
        raise RuntimeError("the adapter's parameters are not float32, the precision they trained at")


def _switched_off_is_base(wrapped, torch) -> bool:
    """Whether one adapter on and the rest off is that adapter loaded alone: each layer's adapters sit at the
    positions their indexes name (the loader keys their weights by index, so an adapter at another position would
    load none), and at strength 0 every adapted weight equals its base."""
    model = wrapped.model
    layers = [(mod, plist) for mod in model.modules()
              if (plist := getattr(getattr(mod, "parametrizations", None), "weight", None)) is not None]
    if any(getattr(p, "lora_index", pos) != pos for _mod, plist in layers for pos, p in enumerate(plist)):
        return False
    wrapped.set_lora_strength(0.0)
    with torch.no_grad():
        return bool(layers) and all(torch.equal(mod.weight, plist.original) for mod, plist in layers)


def heldout_torch_many(req: dict) -> dict:
    """The held-out losses under each of several adapters from one load of the base: every adapter loaded at its own
    index, then each switched on alone; a fallback reply when switched-off adapters would not leave the base whole."""
    import torch  # noqa: PLC0415
    from khaos_attribution.engines.sa3.loading import load_measured  # noqa: PLC0415
    wrapped = load_measured(req["model"])
    _load_adapters(wrapped, req["adapters"], torch)
    if not _switched_off_is_base(wrapped, torch):
        return {"per_adapter": [], "fallback": "an adapter switched off would not leave the base whole", "runtime": "torch"}
    per_adapter = []
    for i in range(len(req["adapters"])):
        wrapped.set_lora_strength(0.0)
        wrapped.set_lora_strength(1.0, lora_index=i)
        per_adapter.append(_score_torch(wrapped, req, torch))
    return {"per_adapter": per_adapter, "runtime": "torch"}


def heldout_torch(req: dict) -> dict:
    import torch  # noqa: PLC0415
    from khaos_attribution.engines.sa3.loading import load_measured  # noqa: PLC0415

    wrapped = load_measured(req["model"])
    if req.get("adapter"):
        _load_adapters(wrapped, [req["adapter"]], torch)
    return {"per_sample": _score_torch(wrapped, req, torch), "runtime": "torch"}


def _score_torch(wrapped, req: dict, torch) -> dict:
    """Per-sample bucket losses of the clips under the model as it stands: the trainer's forward, noise seeded per
    clip by torch's own generator, the input at the model's precision."""
    from khaos_attribution.engines.sa3 import forward_torch as F  # noqa: PLC0415
    model = wrapped.model
    model.eval()
    device = wrapped.device
    timesteps = [float(t) for t in req["timesteps"]]
    per_sample = {}
    with torch.no_grad():
        for wav in sorted(Path(req["clips_dir"]).glob("*.wav")):
            latents, n, valid = F.encode(model, torch, wav, device)
            cond = F.condition(model, torch, latents, F.caption(wav), n / int(model.sample_rate), device)
            mask = F.padding_mask(torch, latents, valid)
            gen = torch.Generator(device="cpu").manual_seed(pair_noise_seed(int(req["seed"]), wav.name))
            noise = torch.randn(latents.shape, generator=gen, dtype=torch.float32).to(latents.device)
            target = noise - latents
            row = {}
            for t in timesteps:
                out = F.forward(model, torch, latents, cond, mask, noise, t,
                                input_dtype=getattr(model, "dtype", latents.dtype))
                mse = ((out - target) ** 2).mean(dim=1)
                row[f"{t:.2f}"] = float(mse[mask].mean())
            per_sample[wav.name] = row
    return per_sample


def heldout_mlx(req: dict) -> dict:
    """The MLX trainer's own loss on its pre-encoded latents, with the model, the conditioner and the adapter loaded
    the way lora_train_mlx.py loads them in main(); t a bucket centre, noise seeded per clip, no CFG dropout."""
    import numpy as np  # noqa: PLC0415
    mlx_dir = Path.cwd() / "optimized" / "mlx"
    sys.path.insert(0, str(mlx_dir))
    sys.path.insert(0, str(mlx_dir / "scripts"))
    import mlx.core as mx  # noqa: PLC0415
    from lora_train_mlx import rectified_flow_loss  # noqa: PLC0415
    from khaos_attribution.engines import mlx_cache  # noqa: PLC0415
    from khaos_attribution.engines.sa3 import forward_mlx as FM  # noqa: PLC0415
    mlx_cache.apply(mx)

    latents_dir = Path(req["latents_dir"])
    items = []
    for npy in sorted(latents_dir.glob("*.npy")):
        meta = json.loads(npy.with_suffix(".json").read_text(encoding="utf-8"))
        items.append((npy, meta, np.load(str(npy)).astype(np.float32)))
    if not items:
        raise RuntimeError(f"no pre-encoded latents under {latents_dir}")
    t_lat = max(arr.shape[-1] for _n, _m, arr in items)
    bundle, t5, padding_emb, secs_embedder, _state = FM.load_bundle(req, t_lat, mx.float16)
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
        seconds = float(meta.get("seconds_total") or 0.0)
        cross16, global16 = FM.conditioning(mx, prompt, seconds, t5, padding_emb, secs_embedder, bundle, mx.float16)
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
    from khaos_attribution.engines.worker_process import answer  # noqa: PLC0415
    return answer(lambda req: heldout_mlx(req) if req.get("runtime") == "mlx"
                  else heldout_torch_many(req) if "adapters" in req else heldout_torch(req))


if __name__ == "__main__":
    sys.exit(main())
