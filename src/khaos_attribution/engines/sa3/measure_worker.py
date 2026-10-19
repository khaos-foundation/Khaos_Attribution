"""Gradient features on Stable Audio 3, inside the engine's environment: the request as JSON on stdin, the reply as
one JSON line last on stdout. ``index`` writes one projected adapter gradient per clip (the D-TRAK feature, the
contract's count-sketch), ``feature`` the mean feature of an output's clips, ``loss`` the output's rectified-flow
loss with the adapter at full strength and with it off. The forward is the trainer's with the randomness fixed,
as the held-out worker takes it; the adapter's parameters are the ones Stability's trainer trains."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(req: dict):
    import torch  # noqa: PLC0415
    from khaos_attribution.engines.sa3.loading import load_measured  # noqa: PLC0415
    from stable_audio_3.models.lora.utils import get_lora_params  # noqa: PLC0415

    from khaos_attribution import gradient as G  # noqa: PLC0415
    from stable_audio_3.models.lora.utils import load_lora_checkpoint, name_is_lora  # noqa: PLC0415

    wrapped = load_measured(req["model"])
    if not req.get("adapter"):
        raise RuntimeError("a feature needs the adapter; none was named")
    wrapped.load_lora([str(req["adapter"])])
    wrapped.set_lora_strength(1.0)
    model = wrapped.model
    model.eval()
    named = [(n, p) for n, p in model.named_parameters() if name_is_lora(n)]
    if not named or len(named) != len(list(get_lora_params(model))):
        raise RuntimeError("the adapter's parameters could not be named")
    state, _cfg = load_lora_checkpoint(str(req["adapter"]))
    # The adapter file's key order, shared with the MLX worker, so one adapter projects alike on either runtime.
    order = G.checkpoint_order([n for n, _p in named], list(state))
    params = [named[i][1] for i in order]
    if any(p.dtype != torch.float32 for p in params):
        raise RuntimeError("the adapter's parameters are not float32, the precision they trained at")
    for p in params:
        p.requires_grad_(True)
    return wrapped, model, params, torch, sorted(state)


def _clip(model, torch, wav: Path, device, pair_seed: int):
    """One clip as the trainer sees it: latents, conditioning with an empty inpaint mask, the padding mask, the
    seeded noise."""
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
    caption = wav.with_suffix(".txt").read_text(encoding="utf-8").strip() if wav.with_suffix(".txt").exists() else ""
    with torch.no_grad():
        with torch.autocast(device_type=device if isinstance(device, str) else device.type, enabled=str(device).startswith("cuda")):
            latents = model.pretransform.encode(x)
    latents = latents.float().detach()
    # The conditioning is taken once and detached: the feature is the DiT adapter's gradient, and a conditioner
    # graph reused across timesteps would be walked twice.
    from khaos_attribution.engines.sa3.loading import measure_autocast  # noqa: PLC0415
    with torch.no_grad(), measure_autocast(device):
        cond = model.conditioner([{"prompt": caption, "seconds_total": n / sr}], device)
    cond = _detached(cond, torch)
    cond["inpaint_mask"] = [torch.zeros(1, 1, latents.shape[-1], dtype=torch.bool, device=latents.device)]
    cond["inpaint_masked_input"] = [torch.zeros_like(latents)]
    valid = max(1, min(int(round(n / ratio)), latents.shape[-1]))
    mask = torch.zeros(1, latents.shape[-1], dtype=torch.bool, device=latents.device)
    mask[:, :valid] = True
    from khaos_attribution import gradient as G  # noqa: PLC0415
    noise = torch.from_numpy(G.pair_noise(pair_seed, latents.shape)).to(latents.device)
    return latents, cond, mask, noise


def _detached(value, torch):
    if isinstance(value, dict):
        return {k: _detached(v, torch) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detached(v, torch) for v in value)
    if hasattr(value, "detach"):
        return value.detach()
    return value


def _forward(model, torch, latents, cond, mask, noise, t: float):
    from khaos_attribution.engines.sa3.loading import measure_autocast  # noqa: PLC0415
    tt = torch.full((1,), t, device=latents.device, dtype=torch.float32)
    noised = latents * (1.0 - t) + noise * t
    with measure_autocast(latents.device):
        return model(noised, tt, cond=cond, cfg_dropout_prob=0.0, padding_mask=mask).float()


def _gradient(model, torch, params, clip, timesteps, target: str):
    """The mean adapter gradient over the timesteps, flat, float32, on the CPU."""
    latents, cond, mask, noise = clip
    total = None
    for t in timesteps:
        with torch.enable_grad():
            out = _forward(model, torch, latents, cond, mask, noise, float(t))
            if target == "vnorm":
                loss = out[mask.unsqueeze(1).expand_as(out)].pow(2).mean()
            else:
                loss = ((out - (noise - latents)) ** 2)[mask.unsqueeze(1).expand_as(out)].mean()
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        flat = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).detach().to("cpu", torch.float32)
                          for g, p in zip(grads, params)])
        total = flat if total is None else total + flat
    return (total / float(len(timesteps))).numpy()


def _loss(model, torch, clip, timesteps) -> float:
    latents, cond, mask, noise = clip
    total = 0.0
    with torch.no_grad():
        for t in timesteps:
            out = _forward(model, torch, latents, cond, mask, noise, float(t))
            total += float(((out - (noise - latents)) ** 2)[mask.unsqueeze(1).expand_as(out)].mean())
    return total / float(len(timesteps))


def _base_dtype(model) -> str:
    """The precision the base model's weights are held at, as a result's name records it."""
    from khaos_attribution.gradient import dtype_tag  # noqa: PLC0415
    from stable_audio_3.models.lora.utils import name_is_lora  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415
    lora = {id(p) for n, p in model.named_parameters() if name_is_lora(n)}
    counts = Counter(str(p.dtype) for p in model.parameters() if id(p) not in lora)
    # The dtype most of the base's weights are held at; a few kept at full precision do not change the label.
    return dtype_tag(counts.most_common(1)[0][0]) if counts else "unknown"


def _peak_gb() -> float | None:
    """The worker's peak device memory, so the room can plan for it."""
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    except Exception:  # noqa: BLE001
        return None
    return None


def main() -> int:
    req = json.loads(sys.stdin.read())
    try:
        reply = run(req)
        reply["peak_gb"] = _peak_gb()
    except Exception as exc:  # noqa: BLE001 — the reply carries it
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("\n" + json.dumps(reply) + "\n")
    sys.stdout.flush()
    return 0 if not reply.get("error") else 1


def run(req: dict) -> dict:
    import numpy as np  # noqa: PLC0415
    from khaos_attribution.engines.sa3.measure import ROW_PREFIX  # noqa: PLC0415
    from khaos_attribution import gradient as G  # noqa: PLC0415

    wrapped, model, params, torch, keys = _load(req)
    device = wrapped.device
    timesteps = [float(t) for t in req["timesteps"]]
    noise_seed = int(req["noise_seed"])
    target = str(req.get("target") or "vnorm")
    files = [Path(f) for f in req["files"]]
    n_params = int(sum(int(p.numel()) for p in params))
    signature = G.parameter_signature(params)
    command = req["command"]
    if command == "parameters":
        return {"n_params": n_params, "n_tensors": len(params), "signature": signature, "device": str(device).split(":")[0],
                "compute_dtype": _base_dtype(model)}
    if command in ("index", "feature"):
        projection = (G.make_projection(n_params, int(req["projection"]["dim"]), int(req["projection"]["seed"]))
                      if command == "index" else G.projection_from_meta(req["index_meta"]))
        if projection.n_params != n_params:
            raise RuntimeError(f"the index was built on {projection.n_params} adapter parameters; this adapter has {n_params}")
        rows, vectors, raw = [], [], []
        for i, wav in enumerate(files):
            clip = _clip(model, torch, wav, device, G.pair_noise_seed(noise_seed, wav.name))
            flat = _gradient(model, torch, params, clip, timesteps, target)
            if req.get("keep_flat"):
                raw.append(flat)
            projected = G.project(flat, projection)
            if command == "index":
                norm = float(np.linalg.norm(projected))
                vectors.append(G.l2_normalise(projected))
                rows.append({"row": i, "pair_id": wav.stem, "track_id": G.track_of_pair(wav.stem), "gradient_norm": round(norm, 8)})
                sys.stdout.write(ROW_PREFIX + json.dumps(rows[-1]) + "\n")
                sys.stdout.flush()
            else:
                vectors.append(projected)
            del clip
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        extra = {}
        if raw:
            np.save(str(req["keep_flat"]), np.stack(raw).astype(np.float32))
            extra = {"flat": str(req["keep_flat"]), "keys": keys}
        if command == "index":
            out = Path(req["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out), np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, projection.dim), np.float32))
            return {**extra, "rows": rows, "matrix": str(out), "n_params": n_params, "n_tensors": len(params),
                    "signature": signature, "device": str(device).split(":")[0], "compute_dtype": _base_dtype(model)}
        return {**extra, "feature": np.mean(np.stack(vectors), axis=0).astype(float).tolist(), "n": len(vectors),
                "signature": signature}
    if command == "loss":
        # A weight-decomposed adapter at strength 0 is still not the base model: the reading is refused, never faked.
        from stable_audio_3.models.lora.utils import load_lora_checkpoint  # noqa: PLC0415
        state, _cfg = load_lora_checkpoint(str(req["adapter"]))
        # Any magnitude tensor means weight decomposition, whatever its axis (Stability's detector gives up on 1-D ones).
        if any(str(k).endswith(".magnitude") for k in state):
            raise RuntimeError("a weight-decomposed adapter cannot be switched off for a measurement")
        clips = [_clip(model, torch, wav, device, G.pair_noise_seed(noise_seed, wav.name)) for wav in files]
        wrapped.set_lora_strength(1.0)
        with_adapter = float(np.mean([_loss(model, torch, c, timesteps) for c in clips]))
        wrapped.set_lora_strength(0.0)
        without = float(np.mean([_loss(model, torch, c, timesteps) for c in clips]))
        wrapped.set_lora_strength(1.0)
        return {"with": with_adapter, "without": without, "n": len(clips)}
    raise RuntimeError(f"unknown command {command!r}")


if __name__ == "__main__":
    sys.exit(main())
