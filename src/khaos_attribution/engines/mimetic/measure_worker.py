"""Gradient features on mimetic-a0, inside the engine's environment: the request as JSON on stdin, the reply as one
JSON line last on stdout. ``index`` writes one projected adapter gradient per clip (the D-TRAK feature, the
contract's count-sketch), ``feature`` the mean feature of an output's clips, ``loss`` the output's rectified-flow
loss with the adapter at full strength and with it off. Clips are taken as the trainer takes them, at float32."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(req: dict):
    import torch  # noqa: PLC0415
    from stable_audio_tools.models.lora.utils import get_lora_params, load_lora_checkpoint, name_is_lora  # noqa: PLC0415

    from khaos_attribution import gradient as G  # noqa: PLC0415
    from khaos_attribution.engines.mimetic.loading import apply_adapter, load_model  # noqa: PLC0415

    if not req.get("adapter"):
        raise RuntimeError("a feature needs the adapter; none was named")
    model, config, device = load_model(req.get("device"))
    apply_adapter(model, config, req["adapter"], 1.0)
    named = [(n, p) for n, p in model.named_parameters() if name_is_lora(n)]
    if not named or len(named) != len(list(get_lora_params(model))):
        raise RuntimeError("the adapter's parameters could not be named")
    state, _cfg = load_lora_checkpoint(str(req["adapter"]))
    # The adapter file's key order, so one adapter projects alike wherever it is measured.
    order = G.checkpoint_order([n for n, _p in named], list(state))
    params = [named[i][1] for i in order]
    if any(p.dtype != torch.float32 for p in params):
        raise RuntimeError("the adapter's parameters are not float32, the precision they trained at")
    for p in params:
        p.requires_grad_(True)
    return model, config, device, params, torch, sorted(state)


def _gradient(torch, params, clip, timesteps, target: str):
    """The mean adapter gradient over the timesteps, flat, float32, on the CPU."""
    from khaos_attribution.engines.mimetic import clip as C  # noqa: PLC0415
    latents, meta, mask, noise, model = clip
    device = latents.device
    total = None
    for t in timesteps:
        with torch.enable_grad():
            # Taken with gradient: the adapter's conditioner tensors learn in training, so they count here too.
            cond = model.conditioner(meta, device)
            out = C.forward(model, latents, cond, mask, noise, float(t))
            if target == "vnorm":
                loss = out[mask.unsqueeze(1).expand_as(out)].pow(2).mean()
            else:
                loss = C.velocity_mse(out, latents, noise, mask)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        flat = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).detach().to("cpu", torch.float32)
                          for g, p in zip(grads, params)])
        total = flat if total is None else total + flat
    return (total / float(len(timesteps))).numpy()


def run(req: dict) -> dict:
    import numpy as np  # noqa: PLC0415

    from khaos_attribution import gradient as G  # noqa: PLC0415
    from khaos_attribution.engines.mimetic import clip as C  # noqa: PLC0415
    from khaos_attribution.engines.mimetic.measure import ROW_PREFIX  # noqa: PLC0415

    model, config, device, params, torch, keys = _load(req)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")   # as the trainer runs
    timesteps = [float(t) for t in req["timesteps"]]
    noise_seed = int(req["noise_seed"])
    target = str(req.get("target") or "vnorm")
    files = [Path(f) for f in req["files"]]
    n_params = int(sum(int(p.numel()) for p in params))
    signature = G.parameter_signature(params)

    def clip_of(wav: Path):
        latents, _cond, mask, noise, meta = C.prepare(model, config, wav, device, G.pair_noise_seed(noise_seed, wav.name),
                                                      with_meta=True)
        return latents, meta, mask, noise, model

    command = req["command"]
    if command == "parameters":
        return {"n_params": n_params, "n_tensors": len(params), "signature": signature, "device": device,
                "compute_dtype": "fp32"}
    if command in ("index", "feature"):
        projection = (G.make_projection(n_params, int(req["projection"]["dim"]), int(req["projection"]["seed"]))
                      if command == "index" else G.projection_from_meta(req["index_meta"]))
        if projection.n_params != n_params:
            raise RuntimeError(f"the index was built on {projection.n_params} adapter parameters; this adapter has {n_params}")
        rows, vectors, raw = [], [], []
        for i, wav in enumerate(files):
            flat = _gradient(torch, params, clip_of(wav), timesteps, target)
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
        extra = {}
        if raw:
            np.save(str(req["keep_flat"]), np.stack(raw).astype(np.float32))
            extra = {"flat": str(req["keep_flat"]), "keys": keys}
        if command == "index":
            out = Path(req["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out), np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, projection.dim), np.float32))
            return {**extra, "rows": rows, "matrix": str(out), "n_params": n_params, "n_tensors": len(params),
                    "signature": signature, "device": device, "compute_dtype": "fp32"}
        return {**extra, "feature": np.mean(np.stack(vectors), axis=0).astype(float).tolist(), "n": len(vectors),
                "signature": signature}
    if command == "loss":
        from stable_audio_tools.models.lora import set_lora_strength  # noqa: PLC0415
        clips = [clip_of(wav) for wav in files]

        def mean_loss() -> float:
            # The conditioning is taken at the strength set: the adapter reaches the conditioner as well as the model.
            with torch.no_grad():
                return float(np.mean([float(C.velocity_mse(C.forward(m, lat, m.conditioner(meta, lat.device), mask, noise, t),
                                                           lat, noise, mask))
                                      for lat, meta, mask, noise, m in clips for t in timesteps]))
        set_lora_strength(model, 1.0)
        with_adapter = mean_loss()
        # Strength 0 gives the base model exactly (checked against no adapter at all).
        set_lora_strength(model, 0.0)
        without = mean_loss()
        set_lora_strength(model, 1.0)
        return {"with": with_adapter, "without": without, "n": len(clips)}
    raise RuntimeError(f"unknown command {command!r}")


def main() -> int:
    req = json.loads(sys.stdin.read())
    try:
        reply = run(req)
        try:
            import torch  # noqa: PLC0415
            reply["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if torch.cuda.is_available() else None
        except Exception:  # noqa: BLE001
            reply["peak_gb"] = None
    except Exception as exc:  # noqa: BLE001 — the reply carries it
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("\n" + json.dumps(reply) + "\n")
    sys.stdout.flush()
    return 0 if not reply.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
