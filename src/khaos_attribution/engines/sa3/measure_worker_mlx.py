"""Gradient features on Stable Audio 3's MLX runtime, inside the engine's MLX environment: the same commands and
reply as the torch worker. The forward is the MLX trainer's own (its pre-encoded latents, its conditioning, the
generation mask), the gradient is MLX's over the adapter's parameters, flattened in the adapter file's key order
and projected by the contract, so a Mac-trained adapter is measured on the weights it was trained and renders with."""

from __future__ import annotations

import json
import sys
from pathlib import Path

MLX_DIR = Path.cwd() / "optimized" / "mlx"
LATENTS_DIRNAME = "latents"
CHECKPOINT_ABOVE_LATENTS = 1024
# "torch" counts a clip's valid frames by the torch trainer's rounding, for comparing the two runtimes.
CONVENTIONS = ("mlx", "torch")
# The trainer's half-precision base; "fp32" is for checks that need the arithmetic exact enough to compare.
PRECISIONS = ("fp16", "fp32")
SAMPLES_PER_LATENT = 4096
_HALF = []


def _half(mx):
    return _HALF[0] if _HALF else mx.float16


def _imports():
    sys.path.insert(0, str(MLX_DIR))
    sys.path.insert(0, str(MLX_DIR / "scripts"))
    import mlx.core as mx  # noqa: PLC0415
    import mlx.nn as nn  # noqa: PLC0415
    return mx, nn


def _latents(files, encoder_box):
    """Each clip's latents, padding mask, prompt and seconds: the trainer's own pre-encoded files when they sit
    beside the clip, else the same pre-encoder run here."""
    import numpy as np  # noqa: PLC0415
    out = []
    for wav in files:
        npy = wav.parent / LATENTS_DIRNAME / (wav.stem + ".npy")
        txt = wav.with_suffix(".txt")
        if npy.is_file() and npy.with_suffix(".json").is_file():
            meta = json.loads(npy.with_suffix(".json").read_text(encoding="utf-8"))
            arr = np.load(str(npy)).astype(np.float32)
        else:
            from pre_encode_mlx import encode_file, load_audio, load_codec_encoder  # noqa: PLC0415
            if not encoder_box:
                encoder_box.append(load_codec_encoder("same-l"))
            encoder, pad_modulo = encoder_box[0]
            audio = load_audio(wav)
            arr, padding_mask, seconds = encode_file(encoder, audio, pad_modulo=pad_modulo)
            meta = {"padding_mask": padding_mask, "seconds_total": seconds, "audio_samples": int(audio.shape[-1])}
            if txt.is_file():
                meta["prompt"] = txt.read_text(encoding="utf-8", errors="replace").strip()
        prompt = str(meta.get("prompt") or (txt.read_text(encoding="utf-8").strip() if txt.is_file() else ""))
        out.append({"wav": wav, "latents": arr, "padding_mask": list(meta.get("padding_mask") or []),
                    "samples": int(meta.get("audio_samples") or 0),
                    "seconds": float(meta.get("seconds_total") or 0.0), "prompt": prompt})
    return out


def _load(req: dict, t_lat: int):
    """The DiT, the conditioner and the adapter, loaded as lora_train_mlx.py's main() loads them."""
    import importlib  # noqa: PLC0415
    mx, _nn = _imports()
    from lora_train_mlx import (DIT_CHOICES, T5GEMMA_NPZ_REL, T5Gemma, TrainBundle, TrainableSecondsEmbedder,  # noqa: PLC0415
                                ensure_local, inject_from_lora_config, load_conditioner_from_npz, load_lora_checkpoint,
                                load_trainable_lora_state, underfit_lora_config)
    if not req.get("adapter"):
        raise RuntimeError("a feature needs the adapter; none was named")
    base_path = str(req["base_npz"])
    mod = importlib.import_module(DIT_CHOICES["medium"]["loader"])
    dit = mod.load_dit(base_path, T_lat=int(t_lat), dtype=_half(mx), compile_=False)
    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    padding_emb, secs_embedder = load_conditioner_from_npz(base_path, prefix="cond.")
    state, cfg = load_lora_checkpoint(str(req["adapter"]))
    lora_config = underfit_lora_config()
    lora_config.update({k: cfg[k] for k in ("adapter_type", "rank", "alpha", "include", "exclude")
                        if (cfg or {}).get(k) is not None})
    inject_from_lora_config(dit, lora_config, checkpoint_prefix="model.")
    secs = None
    if any(str(k).startswith("conditioners.seconds_total.") for k in state):
        secs = TrainableSecondsEmbedder(secs_embedder.W, secs_embedder.b)
        inject_from_lora_config(secs, lora_config, checkpoint_prefix="conditioners.seconds_total.")
    bundle = TrainBundle(dit, secs)
    load_trainable_lora_state(bundle, state)
    if int(t_lat) > CHECKPOINT_ABOVE_LATENTS:
        from lora_train_mlx import grad_checkpoint  # noqa: PLC0415
        grad_checkpoint(dit.transformer.layers[0])
    return bundle, t5, padding_emb, secs_embedder, state


def _keyed(bundle) -> dict:
    """Each trainable parameter's path in the bundle mapped to its key in the adapter file."""
    from models.defs.lora import iter_trainable_lora_layers  # noqa: PLC0415
    layer_ids = {id(layer) for layer in iter_trainable_lora_layers(bundle)}
    names = {}
    for path, module in bundle.named_modules():
        if id(module) in layer_ids:
            names[path] = module.checkpoint_name
    return names


def _ordered_names(bundle, state) -> tuple:
    """(paths, keys) of the trainable parameters in the adapter file's key order, refused unless they are exactly
    the file's keys."""
    from mlx.utils import tree_flatten  # noqa: PLC0415
    from khaos_attribution import gradient as G  # noqa: PLC0415
    by_module = _keyed(bundle)
    paths, keys = [], []
    for path, _value in tree_flatten(bundle.trainable_parameters()):
        module, _, param = path.rpartition(".")
        if module not in by_module:
            raise RuntimeError(f"trainable parameter {path} is not an adapter's")
        paths.append(path)
        keys.append(f"{by_module[module]}.parametrizations.weight.0.{param}")
    order = G.checkpoint_order(keys, list(state))
    return [paths[i] for i in order], [keys[i] for i in order]


def _conditioning(mx, item, t5, padding_emb, secs_embedder, bundle):
    """cross_attn, global_cond and the local inpaint conditioning, as the trainer builds them, detached."""
    import numpy as np  # noqa: PLC0415
    from lora_train_mlx import build_conditioning  # noqa: PLC0415
    prompt_cond = build_conditioning(t5, padding_emb, [item["prompt"]])
    if bundle.secs is not None:
        sec_tok = bundle.secs(mx.array([item["seconds"]], dtype=mx.float32))
    else:
        sec_tok = secs_embedder([item["seconds"]])
    sec_tok = mx.stop_gradient(sec_tok.astype(mx.float32))
    cross16 = mx.concatenate([prompt_cond, sec_tok], axis=1).astype(_half(mx))
    global16 = sec_tok[:, 0, :].astype(_half(mx))
    # The generation mask: what the renderer, the torch trainer and Khaos's patched MLX trainer condition on.
    t_lat = item["latents"].shape[-1]
    lac = mx.array(np.zeros((1, 1 + item["latents"].shape[0], t_lat), np.float32).transpose(0, 2, 1)).astype(_half(mx))
    return cross16, global16, lac


def _prepared(mx, item, pair_seed: int, convention: str = "mlx"):
    """(clean, noise, loss mask) of one clip whole, at its own length as the torch worker takes it, never padded
    to another clip's; the loss mask is the MLX encoder's, or under "torch" the torch trainer's rounding of the clip's
    samples to latent frames. The noise is the contract's seeded draw."""
    import numpy as np  # noqa: PLC0415
    from khaos_attribution import gradient as G  # noqa: PLC0415
    arr = item["latents"]
    t_lat = arr.shape[-1]
    mask_np = np.zeros((1, t_lat), bool)
    if convention == "torch" and item.get("samples"):
        pm = np.ones(max(1, min(int(round(item["samples"] / SAMPLES_PER_LATENT)), t_lat)), bool)
    else:
        pm = np.asarray(item["padding_mask"] or [1] * t_lat, bool)[:t_lat]
    mask_np[0, :len(pm)] = pm
    noise = G.pair_noise(pair_seed, (1,) + arr.shape)
    return mx.array(arr[None]).astype(_half(mx)), mx.array(noise).astype(_half(mx)), mx.array(mask_np)


def _objective(mx, bundle, clean, noise, mask, t: float, cond, target: str):
    cross16, global16, lac = cond
    tt = mx.array([t], dtype=mx.float32)
    noised = clean * (1.0 - t) + noise * t
    pred = bundle.dit(noised, tt.astype(noised.dtype), cross16, global16, local_add_cond=lac).astype(mx.float32)
    if target == "vnorm":
        err = pred * pred
    else:
        err = (pred - (noise.astype(mx.float32) - clean.astype(mx.float32))) ** 2
    m = mask[:, None, :].astype(mx.float32)
    return mx.sum(err * m) / mx.maximum(mx.sum(m) * err.shape[1], 1.0)


def _gradient(mx, nn, bundle, paths, prepared, cond, timesteps, target: str):
    """The mean adapter gradient over the timesteps, flat, float32, in the adapter file's key order."""
    import numpy as np  # noqa: PLC0415
    from mlx.utils import tree_flatten  # noqa: PLC0415
    clean, noise, mask = prepared
    total = None
    for t in timesteps:
        def loss_fn(model):
            return _objective(mx, model, clean, noise, mask, float(t), cond, target)
        _loss, grads = nn.value_and_grad(bundle, loss_fn)(bundle)
        flat_tree = dict(tree_flatten(grads))
        mx.eval(list(flat_tree.values()))
        flat = np.concatenate([np.asarray(flat_tree[p].astype(mx.float32)).reshape(-1) for p in paths])
        total = flat if total is None else total + flat
    return total / float(len(timesteps))


def _set_strength(bundle, strength: float) -> None:
    from models.defs.lora import iter_trainable_lora_layers  # noqa: PLC0415
    for layer in iter_trainable_lora_layers(bundle):
        layer.scaling = float(strength) * layer.alpha / layer.rank


def _peak_gb(mx) -> float | None:
    fn = getattr(mx, "get_peak_memory", None) or getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    try:
        return round(fn() / 1024 ** 3, 2) if fn else None
    except Exception:  # noqa: BLE001
        return None


def run(req: dict) -> dict:
    import numpy as np  # noqa: PLC0415
    from khaos_attribution import gradient as G  # noqa: PLC0415
    mx, nn = _imports()
    convention = str(req.get("convention") or "mlx")
    if convention not in CONVENTIONS:
        raise RuntimeError(f"unknown convention {convention!r}")
    precision = str(req.get("precision") or "fp16")
    if precision not in PRECISIONS:
        raise RuntimeError(f"unknown precision {precision!r}")
    _HALF[:] = [mx.float32 if precision == "fp32" else mx.float16]
    timesteps = [float(t) for t in req["timesteps"]]
    noise_seed = int(req["noise_seed"])
    target = str(req.get("target") or "vnorm")
    files = [Path(f) for f in req["files"]]
    items = _latents(files, [])
    t_lat = max([it["latents"].shape[-1] for it in items] or [256])
    bundle, t5, padding_emb, secs_embedder, state = _load(req, t_lat)
    paths, keys = _ordered_names(bundle, state)
    from mlx.utils import tree_flatten  # noqa: PLC0415
    shapes = dict(tree_flatten(bundle.trainable_parameters()))
    sized = [shapes[p] for p in paths]
    n_params = int(sum(int(v.size) for v in sized))
    signature = G.parameter_signature(sized)
    command = req["command"]
    base = {"device": "mps", "compute_dtype": f"{precision}-base/fp32-adapter", "runtime": "mlx", "convention": convention}
    if command == "parameters":
        return {**base, "n_params": n_params, "n_tensors": len(paths), "signature": signature}
    if command in ("index", "feature"):
        projection = (G.make_projection(n_params, int(req["projection"]["dim"]), int(req["projection"]["seed"]))
                      if command == "index" else G.projection_from_meta(req["index_meta"]))
        if projection.n_params != n_params:
            raise RuntimeError(f"the index was built on {projection.n_params} adapter parameters; this adapter has {n_params}")
        rows, vectors, raw = [], [], []
        for i, item in enumerate(items):
            cond = _conditioning(mx, item, t5, padding_emb, secs_embedder, bundle)
            prepared = _prepared(mx, item, G.pair_noise_seed(noise_seed, item["wav"].name), convention)
            flat = _gradient(mx, nn, bundle, paths, prepared, cond, timesteps, target)
            if req.get("keep_flat"):
                raw.append(flat)
            projected = G.project(flat, projection)
            if command == "index":
                rows.append({"row": i, "pair_id": item["wav"].stem, "track_id": G.track_of_pair(item["wav"].stem),
                             "gradient_norm": round(float(np.linalg.norm(projected)), 8)})
                sys.stdout.write("khaos-row " + json.dumps(rows[-1]) + "\n")
                sys.stdout.flush()
                vectors.append(G.l2_normalise(projected))
            else:
                vectors.append(projected)
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        extra = {}
        if raw:
            flat_out = Path(req["keep_flat"])
            np.save(str(flat_out), np.stack(raw).astype(np.float32))
            extra = {"flat": str(flat_out), "keys": keys}
        if command == "index":
            out = Path(req["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out), np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, projection.dim), np.float32))
            return {**base, **extra, "rows": rows, "matrix": str(out), "n_params": n_params, "n_tensors": len(paths),
                    "signature": signature}
        return {**base, **extra, "feature": np.mean(np.stack(vectors), axis=0).astype(float).tolist(), "n": len(vectors)}
    if command == "loss":
        if any(str(k).endswith((".magnitude", ".magnitude_r", ".magnitude_c")) for k in state):
            raise RuntimeError("a weight-decomposed adapter cannot be switched off for a measurement")
        prepared = [(_conditioning(mx, it, t5, padding_emb, secs_embedder, bundle),
                     _prepared(mx, it, G.pair_noise_seed(noise_seed, it["wav"].name), convention)) for it in items]

        def mean_loss() -> float:
            vals = []
            for cond, (clean, noise, mask) in prepared:
                per_t = [_objective(mx, bundle, clean, noise, mask, t, cond, "loss") for t in timesteps]
                mx.eval(per_t)
                vals.append(float(np.mean([float(v) for v in per_t])))
            return float(np.mean(vals))

        _set_strength(bundle, 1.0)
        with_adapter = mean_loss()
        _set_strength(bundle, 0.0)
        without = mean_loss()
        _set_strength(bundle, 1.0)
        return {**base, "with": with_adapter, "without": without, "n": len(items)}
    raise RuntimeError(f"unknown command {command!r}")


def main() -> int:
    req = json.loads(sys.stdin.read())
    try:
        reply = run(req)
        mx, _nn = _imports()
        reply["peak_gb"] = _peak_gb(mx)
    except Exception as exc:  # noqa: BLE001 — the reply carries it
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("\n" + json.dumps(reply) + "\n")
    sys.stdout.flush()
    return 0 if not reply.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
