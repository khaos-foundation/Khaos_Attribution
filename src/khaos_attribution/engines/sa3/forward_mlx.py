"""The MLX trainer's model and conditioning on Stable Audio 3, in one place for every worker that takes them: the
DiT at a precision and length, T5Gemma, the conditioner and the adapter, loaded as lora_train_mlx.py's main() loads
them, and one clip's cross-attention and global conditioning. The caller has put the MLX port on sys.path."""

from __future__ import annotations


def load_bundle(req: dict, t_lat: int, dtype) -> tuple:
    """(bundle, t5, padding embedding, seconds embedder, adapter state): the base alone, its state None, when the
    request names no adapter."""
    import importlib  # noqa: PLC0415
    from lora_train_mlx import (DIT_CHOICES, T5GEMMA_NPZ_REL, T5Gemma, TrainBundle, TrainableSecondsEmbedder,  # noqa: PLC0415
                                ensure_local, inject_from_lora_config, load_conditioner_from_npz, load_lora_checkpoint,
                                load_trainable_lora_state, underfit_lora_config)
    base_path = str(req["base_npz"])
    mod = importlib.import_module(DIT_CHOICES["medium"]["loader"])
    dit = mod.load_dit(base_path, T_lat=int(t_lat), dtype=dtype, compile_=False)
    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    padding_emb, secs_embedder = load_conditioner_from_npz(base_path, prefix="cond.")
    secs = state = None
    if req.get("adapter"):
        state, cfg = load_lora_checkpoint(str(req["adapter"]))
        lora_config = underfit_lora_config()
        lora_config.update({k: cfg[k] for k in ("adapter_type", "rank", "alpha", "include", "exclude")
                            if (cfg or {}).get(k) is not None})
        inject_from_lora_config(dit, lora_config, checkpoint_prefix="model.")
        if any(str(k).startswith("conditioners.seconds_total.") for k in state):
            secs = TrainableSecondsEmbedder(secs_embedder.W, secs_embedder.b)
            inject_from_lora_config(secs, lora_config, checkpoint_prefix="conditioners.seconds_total.")
    bundle = TrainBundle(dit, secs)
    if state is not None:
        load_trainable_lora_state(bundle, state)
    return bundle, t5, padding_emb, secs_embedder, state


def conditioning(mx, prompt: str, seconds: float, t5, padding_emb, secs_embedder, bundle, dtype) -> tuple:
    """(cross-attention, global) conditioning of one clip as the trainer builds it, at ``dtype``, detached."""
    from lora_train_mlx import build_conditioning  # noqa: PLC0415
    prompt_cond = build_conditioning(t5, padding_emb, [prompt])
    if bundle.secs is not None:
        sec_tok = bundle.secs(mx.array([seconds], dtype=mx.float32))
    else:
        sec_tok = secs_embedder([seconds])
    sec_tok = mx.stop_gradient(sec_tok.astype(mx.float32))
    return mx.concatenate([prompt_cond, sec_tok], axis=1).astype(dtype), sec_tok[:, 0, :].astype(dtype)
