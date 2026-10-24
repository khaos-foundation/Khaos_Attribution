"""Gradient features of a live training module: one pair's adapter gradient
at fixed timesteps, the seeded count-sketch projection, and an output's
feature as the mean over its windows. Needs torch and the engine at run
time (the ``gradient`` extra); the module object is the trainer's own.
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

NORM_EPS = 1e-12
PROJECTION_KIND = "count_sketch"
TARGETS = ("loss", "vnorm")
SAMPLE_KEYS = ("target_latents", "attention_mask", "encoder_hidden_states",
               "encoder_attention_mask", "context_latents")


class GradientUnavailable(RuntimeError):
    """The module, the pair or the projection cannot give a feature, with the reason."""


def bucket_timesteps(n_buckets: int) -> list:
    """Bucket centres on (0, 1): ``(i + 0.5) / n``; the trainer's held-out grid."""
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")
    return [round((i + 0.5) / n_buckets, 6) for i in range(int(n_buckets))]


def pair_noise_seed(seed: int, pair_name: str) -> int:
    """Per-pair noise seed: the run seed mixed with a CRC of the pair's file
    name, so the noise never depends on iteration order."""
    return (int(seed) * 1_000_003 + zlib.crc32(pair_name.encode("utf-8"))) % (2**31 - 1)


def track_of_pair(pair_id: str) -> str:
    """The track a pair belongs to: everything before the last ``_seg_``."""
    head, sep, _ = str(pair_id).rpartition("_seg_")
    return head if sep else str(pair_id)


def pair_noise(seed: int, shape: Sequence[int]) -> np.ndarray:
    """A pair's noise drawn by NumPy's PCG64, so every runtime of an engine takes its feature on the same draw."""
    return np.random.default_rng(int(seed)).standard_normal(tuple(int(s) for s in shape)).astype(np.float32)


def checkpoint_order(names: Sequence[str], checkpoint_keys: Sequence[str]) -> list:
    """The positions of a runtime's adapter parameters sorted by the adapter file's own keys, so every runtime
    flattens one adapter in one order; a parameter the file lacks, or a key no parameter carries, is refused."""
    keys = set(checkpoint_keys)
    matched = {}
    for i, name in enumerate(names):
        hits = [k for k in keys if name == k or name.endswith("." + k)]
        if not hits:
            raise GradientUnavailable(f"adapter parameter {name} has no key in the adapter file")
        key = max(hits, key=len)
        if key in matched:
            raise GradientUnavailable(f"two adapter parameters claim the file's key {key}")
        matched[key] = i
    missing = sorted(keys - set(matched))
    if missing:
        raise GradientUnavailable(f"the adapter file holds {len(missing)} key(s) no parameter carries, e.g. {missing[0]}")
    return [matched[k] for k in sorted(matched)]


@dataclass
class SketchProjection:
    """A seeded count-sketch from ``n_params`` to ``dim``, reproducible from
    the three numbers alone; the random signs make collision terms cancel."""
    n_params: int
    dim: int
    seed: int
    buckets: np.ndarray
    signs: np.ndarray

    def fingerprint(self) -> str:
        return hashlib.sha256(
            f"{PROJECTION_KIND}|{self.n_params}|{self.dim}|{self.seed}".encode("utf-8")).hexdigest()[:16]


def make_projection(n_params: int, dim: int, seed: int) -> SketchProjection:
    """``default_rng`` (PCG64): its stream is fixed by NumPy's policy, so the
    same seed gives the same projection on every machine."""
    n_params, dim, seed = int(n_params), int(dim), int(seed)
    if n_params < 1 or dim < 1:
        raise ValueError("n_params and dim must be >= 1")
    rng = np.random.default_rng(seed)
    buckets = rng.integers(0, dim, size=n_params, dtype=np.int64)
    signs = (rng.integers(0, 2, size=n_params, dtype=np.int8).astype(np.float32) * 2.0) - 1.0
    return SketchProjection(n_params=n_params, dim=dim, seed=seed, buckets=buckets, signs=signs)


def projection_from_meta(meta: dict) -> SketchProjection:
    """Rebuild an index's projection from its record and refuse a fingerprint that disagrees."""
    p = meta["projection"]
    proj = make_projection(int(p["n_params"]), int(p["dim"]), int(p["seed"]))
    if p.get("fingerprint") and proj.fingerprint() != p["fingerprint"]:
        raise GradientUnavailable("the index's projection fingerprint does not match its own parameters")
    return proj


def project(values: Sequence[float] | np.ndarray, projection: SketchProjection) -> np.ndarray:
    """Project one flat parameter-space vector, accumulating in float64."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size != projection.n_params:
        raise ValueError(
            f"vector has {v.size} entries but this projection was built for "
            f"{projection.n_params} parameters — it is not from the same adapter")
    out = np.bincount(projection.buckets, weights=v * projection.signs, minlength=projection.dim)
    return out.astype(np.float32)


def param_filter_mask(names: Sequence[str], pattern: str | None) -> list:
    """Which adapter tensors a research filter keeps: a regex over the parameter names, every tensor when None."""
    import re  # noqa: PLC0415
    if not pattern:
        return [True] * len(names)
    rx = re.compile(pattern)
    kept = [bool(rx.search(str(n))) for n in names]
    if not any(kept):
        raise GradientUnavailable(f"the parameter filter {pattern!r} matches none of the adapter's {len(names)} tensors")
    return kept


def combine_timesteps(flats: Sequence[np.ndarray], *, timestep_norm: bool) -> np.ndarray:
    """One gradient per pair from its timesteps' gradients: their mean, each first scaled to unit length when asked."""
    if not flats:
        raise ValueError("combine_timesteps needs at least one timestep")
    rows = [np.asarray(f, dtype=np.float32) for f in flats]
    if timestep_norm:
        rows = [r / (float(np.linalg.norm(r)) + NORM_EPS) for r in rows]
    return (np.sum(rows, axis=0) / float(len(rows))).astype(np.float32)


def l2_normalise(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= NORM_EPS:
        return np.zeros_like(v)
    return (v / norm).astype(np.float32)


def adapter_parameters(module: Any) -> list:
    """The adapter's parameters by the trainer's rule: ``requires_grad`` is the boundary."""
    params = [p for p in module.model.parameters() if getattr(p, "requires_grad", False)]
    if not params:
        raise GradientUnavailable("this module has no trainable parameters: there is no adapter to take a gradient of")
    return params


def parameter_signature(params: Sequence[Any]) -> str:
    text = "|".join(f"{tuple(p.shape)}" for p in params)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_sample(path: Path | str) -> dict:
    """One preprocessed pair from disk: the five keys the training step reads."""
    import torch  # noqa: PLC0415

    data = torch.load(str(path), map_location="cpu", weights_only=True)
    missing = [k for k in SAMPLE_KEYS if k not in data]
    if missing:
        raise GradientUnavailable(
            f"{Path(path).name} is missing {missing} — it was not written by ACE-Step's preprocessor")
    return {k: data[k] for k in SAMPLE_KEYS}


def pair_gradient(module: Any, sample: dict, *, timesteps: Sequence[float], noise_seed: int,
                  params: Sequence[Any] | None = None, target: str = "vnorm",
                  timestep_norm: bool = False) -> np.ndarray:
    """The mean adapter gradient of one pair over ``timesteps``, flat, float32,
    from the trainer's forward with the randomness removed; ``vnorm``
    differentiates the predicted velocity's squared norm, ``loss`` the MSE."""
    import torch  # noqa: PLC0415
    from contextlib import nullcontext  # noqa: PLC0415

    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, not {target!r}")
    params = list(params) if params is not None else adapter_parameters(module)
    ts = [float(t) for t in timesteps]
    if not ts:
        raise ValueError("pair_gradient needs at least one timestep")
    dev, dtype = module.device, module.dtype
    x0_cpu = sample["target_latents"]
    gen = torch.Generator(device="cpu").manual_seed(int(noise_seed))
    noise = torch.randn(x0_cpu.shape, generator=gen, dtype=torch.float32)
    x0 = x0_cpu.unsqueeze(0).to(dev, dtype=dtype)
    x1 = noise.unsqueeze(0).to(dev, dtype=dtype)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(dev, dtype=dtype)
    ehs = sample["encoder_hidden_states"].unsqueeze(0).to(dev, dtype=dtype)
    eam = sample["encoder_attention_mask"].unsqueeze(0).to(dev, dtype=dtype)
    ctx = sample["context_latents"].unsqueeze(0).to(dev, dtype=dtype)
    flow = x1 - x0
    decoder = module.model.decoder
    was_training = bool(getattr(decoder, "training", True))
    if hasattr(decoder, "eval"):
        decoder.eval()
    total: Any = None
    try:
        for t in ts:
            tt = torch.full((1,), t, device=dev, dtype=dtype)
            t_ = tt.unsqueeze(-1).unsqueeze(-1)
            xt = t_ * x1 + (1.0 - t_) * x0
            if module.device_type in ("cuda", "xpu", "mps") and dtype != torch.float32:
                ctx_mgr = torch.autocast(device_type=module.device_type, dtype=dtype)
            else:
                ctx_mgr = nullcontext()
            with torch.enable_grad(), ctx_mgr:
                out = decoder(hidden_states=xt, timestep=tt, timestep_r=tt, attention_mask=attention_mask,
                              encoder_hidden_states=ehs, encoder_attention_mask=eam, context_latents=ctx)
                if target == "vnorm":
                    loss = out[0].float().pow(2).mean()
                else:
                    loss = torch.nn.functional.mse_loss(out[0], flow).float()
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            flat = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).detach().to("cpu", torch.float32)
                              for g, p in zip(grads, params)])
            if timestep_norm:
                flat = flat / (float(torch.linalg.norm(flat)) + NORM_EPS)
            total = flat if total is None else total + flat
    finally:
        if was_training and hasattr(decoder, "train"):
            decoder.train()
    return (total / float(len(ts))).numpy()


def dtype_tag(dtype: Any) -> str:
    """``torch.bfloat16`` → ``"bf16"``: the short name a precision is recorded under."""
    name = str(dtype).replace("torch.", "")
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(name, name)


def pair_loss(module: Any, sample: dict, *, timesteps: Sequence[float], noise_seed: int) -> float:
    """The flow-matching loss of one pair, the mean over ``timesteps``: the
    trainer's forward with the randomness removed and no gradient taken."""
    import torch  # noqa: PLC0415
    from contextlib import nullcontext  # noqa: PLC0415

    ts = [float(t) for t in timesteps]
    if not ts:
        raise ValueError("pair_loss needs at least one timestep")
    dev, dtype = module.device, module.dtype
    x0_cpu = sample["target_latents"]
    gen = torch.Generator(device="cpu").manual_seed(int(noise_seed))
    noise = torch.randn(x0_cpu.shape, generator=gen, dtype=torch.float32)
    target = noise - x0_cpu.to(torch.float32)
    x0 = x0_cpu.unsqueeze(0).to(dev, dtype=dtype)
    x1 = noise.unsqueeze(0).to(dev, dtype=dtype)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(dev, dtype=dtype)
    ehs = sample["encoder_hidden_states"].unsqueeze(0).to(dev, dtype=dtype)
    eam = sample["encoder_attention_mask"].unsqueeze(0).to(dev, dtype=dtype)
    ctx = sample["context_latents"].unsqueeze(0).to(dev, dtype=dtype)
    decoder = module.model.decoder
    was_training = bool(getattr(decoder, "training", True))
    if hasattr(decoder, "eval"):
        decoder.eval()
    total = 0.0
    try:
        for t in ts:
            tt = torch.full((1,), t, device=dev, dtype=dtype)
            t_ = tt.unsqueeze(-1).unsqueeze(-1)
            xt = t_ * x1 + (1.0 - t_) * x0
            if module.device_type in ("cuda", "xpu", "mps") and dtype != torch.float32:
                ctx_mgr = torch.autocast(device_type=module.device_type, dtype=dtype)
            else:
                ctx_mgr = nullcontext()
            with torch.no_grad(), ctx_mgr:
                out = decoder(hidden_states=xt, timestep=tt, timestep_r=tt, attention_mask=attention_mask,
                              encoder_hidden_states=ehs, encoder_attention_mask=eam, context_latents=ctx)
            pred = out[0].squeeze(0).detach().to("cpu", torch.float32)
            total += float(torch.nn.functional.mse_loss(pred, target))
    finally:
        if was_training and hasattr(decoder, "train"):
            decoder.train()
    return total / float(len(ts))


def output_loss(module: Any, meta: dict, files: Sequence[Path | str]) -> float:
    """An output's loss under the module's current weights: the mean over its
    windows, at the index's timesteps and per-pair noise seeds."""
    timesteps = [float(t) for t in meta["timesteps"]]
    noise_seed = int(meta["noise_seed"])
    losses = [pair_loss(module, load_sample(p), timesteps=timesteps,
                        noise_seed=pair_noise_seed(noise_seed, Path(p).name)) for p in files]
    if not losses:
        raise GradientUnavailable("an output needs at least one preprocessed window")
    return float(np.mean(losses))


def paired_loss(module: Any, meta: dict, files: Sequence[Path | str]) -> dict:
    """An output's loss with the adapter at full strength and with it switched
    off, on the same timesteps and noise. The base model is read, never rendered."""
    net = getattr(module, "lycoris_net", None)
    if net is None or not hasattr(net, "set_multiplier"):
        raise GradientUnavailable("this module's adapter cannot be switched off for a measurement")
    # A weight-decomposed adapter at multiplier 0 is still not the base model.
    if any(getattr(layer, "wd", False) for layer in getattr(net, "loras", ())):
        raise GradientUnavailable("a weight-decomposed adapter cannot be switched off for a measurement")
    try:
        net.set_multiplier(1.0)
        with_adapter = output_loss(module, meta, files)
        net.set_multiplier(0.0)
        without = output_loss(module, meta, files)
    finally:
        net.set_multiplier(1.0)
    return {"with": with_adapter, "without": without}


def output_feature(module: Any, meta: dict, files: Sequence[Path | str], *,
                   params: Sequence[Any] | None = None) -> np.ndarray:
    """An output's feature: the mean projected gradient over its windows, at
    the index's timesteps, target and per-pair noise seeds, under the
    module's current weights."""
    proj = projection_from_meta(meta)
    timesteps = [float(t) for t in meta["timesteps"]]
    noise_seed = int(meta["noise_seed"])
    target = str(meta.get("target", "vnorm"))
    if meta.get("param_filter"):
        raise GradientUnavailable("a parameter filter is not built on this engine's measure; rebuild the index without it")
    params = list(params) if params is not None else adapter_parameters(module)
    # The features are read as the index was built, normalisation included.
    vecs = [project(pair_gradient(module, load_sample(p), timesteps=timesteps,
                                  noise_seed=pair_noise_seed(noise_seed, Path(p).name),
                                  params=params, target=target, timestep_norm=bool(meta.get("timestep_norm", False))), proj)
            for p in files]
    if not vecs:
        raise GradientUnavailable("an output needs at least one preprocessed window")
    return np.mean(np.stack(vecs), axis=0)
