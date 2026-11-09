"""Stability's MLX render script, run inside the engine's MLX environment with what its command line lacks: the
base checkpoint an adapter trained on, integrated with Euler as Stability's own MLX training demos integrate it;
several parts masked in one pass; and the source mixed into the starting noise when the parts start below pure
noise, as the torch package does. Stability's code is not changed: the weights lookup, the sampler and the DiT call
the script imports are wrapped before it runs, and the script is given the span of the parts as its one range."""

from __future__ import annotations

import importlib
import math
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The script's own constants, read from it when the driver runs; these are what it holds at the pin.
SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096


def split_khaos_flags(argv: list) -> tuple:
    """``(script_args, regions, base)``: the script's own arguments, the parts as ``[(start, end), ...]`` seconds,
    and whether to render on the base checkpoint."""
    args, regions, base = [], [], False
    items = iter(argv)
    for item in items:
        if item == "--khaos-base":
            base = True
        elif item == "--khaos-regions":
            for part in next(items).split(";"):
                start, end = part.split(",")
                regions.append((float(start), float(end)))
        else:
            args.append(item)
    return args, regions, base


def flag_value(args: list, flag: str, cast, default):
    return cast(args[args.index(flag) + 1]) if flag in args else default


def latent_ranges(regions: list, frames: int, rate: int = SAMPLE_RATE, per_latent: int = SAMPLES_PER_LATENT) -> list:
    """Each part as latent frames: the start rounded as the script rounds, the end up to cover a partial frame."""
    out = []
    for start, end in regions:
        s0 = max(0, int(round(start * rate / per_latent)))
        s1 = min(frames, math.ceil(end * rate / per_latent - 1e-6))
        if s0 < s1:
            out.append((s0, s1))
    return out


def base_checkpoint(ensure_local, is_present, distilled: str, base: str) -> tuple:
    """The script's weights lookups with its distilled DiT file answered by the base one."""
    swap = lambda rel: base if str(rel) == distilled else rel  # noqa: E731
    return (lambda rel, *a, **kw: ensure_local(swap(rel), *a, **kw)), (lambda rel, *a, **kw: is_present(swap(rel), *a, **kw))


def euler_sampler(mx):
    """Euler integration of the base model's velocity, x += dt * v, as Stability's MLX demos run the base; with the
    script's sampler signature, so its paste-back and progress hooks hold."""
    def sample(model_fn, x, sigmas, seed=0, paste_back=None, on_step=None, before_step=None):
        steps = sigmas.shape[0] - 1
        for i in range(steps):
            t_curr, t_next = sigmas[i], sigmas[i + 1]
            if before_step is not None:
                before_step(i)
            v = model_fn(x, t_curr * mx.ones((x.shape[0],), dtype=x.dtype))
            x = x + (t_next - t_curr).astype(x.dtype) * v
            mx.eval(x)
            if on_step is not None:
                on_step(i + 1, steps)
        if paste_back is not None:
            init, mask = paste_back
            keep = mask.astype(x.dtype)
            x = init.astype(x.dtype) * keep + x * (1.0 - keep)
            mx.eval(x)
        return x
    return sample


def masked_sampler(sample, regions: list, sigma_max: float, state: dict, mx, np, rate: int = SAMPLE_RATE,
                   per_latent: int = SAMPLES_PER_LATENT):
    """The script's sampler with every part masked and, below pure noise, the source mixed into the start."""
    def sampler(model_fn, x, sigmas, seed=0, paste_back=None, on_step=None, before_step=None):
        if paste_back is not None:
            init, _ = paste_back
            frames = init.shape[-1]
            keep = np.ones((1, 1, frames), dtype=np.float32)
            for s0, s1 in latent_ranges(regions, frames, rate, per_latent):
                keep[:, :, s0:s1] = 0.0
            mask = mx.array(keep)
            masked = init.astype(mx.float32) * mask
            state["local_add_cond"] = mx.concatenate([mask, masked], axis=1).transpose(0, 2, 1)
            paste_back = (init, mask)
            if sigma_max < 1.0:
                x = (init * (1.0 - sigma_max) + x * sigma_max).astype(x.dtype)
        return sample(model_fn, x, sigmas, seed=seed, paste_back=paste_back, on_step=on_step, before_step=before_step)
    return sampler


def masked_call(call, state: dict, mx):
    """The DiT's call with the one-range conditioning the script built replaced by the every-part one."""
    def wrapped(self, x, t, cross_attn_cond_raw, global_cond_raw, local_add_cond=None):
        lac = state.get("local_add_cond")
        if lac is not None and local_add_cond is not None:
            local_add_cond = mx.broadcast_to(lac.astype(local_add_cond.dtype), (x.shape[0], *lac.shape[1:]))
        return call(self, x, t, cross_attn_cond_raw, global_cond_raw, local_add_cond=local_add_cond)
    return wrapped


def main() -> int:
    script = Path(sys.argv[1]).resolve()
    args, regions, base = split_khaos_flags(sys.argv[2:])
    # This file's folder holds the contract's own modules; Stability's must not meet them.
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
    sys.path[:0] = [str(script.parent.parent), str(script.parent)]
    import mlx.core as mx  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    state: dict = {}
    script_module = importlib.import_module("sa3_mlx")
    pipeline = importlib.import_module("models.defs.sa3_pipeline")
    dit = flag_value(args, "--dit", str, "medium")
    if base:
        # The trainer's own name for the base file of a DiT family; it loads through the same loader.
        weights = importlib.import_module("weights")
        weights.ensure_local, weights.is_present = base_checkpoint(
            weights.ensure_local, weights.is_present, script_module.DIT_CHOICES[dit]["ckpt"],
            f"models/mlx/dit_{dit}-base_f16.npz")
        pipeline.sample_flow_pingpong = euler_sampler(mx)
    if regions:
        pipeline.sample_flow_pingpong = masked_sampler(pipeline.sample_flow_pingpong, regions,
                                                       flag_value(args, "--init-noise-level", float, 1.0), state, mx,
                                                       np, script_module.SAMPLE_RATE, script_module.SAMPLES_PER_LATENT)
        dit_cls = importlib.import_module(script_module.DIT_CHOICES[dit]["loader"]).DiT
        dit_cls.__call__ = masked_call(dit_cls.__call__, state, mx)
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
