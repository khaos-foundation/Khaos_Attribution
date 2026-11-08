"""Stability's MLX trainer, run inside the engine's MLX environment with what its command line lacks: the warm-up then
cosine rate, and a batch over one (its local conditioning is built for one clip, so the DiT call widens it).
Stability's code is not changed: its lora_train_mlx.py is run as a script with the arguments that remain once Khaos's
own are taken out. The AdamW it builds carries the rate as a function of the optimiser's own step, which the compiled
step reads from the optimiser's state; the flags and the curve are the CUDA driver's (train_driver.py)."""

from __future__ import annotations

import importlib.util
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MLX_TRAINER = Path("optimized") / "mlx" / "scripts" / "lora_train_mlx.py"


def cuda_driver():
    """train_driver.py beside this file, loaded by path: the engine's environment may hold an older contract."""
    spec = importlib.util.spec_from_file_location("khaos_sa3_train_driver", HERE / "train_driver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule_table(total: int, curve) -> list:
    """The rate multiplier at every step of a run of ``total`` steps, from the CUDA driver's curve."""
    return [curve(step, total) for step in range(max(1, int(total)))]


def flag_value(args: list, flag: str, cast, default):
    return cast(args[args.index(flag) + 1]) if flag in args else default


def scheduled_adamw(adamw, mx, table: list, lr: float, offset: int):
    """AdamW on the table's rate, read at the optimiser's step plus the steps taken before this process."""
    rates = mx.array(table, dtype=mx.float32)
    last = len(table) - 1

    def rate(step):
        return lr * mx.take(rates, mx.minimum(step + offset, last))

    class ScheduledAdamW(adamw):
        def __init__(self, learning_rate=None, **kwargs):
            super().__init__(learning_rate=rate, **kwargs)

    return ScheduledAdamW


def batch_wide_conditioning(dit_cls, mx):
    """The DiT's call with a one-clip local conditioning widened to the batch it is called with."""
    call = dit_cls.__call__

    def widened(self, x, t, cross_attn_cond_raw, global_cond_raw, local_add_cond=None):
        if local_add_cond is not None and local_add_cond.shape[0] == 1 and x.shape[0] > 1:
            local_add_cond = mx.broadcast_to(local_add_cond, (x.shape[0], *local_add_cond.shape[1:]))
        return call(self, x, t, cross_attn_cond_raw, global_cond_raw, local_add_cond=local_add_cond)

    return widened


def main() -> int:
    driver = cuda_driver()
    try:
        args, opts = driver.split_khaos_flags(sys.argv[1:])
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if opts["common_random_numbers"]:
        raise SystemExit("common random numbers are a CUDA option")
    # This file's folder holds the contract's own train.py and render.py; Stability's modules must not meet them.
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
    script = Path(os.environ.get("SA3_ROOT") or Path.cwd()) / MLX_TRAINER
    if opts["lr_schedule"] == "cosine":
        import mlx.core as mx  # noqa: PLC0415
        import mlx.optimizers as optim  # noqa: PLC0415
        total = opts["schedule_total"] or flag_value(args, "--max-steps", int, 0)
        table = schedule_table(total, driver.warmup_cosine)
        # The script names optim.AdamW when it builds its optimiser; this process is the run's alone.
        optim.AdamW = scheduled_adamw(optim.AdamW, mx, table, flag_value(args, "--lr", float, 0.0),
                                      opts["schedule_offset"])
    if flag_value(args, "--batch-size", int, 1) > 1:
        import importlib  # noqa: PLC0415

        import mlx.core as mx  # noqa: PLC0415
        # The DiT module Stability's loader imports for --dit, imported first so the trainer gets the widened call.
        sys.path[:0] = [str(script.parent.parent), str(script.parent)]
        loader = importlib.import_module("sa3_mlx").DIT_CHOICES[flag_value(args, "--dit", str, "medium")]["loader"]
        dit_cls = importlib.import_module(loader).DiT
        dit_cls.__call__ = batch_wide_conditioning(dit_cls, mx)
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
