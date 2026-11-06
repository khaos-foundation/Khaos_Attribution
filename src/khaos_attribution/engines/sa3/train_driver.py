"""Stability's CUDA trainer, run inside the engine's environment with what Khaos needs that its script does not give:
every step's loss written to khaos_losses.csv beside the checkpoints (the script logs no loss), the demo callback kept
quiet (it renders unmarked audio at step 1 whatever --demo_every says), and two opt-in recipe options its command line
lacks: a warm-up then cosine learning rate, and common random numbers across runs. Stability's code is not changed;
its train_lora.py is imported and its main() run with the arguments that remain once Khaos's own are taken out."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import sys
from pathlib import Path

LOSSES = "khaos_losses.csv"
SCHEDULE_FLAG = "--khaos_lr_schedule"
CRN_FLAG = "--khaos_common_random_numbers"
SCHEDULES = ("constant", "cosine")
WARMUP_FRACTION = 0.05


def split_khaos_flags(args: list) -> tuple[list, dict]:
    """(the arguments Stability's script takes, Khaos's options) from one command line."""
    rest, opts, i = [], {"lr_schedule": "constant", "common_random_numbers": False}, 0
    while i < len(args):
        if args[i] == SCHEDULE_FLAG:
            if i + 1 >= len(args) or args[i + 1] not in SCHEDULES:
                raise ValueError(f"{SCHEDULE_FLAG} takes one of {', '.join(SCHEDULES)}")
            opts["lr_schedule"] = args[i + 1]
            i += 2
        elif args[i] == CRN_FLAG:
            opts["common_random_numbers"] = True
            i += 1
        else:
            rest.append(args[i])
            i += 1
    return rest, opts


def warmup_cosine(step: int, total: int, warmup_fraction: float = WARMUP_FRACTION) -> float:
    """The learning-rate multiplier at an optimiser step: linear from zero over the warm-up, then cosine to zero."""
    total = max(1, int(total))
    warm = max(1, int(round(total * warmup_fraction)))
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))


def crn_seed(seed: int, epoch: int, key: str, purpose: str) -> int:
    """One clip's random numbers in one epoch, the same in every run that trains on that clip under that seed."""
    digest = hashlib.sha256(f"{int(seed)}|{int(epoch)}|{key}|{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & (2 ** 63 - 1)


def crn_order(keys: list, seed: int, epoch: int) -> list:
    """An epoch's order: one fixed shuffle of every clip name, so a run on fewer clips keeps the others' order."""
    return sorted(range(len(keys)), key=lambda i: crn_seed(seed, epoch, keys[i], "order"))


def clip_key(path) -> str:
    return Path(str(path)).name


def _seed_everything(value: int) -> None:
    import random  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    random.seed(value)
    np.random.seed(value % (2 ** 32))
    torch.manual_seed(value)


def main() -> int:
    try:
        args, opts = split_khaos_flags(sys.argv[1:])
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    import pytorch_lightning as pl  # noqa: PLC0415
    import torch  # noqa: PLC0415

    root = Path(os.environ.get("SA3_ROOT") or Path.cwd())
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    import train_lora  # noqa: PLC0415  — Stability's scripts/train_lora.py, found through the checkout on the path

    sys.argv = [sys.argv[0], *args]
    save_dir = Path(args[args.index("--save_dir") + 1]) if "--save_dir" in args else Path("./lora_checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    losses_path = save_dir / LOSSES
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 42
    batch = int(args[args.index("--batch_size") + 1]) if "--batch_size" in args else 1
    crn = opts["common_random_numbers"]
    if crn and batch != 1:
        raise SystemExit("common random numbers key each step to one clip, so they need --batch_size 1")
    epoch_now = {"epoch": 0}

    class KhaosLosses(pl.Callback):
        """One row per optimiser step: the step (1-based) and the loss the wrapper returned."""

        def __init__(self):
            self._fh = losses_path.open("a", newline="", encoding="utf-8")
            self._w = csv.writer(self._fh)
            if losses_path.stat().st_size == 0:
                self._w.writerow(["step", "loss"])

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            try:
                value = float(loss.detach().float().cpu()) if hasattr(loss, "detach") else float(loss)
            except (TypeError, ValueError):
                return
            self._w.writerow([int(trainer.global_step), f"{value:.6f}"])
            self._fh.flush()

        def on_train_end(self, trainer, module):
            self._fh.close()

    class CommonStepNumbers(pl.Callback):
        """Each step's noise, timestep and mask draws keyed to its clip and epoch, not to the run's history."""

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            _reals, metadata = batch
            _seed_everything(crn_seed(seed, epoch_now["epoch"], clip_key(metadata[0]["path"]), "step"))

    class QuietDemo(pl.Callback):
        """Stability's demo callback replaced: a Khaos run renders nothing unmarked."""

        def __init__(self, *a, **k):
            super().__init__()

    real_trainer = pl.Trainer

    # Stability's checkpoint callback saves on its cadence alone: when the run's length is not a multiple of it,
    # a second one saves the final step, so the last epoch's weights always exist to export.
    total = int(args[args.index("--steps") + 1]) if "--steps" in args else 0
    every = int(args[args.index("--checkpoint_every") + 1]) if "--checkpoint_every" in args else 0
    final = [pl.callbacks.ModelCheckpoint(every_n_train_steps=total, save_top_k=-1)] if total and every and total % every else []

    class TrainerWithLosses(real_trainer):
        def __init__(self, *a, **k):
            k["callbacks"] = [*(k.get("callbacks") or []), *final, KhaosLosses(), *([CommonStepNumbers()] if crn else [])]
            super().__init__(*a, **k)

    real_wrapper = train_lora.DiffusionCondTrainingWrapper

    class KhaosWrapper(real_wrapper):
        def __init__(self, *a, **k):
            if crn:
                # The adapter's starting weights depend on the seed alone, not on what loading the data drew first.
                _seed_everything(crn_seed(seed, 0, "", "init"))
            super().__init__(*a, **k)

        def configure_optimizers(self):
            out = super().configure_optimizers()
            if opts["lr_schedule"] == "constant":
                return out
            opt = out[0][0] if isinstance(out, tuple) or (isinstance(out, list) and isinstance(out[0], list)) else out[0]
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: warmup_cosine(step, total))
            return [opt], [{"scheduler": sched, "interval": "step"}]

    if crn:
        dataset_cls = train_lora.SampleDataset
        real_getitem = dataset_cls.__getitem__

        def seeded_getitem(self, index):
            # The crop and any other draw in loading follow the clip and epoch, so shared clips load alike.
            _seed_everything(crn_seed(seed, epoch_now["epoch"], clip_key(self.filenames[index]), "load"))
            return real_getitem(self, index)

        dataset_cls.__getitem__ = seeded_getitem

        class CommonOrder(torch.utils.data.Sampler):
            def __init__(self, dataset):
                self.keys = [clip_key(f) for f in dataset.filenames]

            def __iter__(self):
                epoch_now["epoch"] += 1
                return iter(crn_order(self.keys, seed, epoch_now["epoch"]))

            def __len__(self):
                return len(self.keys)

        real_loader = torch.utils.data.DataLoader

        class CommonLoader(real_loader):
            def __init__(self, dataset, *a, shuffle=False, **k):
                if shuffle and hasattr(dataset, "filenames"):
                    # In-process loading: a worker's own generator would draw the crops apart.
                    k.update(sampler=CommonOrder(dataset), num_workers=0)
                    k.pop("worker_init_fn", None)
                    shuffle = False
                super().__init__(dataset, *a, shuffle=shuffle, **k)

        train_lora.torch.utils.data.DataLoader = CommonLoader

    train_lora.pl.Trainer = TrainerWithLosses
    train_lora.DiffusionCondTrainingWrapper = KhaosWrapper
    # The text encoder is read from Stability's open base repository: the same files, no account.
    from khaos_attribution.engines.sa3.loading import load_for_training  # noqa: PLC0415
    train_lora.load_model = load_for_training
    train_lora.DiffusionCondInpaintDemoCallback = QuietDemo
    train_lora.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
