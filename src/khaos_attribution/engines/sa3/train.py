"""Stable Audio 3's train slot. Stability's trainers run as processes in the engine's own environment (the CUDA
trainer, or the MLX one on Apple silicon); this module hands them Khaos's pairs, follows their losses and
checkpoints, lays the checkpoints out as the Workshop expects, and gives it the same session ACE-Step gives it.
Khaos counts in epochs; Stability's trainers count in steps, and the session converts at the pair count."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

from khaos_attribution.engines.train import NoLossHooks
from khaos_attribution.engines.train import adapter_file_in as _adapter_file_in
from khaos_attribution.engines.render import RenderFailed
from khaos_attribution.engines.sa3 import DESCRIPTOR
from khaos_attribution.engines.sa3 import train_driver as TD
from khaos_attribution.engines.sa3.render import MLX_SUBDIR, WINDOWS, find_root, python_for, runtime_for
from khaos_attribution.engines import worker_process as W
from khaos_attribution.engines.worker_process import stop_tree
from khaos_attribution.engines import process_engine as PE
from khaos_attribution.engines import process_trainer as PT
from khaos_attribution.engines.process_engine import HeldoutEvaluator, TrainingApi  # noqa: F401
from khaos_attribution.engines.heldout import DEFAULT_BUCKETS, HIGH_NOISE_T, bucket_timesteps, summarise  # noqa: F401
from khaos_attribution.engines.train import TrainingUnavailable

log = logging.getLogger(__name__)

PAIR_SUFFIX = ".wav"
LATENTS_DIRNAME = "latents"
TRAINER_DIRNAME = "sa3"
BASE_NPZ = "medium-base.npz"
TELEMETRY = "loss_by_timestep.bin"
HELDOUT_WORKER = Path(__file__).with_name("heldout_worker.py")
CONVERT_WORKER = Path(__file__).with_name("convert_worker.py")
TRAIN_DRIVER = Path(__file__).with_name("train_driver.py")
TRAIN_DRIVER_MLX = Path(__file__).with_name("train_driver_mlx.py")
# The MLX trainer adapts the seconds conditioner apart from the DiT; a recipe excludes it under this pattern.
SECONDS_PATTERN = "seconds_total"
# A Mac run's command names Stability's script, or the driver that runs it.
MLX_TRAINER_SCRIPTS = ("lora_train_mlx.py", "train_driver_mlx.py")
LOSSES_CSV = "khaos_losses.csv"
POLL_SECONDS = 2.0
TORCH_MIN_PAIRS = 4
# Opt-in for research runs on one short track: the pairs are repeated to the floor instead of refused.
REPEAT_SMALL_RUNS_ENV = "KHAOS_SA3_REPEAT_SMALL_RUNS"
# The trainer saves weights only on its checkpoint cadence, so a held-out score exists only there.
HELDOUT_ON_CHECKPOINTS = True

__all__ = ["HELDOUT_ON_CHECKPOINTS",
           "LossHooks", "Session", "TrainingApi", "TrainingUnavailable", "adapter_file_in", "adapter_kwargs", "heldout_summary",
           "open_module", "open_session", "pair_files", "preprocess_pairs", "run_refusal", "training_api"]


def training_api(repo_dir=None, device: str = "auto") -> TrainingApi:
    root = find_root(repo_dir)
    if not root.is_dir():
        raise TrainingUnavailable(f"Stable Audio 3 not found at {root}")
    runtime = runtime_for(device)
    try:
        python = python_for(root, runtime)
    except RenderFailed as exc:
        raise TrainingUnavailable(str(exc)) from exc
    return TrainingApi(root=root, runtime=runtime, python=Path(python))


def adapter_kwargs(cfg: dict) -> tuple:
    """(adapter type, Stability's trainer arguments) a recipe describes: rank and alpha are the recipe's, the type
    the descriptor's unless the recipe names one of Stability's."""
    adapter_type = str(cfg.get("adapter_type") or DESCRIPTOR.adapter["type"])
    if adapter_type == "lokr":
        adapter_type = DESCRIPTOR.adapter["type"]
    return adapter_type, {"rank": int(cfg["adapter_rank"]), "alpha": int(cfg["adapter_alpha"]),
                          "dropout": float(cfg.get("dropout", 0.0))}


def adapter_file_in(directory: Path | str) -> Path | None:
    return _adapter_file_in(directory, DESCRIPTOR)


def run_refusal(run_dir: Path | str) -> str | None:
    """Why a run folder's adapter cannot be measured, or None: a run whose trainer record is missing, or one trained
    on a Mac under the all-ones inpaint mask the renderer never uses. A folder that is no run (a bundle) passes."""
    run_dir = Path(run_dir)
    if (run_dir / "training_config.yaml").is_file() and not (run_dir / TRAINER_DIRNAME / "train_command.json").is_file():
        return f"{run_dir.name}'s trainer record is missing, so how it was trained is unknown; it cannot be measured"
    if trained_mask(run_dir) == "ones":
        return (f"{run_dir.name} was trained on a Mac under the all-ones inpaint mask the renderer never uses; its "
                "gradients describe that mask, not the music. Retrain it, then measure it")
    return None


def mask_to_record(trainer_now: str | None, run_dir: Path | str, *, resuming: bool) -> str | None:
    """The mask a Mac run records: the trainer's today, except that a resumed run is only as clean as its first
    steps, so an earlier all-ones stretch stays on the record."""
    if resuming and trained_mask(run_dir) == "ones":
        return "ones"
    return trainer_now


def trained_mask(run_dir: Path | str) -> str | None:
    """The inpaint mask a run's MLX trainer conditioned on ("ones" for a Mac run from before Khaos recorded it), or
    None for a run not trained by the MLX trainer here."""
    path = Path(run_dir) / TRAINER_DIRNAME / "train_command.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not any(Path(str(part)).name in MLX_TRAINER_SCRIPTS for part in record.get("command") or []):
        return None
    return record.get("mlx_inpaint_mask") or "ones"


def trained_values(run_dir: Path | str) -> dict:
    """The rate schedule and common random numbers a run's trainer was given, read from its command record, which
    outranks a training_config.yaml that names a schedule the trainer never received; empty with no record."""
    path = Path(run_dir) / TRAINER_DIRNAME / "train_command.json"
    try:
        command = json.loads(path.read_text(encoding="utf-8")).get("command")
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            return {}
        opts = TD.split_khaos_flags(command)[1] if command else {}
        return {k: opts[k] for k in ("lr_schedule", "common_random_numbers") if k in opts}
    except (OSError, ValueError, AttributeError):
        return {}


def pair_files(tensor_dir: Path | str) -> list:
    return sorted(Path(tensor_dir).glob(f"*{PAIR_SUFFIX}"))


def repeat_pairs_to(tensor_dir: Path | str, n: int) -> list:
    """Copy the pairs under ``tensor_dir`` round-robin as ``<name>_rep<i>`` until ``n`` exist; the copies made."""
    tensor_dir = Path(tensor_dir)
    originals = [f for f in pair_files(tensor_dir) if not re.search(r"_rep\d+$", f.stem)]
    if not originals:
        raise TrainingUnavailable(f"no pairs under {tensor_dir} to repeat")
    made = []
    i = 0
    while len(pair_files(tensor_dir)) < n:
        src = originals[i % len(originals)]
        i += 1
        stem = f"{src.stem}_rep{i}"
        for sidecar in sorted(tensor_dir.glob(f"{src.stem}.*")):
            target = tensor_dir / f"{stem}{sidecar.suffix}"
            shutil.copy2(sidecar, target)
            if sidecar.suffix == PAIR_SUFFIX:
                made.append(target)
    return made


def open_module(api: TrainingApi, cfg: dict, adapter_file, *, checkpoint_root=None, device: str = "cuda",
                dataset_dir=None, output_dir=None, precision: str | None = None):
    """What a feature is taken on: a handle naming the engine's environment, its runtime and the adapter, which the
    measure worker of that runtime loads per call."""
    from khaos_attribution.engines.sa3.measure import Handle  # noqa: PLC0415
    adapter_file = Path(adapter_file)
    if not adapter_file.is_file():
        raise TrainingUnavailable(f"no adapter file at {adapter_file}")
    why = run_refusal(output_dir) if output_dir is not None else None
    if why:
        raise TrainingUnavailable(why)
    if api.runtime == "mlx":
        base_npz = Path(checkpoint_root or "") / BASE_NPZ
        if not base_npz.is_file():
            raise TrainingUnavailable(f"the MLX measure needs the base weights as {BASE_NPZ} under {checkpoint_root}; "
                                      "provision the engine first")
        return Handle(root=api.root, python=api.python, adapter=adapter_file.resolve(), device_type="mps",
                      dtype="float16", runtime="mlx", base_npz=base_npz.absolute())
    device_type = str(device).split(":")[0]
    # The torch worker holds the base as the trainer does on CUDA (bfloat16) and at full precision elsewhere.
    return Handle(root=api.root, python=api.python, adapter=adapter_file.resolve(), device_type=device_type,
                  dtype="bfloat16" if device_type == "cuda" else "float32")


# ── preprocessing: Khaos's dataset into Stability's clips ─────────────────────

def write_clips(dataset_json: Path | str, out_dir: Path | str) -> dict:
    """Every sample of a Khaos dataset.json as a clip beside its caption, at the engine's rate."""
    return PT.write_clips(dataset_json, out_dir, sample_rate=DESCRIPTOR.sample_rate, pair_suffix=PAIR_SUFFIX,
                          engine_name="Stable Audio 3", unavailable=TrainingUnavailable, log=log, require_audio=False)


def preprocess_pairs(api: TrainingApi, *, out_dir: Path | str, checkpoint_root: Path | str, variant: str = "base",
                     dataset_json: Path | str | None, device: str = "auto", precision: str | None = None,
                     audio_dir: Path | str | None = None) -> dict:
    """Clips and captions under ``out_dir``; on MLX the latents too (``out_dir/latents``), pre-encoded once."""
    if dataset_json is None:
        return {"processed": len(pair_files(out_dir)), "failed": 0, "reused": True}
    result = write_clips(dataset_json, out_dir)
    if api.runtime == "mlx" and result["processed"]:
        result["latents"] = pre_encode(api, Path(out_dir), Path(checkpoint_root))
    return result


def pre_encode(api: TrainingApi, clips_dir: Path, checkpoint_root: Path) -> dict:
    """Stability's MLX pre-encoder over the clips: ``.npy`` latents and ``.json`` sidecars under ``latents/``."""
    mlx = api.root / MLX_SUBDIR
    out = clips_dir / LATENTS_DIRNAME
    # No limit: a catalogue's encode has no measured bound, and the room's stop reaches the tree.
    code, _stdout, stderr = W.call(api.python, mlx / "scripts" / "pre_encode_mlx.py",
                                   ["-i", clips_dir, "-o", out, "--codec", "same-l"], cwd=mlx, env=_env(api),
                                   label="Stable Audio 3 pre-encoder", timeout=None)
    if code != 0:
        raise TrainingUnavailable("pre-encoding for the MLX trainer failed: " + "\n".join(stderr.strip().splitlines()[-6:]))
    return {"dir": str(out), "n": len(list(out.glob("*.npy")))}


def _env(api: TrainingApi) -> dict:
    return W.engine_env(api.root, SA3_ROOT=str(api.root))


# ── the session ──────────────────────────────────────────────────────────────

LossHooks = NoLossHooks


def steps_for(cfg: dict, n_pairs: int) -> dict:
    return PT.steps_for(cfg, n_pairs)


def longest_clip_seconds(clips_dir: Path) -> float:
    """The trainer pads every clip to one window; the longest clip, whole seconds, so no window is mostly silence."""
    import math  # noqa: PLC0415
    import wave  # noqa: PLC0415
    longest = 0.0
    for wav in pair_files(clips_dir):
        try:
            with wave.open(str(wav), "rb") as fh:
                longest = max(longest, fh.getnframes() / fh.getframerate())
        except (OSError, wave.Error, ZeroDivisionError):
            continue
    return float(math.ceil(longest)) if longest else 30.0


def recipe_options(cfg: dict, runtime: str, *, resuming: bool) -> dict:
    """The recipe's options beyond the adapter's shape and rate; one this runtime cannot honour is refused, never
    dropped, so a run's record says what it trained under."""
    schedule = str(cfg.get("lr_schedule") or "constant")
    if schedule not in TD.SCHEDULES:
        raise TrainingUnavailable(f"Stable Audio 3 trains at a constant or cosine rate, not {schedule!r}")
    given = cfg.get("adapter_exclude") or []
    # Stability matches a pattern by substring, so a blank one would take the adapter off every layer.
    exclude = [str(x).strip() for x in (given.split(",") if isinstance(given, str) else given) if str(x).strip()]
    if given and not exclude:
        raise TrainingUnavailable(f"adapter_exclude names no module pattern: {given!r}")
    crn = bool(cfg.get("common_random_numbers"))
    if runtime != "torch" and crn:
        raise TrainingUnavailable("common random numbers are a CUDA option; the Mac trainer draws its own")
    if resuming and crn:
        raise TrainingUnavailable("a common random numbers run cannot be resumed: the new process counts its epochs "
                                  "from one again, so the continuation would repeat epoch 1's order, crops and noise")
    if crn and max(1, int(cfg.get("batch_size") or 1)) != 1:
        raise TrainingUnavailable("common random numbers key each step to one clip, so they need batch_size 1")
    return {"lr_schedule": schedule, "adapter_exclude": exclude, "common_random_numbers": crn}


def schedule_flags(opts: dict, schedule: dict | None) -> list:
    """A cosine rate's flags for either driver: the whole run's length and the steps taken before this process."""
    if opts["lr_schedule"] == "constant":
        return []
    out = [TD.SCHEDULE_FLAG, opts["lr_schedule"]]
    if schedule:
        out += [TD.SCHEDULE_TOTAL_FLAG, str(int(schedule["total"])), TD.SCHEDULE_OFFSET_FLAG, str(int(schedule["offset"]))]
    return out


def trainer_command(api: TrainingApi, cfg: dict, *, tensor_dir: Path, save_dir: Path, checkpoint_root: Path,
                    steps: dict, seed: int, window_seconds: float, resume_from: Path | None,
                    schedule: dict | None = None) -> list:
    """Stability's trainer for this runtime, at the recipe's adapter shape and the planned steps; ``schedule`` is
    the whole run's {total, offset} in steps, so a resumed cosine run continues its curve."""
    adapter_type, a = adapter_kwargs(cfg)
    lr = float(cfg.get("learning_rate") or 1e-4)
    opts = recipe_options(cfg, api.runtime, resuming=bool(resume_from))
    batch = max(1, int(cfg.get("batch_size") or 1))
    if api.runtime == "torch":
        # Through the driver: Stability's script logs no loss and its demo callback renders at step 1 regardless.
        return [str(api.python), str(TRAIN_DRIVER),
                "--model", DESCRIPTOR.variant_config("base"), "--data_dir", str(tensor_dir),
                "--rank", str(a["rank"]), "--lora_alpha", str(a["alpha"]), "--adapter_type", adapter_type,
                "--dropout", str(a["dropout"]), "--lr", str(lr), "--steps", str(steps["total"]),
                "--checkpoint_every", str(steps["checkpoint_every"]), "--duration", str(window_seconds),
                "--seed", str(int(seed)), "--logger", "csv", "--name", "khaos", "--save_dir", str(save_dir),
                "--batch_size", str(batch),
                "--demo_every", str(10 ** 9), "--num_workers", "0" if WINDOWS or opts["common_random_numbers"] else "4",
                *(["--exclude", *opts["adapter_exclude"]] if opts["adapter_exclude"] else []),
                *schedule_flags(opts, schedule),
                *([TD.CRN_FLAG] if opts["common_random_numbers"] else []),
                *(["--lora_checkpoint", str(resume_from)] if resume_from else [])]
    mlx = api.root / MLX_SUBDIR
    crop = max(2, int(round(window_seconds * DESCRIPTOR.sample_rate / int(DESCRIPTOR.rework["frame_samples"]))))
    # The MLX trainer builds each clip's prompt from its sidecar tags; without this config it would train on an
    # EMPTY prompt. The caption (the pre-encoder's "prompt" tag, read from the .txt beside the clip) goes through
    # verbatim: Khaos's captions already carry the trigger phrase.
    save_dir.mkdir(parents=True, exist_ok=True)
    prompt_config = save_dir / "prompt_config.json"
    prompt_config.write_text(json.dumps({"use_tags": True, "tag_keys": ["prompt"], "hide_tag_names": True,
                                         "shuffle": False, "balance": {"tags": 100}}, indent=1) + "\n")
    # A recipe's list replaces the MLX trainer's own default, as Stability's --exclude does on CUDA.
    exclude = opts["adapter_exclude"]
    dit_exclude = [p for p in exclude if p != SECONDS_PATTERN]
    return [str(api.python), str(TRAIN_DRIVER_MLX), "--dit", "medium",
            "--prompt-config", str(prompt_config),
            "--dit-weights", str(checkpoint_root / BASE_NPZ), "--latents-dir", str(tensor_dir / LATENTS_DIRNAME),
            "--lr", str(lr), "--max-steps", str(steps["total"]), "--rank", str(a["rank"]), "--alpha", str(a["alpha"]),
            "--adapter-type", adapter_type, "--latent-crop-length", str(crop), "--grad-checkpoint",
            "--save-dir", str(save_dir), "--name", "khaos", "--checkpoint-every", str(steps["checkpoint_every"]),
            "--seed", str(int(seed)), "--demo-every", "0", "--batch-size", str(batch),
            *(["--exclude", ",".join(dit_exclude)] if exclude else []),
            *(["--no-conditioner-lora"] if SECONDS_PATTERN in exclude else []),
            *schedule_flags(opts, schedule),
            # Explicit zero offsets: the trainer then counts this run's steps from zero, as the torch one does.
            *(["--lora-ckpt-path", str(resume_from), "--step-offset", "0", "--epoch-offset", "0"] if resume_from else [])]


def losses_from_metrics_csv(save_dir: Path) -> list:
    """(step, loss) rows the driver's callback wrote so far (khaos_losses.csv: step 1-based, loss)."""
    return PT.CsvLossTail(Path(save_dir) / LOSSES_CSV)()


def losses_from_telemetry(path: Path) -> list:
    """(step, loss) from the MLX trainer's loss_by_timestep.bin (struct "Iff": step, t_mean, loss_mean)."""
    return PT.RecordLossTail(path)()


def checkpoint_step(path: Path) -> int | None:
    return PT.checkpoint_step(path)


class Session(PE.ProcessSession):
    """One training run on Stability's trainer, driven as a process: losses read from its logs, each of its
    checkpoints laid out as ``checkpoints/epoch_<n>_loss_<x>/adapter_model.safetensors``, the held-out loss taken
    by a worker on the latest checkpoint."""

    label = "Stable Audio 3"
    descriptor = DESCRIPTOR
    trainer_dirname = TRAINER_DIRNAME

    def __init__(self, api: TrainingApi, cfg: dict, *, checkpoint_root: Path, tensor_dir: Path, output_dir: Path,
                 seed: int, resume_from: Path | None = None) -> None:
        super().__init__(api, cfg, checkpoint_root=checkpoint_root, tensor_dir=tensor_dir, output_dir=output_dir,
                         seed=seed, steps=steps_for(cfg, len(pair_files(tensor_dir))))
        # The room names a checkpoint by its folder; the trainer wants the adapter file inside it.
        resume = Path(resume_from) if resume_from else None
        if resume is not None and resume.is_dir():
            resume = resume / DESCRIPTOR.adapter["filenames"][0]
        self.resume_from = resume
        self.window_seconds = float(cfg.get("window_sec") or 0.0) or longest_clip_seconds(self.tensor_dir)
        self._loss_tail = None

    def pair_files(self, tensor_dir: Path) -> list:
        return pair_files(tensor_dir)

    def run_steps(self) -> dict:
        """The steps this process runs: the epochs left after a resume, at the planned cadence."""
        left = max(1, self.steps["epochs"] - self.start_epoch)
        return {**self.steps, "epochs": left, "total": self.steps["per_epoch"] * left}

    def command(self) -> list:
        return trainer_command(self.api, self.cfg, tensor_dir=self.tensor_dir, save_dir=self.save_dir,
                               checkpoint_root=self.checkpoint_root, steps=self.run_steps(), seed=self.seed,
                               window_seconds=self.window_seconds, resume_from=self.resume_from,
                               schedule={"total": self.steps["total"],
                                         "offset": self.steps["per_epoch"] * self.start_epoch})

    def _losses(self) -> list:
        if self._loss_tail is None:
            self._loss_tail = (PT.RecordLossTail(self.save_dir / TELEMETRY) if self.api.runtime == "mlx"
                               else PT.CsvLossTail(self.save_dir / LOSSES_CSV))
        return self._loss_tail()

    def _set_aside_earlier_run(self) -> None:
        """A resumed run's logs and checkpoints move out of the way, or the old rows would replay as new epochs."""
        aside = self.save_dir / f"before_resume_{time.strftime('%Y%m%d_%H%M%S')}"
        for f in [self.save_dir / LOSSES_CSV, self.save_dir / TELEMETRY,
                  *self.save_dir.rglob("*.ckpt"), *self.save_dir.rglob("*.safetensors")]:
            rel = f.relative_to(self.save_dir)
            if not f.is_file() or any(part.startswith("before_resume_") for part in rel.parts):
                continue
            (aside / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(aside / rel))

    def _checkpoints(self) -> dict:
        found = {}
        for f in [*self.save_dir.rglob("*.ckpt"), *self.save_dir.rglob("*.safetensors")]:
            if any(part.startswith("before_resume_") for part in f.relative_to(self.save_dir).parts):
                continue
            step = checkpoint_step(f)
            if step:
                found[step] = f
        return found

    def _lay_out(self, step: int, source: Path, loss: float) -> Path:
        """The checkpoint in the Workshop's layout, converted to safetensors when the trainer wrote a .ckpt."""
        epoch, target_dir, target = self._checkpoint_target(step, loss)
        if source.suffix == ".ckpt":
            self._convert(CONVERT_WORKER, source, target, _env(self.api))
        else:
            shutil.copy2(source, target)
        self._laid_out(epoch, target)
        return target_dir

    def train(self):
        """The trainer's run as Khaos's updates: info at start, an epoch each time the step count crosses one
        (its loss the mean of the epoch's steps), a checkpoint each time one lands, complete or fail at the end."""
        from khaos_attribution.engines import contract_guard  # noqa: PLC0415
        contract_guard.require(self.api.python, engine="Stable Audio 3", engine_id=DESCRIPTOR.id, error=TrainingUnavailable)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.resume_from is not None:
            self._set_aside_earlier_run()
        self._sizes = {}
        self._loss_tail = None
        offset = self.start_epoch
        cmd = self.command()
        record = {"command": cmd, "steps": self.run_steps(), "window_seconds": self.window_seconds}
        if self.api.runtime == "mlx":
            from khaos_attribution.engines.sa3.provision import trainer_mask  # noqa: PLC0415
            record["mlx_inpaint_mask"] = mask_to_record(trainer_mask(self.api.root), self.output_dir,
                                                        resuming=self.resume_from is not None)
        yield from self._run_trainer(cmd, record, cwd=self.save_dir, env=_env(self.api),
                                     follow=lambda: self._follow(offset), stop_tree=stop_tree)

    def _follow(self, offset: int):
        return PT.follow(self, offset, started=f"[OK] Stable Audio 3 trainer started ({self.api.runtime})",
                         losses=self._losses, run_steps=self.run_steps(), poll=POLL_SECONDS)

    def heldout(self, clips_dir: Path, adapter: Path | None, timesteps: list) -> dict:
        return heldout_summary(self.api, clips_dir, adapter, seed=self.seed, timesteps=timesteps,
                               checkpoint_root=self.checkpoint_root)


def open_session(api: TrainingApi, cfg: dict, *, checkpoint_root, tensor_dir, output_dir, device: str,
                 precision: str, seed: int, resume_from=None, on_model_loaded=None) -> Session:
    """A session on Stability's trainer; nothing is loaded until train() runs, so on_model_loaded fires at once."""
    # Stability's CUDA trainer pre-fetches a demo batch of four whole pairs before the first step.
    if api.runtime == "torch" and len(pair_files(tensor_dir)) < TORCH_MIN_PAIRS:
        if os.environ.get(REPEAT_SMALL_RUNS_ENV) != "1":
            raise TrainingUnavailable(f"Stable Audio 3's trainer needs at least {TORCH_MIN_PAIRS} pairs; "
                                      f"{len(pair_files(tensor_dir))} under {tensor_dir}")
        made = repeat_pairs_to(tensor_dir, TORCH_MIN_PAIRS)
        log.info("%d pair(s) repeated to %d under %s (%s=1)", len(made), TORCH_MIN_PAIRS, tensor_dir,
                 REPEAT_SMALL_RUNS_ENV)
    if api.runtime == "mlx" and not (Path(checkpoint_root) / BASE_NPZ).is_file():
        raise TrainingUnavailable(f"the MLX trainer needs the base weights as {BASE_NPZ} under {checkpoint_root}; "
                                  "provision the engine first")
    if api.runtime == "mlx":
        from khaos_attribution.engines.sa3.provision import trainer_mask  # noqa: PLC0415
        if trainer_mask(api.root) != "zeros":
            raise TrainingUnavailable("the MLX trainer here still conditions on the all-ones inpaint mask, which the "
                                      "renderer never uses; provision the engine to patch it before training")
    if on_model_loaded is not None:
        on_model_loaded(None)
    return Session(api, cfg, checkpoint_root=checkpoint_root, tensor_dir=tensor_dir, output_dir=output_dir,
                   seed=seed, resume_from=resume_from)


# ── the held-out loss, by a worker on the latest checkpoint ──────────────────

def heldout_summary(api: TrainingApi, clips_dir: Path | str, adapter: Path | str | None, *, seed: int,
                    timesteps: list, checkpoint_root: Path | str | None) -> dict:
    """The held-out record of the clips under ``clips_dir`` scored under ``adapter`` (the base alone when None), by
    the worker the trainer's own held-out loss runs: what a datamodel reads per subset adapter."""
    clips_dir = Path(clips_dir).resolve()
    adapter = Path(adapter).resolve() if adapter else None
    root = Path(checkpoint_root) if checkpoint_root else Path(api.root) / "checkpoints"
    request = {"runtime": api.runtime, "model": DESCRIPTOR.variant_config("base"), "clips_dir": str(clips_dir),
               "latents_dir": str(clips_dir / LATENTS_DIRNAME), "adapter": str(adapter) if adapter else None,
               "timesteps": list(timesteps), "seed": int(seed), "high_noise_t": HIGH_NOISE_T,
               "checkpoint_root": str(root.resolve()), "base_npz": str((root / BASE_NPZ).absolute())}
    return PE.heldout_record(api.python, HELDOUT_WORKER, request, cwd=api.root, env=_env(api), label="Stable Audio 3",
                             timesteps=timesteps)
