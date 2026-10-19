"""Stable Audio 3's train slot. Stability's trainers run as processes in the engine's own environment (the CUDA
trainer, or the MLX one on Apple silicon); this module hands them Khaos's pairs, follows their losses and
checkpoints, lays the checkpoints out as the Workshop expects, and gives it the same session ACE-Step gives it.
Khaos counts in epochs; Stability's trainers count in steps, and the session converts at the pair count."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from khaos_attribution.engines.render import RenderFailed
from khaos_attribution.engines.sa3 import DESCRIPTOR
from khaos_attribution.engines.sa3.render import MLX_SUBDIR, WINDOWS, find_root, python_for, runtime_for, stop_tree
from khaos_attribution.engines.train import TrainingUnavailable, TrainingUpdate
from khaos_attribution.identifiers import track_of_segment

log = logging.getLogger(__name__)

PAIR_SUFFIX = ".wav"
LATENTS_DIRNAME = "latents"
TRAINER_DIRNAME = "sa3"
BASE_NPZ = "medium-base.npz"
TELEMETRY = "loss_by_timestep.bin"
HIGH_NOISE_T = 0.8
DEFAULT_BUCKETS = 10
HELDOUT_WORKER = Path(__file__).with_name("heldout_worker.py")
CONVERT_WORKER = Path(__file__).with_name("convert_worker.py")
TRAIN_DRIVER = Path(__file__).with_name("train_driver.py")
LOSSES_CSV = "khaos_losses.csv"
POLL_SECONDS = 2.0
TORCH_MIN_PAIRS = 4

__all__ = ["LossHooks", "Session", "TrainingApi", "TrainingUnavailable", "adapter_file_in", "adapter_kwargs", "heldout_summary",
           "open_module", "open_session", "pair_files", "preprocess_pairs", "run_refusal", "training_api"]


@dataclass(frozen=True)
class TrainingApi:
    """Where the engine's environment is for this machine's runtime; nothing of Stability's is imported here."""
    root: Path
    runtime: str
    python: Path


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
    directory = Path(directory)
    for name in DESCRIPTOR.adapter["filenames"]:
        if (directory / name).is_file():
            return directory / name
    files = sorted(directory.glob("*.safetensors"))
    return files[0] if files else None


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
    if not any("lora_train_mlx.py" in str(part) for part in record.get("command") or []):
        return None
    return record.get("mlx_inpaint_mask") or "ones"


def pair_files(tensor_dir: Path | str) -> list:
    return sorted(Path(tensor_dir).glob(f"*{PAIR_SUFFIX}"))


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

def _resample_stereo(audio, sr: int, target_sr: int):
    import numpy as np  # noqa: PLC0415
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if sr == target_sr:
        return audio
    try:
        import librosa  # noqa: PLC0415
    except ImportError as exc:
        raise TrainingUnavailable("resampling a clip for Stable Audio 3 needs librosa in this environment") from exc
    return np.stack([librosa.resample(audio[:, c], orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
                     for c in range(2)], axis=1)


def write_clips(dataset_json: Path | str, out_dir: Path | str) -> dict:
    """Every sample of a Khaos dataset.json as ``<stem>.wav`` (44.1 kHz stereo, 16-bit) beside ``<stem>.txt``, its
    caption; lyrics never travel, the engine cannot sing."""
    import soundfile as sf  # noqa: PLC0415
    samples = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed = failed = 0
    for s in samples:
        stem = Path(s["filename"]).stem
        try:
            audio, sr = sf.read(str(s["audio_path"]), dtype="float32", always_2d=True)
            audio = _resample_stereo(audio, sr, DESCRIPTOR.sample_rate)
            sf.write(str(out / f"{stem}{PAIR_SUFFIX}"), audio, DESCRIPTOR.sample_rate, subtype="PCM_16")
            (out / f"{stem}.txt").write_text(str(s.get("caption") or "").strip() + "\n", encoding="utf-8")
            processed += 1
        except Exception as exc:  # noqa: BLE001 — one bad clip is counted, the rest go on
            log.error("Stable Audio 3 preprocess: %s failed: %s", stem, exc)
            failed += 1
    return {"processed": processed, "failed": failed}


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
    cmd = [str(api.python), str(mlx / "scripts" / "pre_encode_mlx.py"), "-i", str(clips_dir), "-o", str(out),
           "--codec", "same-l"]
    done = subprocess.run(cmd, cwd=str(mlx), capture_output=True, text=True, env=_env(api))
    if done.returncode != 0:
        raise TrainingUnavailable("pre-encoding for the MLX trainer failed: "
                                  + "\n".join(done.stderr.strip().splitlines()[-6:]))
    return {"dir": str(out), "n": len(list(out.glob("*.npy")))}


def _env(api: TrainingApi) -> dict:
    return {**os.environ, "PYTHONPATH": str(api.root), "SA3_ROOT": str(api.root), "WANDB_MODE": "disabled",
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")}


# ── the session ──────────────────────────────────────────────────────────────

class LossHooks:
    """Stability's trainer takes no seams from Khaos: every flag is off, and the context has nothing to undo."""

    def __init__(self, record_into: dict | None = None) -> None:
        self.flags = {"masked_loss": False, "length_buckets": False}
        self._record_into = record_into

    def manifest_fields(self) -> dict:
        return {}

    def __enter__(self) -> "LossHooks":
        return self

    def __exit__(self, *exc) -> None:
        if self._record_into is not None:
            self._record_into.update(self.manifest_fields())


def steps_for(cfg: dict, n_pairs: int) -> dict:
    """Khaos's epochs as Stability's steps: steps per epoch at the recipe's batch size, the total, and the
    checkpoint cadence in steps."""
    batch = max(1, int(cfg.get("batch_size") or 1))
    # The trainer keeps a short last batch, so an epoch is the batches it runs, not the full ones.
    per_epoch = max(1, -(-n_pairs // batch))
    epochs = int(cfg["max_epochs"])
    every_epochs = max(1, int(cfg.get("checkpoint_every_n_epochs") or 1))
    return {"per_epoch": per_epoch, "total": per_epoch * epochs, "checkpoint_every": per_epoch * every_epochs,
            "epochs": epochs}


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


def trainer_command(api: TrainingApi, cfg: dict, *, tensor_dir: Path, save_dir: Path, checkpoint_root: Path,
                    steps: dict, seed: int, window_seconds: float, resume_from: Path | None) -> list:
    """Stability's trainer for this runtime, at the recipe's adapter shape and the planned steps."""
    adapter_type, a = adapter_kwargs(cfg)
    lr = float(cfg.get("learning_rate") or 1e-4)
    if api.runtime == "torch":
        # Through the driver: Stability's script logs no loss and its demo callback renders at step 1 regardless.
        return [str(api.python), str(TRAIN_DRIVER),
                "--model", DESCRIPTOR.variant_config("base"), "--data_dir", str(tensor_dir),
                "--rank", str(a["rank"]), "--lora_alpha", str(a["alpha"]), "--adapter_type", adapter_type,
                "--dropout", str(a["dropout"]), "--lr", str(lr), "--steps", str(steps["total"]),
                "--checkpoint_every", str(steps["checkpoint_every"]), "--duration", str(window_seconds),
                "--seed", str(int(seed)), "--logger", "csv", "--name", "khaos", "--save_dir", str(save_dir),
                "--batch_size", str(max(1, int(cfg.get("batch_size") or 1))),
                "--demo_every", str(10 ** 9), "--num_workers", "0" if WINDOWS else "4",
                *(["--lora_checkpoint", str(resume_from)] if resume_from else [])]
    mlx = api.root / MLX_SUBDIR
    crop = max(2, int(round(window_seconds * DESCRIPTOR.sample_rate / 4096)))
    # The MLX trainer builds each clip's prompt from its sidecar tags; without this config it would train on an
    # EMPTY prompt. The caption (the pre-encoder's "prompt" tag, read from the .txt beside the clip) goes through
    # verbatim: Khaos's captions already carry the trigger phrase.
    save_dir.mkdir(parents=True, exist_ok=True)
    prompt_config = save_dir / "prompt_config.json"
    prompt_config.write_text(json.dumps({"use_tags": True, "tag_keys": ["prompt"], "hide_tag_names": True,
                                         "shuffle": False, "balance": {"tags": 100}}, indent=1) + "\n")
    return [str(api.python), str(mlx / "scripts" / "lora_train_mlx.py"), "--dit", "medium",
            "--prompt-config", str(prompt_config),
            "--dit-weights", str(checkpoint_root / BASE_NPZ), "--latents-dir", str(tensor_dir / LATENTS_DIRNAME),
            "--lr", str(lr), "--max-steps", str(steps["total"]), "--rank", str(a["rank"]), "--alpha", str(a["alpha"]),
            "--adapter-type", adapter_type, "--latent-crop-length", str(crop), "--grad-checkpoint",
            "--save-dir", str(save_dir), "--name", "khaos", "--checkpoint-every", str(steps["checkpoint_every"]),
            "--seed", str(int(seed)), "--demo-every", "0",
            # Explicit zero offsets: the trainer then counts this run's steps from zero, as the torch one does.
            *(["--lora-ckpt-path", str(resume_from), "--step-offset", "0", "--epoch-offset", "0"] if resume_from else [])]


def losses_from_metrics_csv(save_dir: Path) -> list:
    """(step, loss) rows the driver's callback wrote so far (khaos_losses.csv: step 1-based, loss)."""
    rows = []
    path = save_dir / LOSSES_CSV
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("loss") not in (None, ""):
                    rows.append((int(float(row.get("step") or 0)), float(row["loss"])))
    except (OSError, ValueError):
        return rows
    return rows


def losses_from_telemetry(path: Path) -> list:
    """(step, loss) from the MLX trainer's loss_by_timestep.bin (struct "Iff": step, t_mean, loss_mean)."""
    out = []
    try:
        data = path.read_bytes()
    except OSError:
        return out
    size = struct.calcsize("Iff")
    for i in range(0, len(data) - size + 1, size):
        step, _t, loss = struct.unpack_from("Iff", data, i)
        out.append((int(step), float(loss)))
    return out


def checkpoint_step(path: Path) -> int | None:
    import re  # noqa: PLC0415
    m = re.search(r"step=(\d+)", path.name)
    return int(m.group(1)) if m else None


class Session:
    """One training run on Stability's trainer, driven as a process: losses read from its logs, each of its
    checkpoints laid out as ``checkpoints/epoch_<n>_loss_<x>/adapter_model.safetensors``, the held-out loss taken
    by a worker on the latest checkpoint."""

    def __init__(self, api: TrainingApi, cfg: dict, *, checkpoint_root: Path, tensor_dir: Path, output_dir: Path,
                 seed: int, resume_from: Path | None = None) -> None:
        self.api = api
        self.cfg = cfg
        self.checkpoint_root = Path(checkpoint_root)
        self.tensor_dir = Path(tensor_dir)
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        # The room names a checkpoint by its folder; the trainer wants the adapter file inside it.
        resume = Path(resume_from) if resume_from else None
        if resume is not None and resume.is_dir():
            resume = resume / DESCRIPTOR.adapter["filenames"][0]
        self.resume_from = resume
        self.save_dir = self.output_dir / TRAINER_DIRNAME
        self.steps = steps_for(cfg, len(pair_files(self.tensor_dir)))
        self.window_seconds = float(cfg.get("window_sec") or 0.0) or longest_clip_seconds(self.tensor_dir)
        self.latest_adapter: Path | None = None
        self.latest_epoch = 0
        self.start_epoch = 0
        self._sizes: dict = {}
        self.module = None
        self.training_config = None
        self.process: subprocess.Popen | None = None

    def install_hooks(self, *, start_epoch: int = 0, stem_ids=(), record_into: dict | None = None) -> LossHooks:
        self.start_epoch = max(0, int(start_epoch or 0))
        return LossHooks(record_into=record_into)

    def run_steps(self) -> dict:
        """The steps this process runs: the epochs left after a resume, at the planned cadence."""
        left = max(1, self.steps["epochs"] - self.start_epoch)
        return {**self.steps, "epochs": left, "total": self.steps["per_epoch"] * left}

    def command(self) -> list:
        return trainer_command(self.api, self.cfg, tensor_dir=self.tensor_dir, save_dir=self.save_dir,
                               checkpoint_root=self.checkpoint_root, steps=self.run_steps(), seed=self.seed,
                               window_seconds=self.window_seconds, resume_from=self.resume_from)

    def _losses(self) -> list:
        if self.api.runtime == "mlx":
            return losses_from_telemetry(self.save_dir / TELEMETRY)
        return losses_from_metrics_csv(self.save_dir)

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

    def _settled(self, path: Path, final: bool) -> bool:
        """A checkpoint is read only once its size and mtime held still across two polls (or the trainer has
        exited): the trainer writes it in one long, non-atomic write."""
        try:
            st = path.stat()
        except OSError:
            return False
        seen = self._sizes.get(path)
        self._sizes[path] = (st.st_size, st.st_mtime_ns)
        return final or seen == (st.st_size, st.st_mtime_ns)

    def _epoch_of(self, step: int) -> int:
        return self.start_epoch + max(1, round(step / self.steps["per_epoch"]))

    def _lay_out(self, step: int, source: Path, loss: float) -> Path:
        """The checkpoint in the Workshop's layout, converted to safetensors when the trainer wrote a .ckpt."""
        epoch = self._epoch_of(step)
        target_dir = self.output_dir / "checkpoints" / f"epoch_{epoch}_loss_{loss:.4f}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / DESCRIPTOR.adapter["filenames"][0]
        if source.suffix == ".ckpt":
            done = subprocess.run([str(self.api.python), str(CONVERT_WORKER), str(source), str(target)],
                                  cwd=str(self.api.root), capture_output=True, text=True, env=_env(self.api))
            if done.returncode != 0 or not target.is_file():
                raise TrainingUnavailable("converting the checkpoint to safetensors failed: "
                                          + "\n".join(done.stderr.strip().splitlines()[-4:]))
        else:
            shutil.copy2(source, target)
        self.latest_adapter = target
        self.latest_epoch = epoch
        return target_dir

    def train(self):
        """The trainer's run as Khaos's updates: info at start, an epoch each time the step count crosses one
        (its loss the mean of the epoch's steps), a checkpoint each time one lands, complete or fail at the end."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.resume_from is not None:
            self._set_aside_earlier_run()
        self._sizes = {}
        offset = self.start_epoch
        cmd = self.command()
        record = {"command": cmd, "steps": self.run_steps(), "window_seconds": self.window_seconds}
        if self.api.runtime == "mlx":
            from khaos_attribution.engines.sa3.provision import trainer_mask  # noqa: PLC0415
            record["mlx_inpaint_mask"] = mask_to_record(trainer_mask(self.api.root), self.output_dir,
                                                        resuming=self.resume_from is not None)
        (self.save_dir / "train_command.json").write_text(json.dumps(record, indent=1) + "\n")
        log_file = (self.save_dir / "trainer_stdout.log").open("ab")
        # In the job's own process group, so the room's Stop and Pause signals reach the trainer as they do the job.
        self.process = subprocess.Popen(cmd, cwd=str(self.save_dir), env=_env(self.api), stdout=log_file,
                                        stderr=subprocess.STDOUT)
        try:
            yield from self._follow(offset)
        finally:
            # Whatever ends the loop (a consumer that stops reading, a conversion that fails), no trainer is left running.
            stop_tree(self.process)
            log_file.close()

    def _follow(self, offset: int):
        yield TrainingUpdate(step=0, loss=0.0, msg=f"[OK] Stable Audio 3 trainer started ({self.api.runtime})",
                             kind="info", epoch=offset)
        seen_steps = 0
        epoch_done = offset
        laid_out: set = set()
        epoch_losses: list = []
        losses_by_step: dict = {}
        last_loss = 0.0
        start = time.monotonic()
        per_epoch = self.steps["per_epoch"]
        while True:
            alive = self.process.poll() is None
            losses = self._losses()
            events: list = []
            for step, loss in losses[seen_steps:]:
                last_loss = loss
                losses_by_step[step] = loss
                epoch_losses.append(loss)
                if step % per_epoch == 0:
                    epoch_done = offset + step // per_epoch
                    mean = sum(epoch_losses) / len(epoch_losses)
                    epoch_losses = []
                    events.append((step, 0, TrainingUpdate(step=step, loss=mean, msg=f"[OK] Epoch {epoch_done}", kind="epoch",
                                                           epoch=epoch_done, epoch_time=time.monotonic() - start)))
                    start = time.monotonic()
            seen_steps = len(losses)
            highest = max(losses_by_step) if losses_by_step else 0
            for step, path in sorted(self._checkpoints().items()):
                # Laid out once complete and once its own loss row is in: the name carries that loss.
                if step in laid_out or not self._settled(path, final=not alive) or (alive and highest < step):
                    continue
                laid_out.add(step)
                loss_at = losses_by_step.get(step, last_loss)
                target_dir = self._lay_out(step, path, loss_at)
                events.append((step, 1, TrainingUpdate(step=step, loss=loss_at, msg="[OK] Checkpoint", kind="checkpoint",
                                                       epoch=self._epoch_of(step), checkpoint_path=str(target_dir))))
            # In step order, a checkpoint after the epoch it closes: what the trainer's own stream would say.
            for _step, _order, update in sorted(events, key=lambda e: (e[0], e[1])):
                yield update
            if not alive:
                break
            time.sleep(POLL_SECONDS)
        code = self.process.returncode
        if code != 0:
            tail = "\n".join((self.save_dir / "trainer_stdout.log").read_text(errors="ignore").splitlines()[-8:])
            yield TrainingUpdate(step=seen_steps, loss=last_loss, msg=f"trainer exited {code}: {tail}", kind="fail",
                                 epoch=epoch_done)
            return
        yield TrainingUpdate(step=seen_steps, loss=last_loss, msg="[OK] done", kind="complete",
                             epoch=max(epoch_done, self.steps["epochs"]))

    def heldout_evaluator(self, tensor_val_dir: Path | str, *, n_buckets: int):
        return HeldoutEvaluator(self, Path(tensor_val_dir), n_buckets=int(n_buckets))

    def save_final(self, final_dir: Path | str) -> Path:
        """The last epoch's checkpoint, as the adapter the run exports; an older one is refused so the room's
        salvage path records what it really took."""
        if self.latest_adapter is None or not self.latest_adapter.is_file():
            raise RuntimeError("Stable Audio 3 wrote no checkpoint to export")
        if getattr(self, "latest_epoch", 0) < self.steps["epochs"]:
            raise RuntimeError(f"the last checkpoint is epoch {self.latest_epoch} of {self.steps['epochs']}; "
                               "the final weights were not written")
        final_dir = Path(final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        target = final_dir / DESCRIPTOR.adapter["filenames"][0]
        shutil.copy2(self.latest_adapter, target)
        return target


def open_session(api: TrainingApi, cfg: dict, *, checkpoint_root, tensor_dir, output_dir, device: str,
                 precision: str, seed: int, resume_from=None, on_model_loaded=None) -> Session:
    """A session on Stability's trainer; nothing is loaded until train() runs, so on_model_loaded fires at once."""
    # Stability's CUDA trainer pre-fetches a demo batch of four whole pairs before the first step.
    if api.runtime == "torch" and len(pair_files(tensor_dir)) < TORCH_MIN_PAIRS:
        raise TrainingUnavailable(f"Stable Audio 3's trainer needs at least {TORCH_MIN_PAIRS} pairs; "
                                  f"{len(pair_files(tensor_dir))} under {tensor_dir}")
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

def bucket_timesteps(n_buckets: int = DEFAULT_BUCKETS) -> list:
    return [round((i + 0.5) / n_buckets, 6) for i in range(int(n_buckets))]


class HeldoutEvaluator:
    """The rectified-flow velocity MSE on the held-out clips at fixed timesteps with seeded noise, taken by a
    worker in the engine's environment on the adapter named last (the newest checkpoint, else the base alone)."""

    def __init__(self, session: Session, tensor_val_dir: Path, *, n_buckets: int) -> None:
        self.session = session
        # The worker runs in the engine's checkout, so every path it is handed is absolute.
        self.tensor_val_dir = Path(tensor_val_dir).resolve()
        self.timesteps = bucket_timesteps(n_buckets)
        self.adapter: Path | None = None
        self.samples = pair_files(tensor_val_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def load_adapter_weights(self, weights_path: Path | str) -> bool:
        self.adapter = Path(weights_path)
        return True

    def evaluate(self) -> dict:
        adapter = self.adapter or self.session.latest_adapter
        return heldout_summary(self.session.api, self.tensor_val_dir, adapter, seed=self.session.seed,
                               timesteps=self.timesteps, checkpoint_root=self.session.checkpoint_root)


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
    done = subprocess.run([str(api.python), str(HELDOUT_WORKER)], input=json.dumps(request), cwd=str(api.root),
                          capture_output=True, text=True, env=_env(api))
    last = (done.stdout or "").strip().splitlines()[-1:] or [""]
    try:
        reply = json.loads(last[0])
    except ValueError:
        reply = None
    if done.returncode != 0 or not isinstance(reply, dict) or reply.get("error"):
        why = (reply or {}).get("error") if isinstance(reply, dict) else "\n".join((done.stderr or "").splitlines()[-6:])
        raise RuntimeError(f"the Stable Audio 3 held-out worker failed: {why}")
    return summarise(reply["per_sample"], list(timesteps))


def summarise(per_sample: dict, timesteps: list) -> dict:
    """The evaluator's record from per-sample bucket losses: pooled buckets, mean, the high-noise band, per track."""
    keys = [f"{t:.2f}" for t in timesteps]
    n = len(per_sample)
    if not n:
        raise ValueError("the held-out worker scored no sample")
    sums = {k: 0.0 for k in keys}
    per_track: dict = {}
    for name, row in per_sample.items():
        track = track_of_segment(Path(name).stem)
        tr = per_track.setdefault(track, {"sums": {k: 0.0 for k in keys}, "n_pairs": 0})
        tr["n_pairs"] += 1
        for k in keys:
            sums[k] += float(row[k])
            tr["sums"][k] += float(row[k])
    buckets = {k: v / n for k, v in sums.items()}
    high = [buckets[f"{t:.2f}"] for t in timesteps if t >= HIGH_NOISE_T]
    by_track = {}
    for track, tr in sorted(per_track.items()):
        tb = {k: v / tr["n_pairs"] for k, v in tr["sums"].items()}
        th = [tb[f"{t:.2f}"] for t in timesteps if t >= HIGH_NOISE_T]
        by_track[track] = {"mean": sum(tb.values()) / len(tb), "high_noise": (sum(th) / len(th)) if th else None,
                           "n_pairs": tr["n_pairs"]}
    return {"buckets": buckets, "mean": sum(buckets.values()) / len(buckets),
            "high_noise": (sum(high) / len(high)) if high else None, "n_pairs": n,
            "timesteps": [float(t) for t in timesteps], "by_track": by_track}
