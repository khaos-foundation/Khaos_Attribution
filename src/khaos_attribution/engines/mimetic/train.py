"""mimetic-a0's train slot. stable-audio-tools trains in the engine's own environment through Khaos's driver; this
module hands it Khaos's pairs, follows its losses and checkpoints, lays the checkpoints out as the Workshop expects,
and gives it the same session the other engines give it. Khaos counts in epochs, the trainer in steps."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from khaos_attribution.engines.mimetic import DESCRIPTOR
from khaos_attribution.engines.mimetic.render import WINDOWS, find_root, python_for, worker_env
from khaos_attribution.engines.render import RenderFailed
# The held-out record's arithmetic is the same for every engine whose worker returns per-sample bucket losses.
from khaos_attribution.engines.sa3.train import bucket_timesteps, summarise  # noqa: F401
from khaos_attribution.engines.train import TrainingUnavailable, TrainingUpdate
from khaos_attribution.engines.worker_process import stop_tree

log = logging.getLogger(__name__)

PAIR_SUFFIX = ".wav"
TRAINER_DIRNAME = "mimetic"
DEFAULT_BUCKETS = 10
HELDOUT_WORKER = Path(__file__).with_name("heldout_worker.py")
CONVERT_WORKER = Path(__file__).with_name("convert_worker.py")
TRAIN_DRIVER = Path(__file__).with_name("train_driver.py")
LOSSES_CSV = "khaos_losses.csv"
POLL_SECONDS = 2.0
MIN_PAIRS = 2

__all__ = ["LossHooks", "Session", "TrainingApi", "TrainingUnavailable", "adapter_file_in", "adapter_kwargs", "heldout_summary",
           "open_module", "open_session", "pair_files", "preprocess_pairs", "run_refusal", "training_api"]


@dataclass(frozen=True)
class TrainingApi:
    """Where the engine's environment is; nothing of stable-audio-tools is imported here."""
    root: Path
    runtime: str
    python: Path


def training_api(repo_dir=None, device: str = "auto") -> TrainingApi:
    root = find_root(repo_dir)
    if not root.is_dir():
        raise TrainingUnavailable(f"mimetic-a0 not found at {root}")
    try:
        python = python_for(root)
    except RenderFailed as exc:
        raise TrainingUnavailable(str(exc)) from exc
    return TrainingApi(root=root, runtime="torch", python=Path(python))


def adapter_kwargs(cfg: dict) -> tuple:
    """(adapter type, rank and alpha) a recipe describes; the type is the descriptor's unless the recipe names one of
    stable-audio-tools' own."""
    adapter_type = str(cfg.get("adapter_type") or DESCRIPTOR.adapter["type"])
    if adapter_type not in ("lora", "dora", "lora-xs"):
        adapter_type = DESCRIPTOR.adapter["type"]
    return adapter_type, {"rank": int(cfg["adapter_rank"]), "alpha": int(cfg["adapter_alpha"])}


def adapter_file_in(directory: Path | str) -> Path | None:
    directory = Path(directory)
    for name in DESCRIPTOR.adapter["filenames"]:
        if (directory / name).is_file():
            return directory / name
    files = sorted(directory.glob("*.safetensors"))
    return files[0] if files else None


def run_refusal(run_dir: Path | str) -> str | None:
    """Why a run folder's adapter cannot be measured, or None: a run whose trainer record is missing."""
    run_dir = Path(run_dir)
    if (run_dir / "training_config.yaml").is_file() and not (run_dir / TRAINER_DIRNAME / "train_command.json").is_file():
        return f"{run_dir.name}'s trainer record is missing, so how it was trained is unknown; it cannot be measured"
    return None


def pair_files(tensor_dir: Path | str) -> list:
    return sorted(Path(tensor_dir).glob(f"*{PAIR_SUFFIX}"))


def open_module(api: TrainingApi, cfg: dict, adapter_file, *, checkpoint_root=None, device: str = "cuda",
                dataset_dir=None, output_dir=None, precision: str | None = None):
    """What a feature is taken on: a handle naming the engine's environment and the adapter, loaded per call."""
    from khaos_attribution.engines.mimetic.measure import Handle  # noqa: PLC0415
    adapter_file = Path(adapter_file)
    if not adapter_file.is_file():
        raise TrainingUnavailable(f"no adapter file at {adapter_file}")
    why = run_refusal(output_dir) if output_dir is not None else None
    if why:
        raise TrainingUnavailable(why)
    return Handle(root=api.root, python=api.python, adapter=adapter_file.resolve(),
                  device_type=str(device).split(":")[0], dtype="float32")


# ── preprocessing: Khaos's dataset into the trainer's clips ─────────────────────

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
        raise TrainingUnavailable("resampling a clip for mimetic-a0 needs librosa in this environment") from exc
    return np.stack([librosa.resample(audio[:, c], orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
                     for c in range(2)], axis=1)


def write_clips(dataset_json: Path | str, out_dir: Path | str) -> dict:
    """Every sample of a Khaos dataset.json as ``<stem>.wav`` (48 kHz stereo, 16-bit) beside ``<stem>.txt``, its
    caption; an empty sample is counted as failed, the engine's trainer would skip it."""
    import soundfile as sf  # noqa: PLC0415
    samples = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed = failed = 0
    for s in samples:
        stem = Path(s["filename"]).stem
        try:
            audio, sr = sf.read(str(s["audio_path"]), dtype="float32", always_2d=True)
            if not len(audio):
                raise ValueError("no audio")
            audio = _resample_stereo(audio, sr, DESCRIPTOR.sample_rate)
            sf.write(str(out / f"{stem}{PAIR_SUFFIX}"), audio, DESCRIPTOR.sample_rate, subtype="PCM_16")
            (out / f"{stem}.txt").write_text(str(s.get("caption") or "").strip() + "\n", encoding="utf-8")
            processed += 1
        except Exception as exc:  # noqa: BLE001 — one bad clip is counted, the rest go on
            log.error("mimetic-a0 preprocess: %s failed: %s", stem, exc)
            failed += 1
    return {"processed": processed, "failed": failed}


def preprocess_pairs(api: TrainingApi, *, out_dir: Path | str, checkpoint_root: Path | str, variant: str = "base",
                     dataset_json: Path | str | None, device: str = "auto", precision: str | None = None,
                     audio_dir: Path | str | None = None) -> dict:
    """Clips and captions under ``out_dir``; the trainer encodes them itself."""
    if dataset_json is None:
        return {"processed": len(pair_files(out_dir)), "failed": 0, "reused": True}
    return write_clips(dataset_json, out_dir)


def _env(api: TrainingApi) -> dict:
    return worker_env(api.root)


# ── the session ──────────────────────────────────────────────────────────────

class LossHooks:
    """The trainer takes no seams from Khaos: every flag is off, and the context has nothing to undo."""

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
    """Khaos's epochs as the trainer's steps: steps per epoch at the recipe's batch size, the total, and the
    checkpoint cadence in steps."""
    batch = max(1, int(cfg.get("batch_size") or 1))
    per_epoch = max(1, -(-n_pairs // batch))
    epochs = int(cfg["max_epochs"])
    every_epochs = max(1, int(cfg.get("checkpoint_every_n_epochs") or 1))
    return {"per_epoch": per_epoch, "total": per_epoch * epochs, "checkpoint_every": per_epoch * every_epochs,
            "epochs": epochs}


def trainer_command(api: TrainingApi, cfg: dict, *, tensor_dir: Path, save_dir: Path, steps: dict, seed: int,
                    device: str) -> list:
    adapter_type, a = adapter_kwargs(cfg)
    return [str(api.python), str(TRAIN_DRIVER), "--clips", str(tensor_dir), "--save-dir", str(save_dir),
            "--steps", str(steps["total"]), "--checkpoint-every", str(steps["checkpoint_every"]),
            "--rank", str(a["rank"]), "--alpha", str(a["alpha"]), "--adapter-type", adapter_type,
            "--lr", str(float(cfg.get("learning_rate") or 1e-4)),
            "--batch-size", str(max(1, int(cfg.get("batch_size") or 1))),
            # stable-audio-tools keeps its loader's workers alive between epochs, which needs at least one.
            "--num-workers", "1" if WINDOWS else "2", "--seed", str(int(seed)), "--device", device]


def losses_from_csv(save_dir: Path) -> list:
    """(step, loss) rows the driver wrote so far."""
    rows = []
    try:
        with (save_dir / LOSSES_CSV).open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("loss") not in (None, ""):
                    rows.append((int(float(row.get("step") or 0)), float(row["loss"])))
    except (OSError, ValueError):
        return rows
    return rows


def checkpoint_step(path: Path) -> int | None:
    import re  # noqa: PLC0415
    m = re.search(r"step=(\d+)", path.name)
    return int(m.group(1)) if m else None


class Session:
    """One training run, driven as a process: losses read from the driver's file, each checkpoint laid out as
    ``checkpoints/epoch_<n>_loss_<x>/adapter_model.safetensors``, the held-out loss taken by a worker."""

    def __init__(self, api: TrainingApi, cfg: dict, *, checkpoint_root: Path, tensor_dir: Path, output_dir: Path,
                 seed: int, device: str = "auto") -> None:
        self.api = api
        self.cfg = cfg
        self.checkpoint_root = Path(checkpoint_root)
        self.tensor_dir = Path(tensor_dir)
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.device = device
        self.save_dir = self.output_dir / TRAINER_DIRNAME
        self.steps = steps_for(cfg, len(pair_files(self.tensor_dir)))
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

    def command(self) -> list:
        return trainer_command(self.api, self.cfg, tensor_dir=self.tensor_dir, save_dir=self.save_dir, steps=self.steps,
                               seed=self.seed, device=self.device)

    def _checkpoints(self) -> dict:
        return {step: f for f in self.save_dir.glob("*.ckpt") if (step := checkpoint_step(f))}

    def _settled(self, path: Path, final: bool) -> bool:
        """A checkpoint is read once its size and mtime held still across two polls, or the trainer has exited."""
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
        """The checkpoint in the Workshop's layout, as a float32 safetensors adapter."""
        epoch = self._epoch_of(step)
        target_dir = self.output_dir / "checkpoints" / f"epoch_{epoch}_loss_{loss:.4f}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / DESCRIPTOR.adapter["filenames"][0]
        done = subprocess.run([str(self.api.python), str(CONVERT_WORKER), str(source), str(target)],
                              cwd=str(self.api.root), capture_output=True, text=True, env=_env(self.api))
        if done.returncode != 0 or not target.is_file():
            raise TrainingUnavailable("converting the checkpoint to safetensors failed: "
                                      + "\n".join(done.stderr.strip().splitlines()[-4:]))
        self.latest_adapter = target
        self.latest_epoch = epoch
        return target_dir

    def train(self):
        """The trainer's run as Khaos's updates: info at start, an epoch each time the step count crosses one, a
        checkpoint each time one lands, complete or fail at the end."""
        if self.save_dir.exists():
            aside = self.output_dir / f"{TRAINER_DIRNAME}_before_{time.strftime('%Y%m%d_%H%M%S')}"
            self.save_dir.replace(aside)
        self.save_dir.mkdir(parents=True)
        self._sizes = {}
        cmd = self.command()
        record = {"command": cmd, "steps": self.steps, "precision": "float32", "device": self.device}
        (self.save_dir / "train_command.json").write_text(json.dumps(record, indent=1) + "\n")
        log_file = (self.save_dir / "trainer_stdout.log").open("ab")
        # In the job's own process group, so the room's Stop reaches the trainer as it does the job.
        self.process = subprocess.Popen(cmd, cwd=str(self.api.root), env=_env(self.api), stdout=log_file,
                                        stderr=subprocess.STDOUT)
        try:
            yield from self._follow(self.start_epoch)
        finally:
            stop_tree(self.process)
            log_file.close()

    def _follow(self, offset: int):
        yield TrainingUpdate(step=0, loss=0.0, msg="[OK] mimetic-a0 trainer started", kind="info", epoch=offset)
        seen = 0
        epoch_done = offset
        laid_out: set = set()
        epoch_losses: list = []
        by_step: dict = {}
        last_loss = 0.0
        start = time.monotonic()
        per_epoch = self.steps["per_epoch"]
        while True:
            alive = self.process.poll() is None
            losses = losses_from_csv(self.save_dir)
            events: list = []
            for step, loss in losses[seen:]:
                last_loss = loss
                by_step[step] = loss
                epoch_losses.append(loss)
                if step % per_epoch == 0:
                    epoch_done = offset + step // per_epoch
                    mean = sum(epoch_losses) / len(epoch_losses)
                    epoch_losses = []
                    events.append((step, 0, TrainingUpdate(step=step, loss=mean, msg=f"[OK] Epoch {epoch_done}", kind="epoch",
                                                           epoch=epoch_done, epoch_time=time.monotonic() - start)))
                    start = time.monotonic()
            seen = len(losses)
            highest = max(by_step) if by_step else 0
            for step, path in sorted(self._checkpoints().items()):
                if step in laid_out or not self._settled(path, final=not alive) or (alive and highest < step):
                    continue
                laid_out.add(step)
                loss_at = by_step.get(step, last_loss)
                target_dir = self._lay_out(step, path, loss_at)
                events.append((step, 1, TrainingUpdate(step=step, loss=loss_at, msg="[OK] Checkpoint", kind="checkpoint",
                                                       epoch=self._epoch_of(step), checkpoint_path=str(target_dir))))
            for _step, _order, update in sorted(events, key=lambda e: (e[0], e[1])):
                yield update
            if not alive:
                break
            time.sleep(POLL_SECONDS)
        code = self.process.returncode
        if code != 0:
            tail = "\n".join((self.save_dir / "trainer_stdout.log").read_text(errors="ignore").splitlines()[-8:])
            yield TrainingUpdate(step=seen, loss=last_loss, msg=f"trainer exited {code}: {tail}", kind="fail", epoch=epoch_done)
            return
        yield TrainingUpdate(step=seen, loss=last_loss, msg="[OK] done", kind="complete",
                             epoch=max(epoch_done, self.steps["epochs"]))

    def heldout_evaluator(self, tensor_val_dir: Path | str, *, n_buckets: int):
        return HeldoutEvaluator(self, Path(tensor_val_dir), n_buckets=int(n_buckets))

    def save_final(self, final_dir: Path | str) -> Path:
        """The last epoch's checkpoint, as the adapter the run exports; an older one is refused."""
        if self.latest_adapter is None or not self.latest_adapter.is_file():
            raise RuntimeError("mimetic-a0 wrote no checkpoint to export")
        if self.latest_epoch < self.steps["epochs"]:
            raise RuntimeError(f"the last checkpoint is epoch {self.latest_epoch} of {self.steps['epochs']}; "
                               "the final weights were not written")
        final_dir = Path(final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        target = final_dir / DESCRIPTOR.adapter["filenames"][0]
        shutil.copy2(self.latest_adapter, target)
        return target


def open_session(api: TrainingApi, cfg: dict, *, checkpoint_root, tensor_dir, output_dir, device: str,
                 precision: str, seed: int, resume_from=None, on_model_loaded=None) -> Session:
    """A session on the trainer; nothing is loaded until train() runs, so on_model_loaded fires at once. A resume is
    refused: the descriptor says this engine's adapters do not resume."""
    if resume_from is not None:
        raise TrainingUnavailable("mimetic-a0 adapters do not resume; start the run again")
    if len(pair_files(tensor_dir)) < MIN_PAIRS:
        raise TrainingUnavailable(f"mimetic-a0's trainer needs at least {MIN_PAIRS} pairs; "
                                  f"{len(pair_files(tensor_dir))} under {tensor_dir}")
    if on_model_loaded is not None:
        on_model_loaded(None)
    return Session(api, cfg, checkpoint_root=checkpoint_root, tensor_dir=tensor_dir, output_dir=output_dir,
                   seed=seed, device=str(device).split(":")[0] if device not in (None, "auto") else "auto")


# ── the held-out loss, by a worker ───────────────────────────────────────────

class HeldoutEvaluator:
    """The trainer's velocity MSE on the held-out clips at fixed timesteps with seeded noise, under the adapter
    named last (the newest checkpoint, else the base alone)."""

    def __init__(self, session: Session, tensor_val_dir: Path, *, n_buckets: int) -> None:
        self.session = session
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
                    timesteps: list, checkpoint_root: Path | str | None = None, device: str | None = None) -> dict:
    """The held-out record of the clips under ``clips_dir`` under ``adapter`` (the base alone when None): what the
    trainer's held-out curve and a datamodel's subset losses both read."""
    request = {"clips_dir": str(Path(clips_dir).resolve()), "adapter": str(Path(adapter).resolve()) if adapter else None,
               "timesteps": list(timesteps), "seed": int(seed), "device": device}
    done = subprocess.run([str(api.python), str(HELDOUT_WORKER)], input=json.dumps(request), cwd=str(api.root),
                          capture_output=True, text=True, env=_env(api))
    last = (done.stdout or "").strip().splitlines()[-1:] or [""]
    try:
        reply = json.loads(last[0])
    except ValueError:
        reply = None
    if done.returncode != 0 or not isinstance(reply, dict) or reply.get("error"):
        why = (reply or {}).get("error") if isinstance(reply, dict) else "\n".join((done.stderr or "").splitlines()[-6:])
        raise RuntimeError(f"the mimetic-a0 held-out worker failed: {why}")
    return summarise(reply["per_sample"], list(timesteps))
