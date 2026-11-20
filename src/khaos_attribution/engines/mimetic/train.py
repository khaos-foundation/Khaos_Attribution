"""mimetic-a0's train slot. stable-audio-tools trains in the engine's own environment through Khaos's driver; this
module hands it Khaos's pairs, follows its losses and checkpoints, lays the checkpoints out as the Workshop expects,
and gives it the same session the other engines give it. Khaos counts in epochs, the trainer in steps."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from khaos_attribution.engines.train import NoLossHooks
from khaos_attribution.engines.train import adapter_file_in as _adapter_file_in
from khaos_attribution.engines.mimetic import DESCRIPTOR
from khaos_attribution.engines.mimetic.render import WINDOWS, find_root, python_for, worker_env
from khaos_attribution.engines.render import RenderFailed
from khaos_attribution.engines import process_engine as PE
from khaos_attribution.engines import process_trainer as PT
from khaos_attribution.engines.process_engine import HeldoutEvaluator, TrainingApi  # noqa: F401
from khaos_attribution.engines.heldout import DEFAULT_BUCKETS, bucket_timesteps, summarise  # noqa: F401
from khaos_attribution.engines.train import TrainingUnavailable
from khaos_attribution.engines.worker_process import stop_tree

log = logging.getLogger(__name__)

PAIR_SUFFIX = ".wav"
TRAINER_DIRNAME = "mimetic"
HELDOUT_WORKER = Path(__file__).with_name("heldout_worker.py")
CONVERT_WORKER = Path(__file__).with_name("convert_worker.py")
TRAIN_DRIVER = Path(__file__).with_name("train_driver.py")
LOSSES_CSV = "khaos_losses.csv"
POLL_SECONDS = 2.0
MIN_PAIRS = 2
# The trainer saves weights only on its checkpoint cadence, so a held-out score exists only there.
HELDOUT_ON_CHECKPOINTS = True

__all__ = ["HELDOUT_ON_CHECKPOINTS",
           "LossHooks", "Session", "TrainingApi", "TrainingUnavailable", "adapter_file_in", "adapter_kwargs", "heldout_summary",
           "open_module", "open_session", "pair_files", "preprocess_pairs", "run_refusal", "training_api"]


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
    return _adapter_file_in(directory, DESCRIPTOR)


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

def write_clips(dataset_json: Path | str, out_dir: Path | str) -> dict:
    """Every sample of a Khaos dataset.json as a clip beside its caption, at the engine's rate."""
    return PT.write_clips(dataset_json, out_dir, sample_rate=DESCRIPTOR.sample_rate, pair_suffix=PAIR_SUFFIX,
                          engine_name="mimetic-a0", unavailable=TrainingUnavailable, log=log, require_audio=True)


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

LossHooks = NoLossHooks


def steps_for(cfg: dict, n_pairs: int) -> dict:
    return PT.steps_for(cfg, n_pairs)


def trainer_command(api: TrainingApi, cfg: dict, *, tensor_dir: Path, save_dir: Path, steps: dict, seed: int,
                    device: str) -> list:
    adapter_type, a = adapter_kwargs(cfg)
    return [str(api.python), str(TRAIN_DRIVER), "--clips", str(tensor_dir), "--save-dir", str(save_dir),
            "--steps", str(steps["total"]), "--checkpoint-every", str(steps["checkpoint_every"]),
            "--rank", str(a["rank"]), "--alpha", str(a["alpha"]), "--adapter-type", adapter_type,
            # The rate comes from the recipe, else the descriptor's train block; the driver has no default of its own.
            "--lr", str(float(cfg.get("learning_rate") or DESCRIPTOR.train["learning_rate"])),
            "--batch-size", str(max(1, int(cfg.get("batch_size") or 1))),
            # stable-audio-tools keeps its loader's workers alive between epochs, which needs at least one.
            "--num-workers", "1" if WINDOWS else "2", "--seed", str(int(seed)), "--device", device]


def losses_from_csv(save_dir: Path) -> list:
    """(step, loss) rows the driver wrote so far."""
    return PT.CsvLossTail(Path(save_dir) / LOSSES_CSV)()


def checkpoint_step(path: Path) -> int | None:
    return PT.checkpoint_step(path)


class Session(PE.ProcessSession):
    """One training run, driven as a process: losses read from the driver's file, each checkpoint laid out as
    ``checkpoints/epoch_<n>_loss_<x>/adapter_model.safetensors``, the held-out loss taken by a worker."""
    label = "mimetic-a0"
    descriptor = DESCRIPTOR
    trainer_dirname = TRAINER_DIRNAME

    def __init__(self, api: TrainingApi, cfg: dict, *, checkpoint_root: Path, tensor_dir: Path, output_dir: Path,
                 seed: int, device: str = "auto") -> None:
        super().__init__(api, cfg, checkpoint_root=checkpoint_root, tensor_dir=tensor_dir, output_dir=output_dir,
                         seed=seed, steps=steps_for(cfg, len(pair_files(tensor_dir))))
        self.device = device

    def pair_files(self, tensor_dir: Path) -> list:
        return pair_files(tensor_dir)

    def command(self) -> list:
        return trainer_command(self.api, self.cfg, tensor_dir=self.tensor_dir, save_dir=self.save_dir, steps=self.steps,
                               seed=self.seed, device=self.device)

    def _checkpoints(self) -> dict:
        return {step: f for f in self.save_dir.glob("*.ckpt") if (step := checkpoint_step(f))}

    def _lay_out(self, step: int, source: Path, loss: float) -> Path:
        """The checkpoint in the Workshop's layout, as a float32 safetensors adapter."""
        epoch, target_dir, target = self._checkpoint_target(step, loss)
        self._convert(CONVERT_WORKER, source, target, _env(self.api))
        self._laid_out(epoch, target)
        return target_dir

    def train(self):
        """The trainer's run as Khaos's updates: info at start, an epoch each time the step count crosses one, a
        checkpoint each time one lands, complete or fail at the end."""
        from khaos_attribution.engines import contract_guard  # noqa: PLC0415
        contract_guard.require(self.api.python, engine=DESCRIPTOR.name, engine_id=DESCRIPTOR.id, error=TrainingUnavailable)
        if self.save_dir.exists():
            aside = self.output_dir / f"{TRAINER_DIRNAME}_before_{time.strftime('%Y%m%d_%H%M%S')}"
            self.save_dir.replace(aside)
        self.save_dir.mkdir(parents=True)
        self._sizes = {}
        cmd = self.command()
        record = {"command": cmd, "steps": self.steps, "precision": "float32", "device": self.device}
        yield from self._run_trainer(cmd, record, cwd=self.api.root, env=_env(self.api),
                                     follow=lambda: self._follow(self.start_epoch), stop_tree=stop_tree)

    def _follow(self, offset: int):
        return PT.follow(self, offset, started="[OK] mimetic-a0 trainer started",
                         losses=PT.CsvLossTail(self.save_dir / LOSSES_CSV), run_steps=self.steps, poll=POLL_SECONDS)

    def heldout(self, clips_dir: Path, adapter: Path | None, timesteps: list) -> dict:
        return heldout_summary(self.api, clips_dir, adapter, seed=self.seed, timesteps=timesteps,
                               checkpoint_root=self.checkpoint_root)


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

def heldout_summary(api: TrainingApi, clips_dir: Path | str, adapter: Path | str | None, *, seed: int,
                    timesteps: list, checkpoint_root: Path | str | None = None, device: str | None = None) -> dict:
    """The held-out record of the clips under ``clips_dir`` under ``adapter`` (the base alone when None): what the
    trainer's held-out curve and a datamodel's subset losses both read."""
    request = {"clips_dir": str(Path(clips_dir).resolve()), "adapter": str(Path(adapter).resolve()) if adapter else None,
               "timesteps": list(timesteps), "seed": int(seed), "device": device}
    return PE.heldout_record(api.python, HELDOUT_WORKER, request, cwd=api.root, env=_env(api), label="mimetic-a0",
                             timesteps=timesteps)
