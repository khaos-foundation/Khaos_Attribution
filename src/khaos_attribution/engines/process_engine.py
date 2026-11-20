"""What the engines driven as processes share (Stable Audio 3 and mimetic-a0): the training door, a session that runs
the engine's trainer and lays its checkpoints out in the Workshop's layout, the held-out evaluator a worker serves,
and the measure client's checks around a worker's reply. Each engine keeps its own trainer command, workers, loss
log and the research options it has built."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from khaos_attribution.engines import process_trainer as PT
from khaos_attribution.engines import worker_process as W
from khaos_attribution.engines.heldout import bucket_timesteps, summarise
from khaos_attribution.engines.train import NoLossHooks, TrainingUnavailable

__all__ = ["HeldoutEvaluator", "ProcessSession", "TrainingApi", "heldout_record", "index_refusal", "measure_reply",
           "refuse_unbuilt"]


@dataclass(frozen=True)
class TrainingApi:
    """Where the engine's environment is; nothing of the engine's own code is imported here."""
    root: Path
    runtime: str
    python: Path


class ProcessSession:
    """One training run on an engine's own trainer, driven as a process. A subclass names the engine (``label``,
    ``descriptor``), its trainer's folder and the module that holds its patchable workers and settings."""
    label = ""
    descriptor = None
    trainer_dirname = ""

    def __init__(self, api: TrainingApi, cfg: dict, *, checkpoint_root: Path, tensor_dir: Path, output_dir: Path,
                 seed: int, steps: dict) -> None:
        self.api = api
        self.cfg = cfg
        self.checkpoint_root = Path(checkpoint_root)
        self.tensor_dir = Path(tensor_dir)
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.save_dir = self.output_dir / self.trainer_dirname
        self.steps = steps
        self.latest_adapter: Path | None = None
        self.latest_epoch = 0
        self.start_epoch = 0
        self.epoch_done = 0
        self.checkpoint_by_epoch: dict = {}
        self._sizes: dict = {}
        self.module = None
        self.training_config = None
        self.process: subprocess.Popen | None = None

    def install_hooks(self, *, start_epoch: int = 0, stem_ids=(), record_into: dict | None = None) -> NoLossHooks:
        self.start_epoch = max(0, int(start_epoch or 0))
        return NoLossHooks(record_into=record_into)

    def _settled(self, path: Path, final: bool) -> bool:
        return PT.settled(self._sizes, path, final)

    def _epoch_of(self, step: int) -> int:
        return PT.epoch_of(self.start_epoch, self.steps["per_epoch"], step)

    def _checkpoint_target(self, step: int, loss: float) -> tuple:
        """(epoch, folder, adapter file) a checkpoint at ``step`` is laid out as."""
        epoch = self._epoch_of(step)
        target_dir = self.output_dir / "checkpoints" / f"epoch_{epoch}_loss_{loss:.4f}"
        target_dir.mkdir(parents=True, exist_ok=True)
        return epoch, target_dir, target_dir / self.descriptor.adapter["filenames"][0]

    def _convert(self, convert_worker: Path, source: Path, target: Path, env: dict) -> None:
        """A trainer checkpoint as a safetensors adapter, by the engine's converter in its own environment."""
        code, _stdout, stderr = W.call(self.api.python, convert_worker, [source, target], cwd=self.api.root, env=env,
                                       label=f"{self.label} converter", timeout=None)
        if code != 0 or not target.is_file():
            raise TrainingUnavailable("converting the checkpoint to safetensors failed: "
                                      + "\n".join(stderr.strip().splitlines()[-4:]))

    def _laid_out(self, epoch: int, target: Path) -> None:
        self.latest_adapter = target
        self.latest_epoch = epoch
        self.checkpoint_by_epoch[epoch] = target

    def _run_trainer(self, cmd: list, record: dict, *, cwd: Path, env: dict, follow, stop_tree):
        """The trainer started with its record beside it, its updates followed; no trainer outlives the loop."""
        (self.save_dir / "train_command.json").write_text(json.dumps(record, indent=1) + "\n")
        log_file = (self.save_dir / "trainer_stdout.log").open("ab")
        # In the job's own process group, so the room's Stop and Pause signals reach the trainer as they do the job.
        # The record keeps the trainer's own command; a contract driver is started through the launcher.
        self.process = subprocess.Popen(W.command(*cmd), cwd=str(cwd), env=env, stdout=log_file,
                                        stderr=subprocess.STDOUT)
        try:
            yield from follow()
        finally:
            stop_tree(self.process)
            log_file.close()

    def heldout_evaluator(self, tensor_val_dir: Path | str, *, n_buckets: int):
        return HeldoutEvaluator(self, Path(tensor_val_dir), n_buckets=int(n_buckets))

    def heldout(self, clips_dir: Path, adapter: Path | None, timesteps: list) -> dict:
        raise NotImplementedError

    def pair_files(self, tensor_dir: Path) -> list:
        raise NotImplementedError

    def save_final(self, final_dir: Path | str) -> Path:
        """The last epoch's checkpoint, as the adapter the run exports; an older one is refused so the room's
        salvage path records what it really took."""
        if self.latest_adapter is None or not self.latest_adapter.is_file():
            raise RuntimeError(f"{self.label} wrote no checkpoint to export")
        if self.latest_epoch < self.steps["epochs"]:
            raise RuntimeError(f"the last checkpoint is epoch {self.latest_epoch} of {self.steps['epochs']}; "
                               "the final weights were not written")
        final_dir = Path(final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        target = final_dir / self.descriptor.adapter["filenames"][0]
        shutil.copy2(self.latest_adapter, target)
        return target


class HeldoutEvaluator:
    """The velocity MSE on the held-out clips at fixed timesteps with seeded noise, taken by the engine's worker on
    the adapter named, else the checkpoint of the epoch just finished."""

    def __init__(self, session: ProcessSession, tensor_val_dir: Path, *, n_buckets: int) -> None:
        self.session = session
        # The worker runs in the engine's checkout, so every path it is handed is absolute.
        self.tensor_val_dir = Path(tensor_val_dir).resolve()
        self.timesteps = bucket_timesteps(n_buckets)
        self.adapter: Path | None = None
        self.samples = session.pair_files(tensor_val_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def load_adapter_weights(self, weights_path: Path | str) -> bool:
        self.adapter = Path(weights_path)
        return True

    def evaluate(self) -> dict:
        adapter = PT.scored_adapter(self.session, self.adapter, self.session.descriptor.name)
        return self.session.heldout(self.tensor_val_dir, adapter, self.timesteps)


def heldout_record(python: Path, worker: Path, request: dict, *, cwd: Path, env: dict, label: str, timesteps: list) -> dict:
    """The held-out record from the engine's held-out worker; no time limit, as a pass over a run's clips has no
    measured bound and the room's stop reaches the tree."""
    try:
        reply = W.run(python, worker, request, cwd=cwd, env=env, label=f"{label} held-out", timeout=None,
                      required="per_sample")
    except W.WorkerFailed as exc:
        raise RuntimeError(f"the {label} held-out worker failed: {exc.why}") from exc
    return summarise(reply["per_sample"], list(timesteps))


# ── the measure client ────────────────────────────────────────────────────────

# What each worker command must answer with.
REPLY_KEYS = {"parameters": ("n_params", "n_tensors", "signature"), "index": ("rows", "matrix"), "feature": ("feature",),
              "loss": ("with", "without")}


def measure_reply(python: Path, worker: Path, request: dict, *, cwd: Path, env: dict, label: str, descriptor,
                  engine: str, reply_keys: dict = REPLY_KEYS, on_progress=None) -> dict:
    """The worker's reply, its ``khaos-row`` lines handed to ``on_progress`` as they come; the interpreter's contract
    is checked first and a failure is GradientUnavailable."""
    from khaos_attribution.engines import contract_guard  # noqa: PLC0415
    from khaos_attribution.gradient import GradientUnavailable  # noqa: PLC0415
    contract_guard.require(python, engine=engine, engine_id=descriptor.id, error=GradientUnavailable)
    try:
        return W.stream(python, worker, request, cwd=cwd, env=env, label=f"{label} measure", on_row=on_progress,
                        required=reply_keys.get(request.get("command"), ()))
    except W.WorkerFailed as exc:
        raise GradientUnavailable(str(exc)) from exc


def index_refusal(descriptor, index_meta: dict) -> str | None:
    """Why an index cannot be read against this measure's features, or None: it must have been built by this
    feature version, since the noise and the parameter order are part of what a row means."""
    built = index_meta.get("feature_version")
    if built != descriptor.feature_version:
        return (f"the influence index was built by {descriptor.name}'s measure {built or 'before it was versioned'}, "
                f"and this one is {descriptor.feature_version}; rebuild the index")
    return None


def refuse_unbuilt(worker_name: str, *, built: frozenset, timestep_norm=False, param_filter=None, latent_transform=None,
                   latent_pool=None, noise_draws=None, noise_shared=False, param_groups=None, groups_out=None) -> None:
    """GradientUnavailable for the first research option in this table asked of a worker that has not built it;
    ``built`` names the ones it has, so an option added here is refused by every engine that does not claim it."""
    from khaos_attribution.gradient import GradientUnavailable, research_int  # noqa: PLC0415
    asked = [("param_groups", lambda: bool(param_groups or groups_out), "block sketches are"),
             ("latent_pool", lambda: research_int(latent_pool, "latent_pool") > 1
              or research_int(noise_draws, "noise_draws") > 1, "latent pooling and noise draws are"),
             ("noise_shared", lambda: bool(noise_shared), "shared noise is"),
             ("latent_transform", lambda: bool(latent_transform), "a latent transform is"),
             ("timestep_norm", lambda: bool(timestep_norm), "per-timestep normalisation is"),
             ("param_filter", lambda: bool(param_filter), "a parameter filter is")]
    for option, on, words in asked:
        if option not in built and on():
            raise GradientUnavailable(f"{words} not built on the {worker_name} worker")
