"""The mock engine's train slot: the same names as a real engine's, with pure-Python stand-ins, so the checklist and
the rooms' tests can drive the socket without a model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from khaos_attribution.engines.train import NoLossHooks
from khaos_attribution.engines.train import adapter_file_in as _adapter_file_in
from khaos_attribution.engines.mock import DESCRIPTOR
from khaos_attribution.files import real_glob
from khaos_attribution.engines.train import TrainingUnavailable

__all__ = ["LossHooks", "Session", "TrainingApi", "TrainingUnavailable", "adapter_file_in", "adapter_kwargs",
           "open_module", "open_session", "pair_files", "preprocess_pairs", "training_api"]


@dataclass(frozen=True)
class TrainingApi:
    preprocess: Any
    Module: Any


def _preprocess(*, output_dir, dataset_json, **_ignored) -> dict:
    """One ``<name>.pt``-named file per sample, holding its caption: a pair in shape, not in content."""
    samples = json.loads(Path(dataset_json).read_text(encoding="utf-8")) if dataset_json else []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in samples:
        (out / (Path(s["filename"]).stem + ".pt")).write_text(json.dumps({"caption": s.get("caption", "")}))
    return {"processed": len(samples), "failed": 0}


class _Module(SimpleNamespace):
    pass


def training_api(repo_dir=None) -> TrainingApi:
    return TrainingApi(preprocess=_preprocess, Module=_Module)


def adapter_kwargs(cfg: dict) -> tuple:
    return str(cfg.get("adapter_type") or DESCRIPTOR.adapter["type"]), {"rank": int(cfg.get("adapter_rank") or 1)}


def pair_files(tensor_dir) -> list:
    return real_glob(tensor_dir, "*.pt")


def adapter_file_in(directory: Path | str) -> Path | None:
    return _adapter_file_in(directory, DESCRIPTOR)


def preprocess_pairs(api: TrainingApi, *, out_dir, checkpoint_root=None, variant=None, dataset_json=None, device="cpu",
                     precision=None, audio_dir=None) -> dict:
    return api.preprocess(output_dir=out_dir, dataset_json=dataset_json)


def open_module(api: TrainingApi, cfg: dict, adapter_file, *, checkpoint_root=None, device="cpu", dataset_dir=None,
                output_dir=None, precision=None) -> Any:
    if not Path(adapter_file).is_file():
        raise TrainingUnavailable(f"no adapter file at {adapter_file}")
    return api.Module(dtype="mock", device="cpu", device_type="cpu", adapter=str(adapter_file),
                      model=SimpleNamespace(decoder=SimpleNamespace(config=None)))


# ── a session: the same stream a real engine yields, in milliseconds ─────────

from khaos_attribution.engines.train import TrainingUpdate  # noqa: E402


LossHooks = NoLossHooks


class Session:
    """Epochs that write a tiny checkpoint each, in the Workshop's layout, with a falling loss."""

    def __init__(self, api, cfg: dict, *, tensor_dir, output_dir, seed: int):
        self.api, self.cfg, self.seed = api, cfg, int(seed)
        self.tensor_dir, self.output_dir = Path(tensor_dir), Path(output_dir)
        self.module = None
        self.training_config = None
        self.latest_adapter = None

    def install_hooks(self, *, start_epoch: int = 0, stem_ids=(), record_into=None) -> LossHooks:
        return LossHooks(record_into)

    def train(self):
        n = max(1, len(pair_files(self.tensor_dir)))
        epochs = int(self.cfg.get("max_epochs") or 1)
        every = max(1, int(self.cfg.get("checkpoint_every_n_epochs") or 1))
        yield TrainingUpdate(step=0, loss=0.0, msg="[OK] mock trainer", kind="info", epoch=0)
        for ep in range(1, epochs + 1):
            loss = 1.0 / ep
            yield TrainingUpdate(step=ep * n, loss=loss, msg=f"[OK] Epoch {ep}", kind="epoch", epoch=ep)
            if ep % every == 0 or ep == epochs:
                d = self.output_dir / "checkpoints" / f"epoch_{ep}_loss_{loss:.4f}"
                d.mkdir(parents=True, exist_ok=True)
                f = d / DESCRIPTOR.adapter["filenames"][0]
                f.write_bytes(b"mock-adapter:" + str(ep).encode())
                self.latest_adapter = f
                yield TrainingUpdate(step=ep * n, loss=loss, msg="[OK] Checkpoint", kind="checkpoint", epoch=ep,
                                     checkpoint_path=str(d))
        yield TrainingUpdate(step=epochs * n, loss=1.0 / epochs, msg="[OK] done", kind="complete", epoch=epochs)

    def heldout_evaluator(self, tensor_val_dir, *, n_buckets: int):
        return _Evaluator(tensor_val_dir, n_buckets)

    def save_final(self, final_dir) -> Path:
        if self.latest_adapter is None:
            raise RuntimeError("the mock trainer wrote no checkpoint")
        final_dir = Path(final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        target = final_dir / DESCRIPTOR.adapter["filenames"][0]
        target.write_bytes(self.latest_adapter.read_bytes())
        return target


class _Evaluator:
    def __init__(self, tensor_val_dir, n_buckets: int):
        self.samples = pair_files(tensor_val_dir)
        self.timesteps = [round((i + 0.5) / n_buckets, 6) for i in range(int(n_buckets))]
        self.adapter = None

    def __len__(self):
        return len(self.samples)

    def load_adapter_weights(self, path) -> bool:
        self.adapter = Path(path)
        return True

    def evaluate(self) -> dict:
        buckets = {f"{t:.2f}": 0.5 for t in self.timesteps}
        return {"buckets": buckets, "mean": 0.5, "high_noise": 0.5, "n_pairs": len(self.samples),
                "timesteps": self.timesteps, "by_track": {}}


def open_session(api, cfg: dict, *, checkpoint_root=None, tensor_dir, output_dir, device="cpu", precision="bf16",
                 seed: int, resume_from=None, on_model_loaded=None) -> Session:
    if on_model_loaded is not None:
        on_model_loaded(None)
    return Session(api, cfg, tensor_dir=tensor_dir, output_dir=output_dir, seed=seed)
