"""The train slot in Khaos's words. An engine's train module offers training_api(repo_dir), adapter_kwargs(cfg),
preprocess_pairs(...), open_module(...) and adapter_file_in(dir); these are the refusal, the adapter lookup and the
seamless hooks they share."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class TrainingUnavailable(RuntimeError):
    """The engine's training side cannot be reached here, with the reason."""


@dataclass
class TrainingUpdate:
    """One event from a session's train(): kind is info, epoch, checkpoint, complete, warn or fail."""
    step: int
    loss: float
    msg: str
    kind: str
    epoch: int = 0
    lr: float = 0.0
    epoch_time: float = 0.0
    samples_per_sec: float = 0.0
    checkpoint_path: str | None = None


def adapter_file_in(directory: Path | str, descriptor) -> Path | None:
    """The adapter weights inside a checkpoint directory: the descriptor's names first, then any safetensors."""
    directory = Path(directory)
    for name in descriptor.adapter["filenames"]:
        if (directory / name).is_file():
            return directory / name
    files = sorted(directory.glob("*.safetensors"))
    return files[0] if files else None


class NoLossHooks:
    """The hooks of an engine whose trainer takes no seams from Khaos: every flag off, and nothing to undo."""

    def __init__(self, record_into: dict | None = None) -> None:
        self.flags = {"masked_loss": False, "length_buckets": False}
        self._record_into = record_into

    def manifest_fields(self) -> dict:
        return {}

    def __enter__(self) -> "NoLossHooks":
        return self

    def __exit__(self, *exc) -> None:
        if self._record_into is not None:
            self._record_into.update(self.manifest_fields())


def average_adapters(weight_files: Sequence[Path | str], out_file: Path | str) -> dict:
    """Uniform average of the adapter tensors in ``weight_files`` → ``out_file``
    (safetensors). Metadata is carried from the LAST file so the soup loads as
    a checkpoint does; keys must match across files (a mismatch is a corrupt run)."""
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    files = [Path(f) for f in weight_files]
    if not files:
        raise ValueError("average_adapters needs at least one file")
    acc: dict[str, Any] = {}
    keys: set[str] | None = None
    metadata: dict[str, str] | None = None
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            these = set(sf.keys())
            if keys is None:
                keys = these
            elif these != keys:
                raise ValueError(f"adapter key sets differ: {f.name} vs {files[0].name}")
            metadata = sf.metadata() or metadata
            for k in these:
                t = sf.get_tensor(k).to(torch.float32)
                acc[k] = acc[k] + t if k in acc else t
    n = float(len(files))
    averaged = {k: (v / n) for k, v in acc.items()}
    # Store in the source dtype so the file is the size a checkpoint is.
    with safe_open(str(files[-1]), framework="pt", device="cpu") as sf:
        src_dtype = {k: sf.get_tensor(k).dtype for k in (keys or [])}
    averaged = {k: v.to(src_dtype.get(k, torch.float32)).contiguous() for k, v in averaged.items()}
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(averaged, str(out), metadata=metadata or None)
    return {"n_files": len(files), "n_tensors": len(averaged), "out": str(out)}


__all__ = ["TrainingUnavailable", "TrainingUpdate", "average_adapters"]
