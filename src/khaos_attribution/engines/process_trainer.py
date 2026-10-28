"""What every engine trained as a process shares: clips written from a Khaos dataset, epochs as the trainer's
steps, the checkpoint a file name carries, and the two waits a session makes (a checkpoint settling on disk, the
epoch a step belongs to). Each engine's train.py keeps its own names over these."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path


def resample_stereo(audio, sr: int, target_sr: int, *, engine_name: str, unavailable: type):
    """Two channels at the engine's rate: mono doubled, extra channels dropped, resampled only when the rate differs."""
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
        raise unavailable(f"resampling a clip for {engine_name} needs librosa in this environment") from exc
    return np.stack([librosa.resample(audio[:, c], orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
                     for c in range(2)], axis=1)


def write_clips(dataset_json: Path | str, out_dir: Path | str, *, sample_rate: int, pair_suffix: str, engine_name: str,
                unavailable: type, log: logging.Logger, require_audio: bool = False) -> dict:
    """Every sample of a Khaos dataset.json as ``<stem><pair_suffix>`` (stereo, 16-bit) beside ``<stem>.txt``, its
    caption; one bad clip is counted and the rest go on; with require_audio an empty clip counts as failed."""
    import soundfile as sf  # noqa: PLC0415
    samples = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed = failed = 0
    for s in samples:
        stem = Path(s["filename"]).stem
        try:
            audio, sr = sf.read(str(s["audio_path"]), dtype="float32", always_2d=True)
            if require_audio and not len(audio):
                raise ValueError("no audio")
            audio = resample_stereo(audio, sr, sample_rate, engine_name=engine_name, unavailable=unavailable)
            sf.write(str(out / f"{stem}{pair_suffix}"), audio, sample_rate, subtype="PCM_16")
            (out / f"{stem}.txt").write_text(str(s.get("caption") or "").strip() + "\n", encoding="utf-8")
            processed += 1
        except Exception as exc:  # noqa: BLE001 — one bad clip is counted, the rest go on
            log.error("%s preprocess: %s failed: %s", engine_name, stem, exc)
            failed += 1
    return {"processed": processed, "failed": failed}


def steps_for(cfg: dict, n_pairs: int) -> dict:
    """Khaos's epochs as the trainer's steps: steps per epoch at the recipe's batch size, the total, and the
    checkpoint cadence in steps; a trainer keeps a short last batch, so an epoch is the batches it runs."""
    batch = max(1, int(cfg.get("batch_size") or 1))
    per_epoch = max(1, -(-n_pairs // batch))
    epochs = int(cfg["max_epochs"])
    every_epochs = max(1, int(cfg.get("checkpoint_every_n_epochs") or 1))
    return {"per_epoch": per_epoch, "total": per_epoch * epochs, "checkpoint_every": per_epoch * every_epochs,
            "epochs": epochs}


def checkpoint_step(path: Path) -> int | None:
    """The step a checkpoint's name carries (``…step=1234…``), or None."""
    m = re.search(r"step=(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def settled(sizes: dict, path: Path, final: bool) -> bool:
    """Whether a checkpoint held still across two polls (size and mtime), or the trainer has exited: it is written
    in one long, non-atomic write. ``sizes`` is the session's memory of the last poll."""
    try:
        st = path.stat()
    except OSError:
        return False
    seen = sizes.get(path)
    sizes[path] = (st.st_size, st.st_mtime_ns)
    return final or seen == (st.st_size, st.st_mtime_ns)


def epoch_of(start_epoch: int, per_epoch: int, step: int) -> int:
    """The epoch a trainer step belongs to, counted from the epoch a resume started at; never below the first."""
    return start_epoch + max(1, round(step / per_epoch))


class HeldoutNotScored(RuntimeError):
    """A held-out score was asked at an epoch whose weights the trainer did not save."""


def due_checkpoint(step: int, steps: dict) -> bool:
    """Whether the trainer saves its weights at ``step`` of its process: on the cadence and at the last step."""
    return step % steps["checkpoint_every"] == 0 or step == steps["total"]


def releasable(queue: list, laid_out: set, steps: dict, alive: bool) -> list:
    """The queued ``(step, order, update)`` events in step order, held at the first epoch whose checkpoint is due but
    not yet laid out, so a held-out score taken at that epoch reads that epoch's weights."""
    queue.sort(key=lambda e: (e[0], e[1]))
    out = []
    while queue:
        step, order, _update = queue[0]
        if alive and order == 0 and due_checkpoint(step, steps) and not any(s >= step for s in laid_out):
            break
        out.append(queue.pop(0)[2])
    return out


def scored_adapter(session, named, engine_name: str):
    """The adapter a held-out score reads: the one named, else the checkpoint of the epoch just finished."""
    if named is not None:
        return named
    epoch = int(getattr(session, "epoch_done", 0) or 0)
    adapter = session.checkpoint_by_epoch.get(epoch)
    if adapter is None:
        raise HeldoutNotScored(f"{engine_name} saved no weights at epoch {epoch}; its held-out loss is scored only "
                               "on the epochs it saves (HELDOUT_ON_CHECKPOINTS)")
    return adapter
