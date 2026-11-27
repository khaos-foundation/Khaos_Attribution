"""The embedding contract: how any Khaos surface turns audio into the one
CLAP vector space where catalogue and outputs are comparable.

Both sides must embed identically — same model, same pinned checkpoint,
same deterministic windowing, same pooling — so the constants and the pure
math live here and each app supplies only its model forward pass. Fixed
10 s windows at a 5 s hop, each unit-normalised before mean-pooling,
replace laion_clap's random crop so the same audio always gives the same
vector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

MODEL_ID = "laion/larger_clap_music"
# HF URL for the music-specific CLAP checkpoint (HTSAT-base), pinned to a
# commit SHA so the downloaded binary cannot change silently.
CLAP_CKPT_REVISION = "b3708341862f581175dba5c356a4ebf74a9b6651"
CLAP_CKPT_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"
CLAP_CKPT_URL = (
    f"https://huggingface.co/lukewys/laion_clap/resolve/{CLAP_CKPT_REVISION}/"
    f"{CLAP_CKPT_FILENAME}"
)

# laion_clap loads all three tokenizers when it is imported and builds its text side from roberta-base.
# Each commit is the one fetched when the cache names none; each file is one an offline load cannot do
# without, "a|b" where either spelling serves.
CLAP_TEXT_SNAPSHOTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("roberta-base", "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
     ("config.json", "model.safetensors|pytorch_model.bin", "vocab.json", "merges.txt")),
    ("bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594", ("vocab.txt",)),
    ("facebook/bart-base", "aadd2ab0ae0c8268c7c9693540e9904811f36177", ("vocab.json", "merges.txt")),
)

TARGET_SR = 48000     # CLAP requires 48 kHz
WINDOW_SEC = 10.0     # exactly what the encoder takes
HOP_SEC = 5.0
MAX_WINDOW_BATCH = 16  # ~31 MB of float32 per forward pass


def l2_normalise(vector: np.ndarray) -> np.ndarray:
    """L2-normalise a 1D float32 vector; near-zero vectors pass through."""
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        return vector
    return (vector / norm).astype(np.float32)


def window_starts(n_samples: int, win: int, hop: int) -> list[int]:
    """Deterministic window offsets covering the whole signal.

    The final window is flush with the END of the audio rather than dropped,
    so the last few seconds are never silently unseen.
    """
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, hop))
    if starts[-1] + win < n_samples:
        starts.append(n_samples - win)
    return starts


def embed_windows(audio: np.ndarray, embed_batch) -> np.ndarray:
    """Mean-pool the embeddings of deterministic windows over `audio`.

    Each window is unit-normalised before pooling; the caller normalises the
    pooled result. `embed_batch` is the app's model forward.
    """
    win = int(WINDOW_SEC * TARGET_SR)
    hop = int(HOP_SEC * TARGET_SR)
    starts = window_starts(len(audio), win, hop)
    parts = []
    for k in range(0, len(starts), MAX_WINDOW_BATCH):
        chunk = np.stack([audio[i:i + win]
                          for i in starts[k:k + MAX_WINDOW_BATCH]])
        parts.append(np.asarray(embed_batch(chunk), dtype=np.float32))
    embs = np.concatenate(parts) if len(parts) > 1 else parts[0]
    if embs.ndim == 1:                      # a stub returned a single vector
        return embs
    unit = np.stack([l2_normalise(e) for e in embs])
    return unit.mean(axis=0).astype(np.float32)


def hub_cache_dir() -> Path:
    """Where transformers keeps its cache in this process: its three old overrides in its own order,
    then the hub library's answer (HF_HUB_CACHE, HF_HOME, XDG_CACHE_HOME)."""
    import os  # noqa: PLC0415

    for name in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE"):
        if os.environ.get(name):
            return Path(os.environ[name])
    from huggingface_hub import constants  # noqa: PLC0415

    return Path(constants.HF_HUB_CACHE)


def _repo_dir(cache: Path, repo: str) -> Path:
    return cache / ("models--" + repo.replace("/", "--"))


def _served_commit(cache: Path, repo: str) -> str | None:
    """The commit refs/main names, read as the hub reads it: raw, so a ref with stray whitespace names nothing."""
    try:
        raw = (_repo_dir(cache, repo) / "refs" / "main").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    return raw if len(raw) == 40 and all(c in "0123456789abcdef" for c in raw) else None


def _short_of(cache: Path, repo: str, commit: str | None, files: tuple[str, ...]) -> list[str]:
    """The listed files a snapshot lacks, each by its first spelling; all of them when there is no commit."""
    snapshot = _repo_dir(cache, repo) / "snapshots" / (commit or "")
    return [spec.split("|")[0] for spec in files
            if commit is None or not any((snapshot / name).is_file() for name in spec.split("|"))]


def clap_text_missing(cache_dir: Path | str | None = None) -> list[str]:
    """The CLAP_TEXT_SNAPSHOTS files an offline load would not find, as "repo/file"; empty when it has
    them all. It reads the snapshot refs/main names, as that load does, and the disk only."""
    cache = Path(cache_dir) if cache_dir else hub_cache_dir()
    return [f"{repo}/{name}" for repo, _, files in CLAP_TEXT_SNAPSHOTS
            for name in _short_of(cache, repo, _served_commit(cache, repo), files)]


def clap_text_prefetch(cache_dir: Path | str | None = None, progress: Callable[[str], None] | None = None) -> None:
    """Fetch the files an offline load would not find, into the snapshot it reads; the pin is used
    when the cache names no commit, or one that cannot be completed. Raises when nothing lands."""
    import os  # noqa: PLC0415

    from huggingface_hub import snapshot_download  # noqa: PLC0415

    cache = Path(cache_dir) if cache_dir else hub_cache_dir()

    def lands(repo: str, commit: str, files: tuple[str, ...]) -> list[str]:
        if progress:
            progress(f"{repo} @ {commit[:12]}")
        # Hugging Face, then the Khaos mirror, each file against its pin; the hub library when neither can serve it.
        from khaos_attribution import weights as mirror  # noqa: PLC0415
        try:
            served = mirror.fetch_into_hub_cache(repo, cache, [spec.split("|")[0] for spec in files], revision=commit)
        except mirror.PinMismatch:
            raise
        except mirror.Unserved as exc:
            if exc.pinned:
                raise
            served = None
        except (mirror.MirrorError, OSError):
            served = None
        if served is None:
            with mirror.anonymous_hub():
                snapshot_download(repo, revision=commit, allow_patterns=[spec.split("|")[0] for spec in files], cache_dir=str(cache))
        # The hub hands back a snapshot folder it could not complete without raising.
        return _short_of(cache, repo, commit, files)

    for repo, pin, files in CLAP_TEXT_SNAPSHOTS:
        named = _served_commit(cache, repo)
        if not _short_of(cache, repo, named, files):
            continue
        # The commit another program's cache names is completed where it stands: repointing its ref takes its files away.
        if named and named != pin:
            try:
                whole = not lands(repo, named, files)
            except Exception:  # noqa: BLE001
                whole = False   # a commit the hub cannot serve: the pin is tried, and its failure is the one reported
            if whole:
                continue
        short = lands(repo, pin, files)
        if short:
            raise ConnectionError(f"{repo}: the fetch did not land {', '.join(short)} (offline, blocked, or refused by the hub)")
        if _served_commit(cache, repo) in (None, named):
            # Written only when the ref is still the one that could not serve: a newer one is another program's.
            ref = _repo_dir(cache, repo) / "refs" / "main"
            ref.parent.mkdir(parents=True, exist_ok=True)
            part = ref.with_name(f"main.{os.getpid()}.part")
            try:
                part.write_text(pin, encoding="utf-8")
                os.replace(part, ref)
            finally:
                part.unlink(missing_ok=True)
