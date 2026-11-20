"""Verify keys: a free lane of its own to /api/verify for authorities, researchers and media, far above the public bound.
Only each key's hash is kept; the key itself is shown once, when it is issued."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path

KINDS = ("authority", "research", "media")
FILE_NAME = "verify-keys.json"


def _path(cfg: dict) -> Path:
    return Path(cfg["data"]) / FILE_NAME


def _digest(key: str) -> str:
    return hashlib.sha256(key.strip().encode()).hexdigest()


def load(cfg: dict) -> dict:
    try:
        doc = json.loads(_path(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _save(cfg: dict, doc: dict) -> None:
    path = _path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{FILE_NAME}.")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)


def issue(cfg: dict, holder: str, kind: str) -> str:
    """A new key for one holder; returned once and never stored."""
    if kind not in KINDS:
        raise ValueError(f"kind is one of {', '.join(KINDS)}")
    key = "kv_" + secrets.token_urlsafe(24)
    doc = load(cfg)
    doc[_digest(key)] = {"holder": holder.strip(), "kind": kind}
    _save(cfg, doc)
    return key


def revoke(cfg: dict, holder: str) -> int:
    doc = load(cfg)
    kept = {d: v for d, v in doc.items() if v.get("holder") != holder.strip()}
    _save(cfg, kept)
    return len(doc) - len(kept)


class Allowed:
    """Whether a presented key is on the list, re-reading the file only when it changes."""

    def __init__(self, cfg: dict):
        self.cfg, self._stamp, self._digests = cfg, None, set()

    def __call__(self, key: str | None) -> bool:
        if not key:
            return False
        try:
            stamp = _path(self.cfg).stat().st_mtime_ns
        except OSError:
            return False
        if stamp != self._stamp:
            self._digests, self._stamp = set(load(self.cfg)), stamp
        return _digest(key) in self._digests
