"""The rights stamp: one hash of an artist's track-rights file, the same in every room. The Space serves rights as a
snapshot copied at import; the stamp lets each side say whether that snapshot is the current record without a
connection. Only the `tracks` mapping is hashed, canonically, so a re-save that changes nothing does not read as one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from khaos_attribution.canonical import compact


def rights_sha256(path: Path | str) -> str | None:
    """sha256 of the canonical ``tracks`` mapping in a rights.json, or None
    when the file is missing or unreadable — never a hash of nothing."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    tracks = data.get("tracks")
    if not isinstance(tracks, dict):
        return None
    return hashlib.sha256(compact(tracks)).hexdigest()
