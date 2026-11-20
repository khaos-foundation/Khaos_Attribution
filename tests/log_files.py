"""A mirror's exported log as a verifier reads it, whichever layout the export wrote: its leaf hashes, its bodies, its
delay edge, and one body changed where it lies (a tile or a tail), for the tests that tamper with a copy."""
import json
from pathlib import Path

from khaos_attribution.translog import verify as V


def _mirror(log_dir) -> V.Mirror:
    return V.Mirror(str(Path(log_dir).parent))


def size(log_dir) -> int:
    log_dir = Path(log_dir)
    tail = log_dir / "hashes" / "tail.json"
    doc = json.loads((tail if tail.is_file() else log_dir / "hashes.json").read_text())
    return int(doc["tree_size"])


def leaves(log_dir) -> list:
    return V.read_leaves(_mirror(log_dir), size(log_dir))


def bodies(log_dir) -> list:
    return V.read_bodies(_mirror(log_dir), size(log_dir))[0]


def pending_from(log_dir):
    return V.read_bodies(_mirror(log_dir), size(log_dir))[1]


def body_of(log_dir, leaf_hex: str):
    return bodies(log_dir)[leaves(log_dir).index(leaf_hex)]


def _place(log_dir, index: int) -> tuple:
    """(the file holding body ``index``, the list key in it, the position there)."""
    log_dir = Path(log_dir)
    if not (log_dir / "hashes" / "tail.json").is_file():
        return log_dir / "bodies.json", "bodies", index
    width = json.loads((log_dir / "hashes" / "tail.json").read_text())["tile_width"]
    first = json.loads((log_dir / "bodies" / "tail.json").read_text())["first"]
    if index >= first:
        return log_dir / "bodies" / "tail.json", "bodies", index - first
    return log_dir / "bodies" / f"{index // width}.json", "bodies", index % width


def set_body(log_dir, index: int, text) -> None:
    """Body ``index`` replaced by ``text`` (None to drop it) in whichever file holds it."""
    path, key, at = _place(log_dir, index)
    doc = json.loads(path.read_text())
    doc[key][at] = text
    path.write_text(json.dumps(doc))


def edit_bodies(log_dir, change) -> None:
    """``change(text) -> text`` applied to every published body, wherever it lies."""
    for i, text in enumerate(bodies(log_dir)):
        if text is not None and change(text) != text:
            set_body(log_dir, i, change(text))


def set_pending_from(log_dir, value) -> None:
    log_dir = Path(log_dir)
    path = log_dir / "bodies" / "tail.json" if (log_dir / "hashes" / "tail.json").is_file() else log_dir / "bodies.json"
    doc = json.loads(path.read_text())
    doc["pending_from"] = value
    path.write_text(json.dumps(doc))


def set_leaf(log_dir, index: int, leaf_hex: str) -> None:
    """Leaf hash ``index`` replaced in whichever file holds it."""
    log_dir = Path(log_dir)
    tail = log_dir / "hashes" / "tail.json"
    if not tail.is_file():
        path, at = log_dir / "hashes.json", index
    else:
        doc = json.loads(tail.read_text())
        width, first = doc["tile_width"], doc["first"]
        path, at = (tail, index - first) if index >= first else (log_dir / "hashes" / f"{index // width}.json", index % width)
    doc = json.loads(path.read_text())
    doc["leaves"][at] = leaf_hex
    path.write_text(json.dumps(doc))
