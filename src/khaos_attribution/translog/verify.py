"""khaos-log-verify: prove an entry is in the Guild Register's log from a mirror alone, with the committed key.
    khaos-log-verify <entry.json | record.json | leaf hash | record hash> [--mirror URL-or-folder] [--key log-key.pub]
It also checks the newest head is consistent with the last head this machine saw. It reads static files only: the
register's own mirror first, then the public copy; the register's API is never asked. A remote mirror's finished
tiles are kept under ~/.khaos/log-cache (KHAOS_LOG_CACHE names another folder) and checked on every run.
Exit codes: 0 proved; 1 the proof failed (the log or mirror is not what it claims); 2 not found or bad input;
3 no key or no mirror could be read; 4 proved, but a strike after it could be hidden (entries not yet public, or lost
in a recovery); 5 proved, but the newest head is over 48 hours old and this copy may be frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from khaos_attribution import translog as T

DEFAULT_MIRRORS = ("https://register.khaos.dev/mirror", T.PUBLIC_MIRROR_RAW)
STATE_FILE = Path.home() / ".khaos" / "log-verify-state.json"
PROVED, FAILED, NOT_FOUND, UNREADABLE, HELD_BACK, FROZEN = 0, 1, 2, 3, 4, 5
HELD_BACK_NOTE = "proved, but newer entries are not yet public; a strike among them would not show"
LOST_NOTE = "an entry after this one was lost in a recovery; whether it struck this model is unknown"
STALE_HOURS = 48   # the register appends a heartbeat entry daily, quiet or not
DELAY_ALLOWANCE_DAYS = 9   # the public mirror holds new bodies back at most seven days; two more for the daily head


class Unreadable(RuntimeError):
    pass


class NotFound(LookupError):
    pass


class Mirror:
    """A mirror's log folder, as a local path or a base URL; every read is a file under log/."""

    def __init__(self, where: str, timeout: float = 20.0):
        self.where, self.timeout = where.rstrip("/"), timeout
        self.local = not where.startswith(("http://", "https://"))

    def get(self, name: str):
        if self.local:
            if not (Path(self.where) / "log").is_dir():
                raise Unreadable(f"{self.where} holds no log/ folder")
            path = Path(self.where) / "log" / name
            if not path.is_file():
                raise NotFound(name)
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise Unreadable(f"{path}: {e}") from None
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415
        try:
            with urllib.request.urlopen(f"{self.where}/log/{name}", timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise NotFound(name) from None
            raise Unreadable(f"{self.where}: {e.code}") from None
        except (OSError, ValueError) as e:
            raise Unreadable(f"{self.where}: {e}") from None


_TILE_NAME = re.compile(r"(hashes|bodies)/[0-9]{1,9}\.json")
_KEY_ID = re.compile(r"[0-9a-f]{16}")


class CachedMirror(Mirror):
    """A mirror whose full hash tiles and sealed body tiles are kept under ``base``/<key id>/<tile width>/: a tile's
    bytes never change except a body tile under a withhold, and every tile read is still checked against the head.
    Nothing a mirror says names a path: the width is checked before it is used, and nothing outside ``base`` is touched."""

    def __init__(self, mirror: Mirror, base: Path, key_id: str):
        if not isinstance(key_id, str) or not _KEY_ID.fullmatch(key_id):
            raise ValueError("a log key id is 16 lowercase hex characters")
        self.mirror, self.base, self.key_id, self.width, self.cached = mirror, Path(base), key_id, None, set()
        self.where, self.timeout, self.local = mirror.where, mirror.timeout, mirror.local

    def _folder(self) -> Path | None:
        """This key's and width's folder, when it lies in the cache base with no link on the way; else None."""
        if self.width is None:
            return None
        base = self.base.resolve()
        folder = self.base / self.key_id / str(self.width)
        for step in (self.base / self.key_id, folder):
            if step.is_symlink():
                return None
        return folder if folder.resolve() == base / self.key_id / str(self.width) else None

    def _place(self, name: str) -> Path | None:
        folder = self._folder()
        if folder is None or not _TILE_NAME.fullmatch(name):
            return None
        sub = folder / name.split("/")[0]
        if sub.is_symlink() or (sub.exists() and sub.resolve() != folder.resolve() / name.split("/")[0]):
            return None
        return folder / name

    def get(self, name: str):
        place = self._place(name)
        if place is not None and place.is_file() and not place.is_symlink():
            try:
                doc = json.loads(place.read_text(encoding="utf-8"))
                self.cached.add(name)
                return doc
            except (OSError, ValueError):
                self.evict(name)
        doc = self.mirror.get(name)
        if name == "hashes/tail.json":
            width = _width(doc)
            if self.width is not None and width != self.width:
                raise Unreadable("the mirror's tile width changed while it was read; run again")
            self.width = width
        elif place is not None:
            try:
                place.parent.mkdir(parents=True, exist_ok=True)
                if self._place(name) == place:
                    _write_new(place, json.dumps(doc).encode("utf-8"))
            except OSError:
                pass
        return doc

    def evict(self, name: str | None = None) -> None:
        """One tile, or with no name every tile of this key and width, read from the mirror again next time."""
        import shutil  # noqa: PLC0415
        if name is None:
            folder = self._folder()
            if folder is not None and folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)
            self.cached.clear()
            return
        place = self._place(name)
        if place is not None and (place.is_file() or place.is_symlink()):
            place.unlink()
        self.cached.discard(name)


def _write_new(place: Path, data: bytes) -> None:
    """A new file beside ``place``, created afresh and never through a link, then moved over it: a link planted at
    ``place`` is replaced, never written through."""
    import secrets  # noqa: PLC0415
    tmp = place.with_name(f".{place.name}.{os.getpid()}.{secrets.token_hex(6)}.new")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, place)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class TileUnreadable(Unreadable):
    """One tile that is not what its name says: only that tile is read again."""

    def __init__(self, message: str, tile: str):
        super().__init__(message)
        self.tile = tile


def cache_base() -> Path:
    """Where khaos-log-verify keeps tiles; KHAOS_LOG_CACHE names another folder."""
    return Path(os.environ.get("KHAOS_LOG_CACHE") or Path.home() / ".khaos" / "log-cache")


def _width(tail: dict) -> int:
    w = tail.get("tile_width") if isinstance(tail, dict) else None
    if not isinstance(w, int) or isinstance(w, bool) or w < 2 or w > 4096 or w & (w - 1):
        raise Unreadable(f"the mirror's tile width {w!r} is not a power of two from 2 to 4096")
    return w


def _tail(doc, n: int, key: str) -> tuple:
    """A tail's first index and its items, after the checks every tail must pass."""
    if not isinstance(doc, dict) or not isinstance(doc.get(key), list) or not isinstance(doc.get("first"), int):
        raise Unreadable(f"a tail file holds no {key}")
    first, size = doc["first"], doc.get("tree_size")
    if not isinstance(size, int) or size < n:
        raise Unreadable("the mirror changed while it was read; run again")
    if first < 0 or first > size or len(doc[key]) != size - first:
        raise Unreadable(f"a tail's first ({first}) and length do not fit its size ({size})")
    return first, doc[key]


def _tiles(mirror: Mirror, folder: str, count: int, width: int, key: str) -> list:
    out = []
    for k in range(count):
        name = f"{folder}/{k}.json"
        try:
            tile = mirror.get(name)
        except NotFound:
            # A tile the tail counts and the mirror lacks makes the mirror unreadable, so the next one is asked.
            raise Unreadable(f"the mirror lacks {name}") from None
        if not isinstance(tile, dict) or tile.get("tile") != k or not isinstance(tile.get(key), list) or len(tile[key]) != width:
            raise TileUnreadable(f"{name} is not tile {k} of {width}", name)
        out.extend(tile[key])
    return out


def read_leaves(mirror: Mirror, n: int) -> list:
    """The first ``n`` leaf hashes, from the tiles and the tail; from hashes.json where the mirror has no tiles yet."""
    try:
        tail = mirror.get("hashes/tail.json")
    except NotFound:
        hashes = mirror.get("hashes.json").get("leaves") or []
        if len(hashes) < n:
            raise Unreadable("the mirror changed while it was read; run again") from None
        return hashes[:n]
    width = _width(tail)
    first, rest = _tail(tail, n, "leaves")
    if first % width:
        raise Unreadable(f"the hash tail's first ({first}) is not a tile's start")
    return (_tiles(mirror, "hashes", first // width, width, "leaves") + rest)[:n]


def read_bodies(mirror: Mirror, n: int) -> tuple:
    """([body text or None] for the first ``n`` leaves, pending_from), from the sealed tiles and the tail; from
    bodies.json where the mirror has no tiles yet. Only the tiles' absence falls back: a missing tail is an error."""
    try:
        tail = mirror.get("hashes/tail.json")
    except NotFound:
        bundle = mirror.get("bodies.json")
        return list(bundle.get("bodies") or [])[:n], bundle.get("pending_from")
    width = _width(tail)
    try:
        bodies = mirror.get("bodies/tail.json")
    except NotFound:
        raise Unreadable("the mirror has hash tiles but no body tail") from None
    first, rest = _tail(bodies, n, "bodies")
    if first % width:
        raise Unreadable(f"the body tail's first ({first}) is not a tile's start")
    return (_tiles(mirror, "bodies", first // width, width, "bodies") + rest)[:n], bodies.get("pending_from")


def head_at(mirror: Mirror, size: int, pem: str) -> dict:
    head = mirror.get(f"sth/{int(size)}.json")
    T.verify_head(head, pem)
    if head["tree_size"] != size:
        raise T.LogError(f"the head filed as size {size} says {head['tree_size']}")
    return head


def load_log(mirror: Mirror, pem: str) -> dict:
    """The newest signed head and every leaf and published body, each checked against it: the leaf hashes must make
    the signed root, each body must hash to its leaf, and a missing body needs a proved withhold or the delay."""
    if not isinstance(mirror, CachedMirror):
        return _load_log(mirror, pem)
    try:
        log = _load_log(mirror, pem)
    except (T.LogError, Unreadable, KeyError, TypeError, ValueError) as e:
        if not mirror.cached:
            raise
        # A kept tile that no longer checks is read from the mirror again; only the mirror's own copy can fail.
        mirror.evict(e.tile if isinstance(e, TileUnreadable) and e.tile in mirror.cached else None)
        log = _load_log(mirror, pem)
    stale = {_body_tile(log, leaf, mirror.width) for leaf in log["withheld"]} & mirror.cached
    if stale:
        # A body kept before its withhold was proved goes; the mirror's tile holds it null.
        for name in stale:
            mirror.evict(name)
        log = _load_log(mirror, pem)
    return log


def _body_tile(log: dict, leaf: str, width: int | None) -> str | None:
    i = log["leaves"].index(leaf)
    return f"bodies/{i // width}.json" if width and log["entries"][i] is not None else None


def _load_log(mirror: Mirror, pem: str) -> dict:
    latest = mirror.get("sth/latest.json")
    T.verify_head(latest, pem)
    n = latest["tree_size"]
    leaves = read_leaves(mirror, n)
    if len(leaves) < n:
        raise Unreadable("the mirror changed while it was read; run again")
    if not all(T.is_hex256(h) for h in leaves) or T.Tree.from_leaves([bytes.fromhex(h) for h in leaves]).root(n).hex() != latest["root_sha256"]:
        raise T.LogError("the mirror's leaf hashes do not make its signed root")
    texts, pending = read_bodies(mirror, n)
    entries = [None] * n
    for i in range(min(n, len(texts))):
        if texts[i] is not None:
            body = str(texts[i]).encode("utf-8")
            if T.leaf_hash(body).hex() != leaves[i]:
                raise T.LogError(f"the body of entry {i} does not hash to its leaf")
            entries[i] = json.loads(body)
    withheld = {}
    for i, e in enumerate(entries):
        if e and e.get("kind") == "withhold" and e.get("withholds") in leaves and leaves.index(e["withholds"]) < i:
            withheld.setdefault(e["withholds"], e)
    if pending is not None and int(pending) < n:
        pending = int(pending)
        first = head_at(mirror, pending + 1, pem)
        if _hours(first["timestamp"], latest["timestamp"]) > DELAY_ALLOWANCE_DAYS * 24:
            raise T.LogError(f"the mirror holds back bodies from entry {pending} on, beyond the publication delay")
    else:
        pending = None
    for i in range(n):
        if entries[i] is None and leaves[i] not in withheld and (pending is None or i < pending):
            raise T.LogError(f"entry {i}'s body is missing and no proved withhold names it")
    lost = {leaf for leaf, e in withheld.items() if e.get("lost") is True}
    return {"latest": latest, "leaves": leaves, "entries": entries, "withheld": withheld, "pending": pending, "lost": lost}


def _hours(earlier: str, later: str) -> float:
    from datetime import datetime  # noqa: PLC0415
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (datetime.strptime(later, fmt) - datetime.strptime(earlier, fmt)).total_seconds() / 3600


def resolve(target: str, log: dict) -> tuple:
    """(leaf index, how it was found, the entry if the target carried it, the record hash asked for or None)."""
    text = target
    path = Path(target)
    if len(target) != T.SHA256_HEX and path.is_file():
        text = path.read_text(encoding="utf-8")
    text = text.strip()
    if T.is_hex256(text.lower()):
        h = text.lower()
        if h in log["leaves"]:
            return log["leaves"].index(h), "the leaf hash given", None, None
        return _by_record(h, log)
    try:
        doc = json.loads(text)
    except ValueError:
        raise NotFound("not a file, a JSON document or a sha256 hex digest") from None
    if isinstance(doc, dict) and doc.get("version") == T.ENTRY_VERSION:
        h = T.entry_leaf_hex(doc)
        if h not in log["leaves"]:
            raise NotFound("that entry is not in the log")
        return log["leaves"].index(h), "the entry body given, hashed here", doc, None
    if isinstance(doc, dict) and T.is_hex256(doc.get("record_sha256")):
        if isinstance(doc.get("record"), dict) and doc.get("signed_version"):
            from khaos_attribution import registry as R  # noqa: PLC0415
            if R.sha256_hex(R.canonical_bytes(doc["record"])) != doc["record_sha256"]:
                raise T.LogError("the signed record's hash is not over its own record: it was changed, or written again by a "
                                 "tool that rewrote its numbers (50.0 as 50)")
        return _by_record(doc["record_sha256"], log)
    raise NotFound("the document is neither a log entry nor a register record")


def _by_record(record_sha256: str, log: dict) -> tuple:
    """The newest proved admit of the record; a hint file is never trusted for this."""
    found = [i for i, e in enumerate(log["entries"]) if e and e.get("kind") == "admit" and e.get("record_sha256") == record_sha256]
    if not found:
        more = " (some recent bodies are not yet published)" if log["pending"] is not None else ""
        raise NotFound(f"no admitted entry for record {record_sha256}{more}")
    return found[-1], "the record's admit entry", None, record_sha256


def strike_of(log: dict, index: int) -> dict | None:
    """The proved revoke after the entry that names it, or a withhold of a revoke that still names it. A revoke no
    warden's machine appended is the account's own withdrawal (`by_account`); a withheld one cannot say."""
    leaf = log["leaves"][index]
    for i in range(index + 1, len(log["leaves"])):
        e = log["entries"][i]
        if e and e.get("kind") == "revoke" and e.get("revokes") == leaf:
            return {"leaf_hash": log["leaves"][i], "reason": e.get("reason"), "sitting": e.get("sitting"), "reason_withheld": False,
                    "by_account": not e.get("appended_by")}
        if e and e.get("kind") == "withhold" and e.get("revokes") == leaf:
            return {"leaf_hash": e["withholds"], "reason": None, "sitting": e.get("sitting"), "reason_withheld": True,
                    "by_account": None}
    return None


def withdrawn_after(log: dict, index: int) -> str | None:
    """The proved withdraw entry after an admit that names its record: its account withdrew a model already struck."""
    target = log["entries"][index] or {}
    if target.get("kind") != "admit":
        return None
    for i in range(index + 1, len(log["leaves"])):
        e = log["entries"][i]
        if e and e.get("kind") == "withdraw" and e.get("record_sha256") == target.get("record_sha256"):
            return log["leaves"][i]
    return None


def lost_after(log: dict, index: int) -> bool:
    """Whether an entry after this one was lost in a recovery and no later confirm answers that loss, for every admit
    (naming none) or for this one."""
    leaf, leaves = log["leaves"][index], log["leaves"]
    for h in (log["entries"][index] or {}).get("pending_lost") or []:
        # An admit appended after a recovery waits on the losses it names until a confirm after it answers them.
        if not any(e and e.get("kind") == "confirm" and e.get("answers") == h and e.get("confirms") in (None, leaf)
                   for e in log["entries"][index + 1:]):
            return True
    for i in range(index + 1, len(leaves)):
        if leaves[i] not in log["lost"]:
            continue
        if not any(e and e.get("kind") == "confirm" and e.get("answers") == leaves[i] and e.get("confirms") in (None, leaf)
                   for e in log["entries"][i + 1:]):
            return True
    return False


def prove(target: str, mirror: Mirror, pem: str, state: dict | None = None, now: str | None = None) -> dict:
    """Every check, from the mirror's files alone; raises LogError on a failed proof, NotFound or Unreadable otherwise."""
    log = load_log(mirror, pem)
    latest = log["latest"]
    index, how, given, record = resolve(target, log)
    leaf_hex = log["leaves"][index]
    proof, head = _included(mirror, leaf_hex, latest, pem)
    if proof["leaf_index"] != index:
        raise T.LogError("the inclusion proof is for another position than the leaf holds")
    entry = log["entries"][index] or given
    if record is not None and not (entry and entry.get("kind") == "admit" and entry.get("record_sha256") == record):
        raise T.LogError("the entry proved is not an admit of the record asked for")
    warnings = []
    out = {"leaf_hash": leaf_hex, "found_by": how, "leaf_index": index, "tree_size": head["tree_size"],
           "head_timestamp": head["timestamp"], "key_id": head["key_id"], "entry": log["entries"][index],
           "withheld": leaf_hex in log["withheld"], "pending": log["entries"][index] is None and leaf_hex not in log["withheld"],
           "struck": strike_of(log, index), "withdrawn": withdrawn_after(log, index), "warnings": warnings}
    held = [i for i in range(index + 1, len(log["leaves"])) if log["entries"][i] is None and log["leaves"][i] not in log["withheld"]]
    out["held_back_after"] = len(held)
    out["lost_after"] = lost_after(log, index) and not out["struck"]
    if held and not out["struck"]:
        warnings.append(f"{HELD_BACK_NOTE} ({len(held)} held back by the publication delay)")
    if out["lost_after"]:
        warnings.append(LOST_NOTE)
    from datetime import datetime, timezone  # noqa: PLC0415
    out["stale_hours"] = round(_hours(latest["timestamp"], now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")), 1)
    out["frozen"] = out["stale_hours"] > STALE_HOURS
    if out["frozen"]:
        warnings.append(f"the newest head is {out['stale_hours']:.0f} hours old; this copy may be frozen (the log takes a "
                        "heartbeat entry daily)")
    if state and state.get("tree_size"):
        seen = int(state["tree_size"])
        if seen > latest["tree_size"]:
            raise T.LogError(f"the mirror's newest head ({latest['tree_size']}) is older than one already seen ({seen})")
        root = T.Tree.from_leaves([bytes.fromhex(h) for h in log["leaves"][:seen]]).root(seen).hex()
        if root != state.get("root_sha256"):
            raise T.LogError(f"the log at size {seen} now has another root than when it was seen: the log was rewritten")
        out["consistent_since"] = seen
    out["latest"] = {"tree_size": latest["tree_size"], "root_sha256": latest["root_sha256"], "timestamp": latest["timestamp"],
                     "key_id": latest["key_id"]}
    return out


def _included(mirror: Mirror, leaf_hex: str, latest: dict, pem: str) -> tuple:
    """The leaf's proof and the signed head it proves against, after the proof is checked."""
    proof = mirror.get(f"proofs/{leaf_hex}.json")
    head = latest if proof["tree_size"] == latest["tree_size"] else head_at(mirror, proof["tree_size"], pem)
    T.verify_inclusion(bytes.fromhex(leaf_hex), int(proof["leaf_index"]), head["tree_size"], T.unhex(proof["audit_path"]),
                       bytes.fromhex(head["root_sha256"]))
    return proof, head


def _load_key(path: str | None) -> T.PinnedKey | None:
    if path:
        from khaos_attribution import registry as R  # noqa: PLC0415
        pem = Path(path).read_text(encoding="ascii")
        return T.PinnedKey(pem, R.key_id_of(R.public_raw(R.load_public_pem(pem))))
    return T.committed_key()


def _read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_state(path: Path, latest: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".new")
        tmp.write_text(json.dumps(latest, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="khaos-log-verify", description=__doc__.split("\n", 1)[0])
    p.add_argument("target", help="an entry or record JSON file, or a leaf hash or record hash")
    p.add_argument("--mirror", action="append", help="a mirror URL or folder; repeat to try several in order")
    p.add_argument("--key", help="the log's public key (PEM); default the key committed in this package")
    p.add_argument("--state", default=str(STATE_FILE), help="where the last head seen is kept; 'none' to keep nothing")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    try:
        key = _load_key(a.key)
    except (OSError, ValueError) as e:
        print(f"cannot read the log key: {e}", file=sys.stderr)
        return UNREADABLE
    if key is None:
        print("no log key is committed in this package yet; pass the published key with --key", file=sys.stderr)
        return UNREADABLE
    state_path = None if a.state == "none" else Path(a.state)
    state = _read_state(state_path) if state_path else {}
    mirrors, last = a.mirror or list(DEFAULT_MIRRORS), None
    for where in mirrors:
        try:
            mirror = Mirror(where)
            if not mirror.local:
                mirror = CachedMirror(mirror, cache_base(), T.key_id_of_pem(key.public_pem))
            out = prove(a.target, mirror, key.public_pem, state.get(where))
        except Unreadable as e:
            last = e
            continue
        except NotFound as e:
            print(f"not found: {e}", file=sys.stderr)
            return NOT_FOUND
        except (T.LogError, KeyError, TypeError, ValueError) as e:
            print(f"FAILED against {where}: {e}", file=sys.stderr)
            return FAILED
        if state_path:
            state[where] = out["latest"]
            _write_state(state_path, state)
        out["mirror"] = where
        if a.json:
            print(json.dumps(out, indent=1, sort_keys=True))
        else:
            entry = out["entry"] or {}
            what = f"{entry.get('kind', 'unpublished')} entry" + (f" for No. {entry['serial']}" if entry.get("serial") else "")
            print(f"PROVED: the {what} is leaf {out['leaf_index']} of the log at size {out['tree_size']}, "
                  f"head signed {out['head_timestamp']} by log key {out['key_id']} ({where}).")
            if out["struck"]:
                reason = "its reason is withheld" if out["struck"]["reason_withheld"] else out["struck"]["reason"]
                what = "WITHDRAWN by the account that published it" if out["struck"]["by_account"] else "STRUCK"
                print(f"{what}: {reason} (entry {out['struck']['leaf_hash']})")
            if out["withdrawn"]:
                print(f"WITHDRAWN by the account that published it (entry {out['withdrawn']})")
            if out["withheld"]:
                print("The body is withheld, by a withhold entry in the log; the proof stands without it.")
            elif out["entry"] is None:
                print("The body is not yet published in this mirror; the proof stands without it.")
            if "consistent_since" in out:
                print(f"The newest head ({out['latest']['tree_size']}) extends the one seen before ({out['consistent_since']}).")
            for w in out["warnings"]:
                print(f"WARNING: {w}")
        if out["frozen"]:
            return FROZEN
        if (out["held_back_after"] and not out["struck"]) or out["lost_after"]:
            return HELD_BACK
        return PROVED
    print(f"no mirror could be read: {last}", file=sys.stderr)
    return UNREADABLE


if __name__ == "__main__":
    sys.exit(main())
