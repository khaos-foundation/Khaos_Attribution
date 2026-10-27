"""khaos-log-verify: prove an entry is in the Guild Register's log from a mirror alone, with the committed key.
    khaos-log-verify <entry.json | record.json | leaf hash | record hash> [--mirror URL-or-folder] [--key log-key.pub]
It also checks the newest head is consistent with the last head this machine saw. The register itself is never asked.
Exit codes: 0 proved; 1 the proof failed (the log or mirror is not what it claims); 2 not found or bad input;
3 no key or no mirror could be read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from khaos_attribution import translog as T

DEFAULT_MIRRORS = ("https://register.khaos.dev/mirror", "https://raw.githubusercontent.com/guild-register/ledger/main")
STATE_FILE = Path.home() / ".khaos" / "log-verify-state.json"
PROVED, FAILED, NOT_FOUND, UNREADABLE = 0, 1, 2, 3
STALE_HOURS = 48   # the register signs a fresh head daily, quiet or not
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
            return json.loads(path.read_text(encoding="utf-8"))
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

    def raw(self, name: str) -> bytes:
        """A file's exact bytes: an entry body is hashed as it is stored, never re-encoded."""
        if self.local:
            path = Path(self.where) / "log" / name
            if not path.is_file():
                raise NotFound(name)
            return path.read_bytes()
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415
        try:
            with urllib.request.urlopen(f"{self.where}/log/{name}", timeout=self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise NotFound(name) from None
            raise Unreadable(f"{self.where}: {e.code}") from None
        except OSError as e:
            raise Unreadable(f"{self.where}: {e}") from None


def head_at(mirror: Mirror, size: int, pem: str) -> dict:
    head = mirror.get(f"sth/{int(size)}.json")
    T.verify_head(head, pem)
    if head["tree_size"] != size:
        raise T.LogError(f"the head filed as size {size} says {head['tree_size']}")
    return head


def load_log(mirror: Mirror, pem: str) -> dict:
    """The newest signed head and every leaf and published body, each checked against it: the leaf hashes must make
    the signed root, each body must hash to its leaf, and a missing body needs a proved withhold or the delay."""
    latest = mirror.get("sth/latest.json")
    T.verify_head(latest, pem)
    n = latest["tree_size"]
    hashes = mirror.get("hashes.json").get("leaves") or []
    if len(hashes) < n:
        raise Unreadable("the mirror changed while it was read; run again")
    leaves = hashes[:n]
    if not all(T.is_hex256(h) for h in leaves) or T.Tree.from_leaves([bytes.fromhex(h) for h in leaves]).root(n).hex() != latest["root_sha256"]:
        raise T.LogError("the mirror's leaf hashes do not make its signed root")
    bundle = mirror.get("bodies.json")
    texts = list(bundle.get("bodies") or [])
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
    pending = bundle.get("pending_from")
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
    return {"latest": latest, "leaves": leaves, "entries": entries, "withheld": withheld, "pending": pending}


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
                raise T.LogError("the signed record's hash is not over its own record")
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
    """The proved revoke after the entry that names it, or a withhold of a revoke that still names it."""
    leaf = log["leaves"][index]
    for i in range(index + 1, len(log["leaves"])):
        e = log["entries"][i]
        if e and e.get("kind") == "revoke" and e.get("revokes") == leaf:
            return {"leaf_hash": log["leaves"][i], "reason": e.get("reason"), "sitting": e.get("sitting"), "reason_withheld": False}
        if e and e.get("kind") == "withhold" and e.get("revokes") == leaf:
            return {"leaf_hash": e["withholds"], "reason": None, "sitting": e.get("sitting"), "reason_withheld": True}
    return None


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
           "struck": strike_of(log, index), "warnings": warnings}
    if log["pending"] is not None and log["pending"] < len(log["leaves"]):
        later = len(log["leaves"]) - max(log["pending"], index + 1)
        if later > 0 or out["pending"]:
            warnings.append(f"{len(log['leaves']) - log['pending']} recent entries are not yet published in this mirror "
                            "(the publication delay); a strike among them would not show yet")
    from datetime import datetime, timezone  # noqa: PLC0415
    out["stale_hours"] = round(_hours(latest["timestamp"], now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")), 1)
    if out["stale_hours"] > STALE_HOURS:
        warnings.append(f"the newest head is {out['stale_hours']:.0f} hours old; the log is re-signed daily, so this mirror "
                        "may be stale")
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
            out = prove(a.target, Mirror(where), key.public_pem, state.get(where))
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
                print(f"STRUCK: {reason} (entry {out['struck']['leaf_hash']})")
            if out["withheld"]:
                print("The body is withheld, by a withhold entry in the log; the proof stands without it.")
            elif out["entry"] is None:
                print("The body is not yet published in this mirror; the proof stands without it.")
            if "consistent_since" in out:
                print(f"The newest head ({out['latest']['tree_size']}) extends the one seen before ({out['consistent_since']}).")
            for w in out["warnings"]:
                print(f"WARNING: {w}")
        return PROVED
    print(f"no mirror could be read: {last}", file=sys.stderr)
    return UNREADABLE


if __name__ == "__main__":
    sys.exit(main())
