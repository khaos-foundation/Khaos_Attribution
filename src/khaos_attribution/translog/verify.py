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


def resolve(target: str, mirror: Mirror) -> tuple:
    """(leaf hash, how it was found) for an entry body, a record document, a leaf hash or a record hash."""
    text = target
    path = Path(target)
    if len(target) != T.SHA256_HEX and path.is_file():
        text = path.read_text(encoding="utf-8")
    text = text.strip()
    if T.is_hex256(text.lower()):
        h = text.lower()
        if h in set(mirror.get("hashes.json")["leaves"]):
            return h, "the leaf hash given"
        return _by_record(h, mirror)
    try:
        doc = json.loads(text)
    except ValueError:
        raise NotFound("not a file, a JSON document or a sha256 hex digest") from None
    if isinstance(doc, dict) and doc.get("version") == T.ENTRY_VERSION:
        return T.entry_leaf_hex(doc), "the entry body given, hashed here"
    if isinstance(doc, dict) and T.is_hex256(doc.get("record_sha256")):
        if isinstance(doc.get("record"), dict) and doc.get("signed_version"):
            from khaos_attribution import registry as R  # noqa: PLC0415
            if R.sha256_hex(R.canonical_bytes(doc["record"])) != doc["record_sha256"]:
                raise T.LogError("the signed record's hash is not over its own record")
        return _by_record(doc["record_sha256"], mirror)
    raise NotFound("the document is neither a log entry nor a register record")


def _by_record(record_sha256: str, mirror: Mirror) -> tuple:
    found = mirror.get("records.json").get(record_sha256) or []
    admits = [x["leaf_hash"] for x in found if x.get("kind") == "admit"]
    if not admits:
        raise NotFound(f"no admitted entry for record {record_sha256}")
    return admits[0], "the record's admit entry"


def prove(target: str, mirror: Mirror, pem: str, state: dict | None = None) -> dict:
    """Every check, from the mirror's files alone; raises LogError on a failed proof, NotFound or Unreadable otherwise."""
    latest = mirror.get("sth/latest.json")
    T.verify_head(latest, pem)
    leaf_hex, how = resolve(target, mirror)
    proof, head = _included(mirror, leaf_hex, latest, pem)
    out = {"leaf_hash": leaf_hex, "found_by": how, "leaf_index": proof["leaf_index"], "tree_size": head["tree_size"],
           "head_timestamp": head["timestamp"], "key_id": head["key_id"], "entry": None, "struck": None}
    try:
        body = mirror.raw(f"entries/{leaf_hex}.json")
        if T.leaf_hash(body).hex() != leaf_hex:
            raise T.LogError("the mirror's entry body does not hash to its leaf")
        out["entry"] = json.loads(body)
    except NotFound:
        out["entry"] = None   # withheld: the proof stands without it
    struck = mirror.get("struck.json").get(leaf_hex)
    if struck:
        _included(mirror, struck["leaf_hash"], latest, pem)
        out["struck"] = struck
    if head is not latest:
        _consistent(mirror, head, latest, pem)
    if state and state.get("tree_size"):
        seen = int(state["tree_size"])
        if seen > latest["tree_size"]:
            raise T.LogError(f"the mirror's newest head ({latest['tree_size']}) is older than one already seen ({seen})")
        earlier = head_at(mirror, seen, pem)
        if earlier["root_sha256"] != state["root_sha256"]:
            raise T.LogError(f"the head of size {seen} now has another root than when it was seen: the log was rewritten")
        _consistent(mirror, earlier, latest, pem)
        out["consistent_since"] = seen
    out["latest"] = {"tree_size": latest["tree_size"], "root_sha256": latest["root_sha256"], "timestamp": latest["timestamp"]}
    return out


def _included(mirror: Mirror, leaf_hex: str, latest: dict, pem: str) -> tuple:
    """The leaf's proof and the signed head it proves against, after the proof is checked."""
    proof = mirror.get(f"proofs/{leaf_hex}.json")
    head = latest if proof["tree_size"] == latest["tree_size"] else head_at(mirror, proof["tree_size"], pem)
    T.verify_inclusion(bytes.fromhex(leaf_hex), int(proof["leaf_index"]), head["tree_size"], T.unhex(proof["audit_path"]),
                       bytes.fromhex(head["root_sha256"]))
    return proof, head


def _consistent(mirror: Mirror, first: dict, second: dict, pem: str) -> None:
    if first["tree_size"] == second["tree_size"]:
        T.verify_consistency(first["tree_size"], second["tree_size"], bytes.fromhex(first["root_sha256"]),
                             bytes.fromhex(second["root_sha256"]), [])
        return
    proof = mirror.get(f"consistency/{first['tree_size']}.json")
    if proof["second"] != second["tree_size"]:
        raise Unreadable("the mirror changed while it was read; run again")
    T.verify_consistency(first["tree_size"], second["tree_size"], bytes.fromhex(first["root_sha256"]),
                         bytes.fromhex(second["root_sha256"]), T.unhex(proof["proof"]))


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
            what = f"{entry.get('kind', 'withheld')} entry" + (f" for No. {entry['serial']}" if entry.get("serial") else "")
            print(f"PROVED: the {what} is leaf {out['leaf_index']} of the log at size {out['tree_size']}, "
                  f"head signed {out['head_timestamp']} by log key {out['key_id']} ({where}).")
            if out["struck"]:
                print(f"STRUCK: {out['struck'].get('reason', '')} (entry {out['struck']['leaf_hash']})")
            if out["entry"] is None:
                print("The body is withheld; the proof stands without it.")
            if "consistent_since" in out:
                print(f"The newest head ({out['latest']['tree_size']}) extends the one seen before ({out['consistent_since']}).")
        return PROVED
    print(f"no mirror could be read: {last}", file=sys.stderr)
    return UNREADABLE


if __name__ == "__main__":
    sys.exit(main())
