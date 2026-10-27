"""The register's side of the signed append-only log: leaves, perfect-subtree nodes, every signed head, withheld
bodies and anchor receipts in the register's own SQLite file, and the static export a verifier needs.
Every append happens inside the Store's write transaction, so a record and its log entry stand or fall together.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from khaos_attribution import translog as T

SCHEMA = """
CREATE TABLE IF NOT EXISTS log_leaves (idx INTEGER PRIMARY KEY, leaf_hash TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
  serial INTEGER, record_sha256 TEXT, refers TEXT);
CREATE INDEX IF NOT EXISTS log_leaves_record ON log_leaves (record_sha256);
CREATE INDEX IF NOT EXISTS log_leaves_refers ON log_leaves (refers);
CREATE TABLE IF NOT EXISTS log_bodies (leaf_hash TEXT PRIMARY KEY, body BLOB NOT NULL, withheld_at TEXT, withhold_reason TEXT);
CREATE TABLE IF NOT EXISTS log_nodes (level INTEGER NOT NULL, idx INTEGER NOT NULL, hash BLOB NOT NULL, PRIMARY KEY (level, idx));
CREATE TABLE IF NOT EXISTS log_heads (tree_size INTEGER NOT NULL, root_sha256 TEXT NOT NULL, timestamp TEXT NOT NULL,
  key_id TEXT NOT NULL, signature TEXT NOT NULL, PRIMARY KEY (tree_size, key_id));
CREATE TABLE IF NOT EXISTS log_anchors (tree_size INTEGER PRIMARY KEY, root_sha256 TEXT NOT NULL, backend TEXT NOT NULL,
  stamped_at TEXT NOT NULL, receipt BLOB NOT NULL, state TEXT NOT NULL, completed_at TEXT);
"""
PAGE_MAX = 1000
TEXT_MAX = 400
NAMES_MAX = 12


class EntryRefused(ValueError):
    """An entry the log will not take, in words."""


def _names(value) -> list | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value or len(value) > NAMES_MAX or not all(isinstance(n, str) and n.strip() for n in value):
        raise EntryRefused("wardens is a list of names")
    return [n.strip()[:80] for n in value]


def _text(value, name: str, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EntryRefused(f"{name} is required")
    return value.strip()[:TEXT_MAX]


class TransLog:
    """The log over one Store's connection and lock; `signer` is the log key, set by the service before any append."""

    def __init__(self, store):
        self.store, self.signer = store, None
        store.db.executescript(SCHEMA)

    @property
    def db(self):
        return self.store.db

    # ---- reading ----

    def size(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) AS n FROM log_leaves").fetchone()["n"])

    def _perfect(self, level: int, index: int) -> bytes:
        row = self.db.execute("SELECT hash FROM log_nodes WHERE level = ? AND idx = ?", (level, index)).fetchone()
        if row is None:
            raise T.LogError(f"node ({level}, {index}) is missing")
        return bytes(row["hash"])

    def tree(self, whole: bool = False) -> T.Tree:
        """The tree at its current size; `whole` reads every node at once, for the export."""
        if whole:
            nodes = {(r["level"], r["idx"]): bytes(r["hash"]) for r in self.db.execute("SELECT level, idx, hash FROM log_nodes")}
            return T.Tree(self.size(), lambda level, index: nodes[(level, index)])
        return T.Tree(self.size(), self._perfect)

    def _key_id(self) -> str:
        return self.signer.key_id if self.signer is not None else ""

    def head(self, size: int | None = None) -> dict | None:
        """The head of this size (the newest when None), the one signed by the current log key where there are two."""
        if size is None:
            row = self.db.execute("SELECT * FROM log_heads ORDER BY tree_size DESC, key_id = ? DESC LIMIT 1",
                                  (self._key_id(),)).fetchone()
        else:
            row = self.db.execute("SELECT * FROM log_heads WHERE tree_size = ? ORDER BY key_id = ? DESC LIMIT 1",
                                  (int(size), self._key_id())).fetchone()
        return self._head_doc(row) if row else None

    @staticmethod
    def _head_doc(row) -> dict:
        return {**T.head_payload(row["tree_size"], row["root_sha256"], row["timestamp"]), "key_id": row["key_id"],
                "signature": row["signature"]}

    def heads(self) -> list:
        """One head per size, preferring the current log key's."""
        out = {}
        for r in self.db.execute("SELECT * FROM log_heads ORDER BY tree_size, key_id = ?", (self._key_id(),)):
            out[r["tree_size"]] = self._head_doc(r)
        return list(out.values())

    def all_heads(self) -> list:
        return [self._head_doc(r) for r in self.db.execute("SELECT * FROM log_heads ORDER BY tree_size, timestamp")]

    def leaf(self, leaf_hex: str) -> dict | None:
        row = self.db.execute("SELECT * FROM log_leaves WHERE leaf_hash = ?", (leaf_hex,)).fetchone()
        return dict(row) if row else None

    def body(self, leaf_hex: str) -> bytes | None:
        """The entry's exact bytes, or None when withheld."""
        row = self.db.execute("SELECT body, withheld_at FROM log_bodies WHERE leaf_hash = ?", (leaf_hex,)).fetchone()
        return None if row is None or row["withheld_at"] else bytes(row["body"])

    def withheld(self) -> dict:
        return {r["leaf_hash"]: r["withhold_reason"] for r in self.db.execute(
            "SELECT leaf_hash, withhold_reason FROM log_bodies WHERE withheld_at IS NOT NULL")}

    def inclusion(self, leaf_hex: str, size: int | None = None) -> dict:
        leaf = self.leaf(leaf_hex)
        if leaf is None:
            raise KeyError("no such leaf")
        tree = self.tree()
        size = tree.size if size is None else int(size)
        if leaf["idx"] >= size:
            raise KeyError(f"the leaf was appended after the head of size {size}")
        return {"leaf_hash": leaf_hex, "leaf_index": leaf["idx"], "tree_size": size,
                "audit_path": T.hexes(tree.inclusion(leaf["idx"], size))}

    def consistency(self, first: int, second: int) -> dict:
        return {"first": int(first), "second": int(second), "proof": T.hexes(self.tree().consistency(int(first), int(second)))}

    def entries(self, start: int, end: int) -> list:
        """Leaves [start, end): each hash, and its body where not withheld."""
        end = min(end, start + PAGE_MAX)
        out = []
        for r in self.db.execute("SELECT l.idx, l.leaf_hash, b.body, b.withheld_at FROM log_leaves l JOIN log_bodies b"
                                 " ON b.leaf_hash = l.leaf_hash WHERE l.idx >= ? AND l.idx < ? ORDER BY l.idx", (start, end)):
            withheld = r["withheld_at"] is not None
            out.append({"index": r["idx"], "leaf_hash": r["leaf_hash"], "withheld": withheld,
                        "entry": None if withheld else json.loads(bytes(r["body"]))})
        return out

    def of_record(self, record_sha256: str) -> list:
        """A record's admit entries and every revoke of them, oldest first, as the model page shows them."""
        out = []
        for r in self.db.execute("SELECT * FROM log_leaves WHERE record_sha256 = ? ORDER BY idx", (record_sha256,)):
            body = self.body(r["leaf_hash"])
            out.append({**dict(r), "entry": json.loads(body) if body else None})
        return out

    def struck(self) -> dict:
        """Every struck leaf: the revoke that strikes it, its reason and sitting."""
        out = {}
        for r in self.db.execute("SELECT leaf_hash, refers FROM log_leaves WHERE kind = 'revoke' ORDER BY idx"):
            body = self.body(r["leaf_hash"])
            entry = json.loads(body) if body else {}
            out.setdefault(r["refers"], {"leaf_hash": r["leaf_hash"], "reason": entry.get("reason"), "sitting": entry.get("sitting"),
                                         "time": entry.get("time")})
        return out

    # ---- appending (inside the Store's transaction) ----

    def _store_node(self, level: int, index: int, h: bytes) -> None:
        self.db.execute("INSERT INTO log_nodes (level, idx, hash) VALUES (?, ?, ?)", (level, index, h))

    def append_locked(self, entry: dict) -> dict:
        """One leaf, its completed subtrees and a new signed head. The caller holds the Store's write transaction."""
        if self.signer is None:
            raise RuntimeError("the log key is not loaded; the log takes no entry")
        body = T.jcs(entry)
        leaf = T.leaf_hash(body)
        index = self.size()
        self.db.execute("INSERT INTO log_leaves (idx, leaf_hash, kind, serial, record_sha256, refers) VALUES (?, ?, ?, ?, ?, ?)",
                        (index, leaf.hex(), entry["kind"], entry.get("serial"), entry.get("record_sha256"),
                         entry.get("revokes") or entry.get("withholds")))
        self.db.execute("INSERT INTO log_bodies (leaf_hash, body) VALUES (?, ?)", (leaf.hex(), body))
        level, i, h = 0, index, leaf
        self._store_node(0, index, leaf)
        while i % 2 == 1:
            h = T.node_hash(self._perfect(level, i - 1), h)
            level, i = level + 1, i // 2
            self._store_node(level, i, h)
        self._sign_head_locked()
        return {"leaf_hash": leaf.hex(), "index": index, "tree_size": index + 1}

    def _sign_head_locked(self) -> dict:
        head = T.sign_head(self.size(), self.tree().root(), self.signer)
        self.db.execute("INSERT INTO log_heads (tree_size, root_sha256, timestamp, key_id, signature) VALUES (?, ?, ?, ?, ?)",
                        (head["tree_size"], head["root_sha256"], head["timestamp"], head["key_id"], head["signature"]))
        return head

    def resign_locked(self) -> dict | None:
        """After a key change, the current tree signed afresh by the new key: it commits to every entry before it."""
        latest = self.head()
        if latest is None or latest["key_id"] == self.signer.key_id:
            return None
        return self._sign_head_locked()

    def admit_locked(self, serial: int, record_sha256: str, counter: dict, time: str | None = None, wardens=None,
                     sitting=None) -> dict:
        ref = {k: counter[k] for k in ("register_key_id", "register_signature", "countersigned_at_utc")}
        return self.append_locked(T.make_entry("admit", time=time, serial=int(serial), record_sha256=record_sha256,
                                               countersignature=ref, wardens=_names(wardens), sitting=_text(sitting, "sitting", False)))

    def revoke_locked(self, leaf_hex: str, reason: str, time: str | None = None, wardens=None, sitting=None) -> dict:
        """A strike is an append naming the admitted leaf; the leaf itself is never changed."""
        leaf = self.leaf(leaf_hex)
        if leaf is None or leaf["kind"] != "admit":
            raise EntryRefused("a revoke names an admitted entry's leaf hash")
        if self.db.execute("SELECT 1 FROM log_leaves WHERE kind = 'revoke' AND refers = ?", (leaf_hex,)).fetchone():
            raise EntryRefused("that entry is already struck")
        return self.append_locked(T.make_entry("revoke", time=time, revokes=leaf_hex, serial=leaf["serial"],
                                               record_sha256=leaf["record_sha256"], reason=_text(reason, "reason", True),
                                               wardens=_names(wardens), sitting=_text(sitting, "sitting", False)))

    def withhold_locked(self, leaf_hex: str, reason: str, wardens=None, sitting=None) -> dict:
        """The body leaves public view; the leaf, every proof and every head stay as they were."""
        leaf = self.leaf(leaf_hex)
        if leaf is None or leaf["kind"] not in ("admit", "revoke"):
            raise EntryRefused("a withhold names an admit or revoke entry's leaf hash")
        reason = _text(reason, "reason", True)
        row = self.db.execute("SELECT withheld_at FROM log_bodies WHERE leaf_hash = ?", (leaf_hex,)).fetchone()
        if row["withheld_at"]:
            raise EntryRefused("that body is already withheld")
        out = self.append_locked(T.make_entry("withhold", withholds=leaf_hex, reason=reason, wardens=_names(wardens),
                                              sitting=_text(sitting, "sitting", False)))
        self.db.execute("UPDATE log_bodies SET withheld_at = ?, withhold_reason = ? WHERE leaf_hash = ?",
                        (T.now_rfc3339(), reason, leaf_hex))
        return out

    def admit_leaf_of(self, record_sha256: str) -> str | None:
        row = self.db.execute("SELECT leaf_hash FROM log_leaves WHERE kind = 'admit' AND record_sha256 = ? ORDER BY idx LIMIT 1",
                              (record_sha256,)).fetchone()
        return row["leaf_hash"] if row else None

    def genesis_locked(self) -> int:
        """Once, on an empty log: the old ledger's tip, then every record in serial order with its own times."""
        if self.size():
            return 0
        tip = self.db.execute("SELECT seq, entry_sha256 FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        count = int(self.db.execute("SELECT COUNT(*) AS n FROM ledger").fetchone()["n"])
        self.append_locked(T.make_entry("genesis", ledger_tip_sha256=tip["entry_sha256"] if tip else "", ledger_length=count))
        n = 1
        for r in self.db.execute("SELECT * FROM records ORDER BY serial").fetchall():
            counter = json.loads(r["counter_json"])
            out = self.admit_locked(r["serial"], r["record_sha256"], counter, time=r["published_at"])
            n += 1
            if r["state"] == "withdrawn":
                self.revoke_locked(out["leaf_hash"], r["withdraw_reason"] or "withdrawn", time=r["withdrawn_at"])
                n += 1
        return n

    def problems(self, public_pem: str | None = None) -> list:
        """The whole log re-derived from its bodies: every leaf, node and head root, and each head's signature."""
        out, leaves = [], []
        for r in self.db.execute("SELECT l.idx, l.leaf_hash, b.body FROM log_leaves l LEFT JOIN log_bodies b"
                                 " ON b.leaf_hash = l.leaf_hash ORDER BY l.idx"):
            if r["idx"] != len(leaves):
                out.append(f"leaf {len(leaves)} is missing")
            if r["body"] is None or T.leaf_hash(bytes(r["body"])).hex() != r["leaf_hash"]:
                out.append(f"leaf {r['idx']}'s body does not hash to its leaf")
            leaves.append(bytes.fromhex(r["leaf_hash"]))
        nodes = T.perfect_nodes(leaves)
        stored = {(r["level"], r["idx"]): bytes(r["hash"]) for r in self.db.execute("SELECT level, idx, hash FROM log_nodes")}
        if stored != nodes:
            out.append("the stored nodes are not the nodes of the leaves")
        tree = T.Tree(len(leaves), lambda level, index: nodes[(level, index)])
        current = T.key_id_of_pem(public_pem) if public_pem else None
        for head in self.all_heads():
            if head["tree_size"] > len(leaves) or tree.root(head["tree_size"]).hex() != head["root_sha256"]:
                out.append(f"the head of size {head['tree_size']} is not the root of the leaves")
            if public_pem and head["key_id"] == current:
                try:
                    T.verify_head(head, public_pem)
                except T.LogError as e:
                    out.append(f"head {head['tree_size']}: {e}")
        return out

    # ---- anchors ----

    def anchors(self) -> list:
        return [dict(r) for r in self.db.execute("SELECT * FROM log_anchors ORDER BY tree_size")]


# ---- the static export ----

def _write(path: Path, data: bytes) -> None:
    """Temp file then replace: a reader never meets a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".new")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _json(doc) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8")


def export(log: TransLog, mirror: Path, key_doc: dict) -> None:
    """Everything a verifier needs, under mirror/log/: heads, hashes, bodies, every proof against the newest head,
    consistency from every head to it, the key and the anchor receipts."""
    root = Path(mirror) / "log"
    tree = log.tree(whole=True)
    n = tree.size
    leaves = [r["leaf_hash"] for r in log.db.execute("SELECT leaf_hash FROM log_leaves ORDER BY idx")]
    withheld = log.withheld()
    _write(root / "key.json", _json(key_doc))
    for i, leaf_hex in enumerate(leaves):
        path = root / "entries" / f"{leaf_hex}.json"
        body = log.body(leaf_hex)
        if body is None:
            path.unlink(missing_ok=True)
        elif not path.is_file():
            _write(path, body)
        _write(root / "proofs" / f"{leaf_hex}.json",
               _json({"leaf_hash": leaf_hex, "leaf_index": i, "tree_size": n, "audit_path": T.hexes(tree.inclusion(i))}))
    records: dict = {}
    for r in log.db.execute("SELECT leaf_hash, kind, record_sha256 FROM log_leaves WHERE record_sha256 IS NOT NULL ORDER BY idx"):
        if r["leaf_hash"] not in withheld:
            records.setdefault(r["record_sha256"], []).append({"leaf_hash": r["leaf_hash"], "kind": r["kind"]})
    struck = {k: v for k, v in log.struck().items() if k not in withheld}
    _write(root / "records.json", _json(records))
    _write(root / "struck.json", _json(struck))
    _write(root / "withheld.json", _json(sorted(withheld)))
    _write(root / "hashes.json", _json({"tree_size": n, "leaves": leaves}))
    latest = None
    for head in log.heads():
        path = root / "sth" / f"{head['tree_size']}.json"
        if not path.is_file() or path.read_bytes() != _json(head):
            _write(path, _json(head))
        if head["tree_size"] < n:
            _write(root / "consistency" / f"{head['tree_size']}.json",
                   _json({"first": head["tree_size"], "second": n, "proof": T.hexes(tree.consistency(head["tree_size"], n))}))
        latest = head
    export_anchors(log, mirror)
    if latest is not None:
        _write(root / "sth" / "latest.json", _json(latest))


def export_anchors(log: TransLog, mirror: Path) -> None:
    """The anchor receipts, as OpenTimestamps files, and their list."""
    root = Path(mirror) / "log"
    rows = []
    for a in log.anchors():
        name = f"{a['tree_size']}.ots"
        _write(root / "anchors" / name, bytes(a["receipt"]))
        rows.append({"tree_size": a["tree_size"], "root_sha256": a["root_sha256"], "backend": a["backend"], "state": a["state"],
                     "stamped_at": a["stamped_at"], "completed_at": a["completed_at"], "receipt": f"anchors/{name}"})
    _write(root / "anchors.json", _json(rows))


def receipt_b64(receipt: bytes) -> str:
    return base64.b64encode(bytes(receipt)).decode("ascii")
