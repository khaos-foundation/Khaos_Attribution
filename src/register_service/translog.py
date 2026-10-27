"""The register's side of the signed append-only log: leaves, perfect-subtree nodes, every signed head, withheld
bodies and anchor receipts in the register's own SQLite file, and the static export a verifier needs.
Every append happens inside the Store's write transaction, so a record and its log entry stand or fall together.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from khaos_attribution import translog as T

SCHEMA = """
CREATE TABLE IF NOT EXISTS log_leaves (idx INTEGER PRIMARY KEY, leaf_hash TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
  serial INTEGER, record_sha256 TEXT, refers TEXT);
CREATE INDEX IF NOT EXISTS log_leaves_record ON log_leaves (record_sha256);
CREATE INDEX IF NOT EXISTS log_leaves_refers ON log_leaves (refers);
CREATE TABLE IF NOT EXISTS log_bodies (leaf_hash TEXT PRIMARY KEY, body BLOB NOT NULL, withheld_at TEXT, withhold_reason TEXT);
CREATE TABLE IF NOT EXISTS log_nodes (level INTEGER NOT NULL, idx INTEGER NOT NULL, hash BLOB NOT NULL, PRIMARY KEY (level, idx));
CREATE TABLE IF NOT EXISTS log_heads (id INTEGER PRIMARY KEY, tree_size INTEGER NOT NULL, root_sha256 TEXT NOT NULL,
  timestamp TEXT NOT NULL, key_id TEXT NOT NULL, signature TEXT NOT NULL, UNIQUE (tree_size, key_id, timestamp));
CREATE INDEX IF NOT EXISTS log_heads_size ON log_heads (tree_size);
CREATE TABLE IF NOT EXISTS log_anchors (tree_size INTEGER PRIMARY KEY, root_sha256 TEXT NOT NULL, backend TEXT NOT NULL,
  stamped_at TEXT NOT NULL, receipt BLOB NOT NULL, state TEXT NOT NULL, completed_at TEXT);
"""
PAGE_MAX = 1000
TEXT_MAX = 400
NAMES_MAX = 12


class EntryRefused(ValueError):
    """An entry the log will not take, in words."""


class LogBlocked(RuntimeError):
    """The database is behind or apart from the log it already published: appending would fork it."""


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
        self.store, self.signer, self.blocked = store, None, None
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

    def tree(self) -> T.Tree:
        return T.Tree(self.size(), self._perfect)

    def _key_id(self) -> str:
        return self.signer.key_id if self.signer is not None else ""

    _PREFER = "ORDER BY key_id = ? DESC, timestamp DESC, id DESC"

    def head(self, size: int | None = None) -> dict | None:
        """The head of this size (the largest when None): the current key's newest where there are several."""
        if size is None:
            row = self.db.execute(f"SELECT * FROM log_heads WHERE tree_size = (SELECT MAX(tree_size) FROM log_heads) {self._PREFER}"
                                  " LIMIT 1", (self._key_id(),)).fetchone()
        else:
            row = self.db.execute(f"SELECT * FROM log_heads WHERE tree_size = ? {self._PREFER} LIMIT 1",
                                  (int(size), self._key_id())).fetchone()
        return head_doc(row) if row else None

    def heads(self) -> list:
        """One head per size, chosen as head() chooses."""
        out = {}
        for r in self.db.execute(f"SELECT * FROM log_heads {self._PREFER.replace('ORDER BY', 'ORDER BY tree_size,')}",
                                 (self._key_id(),)):
            out.setdefault(r["tree_size"], head_doc(r))
        return [out[k] for k in sorted(out)]

    def all_heads(self) -> list:
        return [head_doc(r) for r in self.db.execute("SELECT * FROM log_heads ORDER BY tree_size, timestamp, id")]

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
        """Every struck leaf: the revoke that strikes it, its reason and sitting (None when the revoke is withheld)."""
        out = {}
        for r in self.db.execute("SELECT leaf_hash, refers FROM log_leaves WHERE kind = 'revoke' ORDER BY idx"):
            body = self.body(r["leaf_hash"])
            entry = json.loads(body) if body else {}
            out.setdefault(r["refers"], {"leaf_hash": r["leaf_hash"], "reason": entry.get("reason"), "sitting": entry.get("sitting"),
                                         "time": entry.get("time"), "reason_withheld": body is None})
        return out

    def standing_admits(self, record_sha256: str) -> list:
        """The record's admit leaves no revoke has struck."""
        return [r["leaf_hash"] for r in self.db.execute(
            "SELECT a.leaf_hash FROM log_leaves a WHERE a.kind = 'admit' AND a.record_sha256 = ? AND NOT EXISTS"
            " (SELECT 1 FROM log_leaves r WHERE r.kind = 'revoke' AND r.refers = a.leaf_hash) ORDER BY a.idx", (record_sha256,))]

    def admit_leaf_of(self, record_sha256: str) -> str | None:
        row = self.db.execute("SELECT leaf_hash FROM log_leaves WHERE kind = 'admit' AND record_sha256 = ? ORDER BY idx LIMIT 1",
                              (record_sha256,)).fetchone()
        return row["leaf_hash"] if row else None

    def max_serial(self) -> int:
        row = self.db.execute("SELECT MAX(serial) AS m FROM log_leaves").fetchone()
        return int(row["m"] or 0)

    # ---- appending (inside the Store's transaction) ----

    def _insert_leaf(self, body: bytes, entry: dict) -> tuple:
        leaf = T.leaf_hash(body)
        index = self.size()
        self.db.execute("INSERT INTO log_leaves (idx, leaf_hash, kind, serial, record_sha256, refers) VALUES (?, ?, ?, ?, ?, ?)",
                        (index, leaf.hex(), entry["kind"], entry.get("serial"), entry.get("record_sha256"),
                         entry.get("withholds") or entry.get("revokes")))
        self.db.execute("INSERT INTO log_bodies (leaf_hash, body) VALUES (?, ?)", (leaf.hex(), body))
        level, i, h = 0, index, leaf
        self.db.execute("INSERT INTO log_nodes (level, idx, hash) VALUES (0, ?, ?)", (index, leaf))
        while i % 2 == 1:
            h = T.node_hash(self._perfect(level, i - 1), h)
            level, i = level + 1, i // 2
            self.db.execute("INSERT INTO log_nodes (level, idx, hash) VALUES (?, ?, ?)", (level, i, h))
        return index, leaf.hex()

    def append_locked(self, entry: dict) -> dict:
        """One leaf, its completed subtrees and a new signed head. The caller holds the Store's write transaction."""
        if self.blocked:
            raise LogBlocked(self.blocked)
        if self.signer is None:
            raise RuntimeError("the log key is not loaded; the log takes no entry")
        index, leaf_hex = self._insert_leaf(T.jcs(entry), entry)
        self._sign_head_locked()
        return {"leaf_hash": leaf_hex, "index": index, "tree_size": index + 1}

    def replay_locked(self, body: bytes) -> str:
        """A published entry put back verbatim, unsigned: the caller checks the result reproduces the published root."""
        _, leaf_hex = self._insert_leaf(body, json.loads(body))
        return leaf_hex

    def insert_head_locked(self, head: dict) -> None:
        self.db.execute("INSERT OR IGNORE INTO log_heads (tree_size, root_sha256, timestamp, key_id, signature) VALUES (?, ?, ?, ?, ?)",
                        (head["tree_size"], head["root_sha256"], head["timestamp"], head["key_id"], head["signature"]))

    def _sign_head_locked(self) -> dict:
        head = T.sign_head(self.size(), self.tree().root(), self.signer)
        self.insert_head_locked(head)
        return head

    def refresh_locked(self) -> dict | None:
        """The current tree signed again with the time now, so a quiet log still shows it is alive."""
        if self.blocked:
            raise LogBlocked(self.blocked)
        return self._sign_head_locked() if self.size() else None

    def resign_locked(self) -> dict | None:
        """After a key change, the current tree signed afresh by the new key: it commits to every entry before it."""
        latest = self.head()
        if latest is None or latest["key_id"] == self.signer.key_id:
            return None
        return self._sign_head_locked()

    def admit_locked(self, serial: int, record_sha256: str, counter: dict, time: str | None = None, wardens=None,
                     sitting=None, appended_by: str | None = None) -> dict:
        ref = {k: counter[k] for k in ("register_key_id", "register_signature", "countersigned_at_utc")}
        return self.append_locked(T.make_entry("admit", time=time, serial=int(serial), record_sha256=record_sha256,
                                               countersignature=ref, wardens=_names(wardens), sitting=_text(sitting, "sitting", False),
                                               appended_by=appended_by))

    def revoke_locked(self, leaf_hex: str, reason: str, time: str | None = None, wardens=None, sitting=None,
                      appended_by: str | None = None) -> dict:
        """A strike is an append naming the admitted leaf; the leaf itself is never changed."""
        leaf = self.leaf(leaf_hex)
        if leaf is None or leaf["kind"] != "admit":
            raise EntryRefused("a revoke names an admitted entry's leaf hash")
        if self.db.execute("SELECT 1 FROM log_leaves WHERE kind = 'revoke' AND refers = ?", (leaf_hex,)).fetchone():
            raise EntryRefused("that entry is already struck")
        return self.append_locked(T.make_entry("revoke", time=time, revokes=leaf_hex, serial=leaf["serial"],
                                               record_sha256=leaf["record_sha256"], reason=_text(reason, "reason", True),
                                               wardens=_names(wardens), sitting=_text(sitting, "sitting", False),
                                               appended_by=appended_by))

    def withhold_locked(self, leaf_hex: str, reason: str, wardens=None, sitting=None, appended_by: str | None = None) -> dict:
        """The body leaves public view; the leaf, every proof and every head stay as they were. Withholding a revoke
        keeps naming what it struck, so the strike itself never disappears."""
        leaf = self.leaf(leaf_hex)
        if leaf is None or leaf["kind"] not in ("admit", "revoke"):
            raise EntryRefused("a withhold names an admit or revoke entry's leaf hash")
        reason = _text(reason, "reason", True)
        row = self.db.execute("SELECT withheld_at FROM log_bodies WHERE leaf_hash = ?", (leaf_hex,)).fetchone()
        if row["withheld_at"]:
            raise EntryRefused("that body is already withheld")
        out = self.append_locked(T.make_entry("withhold", withholds=leaf_hex, reason=reason, wardens=_names(wardens),
                                              sitting=_text(sitting, "sitting", False), appended_by=appended_by,
                                              revokes=leaf["refers"] if leaf["kind"] == "revoke" else None))
        self.db.execute("UPDATE log_bodies SET withheld_at = ?, withhold_reason = ? WHERE leaf_hash = ?",
                        (T.now_rfc3339(), reason, leaf_hex))
        return out

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
                self.revoke_locked(out["leaf_hash"], (r["withdraw_reason"] or "").strip() or "withdrawn", time=r["withdrawn_at"])
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

    # ---- the published log as a witness ----

    def witness(self, places: list, public_pem: str) -> list:
        """What each published copy of the log says about this database: ok, behind, forked or unreadable."""
        from khaos_attribution.translog import verify as V  # noqa: PLC0415
        out = []
        for where in places:
            mirror = V.Mirror(str(where), timeout=5.0)
            try:
                head = mirror.get("sth/latest.json")
            except (V.NotFound, V.Unreadable) as e:
                out.append({"where": str(where), "state": "unreadable", "message": str(e)})
                continue
            try:
                T.verify_head(head, public_pem)
            except T.LogError as e:
                out.append({"where": str(where), "state": "other-key", "message": str(e)})
                continue
            size, n = int(head["tree_size"]), self.size()
            if size > n:
                state = "behind"
                message = (f"this database is behind the log published at {where} ({n} entries against {size}): it was restored or lost "
                           "writes. Nothing is appended until `register-cli log-recover` puts the published entries back")
            elif size and self.tree().root(size).hex() != head["root_sha256"]:
                state = "forked"
                message = (f"the log published at {where} has another root at size {size} than this database: an incident; "
                           "nothing is appended")
            else:
                state, message = "ok", f"{where}: size {size}, consistent"
            out.append({"where": str(where), "state": state, "message": message, "head": head})
        return out

    # ---- anchors ----

    def anchors(self) -> list:
        return [dict(r) for r in self.db.execute("SELECT * FROM log_anchors ORDER BY tree_size")]


def recover(store, places: list, public_pem: str) -> tuple:
    """Puts back the entries a published copy holds and this database lost, verbatim, when they reproduce that copy's
    signed root; (recovered, lines to print). Anything else is an incident for people, not code."""
    from khaos_attribution.translog import verify as V  # noqa: PLC0415
    log = store.log
    seen = log.witness(places, public_pem)
    forked = [w for w in seen if w["state"] == "forked"]
    if forked:
        return False, [f"INCIDENT: {w['message']}" for w in forked]
    behind = sorted((w for w in seen if w["state"] == "behind"), key=lambda w: -w["head"]["tree_size"])
    if not behind:
        return True, ["the database is not behind any published copy of the log; nothing to recover"]
    head, where = behind[0]["head"], behind[0]["where"]
    size, n = int(head["tree_size"]), log.size()
    try:
        leaves = V.Mirror(where).get("hashes.json")["leaves"][:size]
    except (V.NotFound, V.Unreadable, KeyError, TypeError) as e:
        return False, [f"INCIDENT: the leaf hashes at {where} cannot be read: {e}"]
    if len(leaves) < size or T.Tree.from_leaves([bytes.fromhex(x) for x in leaves]).root(size).hex() != head["root_sha256"]:
        return False, [f"INCIDENT: the leaf hashes at {where} do not make its own signed root"]
    mine = [r["leaf_hash"] for r in log.db.execute("SELECT leaf_hash FROM log_leaves ORDER BY idx")]
    if mine != leaves[:n]:
        return False, [f"INCIDENT: this database's {n} entries are not the first {n} published at {where}"]
    found = {}
    for place in places:
        try:
            bundle = V.Mirror(str(place), timeout=5.0).get("bodies.json")["bodies"]
        except (V.NotFound, V.Unreadable, KeyError, TypeError):
            continue
        for i in range(n, min(size, len(bundle))):
            text = bundle[i]
            if i not in found and isinstance(text, str) and T.leaf_hash(text.encode("utf-8")).hex() == leaves[i]:
                found[i] = text.encode("utf-8")
    missing = [i for i in range(n, size) if i not in found]
    if missing:
        return False, [f"INCIDENT: entries {missing[:10]} have no published body (withheld, or not yet public); the log "
                       f"cannot be rebuilt to its signed root of size {size}. Keep the register blocked and call the wardens."]
    lines = []
    try:
        with store._tx():
            for i in range(n, size):
                log.replay_locked(found[i])
                entry = json.loads(found[i])
                if entry["kind"] == "withhold":
                    log.db.execute("UPDATE log_bodies SET withheld_at = ?, withhold_reason = ? WHERE leaf_hash = ?",
                                   (entry["time"], entry.get("reason"), entry["withholds"]))
                if entry["kind"] == "admit" and not log.db.execute("SELECT 1 FROM records WHERE record_sha256 = ?",
                                                                    (entry["record_sha256"],)).fetchone():
                    lines.append(f"record No. {entry['serial']} ({entry['record_sha256']}) is in the log but not in the "
                                 "database: its Workshop publishes it again from its outbox's sent copy")
            if log.tree().root(size).hex() != head["root_sha256"]:
                raise T.LogError("the replayed entries do not reproduce the signed root")
            for k in range(n + 1, size):
                try:
                    old = V.Mirror(where).get(f"sth/{k}.json")
                    T.verify_head(old, public_pem)
                except (V.NotFound, V.Unreadable, T.LogError):
                    continue
                if old["tree_size"] == k and log.tree().root(k).hex() == old["root_sha256"]:
                    log.insert_head_locked(old)
            log.insert_head_locked(head)
    except T.LogError as e:
        return False, [f"INCIDENT: {e}"]
    log.blocked = None
    return True, [f"recovered entries {n} to {size - 1} from {where}; the log is at size {size} with its published root", *lines]


def head_doc(row) -> dict:
    return {**T.head_payload(row["tree_size"], row["root_sha256"], row["timestamp"]), "key_id": row["key_id"],
            "signature": row["signature"]}


# ---- the static export ----

_EXPORT_LOCK = threading.Lock()


def _write(path: Path, data: bytes) -> None:
    """A temp file of this writer's own, then replace: a reader never meets a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.new")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _json(doc) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8")


@contextmanager
def _export_lock(mirror: Path):
    """One export at a time: across threads by the lock, across processes by a lock file."""
    mirror.mkdir(parents=True, exist_ok=True)
    with _EXPORT_LOCK, open(mirror / ".log-export.lock", "w") as fh:
        try:
            import fcntl  # noqa: PLC0415
            fcntl.flock(fh, fcntl.LOCK_EX)
        except ImportError:
            pass
        yield


def snapshot(log: TransLog) -> dict:
    """Everything the export writes, read in one transaction so the files agree with each other."""
    db = log.db
    db.execute("BEGIN")
    try:
        rows = db.execute("SELECT l.idx, l.leaf_hash, l.kind, l.refers, b.body, b.withheld_at FROM log_leaves l"
                          " JOIN log_bodies b ON b.leaf_hash = l.leaf_hash ORDER BY l.idx").fetchall()
        nodes = {(r["level"], r["idx"]): bytes(r["hash"]) for r in db.execute("SELECT level, idx, hash FROM log_nodes")}
        heads = log.heads()
        first_current = db.execute("SELECT MIN(tree_size) AS m FROM log_heads WHERE key_id = ?", (log._key_id(),)).fetchone()["m"]
        struck = log.struck()
        anchors = log.anchors()
    finally:
        db.execute("COMMIT")
    leaves = [r["leaf_hash"] for r in rows]
    return {"leaves": leaves, "kinds": [r["kind"] for r in rows],
            "bodies": [None if r["withheld_at"] else bytes(r["body"]) for r in rows],
            "withheld": sorted(r["leaf_hash"] for r in rows if r["withheld_at"]),
            "tree": T.Tree(len(leaves), lambda level, index: nodes[(level, index)]),
            "heads": heads, "first_current": first_current, "struck": struck, "anchors": anchors}


def pending_from(snap: dict, delay_days: float, now: datetime | None = None) -> int | None:
    """The first leaf whose body is still inside the publication delay; only leaves signed under the current key wait."""
    if delay_days <= 0 or not snap["first_current"]:
        return None
    cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=delay_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    times = {h["tree_size"]: h["timestamp"] for h in snap["heads"]}
    for i in range(snap["first_current"], len(snap["leaves"])):
        if times.get(i + 1, "") >= cutoff:
            return i
    return None


def export(log: TransLog, mirror: Path, key_doc: dict, public: Path | None = None, delay_days: float = 7) -> None:
    """Everything a verifier needs, under mirror/log/, and under public/log/ the same with new bodies held back for
    `delay_days`; a database the published log is ahead of exports nothing."""
    if log.blocked:
        return
    with _export_lock(Path(mirror)):
        snap = snapshot(log)
        _write_log(Path(mirror) / "log", snap, key_doc, None)
        if public is not None:
            _write_log(Path(public) / "log", snap, key_doc, pending_from(snap, delay_days))


def _write_log(root: Path, snap: dict, key_doc: dict, pending: int | None) -> None:
    tree, leaves, n = snap["tree"], snap["leaves"], len(snap["leaves"])
    on_disk = root / "sth" / "latest.json"
    if on_disk.is_file():
        try:
            if json.loads(on_disk.read_text(encoding="utf-8"))["tree_size"] > n:
                return
        except (OSError, ValueError, KeyError):
            pass
    # A withhold names only a leaf and a reason, and a verifier needs it to accept a missing body: it is never held back.
    shown = [None if pending is not None and i >= pending and snap["kinds"][i] != "withhold" else b
             for i, b in enumerate(snap["bodies"])]
    _write(root / "key.json", _json(key_doc))
    for i, leaf_hex in enumerate(leaves):
        path = root / "entries" / f"{leaf_hex}.json"
        if shown[i] is None:
            path.unlink(missing_ok=True)
        elif not path.is_file():
            _write(path, shown[i])
        _write(root / "proofs" / f"{leaf_hex}.json",
               _json({"leaf_hash": leaf_hex, "leaf_index": i, "tree_size": n, "audit_path": T.hexes(tree.inclusion(i))}))
    hidden = {leaves[i] for i in range(n) if shown[i] is None and snap["bodies"][i] is not None}
    struck = {k: v for k, v in snap["struck"].items() if v["leaf_hash"] not in hidden}
    (root / "records.json").unlink(missing_ok=True)
    _write(root / "struck.json", _json(struck))
    _write(root / "withheld.json", _json(snap["withheld"]))
    _write(root / "hashes.json", _json({"tree_size": n, "leaves": leaves}))
    _write(root / "bodies.json", _json({"tree_size": n, "pending_from": pending,
                                        "bodies": [b.decode("utf-8") if b is not None else None for b in shown]}))
    latest = None
    for head in snap["heads"]:
        path = root / "sth" / f"{head['tree_size']}.json"
        if not path.is_file() or path.read_bytes() != _json(head):
            _write(path, _json(head))
        if head["tree_size"] < n:
            _write(root / "consistency" / f"{head['tree_size']}.json",
                   _json({"first": head["tree_size"], "second": n, "proof": T.hexes(tree.consistency(head["tree_size"], n))}))
        latest = head
    _write_anchors(root, snap["anchors"])
    if latest is not None:
        _write(on_disk, _json(latest))


def _write_anchors(root: Path, anchors: list) -> None:
    rows = []
    for a in anchors:
        name = f"{a['tree_size']}.ots"
        _write(root / "anchors" / name, bytes(a["receipt"]))
        rows.append({"tree_size": a["tree_size"], "root_sha256": a["root_sha256"], "backend": a["backend"], "state": a["state"],
                     "stamped_at": a["stamped_at"], "completed_at": a["completed_at"], "receipt": f"anchors/{name}"})
    _write(root / "anchors.json", _json(rows))


def export_anchors(log: TransLog, mirror: Path) -> None:
    """The anchor receipts, as OpenTimestamps files, and their list."""
    with _export_lock(Path(mirror)):
        _write_anchors(Path(mirror) / "log", log.anchors())


def receipt_b64(receipt: bytes) -> str:
    return base64.b64encode(bytes(receipt)).decode("ascii")
