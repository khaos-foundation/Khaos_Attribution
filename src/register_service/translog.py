"""The register's side of the signed append-only log: leaves, perfect-subtree nodes, every signed head, withheld
bodies and anchor receipts in the register's own SQLite file, and the static export a verifier needs.
Every append happens inside the Store's write transaction, so a record and its log entry stand or fall together.
"""

from __future__ import annotations

import base64
import bisect
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
  serial INTEGER, record_sha256 TEXT, refers TEXT, waits INTEGER);
CREATE INDEX IF NOT EXISTS log_leaves_record ON log_leaves (record_sha256);
CREATE INDEX IF NOT EXISTS log_leaves_refers ON log_leaves (refers);
CREATE TABLE IF NOT EXISTS log_bodies (leaf_hash TEXT PRIMARY KEY, body BLOB NOT NULL, withheld_at TEXT, withhold_reason TEXT);
CREATE TABLE IF NOT EXISTS log_nodes (level INTEGER NOT NULL, idx INTEGER NOT NULL, hash BLOB NOT NULL, PRIMARY KEY (level, idx));
CREATE TABLE IF NOT EXISTS log_heads (id INTEGER PRIMARY KEY, tree_size INTEGER UNIQUE NOT NULL, root_sha256 TEXT NOT NULL,
  timestamp TEXT NOT NULL, key_id TEXT NOT NULL, signature TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS log_reserved (serial INTEGER PRIMARY KEY, reason TEXT NOT NULL, reserved_at TEXT NOT NULL,
  returned_at TEXT, returned_to TEXT);
CREATE TABLE IF NOT EXISTS log_keys (key_id TEXT PRIMARY KEY, public_pem TEXT NOT NULL, first_seen TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS log_anchors (tree_size INTEGER PRIMARY KEY, root_sha256 TEXT NOT NULL, backend TEXT NOT NULL,
  stamped_at TEXT NOT NULL, receipt BLOB NOT NULL, state TEXT NOT NULL, completed_at TEXT);
CREATE TABLE IF NOT EXISTS log_exports (place TEXT PRIMARY KEY, format INTEGER NOT NULL, key_id TEXT NOT NULL,
  tree_size INTEGER NOT NULL, pending_from INTEGER);
"""
PAGE_MAX = 1000
# The layout of the static export: a proof names the first head under the current key that holds its leaf and a
# consistency file runs from one head to the next, so neither changes as the log grows.
EXPORT_FORMAT = 2
EXPORT_LOCK_TIMEOUT = 30.0   # seconds; a later export catches up with whatever this one skipped
UNAVAILABLE = "unavailable: recovered without its body"
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
        self._mark_waiting_admits()

    def _mark_waiting_admits(self) -> None:
        """``waits`` on an admit: 1 when its body names lost leaves it waits on (pending_lost), else 0. A column so the
        unresolved models are found without reading every admit; a file from before it is filled here, once."""
        columns = {r["name"] for r in self.db.execute("PRAGMA table_info(log_leaves)")}
        if "waits" not in columns:
            self.db.execute("ALTER TABLE log_leaves ADD COLUMN waits INTEGER")
        self.db.execute("CREATE INDEX IF NOT EXISTS log_leaves_kind_waits ON log_leaves (kind, waits)")
        if not self.db.execute("SELECT 1 FROM log_leaves WHERE kind = 'admit' AND waits IS NULL LIMIT 1").fetchone():
            return
        if self._lost():
            rows = self.db.execute("SELECT l.leaf_hash, b.body FROM log_leaves l LEFT JOIN log_bodies b ON b.leaf_hash = l.leaf_hash"
                                   " WHERE l.kind = 'admit' AND l.waits IS NULL").fetchall()
            for r in rows:
                names = (json.loads(bytes(r["body"])) if r["body"] else {}).get("pending_lost")
                self.db.execute("UPDATE log_leaves SET waits = ? WHERE leaf_hash = ?", (1 if names else 0, r["leaf_hash"]))
        else:
            # An admit names only leaves this log holds as lost: with none lost, none waits.
            self.db.execute("UPDATE log_leaves SET waits = 0 WHERE kind = 'admit' AND waits IS NULL")

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

    def _lost(self) -> dict:
        """The leaves a recovery put back without their bodies, idx -> hash."""
        return {r["idx"]: r["leaf_hash"] for r in self.db.execute("SELECT idx, leaf_hash FROM log_leaves WHERE kind = 'unavailable'")}

    def _lost_and_answers(self) -> tuple:
        """(lost leaves as idx -> hash, confirms as (idx, answers, confirms-or-None))."""
        lost = self._lost()
        answers = []
        for r in self.db.execute("SELECT idx, leaf_hash FROM log_leaves WHERE kind = 'confirm'").fetchall():
            row = self.db.execute("SELECT body FROM log_bodies WHERE leaf_hash = ?", (r["leaf_hash"],)).fetchone()
            body = json.loads(bytes(row["body"])) if row and row["body"] else {}
            answers.append((r["idx"], body.get("answers"), body.get("confirms")))
        return lost, answers

    def open_lost(self, admit_idx: int, admit_leaf: str, lost=None, answers=None) -> list:
        """The lost leaves that leave this admit unresolved: each one after it that no confirm has answered for it (one
        naming no admit answers for all), and each one its own `pending_lost` names that no confirm after it answers."""
        if lost is None:
            lost, answers = self._lost_and_answers()
        out = [h for i, h in sorted(lost.items()) if i > admit_idx and not any(
            c_idx > i and ans == h and conf in (None, admit_leaf) for c_idx, ans, conf in answers)]
        row = self.db.execute("SELECT body FROM log_bodies WHERE leaf_hash = ?", (admit_leaf,)).fetchone()
        body = json.loads(bytes(row["body"])) if row and row["body"] else {}
        for h in body.get("pending_lost") or []:
            if h not in out and not any(c_idx > admit_idx and ans == h and conf in (None, admit_leaf) for c_idx, ans, conf in answers):
                out.append(h)
        return out

    def admit_states(self, record_sha256: str) -> dict:
        """Each admit leaf of the record: struck (a revoke names it), unresolved (an entry after it was lost in a
        recovery and no warden has answered that loss), or standing."""
        lost, answers = self._lost_and_answers()
        out = {}
        for a in self.db.execute("SELECT idx, leaf_hash FROM log_leaves WHERE kind = 'admit' AND record_sha256 = ? ORDER BY idx",
                                 (record_sha256,)).fetchall():
            if self.db.execute("SELECT 1 FROM log_leaves WHERE kind = 'revoke' AND refers = ?", (a["leaf_hash"],)).fetchone():
                out[a["leaf_hash"]] = "struck"
                continue
            out[a["leaf_hash"]] = "unresolved" if self.open_lost(a["idx"], a["leaf_hash"], lost, answers) else "standing"
        return out

    def standing_admits(self, record_sha256: str) -> list:
        """The record's admit leaves neither struck nor left unresolved by a lost entry."""
        return [k for k, v in self.admit_states(record_sha256).items() if v == "standing"]

    def unresolved_records(self) -> list:
        """Records whose every live admit waits on a warden after a recovery lost an entry behind it: admit_states'
        rule, read at the cost of the losses. Only an admit before a loss still open, or one that names a loss
        (``waits``), can be unresolved; every other admit stands unless struck, and none is read."""
        lost = self._lost()
        if not lost:
            return []
        _, answers = self._lost_and_answers()
        # The newest confirm of each lost leaf: the one for every admit, and the one naming a single admit.
        for_all, for_one = {}, {}
        for c_idx, answered, confirms in answers:
            if confirms is None:
                for_all[answered] = max(for_all.get(answered, -1), c_idx)
            else:
                for_one[(answered, confirms)] = max(for_one.get((answered, confirms), -1), c_idx)
        open_at = sorted(i for i, h in lost.items() if for_all.get(h, -1) <= i)
        index_of = {h: i for i, h in lost.items()}
        answered_for: dict = {}
        for (h, admit_leaf), c_idx in for_one.items():
            i = index_of.get(h)
            if i is not None and for_all.get(h, -1) <= i < c_idx:
                answered_for.setdefault(admit_leaf, []).append(i)
        edge = open_at[-1] if open_at else -1
        live = " AND NOT EXISTS (SELECT 1 FROM log_leaves r WHERE r.kind = 'revoke' AND r.refers = l.leaf_hash)"
        states: dict = {}

        def judge(a, named) -> None:
            idx, leaf = a["idx"], a["leaf_hash"]
            waits = len(open_at) - bisect.bisect_right(open_at, idx) - sum(1 for i in answered_for.get(leaf, ()) if i > idx)
            if not waits:
                waits = any(max(for_all.get(h, -1), for_one.get((h, leaf), -1)) <= idx for h in named)
            states.setdefault(a["record_sha256"], set()).add("unresolved" if waits else "standing")
        if open_at:
            for a in self.db.execute("SELECT l.idx, l.leaf_hash, l.record_sha256 FROM log_leaves l WHERE l.kind = 'admit'"
                                     " AND l.waits = 0 AND l.idx < ?" + live, (edge,)):
                judge(a, ())
        for a in self.db.execute("SELECT l.idx, l.leaf_hash, l.record_sha256, b.body FROM log_leaves l JOIN log_bodies b"
                                 " ON b.leaf_hash = l.leaf_hash WHERE l.kind = 'admit' AND l.waits = 1" + live):
            judge(a, (json.loads(bytes(a["body"])) if a["body"] else {}).get("pending_lost") or ())
        waiting = [sha for sha, seen in states.items() if "unresolved" in seen and "standing" not in seen]
        out = []
        for i in range(0, len(waiting), 500):
            chunk = waiting[i:i + 500]
            # An admit past every open loss that names none was not read above: it stands, unless struck.
            standing = {r["record_sha256"] for r in self.db.execute(
                "SELECT DISTINCT l.record_sha256 FROM log_leaves l WHERE l.kind = 'admit' AND l.waits = 0 AND l.idx > ?"
                f" AND l.record_sha256 IN ({','.join('?' * len(chunk))})" + live, [edge, *chunk])}
            out += [sha for sha in chunk if sha not in standing]
        return out

    def admit_leaf_of(self, record_sha256: str) -> str | None:
        row = self.db.execute("SELECT leaf_hash FROM log_leaves WHERE kind = 'admit' AND record_sha256 = ? ORDER BY idx LIMIT 1",
                              (record_sha256,)).fetchone()
        return row["leaf_hash"] if row else None

    def admitted(self, record_sha256: str) -> dict | None:
        """The newest readable admit entry of the record, or None."""
        for r in self.db.execute("SELECT leaf_hash FROM log_leaves WHERE kind = 'admit' AND record_sha256 = ? ORDER BY idx DESC",
                                 (record_sha256,)):
            row = self.db.execute("SELECT body FROM log_bodies WHERE leaf_hash = ?", (r["leaf_hash"],)).fetchone()
            if row and row["body"]:
                return json.loads(bytes(row["body"]))
        return None

    def max_serial(self) -> int:
        row = self.db.execute("SELECT MAX(serial) AS m FROM log_leaves").fetchone()
        return int(row["m"] or 0)

    # ---- appending (inside the Store's transaction) ----

    def _insert_leaf(self, body: bytes, entry: dict) -> tuple:
        leaf = T.leaf_hash(body)
        index = self.size()
        self.db.execute("INSERT INTO log_leaves (idx, leaf_hash, kind, serial, record_sha256, refers, waits) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (index, leaf.hex(), entry["kind"], entry.get("serial"), entry.get("record_sha256"),
                         entry.get("withholds") or entry.get("revokes") or entry.get("answers"),
                         (1 if entry.get("pending_lost") else 0) if entry["kind"] == "admit" else None))
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

    def replay_leaf_locked(self, leaf_hex: str) -> None:
        """A published leaf whose body was never published (withheld, or inside the delay), put back as a leaf alone."""
        index = self.size()
        leaf = bytes.fromhex(leaf_hex)
        self.db.execute("INSERT INTO log_leaves (idx, leaf_hash, kind) VALUES (?, ?, 'unavailable')", (index, leaf_hex))
        self.db.execute("INSERT INTO log_bodies (leaf_hash, body, withheld_at, withhold_reason) VALUES (?, ?, ?, ?)",
                        (leaf_hex, b"", T.now_rfc3339(), UNAVAILABLE))
        level, i, h = 0, index, leaf
        self.db.execute("INSERT INTO log_nodes (level, idx, hash) VALUES (0, ?, ?)", (index, leaf))
        while i % 2 == 1:
            h = T.node_hash(self._perfect(level, i - 1), h)
            level, i = level + 1, i // 2
            self.db.execute("INSERT INTO log_nodes (level, idx, hash) VALUES (?, ?, ?)", (level, i, h))

    def insert_head_locked(self, head: dict) -> None:
        self.db.execute("INSERT OR IGNORE INTO log_heads (tree_size, root_sha256, timestamp, key_id, signature) VALUES (?, ?, ?, ?, ?)",
                        (head["tree_size"], head["root_sha256"], head["timestamp"], head["key_id"], head["signature"]))

    def _sign_head_locked(self) -> dict:
        head = T.sign_head(self.size(), self.tree().root(), self.signer)
        self.insert_head_locked(head)
        return head

    def heartbeat_locked(self) -> dict:
        """An entry with nothing but its time: freshness lives in the tree, and every size keeps exactly one head."""
        return self.append_locked(T.make_entry("heartbeat"))

    def resign_locked(self) -> dict | None:
        """After a key change, a heartbeat signed by the new key: its head commits to every entry before it."""
        latest = self.head()
        if latest is None or latest["key_id"] == self.signer.key_id:
            return None
        return self.heartbeat_locked()

    def remember_key_locked(self) -> None:
        """The log key's public half, kept so a published head signed by an earlier key can still be checked."""
        self.db.execute("INSERT OR IGNORE INTO log_keys (key_id, public_pem, first_seen) VALUES (?, ?, ?)",
                        (self.signer.key_id, self.signer.public_pem(), T.now_rfc3339()))

    def known_key(self, key_id: str) -> str | None:
        row = self.db.execute("SELECT public_pem FROM log_keys WHERE key_id = ?", (key_id,)).fetchone()
        return row["public_pem"] if row else None

    def admit_locked(self, serial: int, record_sha256: str, counter: dict, time: str | None = None, wardens=None,
                     sitting=None, appended_by: str | None = None, restores: dict | None = None,
                     pending_lost: list | None = None) -> dict:
        ref = {k: counter[k] for k in ("register_key_id", "register_signature", "countersigned_at_utc")}
        return self.append_locked(T.make_entry("admit", time=time, serial=int(serial), record_sha256=record_sha256,
                                               countersignature=ref, wardens=_names(wardens), sitting=_text(sitting, "sitting", False),
                                               appended_by=appended_by, restores=restores, pending_lost=pending_lost))

    def reserve_locked(self, count: int, reason: str) -> list:
        """`count` serials above every one this register knows, held back for records a recovery could not see."""
        top = max(int(self.db.execute("SELECT MAX(serial) AS m FROM records").fetchone()["m"] or 0), self.max_serial(),
                  int(self.db.execute("SELECT MAX(serial) AS m FROM log_reserved").fetchone()["m"] or 0))
        from khaos_attribution.registry import FIRST_SERIAL  # noqa: PLC0415
        start = max(top + 1, FIRST_SERIAL)
        out = list(range(start, start + count))
        for serial in out:
            self.db.execute("INSERT INTO log_reserved (serial, reason, reserved_at) VALUES (?, ?, ?)", (serial, reason, T.now_rfc3339()))
        return out

    def sync_reserved_locked(self) -> None:
        """The reservations rebuilt from the log itself: every lost-entry withhold that holds a serial, returned once an
        admit restores it. The log survives a restore that the table alone would not."""
        for r in self.db.execute("SELECT leaf_hash FROM log_leaves WHERE kind IN ('withhold', 'admit')").fetchall():
            row = self.db.execute("SELECT body FROM log_bodies WHERE leaf_hash = ?", (r["leaf_hash"],)).fetchone()
            if not row or not row["body"]:
                continue
            e = json.loads(bytes(row["body"]))
            if e.get("kind") == "withhold" and isinstance(e.get("reserves_serial"), int):
                self.db.execute("INSERT OR IGNORE INTO log_reserved (serial, reason, reserved_at) VALUES (?, ?, ?)",
                                (e["reserves_serial"], "a recovery lost entries whose kind is unknown", e["time"]))
            elif e.get("kind") == "admit" and isinstance(e.get("restores"), dict):
                self.db.execute("UPDATE log_reserved SET returned_at = COALESCE(returned_at, ?), returned_to = COALESCE(returned_to, ?)"
                                " WHERE serial = ?", (e["time"], e["record_sha256"], e["restores"].get("serial")))

    def open_lost_global(self) -> list:
        """Lost leaves no confirm has answered for every admit."""
        lost, answers = self._lost_and_answers()
        return [h for i, h in sorted(lost.items()) if not any(c > i and ans == h and conf is None for c, ans, conf in answers)]

    def admit_leaf_of_latest(self, record_sha256: str) -> str | None:
        row = self.db.execute("SELECT leaf_hash FROM log_leaves WHERE kind = 'admit' AND record_sha256 = ? ORDER BY idx DESC LIMIT 1",
                              (record_sha256,)).fetchone()
        return row["leaf_hash"] if row else None

    def reserved(self) -> list:
        return [r["serial"] for r in self.db.execute("SELECT serial FROM log_reserved WHERE returned_at IS NULL ORDER BY serial")]

    def max_reserved(self) -> int:
        return int(self.db.execute("SELECT MAX(serial) AS m FROM log_reserved").fetchone()["m"] or 0)

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

    def confirm_locked(self, answers: str, confirms: str | None = None, wardens=None, sitting=None,
                       appended_by: str | None = None) -> dict:
        """The wardens' answer to a lost leaf: nothing it held strikes the admits before it (or only `confirms`)."""
        lost = self.leaf(answers)
        if lost is None or lost["kind"] != "unavailable":
            raise EntryRefused("a confirm names the lost leaf it answers")
        sitting = _text(sitting, "sitting", True)
        fields = {}
        if confirms is not None:
            admit = self.leaf(confirms)
            if admit is None or admit["kind"] != "admit" or answers not in self.open_lost(admit["idx"], confirms):
                raise EntryRefused("that admit is not left unresolved by that lost leaf")
            fields = {"confirms": confirms, "serial": admit["serial"], "record_sha256": admit["record_sha256"]}
        return self.append_locked(T.make_entry("confirm", answers=answers, wardens=_names(wardens), sitting=sitting,
                                               appended_by=appended_by, **fields))

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
        for r in self.db.execute("SELECT l.idx, l.leaf_hash, l.kind, b.body FROM log_leaves l LEFT JOIN log_bodies b"
                                 " ON b.leaf_hash = l.leaf_hash ORDER BY l.idx"):
            if r["idx"] != len(leaves):
                out.append(f"leaf {len(leaves)} is missing")
            if r["kind"] == "unavailable":
                pass
            elif r["body"] is None or T.leaf_hash(bytes(r["body"])).hex() != r["leaf_hash"]:
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
        """What each published copy of the log says about this database: ok, behind, forked, empty (nothing published
        there yet), unreadable, or other-key (signed by a key this log never used)."""
        from khaos_attribution.translog import verify as V  # noqa: PLC0415
        out = []
        for where in places:
            mirror = V.Mirror(str(where), timeout=5.0)
            try:
                head = mirror.get("sth/latest.json")
            except V.NotFound as e:
                out.append({"where": str(where), "state": "empty", "message": str(e)})
                continue
            except V.Unreadable as e:
                empty = mirror.local and not (Path(str(where)) / "log").is_dir()
                out.append({"where": str(where), "state": "empty" if empty else "unreadable", "message": str(e)})
                continue
            try:
                if not isinstance(head, dict) or not isinstance(head.get("tree_size"), int):
                    raise T.LogError("not a signed head")
                own = self.head(head["tree_size"]) if head["tree_size"] <= self.size() else None
                # Only the key that signed a size here may speak for it; beyond this database, only the current key.
                expected = own["key_id"] if own else T.key_id_of_pem(public_pem)
                pem = public_pem if expected == T.key_id_of_pem(public_pem) else self.known_key(expected)
                if head.get("key_id") != expected or pem is None:
                    raise T.LogError(f"signed by key {head.get('key_id')}, not the key of size {head['tree_size']} ({expected})")
                T.verify_head(head, pem)
            except (T.LogError, AttributeError) as e:
                out.append({"where": str(where), "state": "other-key", "message": f"{where}: {e}"})
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

    def open_verdict(self, seen: list, unwitnessed: bool = False, started: bool = False) -> str | None:
        """Why this database may not append, or None. It fails closed: some published copy must vouch for it, unless
        the log is new, this register never started one, and nothing anywhere answers that it was published."""
        for w in seen:
            if w["state"] in ("behind", "forked"):
                return w["message"]
        if any(w["state"] == "ok" for w in seen):
            return None
        if self.size() == 0 and not started and all(w["state"] == "empty" for w in seen):
            return None
        if unwitnessed:
            if self.size():
                return ("the one-shot unwitnessed open is refused: the log has entries, so a published copy must vouch for it "
                        "(REGISTER_LOG_WITNESSES)")
            return None
        if self.size() == 0 and started:
            return ("this register started its log before (data/log-started) yet its database holds none, and no published "
                    "copy can be read: restore the database or make a published copy reachable")
        return ("no published copy of the log could be read to show this database is not behind it ("
                + "; ".join(f"{w['where']}: {w['state']}" for w in seen) + "). Make one reachable (REGISTER_LOG_WITNESSES), or, "
                "for a log with no entries and after checking by hand that nothing was published, create data/"
                "log-open-unwitnessed for one start")

    # ---- anchors ----

    def anchors(self) -> list:
        return [dict(r) for r in self.db.execute("SELECT * FROM log_anchors ORDER BY tree_size")]


def recover(store, places: list, public_pem: str) -> tuple:
    """Puts back the entries a published copy holds and this database lost, verbatim (a leaf alone where its body was
    never published), when they reproduce that copy's signed root; (recovered, lines). Anything else is an incident."""
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
    lines = [f"entry {i} recovered without its body (withheld, or not yet public); its leaf is in the signed tree"
             for i in missing]
    try:
        with store._tx():
            for i in range(n, size):
                if i in missing:
                    log.replay_leaf_locked(leaves[i])
                    continue
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
            log.blocked = None
            log.sync_reserved_locked()
            unmarked = [r["leaf_hash"] for r in log.db.execute(
                "SELECT b.leaf_hash FROM log_bodies b JOIN log_leaves l ON l.leaf_hash = b.leaf_hash WHERE b.withhold_reason = ?"
                " ORDER BY l.idx", (UNAVAILABLE,)).fetchall()]
            # A lost leaf may have been an admit holding the next serial; its kind is unknown, so each one holds one.
            held = log.reserve_locked(len(unmarked), "a recovery lost entries whose kind is unknown") if unmarked else []
            for leaf_hex, serial in zip(unmarked, held):
                # The withhold says the body is gone and carries the reservation, so a later restore rebuilds it from the log.
                log.append_locked(T.make_entry("withhold", withholds=leaf_hex, lost=True, reserves_serial=serial,
                                               reason="the body was lost in a restore before it was published"))
                lines.append(f"leaf {leaf_hex} is now marked as lost by a withhold entry, holding serial {serial}")
            if held:
                lines.append(f"serials {held[0]} to {held[-1]} are reserved for records whose admit may be among the lost "
                             "entries; each returns to the record that presents this register's countersignature for it")
    except T.LogError as e:
        log.blocked = log.blocked or "recovery failed"
        return False, [f"INCIDENT: {e}"]
    return True, [f"recovered entries {n} to {size - 1} from {where}; the log is at size {size} with its published root", *lines]


def head_doc(row) -> dict:
    return {**T.head_payload(row["tree_size"], row["root_sha256"], row["timestamp"]), "key_id": row["key_id"],
            "signature": row["signature"]}


# ---- the static export ----

_EXPORT_LOCK = threading.Lock()


def write_whole(path: Path, data: bytes) -> None:
    """A temp file of this writer's own, then replace: a reader never meets a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.new")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _remove(path: Path) -> None:
    path.unlink(missing_ok=True)


def _json(doc) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8")


class ExportBusy(RuntimeError):
    """Another writer held the lock past the timeout."""


@contextmanager
def export_lock(mirror: Path):
    """One writer of a mirror at a time (an export, the ledger, the copy to the public repository): across threads by
    the lock, across processes by the lock file in ``mirror`` (POSIX; the register is served from Linux only), each
    waited on for at most EXPORT_LOCK_TIMEOUT seconds."""
    import time  # noqa: PLC0415
    mirror.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + EXPORT_LOCK_TIMEOUT
    if not _EXPORT_LOCK.acquire(timeout=EXPORT_LOCK_TIMEOUT):
        raise ExportBusy("another export in this process held the lock")
    try:
        with open(mirror / ".log-export.lock", "w") as fh:
            try:
                import fcntl  # noqa: PLC0415
            except ImportError:
                fcntl = None
            while fcntl is not None:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise ExportBusy("another process held the export lock") from None
                    time.sleep(0.05)
            yield
    finally:
        _EXPORT_LOCK.release()


def snapshot(log: TransLog, nodes: bool = True) -> dict:
    """Everything the export writes, read in one transaction so the files agree with each other. Without ``nodes``
    the tree reads its hashes from the database as a proof needs them: they never change once written."""
    db = log.db
    db.execute("BEGIN")
    try:
        rows = db.execute("SELECT l.idx, l.leaf_hash, l.kind, l.refers, b.body, b.withheld_at FROM log_leaves l"
                          " JOIN log_bodies b ON b.leaf_hash = l.leaf_hash ORDER BY l.idx").fetchall()
        held = {(r["level"], r["idx"]): bytes(r["hash"]) for r in db.execute("SELECT level, idx, hash FROM log_nodes")} if nodes else None
        heads = log.heads()
        first_current = db.execute("SELECT MIN(tree_size) AS m FROM log_heads WHERE key_id = ?", (log._key_id(),)).fetchone()["m"]
        struck = log.struck()
        anchors = log.anchors()
    finally:
        db.execute("COMMIT")
    leaves = [r["leaf_hash"] for r in rows]
    return {"leaves": leaves, "kinds": [r["kind"] for r in rows], "refers": [r["refers"] for r in rows],
            "bodies": [None if r["withheld_at"] else bytes(r["body"]) for r in rows],
            "raw": [bytes(r["body"]) for r in rows],
            "withheld": sorted(r["leaf_hash"] for r in rows if r["withheld_at"]),
            "tree": T.Tree(len(leaves), (lambda level, index: held[(level, index)]) if nodes else log._perfect),
            "heads": heads, "first_current": first_current, "struck": struck, "anchors": anchors,
            "current_sizes": [h["tree_size"] for h in heads if h["key_id"] == log._key_id()]}


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


def _written(log: TransLog, root: Path) -> dict | None:
    """What the last export left under ``root``, while the newest head there is the one it wrote and still the
    database's head of that size; else None, and everything is written again."""
    row = log.db.execute("SELECT * FROM log_exports WHERE place = ?", (str(root.resolve()),)).fetchone()
    if row is None or row["format"] != EXPORT_FORMAT or row["key_id"] != log._key_id():
        return None
    try:
        disk = json.loads((root / "sth" / "latest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return dict(row) if disk == log.head(row["tree_size"]) else None


def export(log: TransLog, mirror: Path, key_doc: dict, public: Path | None = None, delay_days: float = 7,
           now: datetime | None = None, full: bool = False) -> bool:
    """Everything a verifier needs, under mirror/log/, and under public/log/ the same with new bodies held back for
    `delay_days`: what the entries since the last export add, or all of it with ``full`` and wherever the files are
    not that export's. False when nothing was written (a blocked log, or the lock not had in time)."""
    import logging  # noqa: PLC0415
    if log.blocked:
        return False
    places = [(Path(mirror) / "log", False)] + ([(Path(public) / "log", True)] if public is not None else [])
    try:
        with export_lock(Path(mirror)):
            done = {root: None if full else _written(log, root) for root, _ in places}
            snap = snapshot(log, nodes=any(d is None for d in done.values()))
            for root, delayed in places:
                pending = pending_from(snap, delay_days, now) if delayed else None
                latest = _write_log(root, snap, key_doc, pending, done[root])
                if latest is not None:
                    with log.store._tx():
                        log.db.execute("INSERT OR REPLACE INTO log_exports (place, format, key_id, tree_size, pending_from)"
                                       " VALUES (?, ?, ?, ?, ?)", (str(root.resolve()), EXPORT_FORMAT, latest["key_id"],
                                                                   latest["tree_size"], pending))
    except ExportBusy as e:
        logging.getLogger("register_service").warning("log export skipped: %s; the next export catches up", e)
        return False
    return True


def _owed_to_the_old_head(root: Path, snap: dict, shown: list, done: dict | None, was_size: int) -> tuple:
    """What a reader of the head still on disk needs that this export takes away: bodies whose withhold that head
    does not hold. (kept, waiting): those it could read, as index -> body, stay until the new head is filed; those
    it knew only as inside the delay, as indices, stay explained by the delay until then."""
    if not was_size:
        return {}, []
    withheld_by = {}
    for w, (kind, names) in enumerate(zip(snap["kinds"], snap["refers"])):
        if kind == "withhold":
            withheld_by.setdefault(names, w)
    index_of = {leaf: i for i, leaf in enumerate(snap["leaves"])}
    late = [index_of[leaf] for leaf, w in withheld_by.items() if leaf in index_of and index_of[leaf] < was_size <= w]
    late = [i for i in late if shown[i] is None]
    if not late:
        return {}, []
    if done is not None:
        # The last export's own record says where its delay stood.
        public = {i for i in late if done["pending_from"] is None or i < done["pending_from"]}
    else:
        try:
            old = json.loads((root / "bodies.json").read_text(encoding="utf-8"))["bodies"]
        except (OSError, ValueError, KeyError, TypeError):
            return {}, []
        public = {i for i in late if i < len(old) and old[i] is not None}
    # A leaf put back without its body has none to keep; one the old head held back is still owed the delay.
    return {i: snap["raw"][i] for i in public if snap["raw"][i]}, sorted(set(late) - public)


def _write_log(root: Path, snap: dict, key_doc: dict, pending: int | None, done: dict | None) -> dict | None:
    """The files under ``root``; with ``done`` (the last export's record) only what changed since. Returns the head
    now filed as the newest, or None when the files there are already ahead of this snapshot. In an order that
    verifies wherever it stops: everything a reader of the old head can live with, the new head, then what is taken away."""
    tree, leaves, n = snap["tree"], snap["leaves"], len(snap["leaves"])
    on_disk, was_size = root / "sth" / "latest.json", 0
    if on_disk.is_file():
        try:
            was_size = int(json.loads(on_disk.read_text(encoding="utf-8"))["tree_size"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if was_size > n:
            return None
    # A withhold names only a leaf and a reason, which a verifier needs; a heartbeat names nothing: neither waits.
    shown = [None if pending is not None and i >= pending and snap["kinds"][i] not in ("withhold", "heartbeat") else b
             for i, b in enumerate(snap["bodies"])]
    was = 0 if done is None else done["tree_size"]
    # A body's place can change behind the newest leaf only where the delay's edge moved, or by a withhold.
    edge = 0 if done is None else min(was, pending if pending is not None else n,
                                      done["pending_from"] if done["pending_from"] is not None else was)

    def bodies(kept: dict, edge_said: int | None) -> bytes:
        return _json({"tree_size": n, "pending_from": edge_said,
                      "bodies": [(kept.get(i) or b).decode("utf-8") if (b is not None or i in kept) else None
                                 for i, b in enumerate(shown)]})
    write_whole(root / "key.json", _json(key_doc))
    for i in range(edge, n):
        path = root / "entries" / f"{leaves[i]}.json"
        if shown[i] is not None and not path.is_file():
            write_whole(path, shown[i])
    current = snap["current_sizes"]
    for i in range(was, n):
        at = bisect.bisect_left(current, i + 1)
        size = current[at] if at < len(current) else n
        write_whole(root / "proofs" / f"{leaves[i]}.json",
                    _json({"leaf_hash": leaves[i], "leaf_index": i, "tree_size": size, "audit_path": T.hexes(tree.inclusion(i, size))}))
    write_whole(root / "hashes.json", _json({"tree_size": n, "leaves": leaves}))
    kept, waiting = _owed_to_the_old_head(root, snap, shown, done, was_size)
    write_whole(root / "bodies.json", bodies(kept, min([pending if pending is not None else n, *waiting]) if waiting else pending))
    latest = before = None
    for head in snap["heads"]:
        if head["tree_size"] > was:
            write_whole(root / "sth" / f"{head['tree_size']}.json", _json(head))
            if before is not None:
                write_whole(root / "consistency" / f"{before['tree_size']}.json",
                            _json({"first": before["tree_size"], "second": head["tree_size"],
                                   "proof": T.hexes(tree.consistency(before["tree_size"], head["tree_size"]))}))
        latest = before = head
    _write_anchors(root, snap["anchors"])
    if latest is not None:
        write_whole(on_disk, _json(latest))
    # From here the new head is the one read, and it proves every withhold: what it hides can go.
    if kept or waiting:
        write_whole(root / "bodies.json", bodies({}, pending))
    for leaf_hex in snap["withheld"] if done is not None else ():
        _remove(root / "entries" / f"{leaf_hex}.json")
    for i in range(edge, n):
        if shown[i] is None:
            _remove(root / "entries" / f"{leaves[i]}.json")
    hidden = {leaves[i] for i in range(n) if shown[i] is None and snap["bodies"][i] is not None}
    if done is None:
        _remove(root / "records.json")
    write_whole(root / "struck.json", _json({k: v for k, v in snap["struck"].items() if v["leaf_hash"] not in hidden}))
    write_whole(root / "withheld.json", _json(snap["withheld"]))
    return latest


def _write_anchors(root: Path, anchors: list) -> None:
    rows = []
    for a in anchors:
        name = f"{a['tree_size']}.ots"
        write_whole(root / "anchors" / name, bytes(a["receipt"]))
        rows.append({"tree_size": a["tree_size"], "root_sha256": a["root_sha256"], "backend": a["backend"], "state": a["state"],
                     "stamped_at": a["stamped_at"], "completed_at": a["completed_at"], "receipt": f"anchors/{name}"})
    write_whole(root / "anchors.json", _json(rows))


def export_anchors(log: TransLog, mirror: Path) -> None:
    """The anchor receipts, as OpenTimestamps files, and their list."""
    try:
        with export_lock(Path(mirror)):
            _write_anchors(Path(mirror) / "log", log.anchors())
    except ExportBusy:
        pass


def receipt_b64(receipt: bytes) -> str:
    return base64.b64encode(bytes(receipt)).decode("ascii")
