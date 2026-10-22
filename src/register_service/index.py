"""The fingerprint index beside the register: released outputs under their model's watermark bucket, the
verify that names one of them, and the training tracks each registered model claims. Fingerprints are
landmark hashes, never audio.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

from khaos_attribution.fingerprint import best_phase_stats, dominant, is_confident
from register_service.store import now_utc

SCHEMA = """
CREATE TABLE IF NOT EXISTS outputs (generation_id TEXT PRIMARY KEY, serial INTEGER NOT NULL, payload INTEGER NOT NULL,
  artist_id TEXT NOT NULL, released_at TEXT NOT NULL, released_by TEXT NOT NULL, kind TEXT NOT NULL, landmarks INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS fingerprints (hash INTEGER NOT NULL, generation_id TEXT NOT NULL, offset_ms INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS fingerprints_hash ON fingerprints (hash);
CREATE INDEX IF NOT EXISTS fingerprints_gen ON fingerprints (generation_id);
CREATE INDEX IF NOT EXISTS outputs_payload ON outputs (payload);
CREATE INDEX IF NOT EXISTS fingerprints_vote ON fingerprints (hash, generation_id, offset_ms);
CREATE TABLE IF NOT EXISTS manifests (generation_id TEXT PRIMARY KEY, store BLOB NOT NULL, sha256 TEXT NOT NULL,
    kept_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS claims (claim_id INTEGER PRIMARY KEY, account_id INTEGER NOT NULL, serial INTEGER NOT NULL,
  track_id TEXT NOT NULL, declaration TEXT NOT NULL, claimed_at TEXT NOT NULL, claimed_by TEXT NOT NULL,
  landmarks INTEGER NOT NULL, UNIQUE (serial, track_id));
CREATE TABLE IF NOT EXISTS claim_fingerprints (hash INTEGER NOT NULL, claim_id INTEGER NOT NULL, offset_ms INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS claim_fingerprints_hash ON claim_fingerprints (hash);
CREATE INDEX IF NOT EXISTS claim_fingerprints_vote ON claim_fingerprints (hash, claim_id, offset_ms);
CREATE INDEX IF NOT EXISTS claim_fingerprints_claim ON claim_fingerprints (claim_id);
CREATE TABLE IF NOT EXISTS claim_shares (claim_id INTEGER NOT NULL, other_claim INTEGER NOT NULL, votes INTEGER NOT NULL,
  at TEXT NOT NULL, PRIMARY KEY (claim_id, other_claim));
"""
MAX_LANDMARKS = 200_000
MAX_PHASES = 8
# Columns a first-version index file gains in place.
_ADDED = (("outputs", "withdrawn", "INTEGER NOT NULL DEFAULT 0"), ("outputs", "removed_at", "TEXT"))
CANDIDATES = 50


class Index:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        for table, column, declaration in _ADDED:
            if column not in {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    @property
    def db(self) -> sqlite3.Connection:
        """This thread's own connection: requests run on the loop and in the threadpool, and one connection shared
        between threads mid-transaction corrupts reads."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), isolation_level=None, timeout=30)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def release(self, generation_id: str, serial: int, payload: int, artist_id: str, fingerprint: list,
                released_by: str, kind: str) -> dict:
        """One released output under its model's bucket; the same id again is the first release, unchanged."""
        rows = [(h, generation_id, o) for h, o in _landmarks(fingerprint)]
        with self._lock:
            have = self.db.execute("SELECT * FROM outputs WHERE generation_id = ?", (generation_id,)).fetchone()
            if have:
                return {**dict(have), "already": True}
            self.db.execute("BEGIN IMMEDIATE")
            try:
                at = now_utc()
                self.db.execute("INSERT INTO outputs (generation_id, serial, payload, artist_id, released_at, released_by, kind, landmarks)"
                                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (generation_id, serial, payload, artist_id, at, released_by, kind, len(rows)))
                self.db.executemany("INSERT INTO fingerprints (hash, generation_id, offset_ms) VALUES (?, ?, ?)", rows)
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        return {"generation_id": generation_id, "serial": serial, "payload": payload, "artist_id": artist_id,
                "released_at": at, "released_by": released_by, "kind": kind, "landmarks": len(rows), "already": False}

    def keep_manifest(self, generation_id: str, store: bytes, released_by: str) -> bool:
        """A released output's Content Credentials manifest store, for the soft-binding lookup; only the machine that
        released the output may set or replace it (a re-signed file carries a new one)."""
        with self._lock:
            out = self.db.execute("SELECT released_by FROM outputs WHERE generation_id = ?", (generation_id,)).fetchone()
            if not out or out["released_by"] != released_by:
                return False
            self.db.execute("INSERT OR REPLACE INTO manifests (generation_id, store, sha256, kept_at) VALUES (?, ?, ?, ?)",
                            (generation_id, store, hashlib.sha256(store).hexdigest(), now_utc()))
        return True

    def manifest(self, generation_id: str) -> bytes | None:
        row = self.db.execute("SELECT store FROM manifests WHERE generation_id = ?", (generation_id,)).fetchone()
        return bytes(row["store"]) if row else None

    def has_manifest(self, generation_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM manifests WHERE generation_id = ?", (generation_id,)).fetchone() is not None

    def withdraw_model(self, serial: int) -> int:
        """A withdrawn model's pieces are not offered by any door: the one rule, kept on the rows."""
        with self._lock:
            return self.db.execute("UPDATE outputs SET withdrawn = 1 WHERE serial = ?", (int(serial),)).rowcount

    def remove_output(self, generation_id: str, serial: int) -> dict | None:
        """A piece its maker took down: its fingerprint stays, so a copy found later is still named, but the answer
        says it was removed and the lookup no longer offers it."""
        with self._lock:
            row = self.db.execute("SELECT * FROM outputs WHERE generation_id = ? AND serial = ?", (generation_id, int(serial))).fetchone()
            if row is None:
                return None
            if row["removed_at"] is None:
                self.db.execute("UPDATE outputs SET removed_at = ? WHERE generation_id = ?", (now_utc(), generation_id))
            return self.output(generation_id)

    def offered(self, generation_id: str) -> bool:
        """Whether a piece may be offered to the public: released, its model not withdrawn, not removed by its maker."""
        row = self.db.execute("SELECT withdrawn, removed_at FROM outputs WHERE generation_id = ?", (generation_id,)).fetchone()
        return bool(row) and not row["withdrawn"] and row["removed_at"] is None

    def output(self, generation_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM outputs WHERE generation_id = ?", (generation_id,)).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM outputs").fetchone()[0])

    def _candidates(self, phases: list, payload: int | None) -> dict:
        """The outputs sharing hashes with the query, by generation id, with the landmarks that share them: a vote
        only ever comes from a shared hash, so the rest of a candidate's landmarks are never read. The bucket first;
        a withdrawn model's pieces never."""
        hashes = sorted({h for q in phases for h, _ in q})
        if not hashes:
            return {}
        rows: dict = {}
        for i in range(0, len(hashes), 900):
            chunk = hashes[i:i + 900]
            marks = ",".join("?" * len(chunk))
            q = ("SELECT f.generation_id AS g, f.hash AS h, f.offset_ms AS o FROM fingerprints f"
                 " JOIN outputs ou ON ou.generation_id = f.generation_id WHERE ou.withdrawn = 0 AND f.hash IN (%s)" % marks)
            params: list = list(chunk)
            if payload is not None:
                q += " AND ou.payload = ?"
                params.append(payload)
            for row in self.db.execute(q, params):
                rows.setdefault(row["g"], []).append((row["h"], row["o"]))
        top = sorted(rows, key=lambda g: len(rows[g]), reverse=True)[:CANDIDATES]
        return {g: rows[g] for g in top}

    def verify(self, phases: list, payload: int | None) -> dict:
        """The three grades: a named output; a watermark alone narrowing to a bucket; nothing."""
        query_len = len(phases[0]) if phases else 0
        scored = []
        for g, fps in self._candidates(phases, payload).items():
            votes, distinct = best_phase_stats(phases, fps)
            scored.append((votes, distinct, g))
        scored.sort(reverse=True)
        if scored:
            votes, distinct, g = scored[0]
            runner = scored[1][0] if len(scored) > 1 else 0
            if is_confident(votes, query_len, distinct) and dominant(votes, runner):
                out = self.output(g) or {}
                return {"grade": "output", "output": out, "votes": votes, "runner_up": runner, "searched": "bucket" if payload is not None else "all"}
        if payload is not None:
            return {"grade": "bucket", "payload": payload, "searched": "bucket"}
        return {"grade": "none", "searched": "all"}

    def _claim_matches(self, phases: list) -> list:
        """Every claim the query confidently matches, strongest first: (votes, claim row). Confidence is read against
        the shorter of the two, so a short claimed track inside a long medley matches either way round."""
        hashes = sorted({h for q in phases for h, _ in q})
        rows: dict = {}
        for i in range(0, len(hashes), 900):
            chunk = hashes[i:i + 900]
            q = "SELECT claim_id AS c, hash AS h, offset_ms AS o FROM claim_fingerprints WHERE hash IN (%s)" % ",".join("?" * len(chunk))
            for row in self.db.execute(q, chunk):
                rows.setdefault(row["c"], []).append((row["h"], row["o"]))
        query_len = len(phases[0]) if phases else 0
        out = []
        for c in sorted(rows, key=lambda c: len(rows[c]), reverse=True)[:CANDIDATES]:
            claim = dict(self.db.execute("SELECT * FROM claims WHERE claim_id = ?", (c,)).fetchone())
            votes, distinct = best_phase_stats(phases, rows[c])
            # The shared rows vote; confidence still reads against the claim's whole fingerprint, as stored.
            if is_confident(votes, min(query_len, int(claim["landmarks"])), distinct):
                out.append((votes, claim))
        out.sort(key=lambda m: m[0], reverse=True)
        return out

    def check_claim(self, phases: list, account_id: int) -> dict:
        """Whether a recording already trains a registered model (none, this account's, or also another account's),
        and the serial of the released output it is, when it is one. Nothing here refuses."""
        matches = self._claim_matches(phases)
        output = self.verify(phases, None)
        released = int(output["output"]["serial"]) if output["grade"] == "output" else None
        others = [(v, c) for v, c in matches if c["account_id"] != account_id]
        if others:
            return {"grade": "other", "claim_ref": others[0][1]["claim_id"], "votes": others[0][0], "released_output": released,
                    "also_mine": len(others) < len(matches)}
        if matches:
            return {"grade": "mine", "serial": matches[0][1]["serial"], "track_id": matches[0][1]["track_id"], "released_output": released}
        return {"grade": "none", "released_output": released}

    def claim(self, account_id: int, serial: int, track_id: str, fingerprint: list, declaration: str, claimed_by: str,
              phases: list | None = None) -> dict:
        """A registered model's training track, claimed on its account's declaration. A recording may train any number
        of models; one another account also claims is linked to that claim, so both models' pages say it is shared."""
        rows = _landmarks(fingerprint)
        query = [_landmarks(p) for p in phases] if phases else [rows]
        have = self.db.execute("SELECT * FROM claims WHERE serial = ? AND track_id = ?", (serial, track_id)).fetchone()
        if have:
            return {"claimed": True, "claim_ref": have["claim_id"], "already": True}
        # Matching reads for seconds; it runs before the lock so the releases and verifies behind it do not wait.
        others = [(v, c) for v, c in self._claim_matches(query) if c["account_id"] != account_id]
        with self._lock:
            have = self.db.execute("SELECT * FROM claims WHERE serial = ? AND track_id = ?", (serial, track_id)).fetchone()
            if have:
                return {"claimed": True, "claim_ref": have["claim_id"], "already": True}
            self.db.execute("BEGIN IMMEDIATE")
            try:
                cur = self.db.execute("INSERT INTO claims (account_id, serial, track_id, declaration, claimed_at, claimed_by, landmarks)"
                                      " VALUES (?, ?, ?, ?, ?, ?, ?)", (account_id, serial, track_id, declaration, now_utc(), claimed_by, len(rows)))
                cid = cur.lastrowid
                self.db.executemany("INSERT INTO claim_fingerprints (hash, claim_id, offset_ms) VALUES (?, ?, ?)",
                                    [(h, cid, o) for h, o in rows])
                self.db.executemany("INSERT OR IGNORE INTO claim_shares (claim_id, other_claim, votes, at) VALUES (?, ?, ?, ?)",
                                    [(cid, c["claim_id"], v, now_utc()) for v, c in others])
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        return {"claimed": True, "claim_ref": cid, "already": False, "shared_with_accounts": len({c["account_id"] for _, c in others})}

    def release_claims(self, serial: int) -> int:
        """A withdrawn model's claims end with it, and so do the links that said another model shares them."""
        with self._lock:
            ids = [r["claim_id"] for r in self.db.execute("SELECT claim_id FROM claims WHERE serial = ?", (serial,))]
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.executemany("DELETE FROM claim_fingerprints WHERE claim_id = ?", [(i,) for i in ids])
                self.db.executemany("DELETE FROM claim_shares WHERE claim_id = ? OR other_claim = ?", [(i, i) for i in ids])
                self.db.execute("DELETE FROM claims WHERE serial = ?", (serial,))
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        return len(ids)

    def claims_of(self, serial: int) -> dict:
        """What a model's page says: how many of its training tracks are claimed, and how many another account's
        registered model also trained on."""
        claimed = int(self.db.execute("SELECT COUNT(*) FROM claims WHERE serial = ?", (serial,)).fetchone()[0])
        shared = int(self.db.execute(
            "SELECT COUNT(DISTINCT c.claim_id) FROM claims c JOIN claim_shares s ON s.claim_id = c.claim_id OR s.other_claim = c.claim_id"
            " WHERE c.serial = ?", (serial,)).fetchone()[0])
        return {"claimed": claimed, "shared": shared}


def phases_of(value) -> list:
    """A query's phases as the index takes them: at most MAX_PHASES lists of at most MAX_LANDMARKS [hash, offset_ms]
    each, so no body can cost more matching than a real query. ValueError says what is wrong."""
    if not isinstance(value, list) or not value or len(value) > MAX_PHASES or not all(isinstance(p, list) for p in value):
        raise ValueError(f"phases is 1 to {MAX_PHASES} lists of landmarks")
    return [_landmarks(p) for p in value]


def _landmarks(fingerprint) -> list:
    if not isinstance(fingerprint, list) or not fingerprint or len(fingerprint) > MAX_LANDMARKS:
        raise ValueError(f"a fingerprint is 1 to {MAX_LANDMARKS} landmarks")
    rows = []
    for item in fingerprint:
        if not (isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in item)):
            raise ValueError("a landmark is [hash, offset_ms], both integers")
        rows.append((int(item[0]), int(item[1])))
    return rows
