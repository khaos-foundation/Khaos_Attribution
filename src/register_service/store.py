"""The register's store: accounts, invites, enrolled machines, records, where
they are served, and the ledger. SQLite, one writer, every change a ledger row.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from khaos_attribution.registry import FIRST_SERIAL, canonical_bytes, sha256_hex
from register_service.translog import LogBlocked, TransLog

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS invites (code TEXT PRIMARY KEY, account_id INTEGER NOT NULL, role TEXT NOT NULL DEFAULT 'workshop',
  created_at TEXT NOT NULL, used_by TEXT, used_at TEXT);
CREATE TABLE IF NOT EXISTS machines (key_id TEXT PRIMARY KEY, account_id INTEGER NOT NULL, role TEXT NOT NULL DEFAULT 'workshop',
  name TEXT NOT NULL, public_pem TEXT NOT NULL, enrolled_at TEXT NOT NULL, revoked_at TEXT);
CREATE TABLE IF NOT EXISTS nonces (key_id TEXT NOT NULL, nonce TEXT NOT NULL, body_sha256 TEXT NOT NULL, answer_json TEXT,
  at TEXT NOT NULL, PRIMARY KEY (key_id, nonce));
CREATE TABLE IF NOT EXISTS records (serial INTEGER PRIMARY KEY, record_sha256 TEXT UNIQUE NOT NULL,
  machine_key_id TEXT NOT NULL, account_id INTEGER NOT NULL, signed_json TEXT NOT NULL, counter_json TEXT NOT NULL,
  state TEXT NOT NULL, published_at TEXT NOT NULL, withdrawn_at TEXT, withdraw_reason TEXT);
CREATE TABLE IF NOT EXISTS served (serial INTEGER NOT NULL, space_url TEXT NOT NULL, state TEXT NOT NULL,
  at TEXT NOT NULL, key_id TEXT NOT NULL, PRIMARY KEY (serial, space_url));
CREATE TABLE IF NOT EXISTS ledger (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, serial INTEGER, record_sha256 TEXT,
  key_id TEXT, at TEXT NOT NULL, detail_json TEXT NOT NULL, entry_sha256 TEXT NOT NULL, prev_sha256 TEXT);
"""


ROLES = ("workshop", "space", "warden")


def _payload_of(signed: dict) -> int:
    return int(signed["record"]["model"]["watermark_payload"])


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self._ledger_status = None
        self._add_payload_column()
        self.log = TransLog(self)

    def _add_payload_column(self) -> None:
        """The watermark bucket as its own indexed column, so a bucket lookup never parses every record."""
        columns = {r["name"] for r in self.db.execute("PRAGMA table_info(records)")}
        if "payload" not in columns:
            self.db.execute("ALTER TABLE records ADD COLUMN payload INTEGER")
        self.db.execute("CREATE INDEX IF NOT EXISTS records_payload ON records (payload)")
        rows = self.db.execute("SELECT serial, signed_json FROM records WHERE payload IS NULL").fetchall()
        for row in rows:
            try:
                payload = _payload_of(json.loads(row["signed_json"]))
            except (KeyError, TypeError, ValueError):
                continue
            self.db.execute("UPDATE records SET payload = ? WHERE serial = ?", (payload, row["serial"]))

    @contextmanager
    def _tx(self):
        """One write transaction under the lock. The connection autocommits, so `with conn:` alone opens nothing and a
        failure half-way would leave its earlier statements in; the lock keeps the ledger chain in order."""
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

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

    # ---- accounts and invites (the operator's, by command line) ----

    def create_account(self, name: str) -> int:
        with self._tx():
            cur = self.db.execute("INSERT INTO accounts (name, created_at) VALUES (?, ?)", (name, now_utc()))
            self._ledger("account", None, None, None, {"account": name})
            return cur.lastrowid

    def create_invite(self, account_name: str, role: str = "workshop") -> str:
        """An invite fixes the role the machine will have: a Workshop publishes and withdraws; a Space reports serving;
        a warden's machine appends the Guild's strikes and manual entries to the log."""
        if role not in ROLES:
            raise ValueError(f"role is one of {ROLES}")
        with self._tx():
            row = self.db.execute("SELECT id FROM accounts WHERE name = ?", (account_name,)).fetchone()
            if not row:
                raise KeyError(f"no account {account_name!r}")
            code = secrets.token_urlsafe(18)
            self.db.execute("INSERT INTO invites (code, account_id, role, created_at) VALUES (?, ?, ?, ?)", (code, row["id"], role, now_utc()))
            return code

    # ---- machines ----

    def enrol(self, invite: str, key_id: str, name: str, public_pem: str) -> dict:
        with self._tx():
            inv = self.db.execute("SELECT * FROM invites WHERE code = ?", (invite,)).fetchone()
            if not inv or inv["used_by"]:
                raise PermissionError("the invite is unknown or already used")
            if self.db.execute("SELECT 1 FROM machines WHERE key_id = ?", (key_id,)).fetchone():
                raise PermissionError("this machine key is already enrolled")
            at = now_utc()
            self.db.execute("INSERT INTO machines (key_id, account_id, role, name, public_pem, enrolled_at) VALUES (?, ?, ?, ?, ?, ?)",
                            (key_id, inv["account_id"], inv["role"], name, public_pem, at))
            self.db.execute("UPDATE invites SET used_by = ?, used_at = ? WHERE code = ?", (key_id, at, invite))
            account = self.db.execute("SELECT name FROM accounts WHERE id = ?", (inv["account_id"],)).fetchone()["name"]
            # The public ledger names the key and the account; the machine's own name stays in the machines table.
            self._ledger("enrol", None, None, key_id, {"account": account, "role": inv["role"]})
            return {"machine_key_id": key_id, "account": account, "role": inv["role"], "enrolled_at": at}

    def nonce_answer(self, key_id: str, nonce: str, body_sha256: str) -> dict | None:
        """A body with a nonce is taken once: the same body again gets its first answer back; a different body
        under a taken nonce is a replay and refused. None means the nonce is new and the caller does the work."""
        if not isinstance(nonce, str) or not 8 <= len(nonce) <= 64:
            raise PermissionError("the body carries no usable nonce")
        with self._lock:
            row = self.db.execute("SELECT body_sha256, answer_json FROM nonces WHERE key_id = ? AND nonce = ?", (key_id, nonce)).fetchone()
        if row is None:
            return None
        if row["body_sha256"] != body_sha256:
            raise PermissionError("this nonce was already taken for another body; a replay")
        return json.loads(row["answer_json"]) if row["answer_json"] else {"taken": True}

    def record_nonce(self, key_id: str, nonce: str, body_sha256: str, answer: dict) -> None:
        with self._tx():
            self.db.execute("INSERT OR REPLACE INTO nonces (key_id, nonce, body_sha256, answer_json, at) VALUES (?, ?, ?, ?, ?)",
                            (key_id, nonce, body_sha256, json.dumps(answer, sort_keys=True), now_utc()))

    def machine(self, key_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM machines WHERE key_id = ? AND revoked_at IS NULL", (key_id,)).fetchone()
        return dict(row) if row else None

    def revoke_machine(self, key_id: str, reason: str) -> None:
        with self._tx():
            self.db.execute("UPDATE machines SET revoked_at = ? WHERE key_id = ?", (now_utc(), key_id))
            self._ledger("revoke", None, None, key_id, {"reason": reason})

    # ---- records ----

    def _next_serial(self) -> int:
        row = self.db.execute("SELECT MAX(serial) AS m FROM records").fetchone()
        # A serial the log already admitted is taken even when its row was lost to a restore.
        return max(FIRST_SERIAL, (row["m"] or 0) + 1, self.log.max_serial() + 1)

    def register(self, signed: dict, machine_key_id: str, account_id: int, countersign) -> tuple:
        """The next serial, the countersignature and the row, as one transaction: (serial, counter, already).
        `countersign(serial, at)` signs; a record the log already admits keeps that admit's serial and time."""
        with self._tx():
            have = self.db.execute("SELECT serial, counter_json FROM records WHERE record_sha256 = ?",
                                   (signed["record_sha256"],)).fetchone()
            if have:
                return have["serial"], json.loads(have["counter_json"]), True
            if self.log.blocked:
                raise LogBlocked(self.log.blocked)
            admitted = self.log.admitted(signed["record_sha256"])
            if admitted is not None:
                return self._register_admitted(signed, machine_key_id, account_id, countersign, admitted)
            serial = self._next_serial()
            counter = countersign(serial, None)
            self.db.execute(
                "INSERT INTO records (serial, record_sha256, machine_key_id, account_id, signed_json, counter_json, state,"
                " published_at, payload) VALUES (?, ?, ?, ?, ?, ?, 'registered', ?, ?)",
                (serial, signed["record_sha256"], machine_key_id, account_id, json.dumps(signed, sort_keys=True),
                 json.dumps(counter, sort_keys=True), counter["countersigned_at_utc"], _payload_of(signed)))
            self._ledger("publish", serial, signed["record_sha256"], machine_key_id,
                         {"artist_id": signed["record"]["artist_id"], "public_view": signed["record"]["public_view"]})
            self.log.admit_locked(serial, signed["record_sha256"], counter, time=counter["countersigned_at_utc"])
            return serial, counter, False

    def _register_admitted(self, signed: dict, machine_key_id: str, account_id: int, countersign, admitted: dict) -> tuple:
        """A record the log admitted before this database lost it: its row back under the same serial, nothing appended."""
        serial = int(admitted["serial"])
        counter = countersign(serial, admitted["countersignature"]["countersigned_at_utc"])
        struck = set(self.log.admit_states(signed["record_sha256"]).values()) == {"struck"}
        self.db.execute(
            "INSERT INTO records (serial, record_sha256, machine_key_id, account_id, signed_json, counter_json, state,"
            " published_at, payload, withdrawn_at, withdraw_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (serial, signed["record_sha256"], machine_key_id, account_id, json.dumps(signed, sort_keys=True),
             json.dumps(counter, sort_keys=True), "withdrawn" if struck else "registered", counter["countersigned_at_utc"],
             _payload_of(signed), now_utc() if struck else None, "struck in the log" if struck else None))
        self._ledger("publish", serial, signed["record_sha256"], machine_key_id,
                     {"artist_id": signed["record"]["artist_id"], "public_view": signed["record"]["public_view"],
                      "restored_from_log": True})
        return serial, counter, False

    def record(self, serial: int) -> dict | None:
        row = self.db.execute("SELECT * FROM records WHERE serial = ?", (serial,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["signed"] = json.loads(out.pop("signed_json"))
        out["countersignature"] = json.loads(out.pop("counter_json"))
        out["served"] = [dict(r) for r in self.db.execute("SELECT * FROM served WHERE serial = ? ORDER BY at", (serial,))]
        mach = self.db.execute("SELECT public_pem FROM machines WHERE key_id = ?", (out["machine_key_id"],)).fetchone()
        out["machine_public_pem"] = mach["public_pem"] if mach else None
        return out

    def serial_of_record(self, record_sha256: str) -> int | None:
        row = self.db.execute("SELECT serial FROM records WHERE record_sha256 = ?", (record_sha256,)).fetchone()
        return int(row["serial"]) if row else None

    def withdraw(self, serial: int, key_id: str, reason: str) -> dict:
        with self._tx():
            row = self.db.execute("SELECT * FROM records WHERE serial = ?", (serial,)).fetchone()
            if not row:
                raise KeyError(f"no record {serial}")
            mach = self.machine(key_id)
            if not mach or mach["account_id"] != row["account_id"]:
                raise PermissionError("only a machine of the account that published a model may withdraw it")
            if row["state"] == "withdrawn":
                return {"serial": serial, "state": "withdrawn", "withdrawn_at": row["withdrawn_at"]}
            at = now_utc()
            self.db.execute("UPDATE records SET state = 'withdrawn', withdrawn_at = ?, withdraw_reason = ? WHERE serial = ?",
                            (at, reason[:400], serial))
            self._ledger("withdraw", serial, row["record_sha256"], key_id, {"reason": reason[:400]})
            if self.log.admit_leaf_of(row["record_sha256"]) is None:
                raise RuntimeError(f"record {serial} has no admit entry in the log")
            for leaf in [k for k, v in self.log.admit_states(row["record_sha256"]).items() if v != "struck"]:
                self.log.revoke_locked(leaf, reason[:400].strip() or "withdrawn by the account that published it", time=at)
            return {"serial": serial, "state": "withdrawn", "withdrawn_at": at}

    def served(self, serial: int, key_id: str, space_url: str, state: str) -> dict:
        with self._tx():
            mach = self.machine(key_id)
            if not mach or mach["role"] != "space":
                raise PermissionError("only an enrolled Space reports where a model is served")
            row = self.db.execute("SELECT record_sha256 FROM records WHERE serial = ?", (serial,)).fetchone()
            if not row:
                raise KeyError(f"no record {serial}")
            at = now_utc()
            self.db.execute("INSERT INTO served (serial, space_url, state, at, key_id) VALUES (?, ?, ?, ?, ?)"
                            " ON CONFLICT(serial, space_url) DO UPDATE SET state = excluded.state, at = excluded.at, key_id = excluded.key_id",
                            (serial, space_url, state, at, key_id))
            self._ledger("served", serial, row["record_sha256"], key_id, {"space_url": space_url, "state": state})
            return {"serial": serial, "space_url": space_url, "state": state, "at": at}

    def serials_in_bucket(self, payload: int) -> list:
        """Every registered, not withdrawn model whose watermark bucket is this one."""
        return [r["serial"] for r in self.db.execute(
            "SELECT serial FROM records WHERE payload = ? AND state != 'withdrawn' ORDER BY serial", (int(payload),))]

    def unresolved_serials(self) -> list:
        """Registered models the log holds unresolved after a recovery: not served until a warden answers."""
        shas = self.log.unresolved_records()
        return sorted(r["serial"] for r in self.db.execute("SELECT serial, record_sha256 FROM records").fetchall()
                      if r["record_sha256"] in shas)

    def withdrawn_serials(self) -> list:
        return [r["serial"] for r in self.db.execute("SELECT serial FROM records WHERE state = 'withdrawn' ORDER BY serial")]

    def serials(self) -> list:
        return [r["serial"] for r in self.db.execute("SELECT serial FROM records ORDER BY serial")]

    def record_key(self, key_id: str, public_pem: str, present: list, note: str = "") -> None:
        """The ceremony's entry: which key countersigns from now on, and who was there when it was made."""
        with self._tx():
            self._ledger("key", None, None, key_id, {"public_key_sha256": sha256_hex(public_pem.encode("utf-8")),
                                                     "present": [str(p)[:80] for p in present][:12], "note": note[:400]})

    # ---- the log ----

    def log_open(self, witnesses: list | None = None, unwitnessed: bool = False, started: bool = False) -> int:
        """Checks the database against the log it published; then the first entries, once, and a heartbeat under a new
        log key. A database no published copy vouches for is blocked, never appended to."""
        with self._tx():
            self.log.remember_key_locked()
        self.log.blocked = self.log.open_verdict(self.log.witness(witnesses or [], self.log.signer.public_pem()), unwitnessed, started)
        if self.log.blocked:
            return 0
        with self._tx():
            n = self.log.genesis_locked()
            self.log.resign_locked()
            return n

    def log_append(self, entry: dict, key_id: str) -> dict:
        """A warden's entry: an admit of a registered record, a revoke or a withhold, each naming its sitting and
        signed into the leaf with the machine that appended it; the ledger gets the row in the same transaction."""
        kind = entry.get("kind")
        with self._tx():
            if kind == "admit":
                row = self.db.execute("SELECT serial, record_sha256, counter_json, state FROM records WHERE record_sha256 = ?",
                                      (str(entry.get("record_sha256") or ""),)).fetchone()
                if row is None:
                    raise KeyError("no record is registered under that record_sha256")
                if row["state"] == "withdrawn":
                    raise ValueError("that record is withdrawn; it is not admitted again")
                if self.log.standing_admits(row["record_sha256"]):
                    raise ValueError("that record already has an admit no revoke has struck")
                out = self.log.admit_locked(row["serial"], row["record_sha256"], json.loads(row["counter_json"]),
                                            wardens=entry.get("wardens"), sitting=entry.get("sitting") or "", appended_by=key_id)
            elif kind == "revoke":
                out = self.log.revoke_locked(str(entry.get("revokes") or ""), entry.get("reason"), wardens=entry.get("wardens"),
                                             sitting=entry.get("sitting") or "", appended_by=key_id)
            elif kind == "confirm":
                out = self.log.confirm_locked(str(entry.get("confirms") or ""), wardens=entry.get("wardens"),
                                              sitting=entry.get("sitting") or "", appended_by=key_id)
            elif kind == "withhold":
                out = self.log.withhold_locked(str(entry.get("withholds") or ""), entry.get("reason"), wardens=entry.get("wardens"),
                                               sitting=entry.get("sitting") or "", appended_by=key_id)
            else:
                raise ValueError("kind is admit, revoke, confirm or withhold")
            self._ledger("log-append", None, None, key_id, {"kind": kind, "leaf_hash": out["leaf_hash"], "index": out["index"]})
            return out

    def log_withhold(self, leaf_hex: str, reason: str) -> dict:
        """The operator's withhold: the body leaves public view, the withholding is itself an entry."""
        with self._tx():
            return self.log.withhold_locked(leaf_hex, reason)

    # ---- the ledger ----

    def _ledger(self, kind: str, serial, record_sha256, key_id, detail: dict) -> None:
        prev = self.db.execute("SELECT entry_sha256 FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        prev_hash = prev["entry_sha256"] if prev else None
        at = now_utc()
        body = {"kind": kind, "serial": serial, "record_sha256": record_sha256, "key_id": key_id, "at": at,
                "detail": detail, "prev_sha256": prev_hash}
        entry_hash = sha256_hex(canonical_bytes(body))
        self.db.execute("INSERT INTO ledger (kind, serial, record_sha256, key_id, at, detail_json, entry_sha256, prev_sha256)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (kind, serial, record_sha256, key_id, at, json.dumps(detail, sort_keys=True), entry_hash, prev_hash))

    def ledger(self, since_seq: int = 0) -> list:
        rows = self.db.execute("SELECT * FROM ledger WHERE seq > ? ORDER BY seq", (since_seq,))
        out = []
        for r in rows:
            d = dict(r)
            d["detail"] = json.loads(d.pop("detail_json"))
            out.append(d)
        return out

    def ledger_ok(self) -> bool:
        """Every entry's hash is over its own body and names the entry before it: the chain is unbroken. Always the
        whole chain: a remembered tip would miss an entry changed behind it."""
        prev = None
        for e in self.ledger():
            body = {"kind": e["kind"], "serial": e["serial"], "record_sha256": e["record_sha256"], "key_id": e["key_id"],
                    "at": e["at"], "detail": e["detail"], "prev_sha256": prev}
            if e["prev_sha256"] != prev or e["entry_sha256"] != sha256_hex(canonical_bytes(body)):
                return False
            prev = e["entry_sha256"]
        return True

    def ledger_status(self, max_age_s: float = 60.0) -> dict:
        """The public pages' reading: the last whole check, repeated at most once a minute, since those pages take
        unlimited calls and the chain grows."""
        now = time.monotonic()
        held = self._ledger_status
        if held is None or now - held["at"] > max_age_s:
            self._ledger_status = held = {"ok": self.ledger_ok(), "at": now, "checked_at_utc": now_utc()}
        return {"ok": held["ok"], "checked_at_utc": held["checked_at_utc"]}
