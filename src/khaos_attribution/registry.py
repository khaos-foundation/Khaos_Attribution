"""The Guild Register's contract: machine keys, the record a Workshop signs at
Publish, the register's countersignature, the outbox that survives being
offline, and the client. The service itself lives in register_service/.
Ed25519 through `cryptography` (the `registry` extra); nothing else here
needs it, so the module imports it lazily.
"""

from __future__ import annotations

import hashlib
import json

from khaos_attribution.base_model import is_base_model_id
import os
import base64
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

RECORD_VERSION = "guild.register.record/1"
SIGNED_VERSION = "guild.register.signed/1"
FIRST_SERIAL = 65536   # above every 16-bit codeword a seal struck before the register carried
PUBLIC_VIEWS = ("model", "tracks", "writers")
OUTBOX_DIR = ".register_outbox"
MACHINE_KEY_FILE = ".register_machine_key.pem"
MACHINE_KEY_PUBLIC_FILE = ".register_machine_key.pub"   # the key file's stem with .pub


def canonical_bytes(doc: dict) -> bytes:
    """One byte form for one document, so a signature over it means one thing everywhere."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ed25519():
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
    from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: PLC0415
    return ed25519, serialization


# ---- keys ----

@dataclass
class SigningKey:
    """An Ed25519 key with its id: the first 16 hex characters of the sha256 of the raw public key."""
    _private: object
    key_id: str

    @classmethod
    def generate(cls) -> "SigningKey":
        ed25519, _ = _ed25519()
        priv = ed25519.Ed25519PrivateKey.generate()
        return cls(priv, key_id_of(public_raw(priv.public_key())))

    @classmethod
    def load(cls, path: Path) -> "SigningKey":
        _, serialization = _ed25519()
        priv = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
        return cls(priv, key_id_of(public_raw(priv.public_key())))

    @classmethod
    def load_or_create(cls, path: Path) -> "SigningKey":
        """A machine makes its key once; the private half never leaves the file."""
        path = Path(path)
        if path.is_file():
            return cls.load(path)
        key = cls.generate()
        _, serialization = _ed25519()
        pem = key._private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                         serialization.NoEncryption())
        tmp = path.with_suffix(".new")
        tmp.write_bytes(pem)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
        path.with_suffix(".pub").write_text(key.public_pem(), encoding="utf-8")
        return key

    def public_pem(self) -> str:
        _, serialization = _ed25519()
        return self._private.public_key().public_bytes(serialization.Encoding.PEM,
                                                       serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii")

    def sign(self, data: bytes) -> str:
        return self._private.sign(data).hex()


def public_raw(public_key) -> bytes:
    _, serialization = _ed25519()
    return public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def key_id_of(raw_public: bytes) -> str:
    return sha256_hex(raw_public)[:16]


def load_public_pem(pem: str):
    _, serialization = _ed25519()
    return serialization.load_pem_public_key(pem.encode("ascii"))


def verify(pem: str, data: bytes, signature_hex: str) -> bool:
    """True when the signature is the key's over the bytes; a bad signature is False, never an exception."""
    try:
        load_public_pem(pem).verify(bytes.fromhex(signature_hex), data)
        return True
    except Exception:  # noqa: BLE001 — every failure means the same thing here
        return False


# ---- the record ----

REQUIRED_RECORD_KEYS = (
    "record_version", "artist_id", "artist_name", "artist_mark", "consent", "model", "provenance",
    "training_set", "measured", "public_view", "published_at_utc",
)


def _is_sha256(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def record_problems(record: dict) -> list:
    """What keeps a record out of the register, in words; empty means it may be signed.
    Every field the public page and the views read is checked here, so a numbered record never breaks them."""
    from khaos_attribution.watermark import DERIVED_PAYLOADS  # noqa: PLC0415
    out = []
    if not isinstance(record, dict):
        return ["the record is not an object"]
    for k in REQUIRED_RECORD_KEYS:
        if k not in record:
            out.append(f"missing {k}")
    if out:
        return out
    if record["record_version"] != RECORD_VERSION:
        out.append(f"record_version is {record['record_version']!r}, not {RECORD_VERSION!r}")
    if record["public_view"] not in PUBLIC_VIEWS:
        out.append(f"public_view must be one of {PUBLIC_VIEWS}")
    for k in ("artist_id", "artist_name", "artist_mark", "published_at_utc"):
        if not isinstance(record[k], str) or not record[k].strip():
            out.append(f"{k} must be a non-empty string")
    for k in ("consent", "model", "provenance", "training_set", "measured"):
        if not isinstance(record[k], dict):
            out.append(f"{k} must be an object")
    if out:
        return out
    model = record["model"]
    for k in ("run_id", "base_model", "base_model_licence", "machine_key_id"):
        if not isinstance(model.get(k), str) or not model[k].strip():
            out.append(f"model.{k} must be a non-empty string")
    for k in ("adapter_sha256", "card_sha256"):
        if not _is_sha256(model.get(k)):
            out.append(f"model.{k} must be a sha256 hex digest")
    payload = model.get("watermark_payload")
    if not isinstance(payload, int) or isinstance(payload, bool) or payload not in DERIVED_PAYLOADS:
        out.append(f"model.watermark_payload must be a derived payload, {DERIVED_PAYLOADS.start} to {DERIVED_PAYLOADS.stop - 1}")
    for k in ("base_model_commit", "workshop_commit", "engine_commit", "trained_at_utc", "card_schema_version"):
        if k in model and model[k] is not None and not isinstance(model[k], str):
            out.append(f"model.{k} must be a string or null")
    if "base_model_id" in model and model["base_model_id"] is not None and not is_base_model_id(model["base_model_id"]):
        out.append("model.base_model_id must be <engine>@<variant>")
    consent = record["consent"]
    if not _is_sha256(consent.get("statement_sha256")):
        out.append("consent.statement_sha256 must be a sha256 hex digest")
    if "scope" in consent and not isinstance(consent["scope"], dict):
        out.append("consent.scope must be an object")
    prov = record["provenance"]
    if not isinstance(prov.get("dataset_hash"), str) or not prov["dataset_hash"]:
        out.append("provenance.dataset_hash must be a string")
    if "config_hash" in prov and prov["config_hash"] is not None and not isinstance(prov["config_hash"], str):
        out.append("provenance.config_hash must be a string or null")
    if "seed" in prov and prov["seed"] is not None and (not isinstance(prov["seed"], int) or isinstance(prov["seed"], bool)):
        out.append("provenance.seed must be an integer or null")
    ts = record["training_set"]
    tracks = ts.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        out.append("training_set.tracks must name at least one track")
    else:
        for i, t in enumerate(tracks):
            if not isinstance(t, dict) or not isinstance(t.get("track_id"), str) or not t["track_id"]:
                out.append(f"training_set.tracks[{i}] must be an object with a track_id")
                continue
            if "title" in t and t["title"] is not None and not isinstance(t["title"], str):
                out.append(f"training_set.tracks[{i}].title must be a string or null")
            if "duration_sec" in t and t["duration_sec"] is not None and not isinstance(t["duration_sec"], (int, float)):
                out.append(f"training_set.tracks[{i}].duration_sec must be a number or null")
            writers = t.get("writers", [])
            if not isinstance(writers, list):
                out.append(f"training_set.tracks[{i}].writers must be a list")
                continue
            for j, w in enumerate(writers):
                if (not isinstance(w, dict) or not isinstance(w.get("name"), str) or not w["name"]
                        or not isinstance(w.get("share"), (int, float)) or isinstance(w.get("share"), bool)):
                    out.append(f"training_set.tracks[{i}].writers[{j}] must have a name and a numeric share")
    if "rights_sha256" in ts and ts["rights_sha256"] is not None and not _is_sha256(ts["rights_sha256"]):
        out.append("training_set.rights_sha256 must be a sha256 hex digest or null")
    decl = ts.get("declaration")
    if decl is not None:
        if (not isinstance(decl, dict) or not isinstance(decl.get("layers"), list) or not decl["layers"]
                or not all(isinstance(x, str) and x for x in decl["layers"])
                or any(not isinstance(decl.get(k), int) or isinstance(decl.get(k), bool) for k in ("declared", "of"))
                or not 0 <= decl["declared"] <= decl["of"]):
            out.append("training_set.declaration must name its layers and count declared of the tracks")
    kinds = ts.get("data_kinds")
    if kinds is not None and (not isinstance(kinds, list) or not all(isinstance(k, str) and k for k in kinds)):
        out.append("training_set.data_kinds must be a list of words")
    if record["public_view"] == "writers" and ts.get("writers_agreed") is not True:
        out.append("public_view 'writers' needs training_set.writers_agreed: true, every named writer having agreed")
    return out


def sign_record(record: dict, key: SigningKey) -> dict:
    """The Workshop's signature over the canonical record; refused with the problems when the record is not whole."""
    problems = record_problems(record)
    if problems:
        raise ValueError("; ".join(problems))
    body = canonical_bytes(record)
    return {"signed_version": SIGNED_VERSION, "record": record, "record_sha256": sha256_hex(body),
            "machine_key_id": key.key_id, "machine_signature": key.sign(body)}


def signed_problems(signed: dict, machine_public_pem: str) -> list:
    """What keeps a signed record out: the hash not matching the record, or the signature not the key's."""
    out = []
    if signed.get("signed_version") != SIGNED_VERSION:
        out.append("not a signed record")
        return out
    if not isinstance(signed.get("record"), dict):
        out.append("record must be an object")
        return out
    body = canonical_bytes(signed["record"])
    if signed.get("record_sha256") != sha256_hex(body):
        out.append("record_sha256 does not match the record")
    if key_id_of(public_raw(load_public_pem(machine_public_pem))) != signed.get("machine_key_id"):
        out.append("machine_key_id is not this key's")
    if not verify(machine_public_pem, body, signed.get("machine_signature", "")):
        out.append("machine_signature is not this key's over this record")
    return out


def countersign(signed: dict, serial: int, register_key: SigningKey, at_utc: str) -> dict:
    """The register's answer: the serial and its signature over the record hash, the serial and the time."""
    if serial < FIRST_SERIAL:
        raise ValueError(f"a serial is {FIRST_SERIAL} or more")
    payload = {"record_sha256": signed["record_sha256"], "serial": serial, "countersigned_at_utc": at_utc}
    return {**payload, "register_key_id": register_key.key_id, "register_signature": register_key.sign(canonical_bytes(payload))}


def countersignature_valid(counter: dict, record_sha256: str, register_public_pem: str) -> bool:
    """True when the register's signature covers this record's hash, this serial and this time."""
    try:
        payload = {"record_sha256": counter["record_sha256"], "serial": int(counter["serial"]),
                   "countersigned_at_utc": counter["countersigned_at_utc"]}
    except (KeyError, TypeError, ValueError):
        return False
    if payload["record_sha256"] != record_sha256 or payload["serial"] < FIRST_SERIAL:
        return False
    return verify(register_public_pem, canonical_bytes(payload), counter.get("register_signature", ""))


# ---- the outbox ----

_OUTBOX_LOCKS: dict = {}
_OUTBOX_LOCKS_GUARD = threading.Lock()


def _outbox_lock(directory: Path) -> threading.Lock:
    """One lock per outbox folder in this process: the rooms append from request threads and background workers,
    and the sequence number is read from the folder."""
    key = str(Path(directory).resolve())
    with _OUTBOX_LOCKS_GUARD:
        return _OUTBOX_LOCKS.setdefault(key, threading.Lock())


class Outbox:
    """Records a machine made while offline, kept in order until the register has taken each one."""

    def __init__(self, root: Path):
        self.dir = Path(root) / OUTBOX_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = _outbox_lock(self.dir)

    def append(self, kind: str, body: dict) -> str:
        entry = {"kind": kind, "body": body, "created_at": time.time()}
        entry_id = sha256_hex(canonical_bytes({"kind": kind, "body": body}))
        with self._lock:
            return self._append_locked(kind, entry, entry_id)

    def _append_locked(self, kind: str, entry: dict, entry_id: str) -> str:
        existing = sorted(self.dir.glob("*.json")) + sorted((self.dir / "sent").glob("*.json"))
        if any(p.name.endswith(f"-{entry_id[:12]}.json") for p in existing):
            return entry_id   # the same record twice is one entry
        # Order is the sequence number, never the clock: two appends in one tick still keep their order.
        seq = max((int(p.name.split("-", 1)[0]) for p in existing if p.name.split("-", 1)[0].isdigit()), default=0) + 1
        path = self.dir / f"{seq:08d}-{entry_id[:12]}.json"
        tmp = path.with_suffix(".new")
        tmp.write_text(json.dumps({"id": entry_id, **entry}, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return entry_id

    def noted(self, key: str) -> bool:
        """Whether `note(key)` was called: a marker, so a caller need not read the entries to know what it did."""
        return (self.dir / "noted" / sha256_hex(key.encode("utf-8"))).is_file()

    def note(self, key: str) -> None:
        marks = self.dir / "noted"
        marks.mkdir(exist_ok=True)
        (marks / sha256_hex(key.encode("utf-8"))).write_bytes(b"")

    def pending(self) -> list:
        """Entries still to send. One also in `sent/` was taken and its move interrupted: it is finished, not resent."""
        sent = {p.name for p in (self.dir / "sent").glob("*.json")}
        out = []
        for p in sorted(self.dir.glob("*.json")):
            if p.name in sent:
                p.unlink(missing_ok=True)
                continue
            try:
                out.append((p, json.loads(p.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
        return out

    def mark_sent(self, path: Path, answer: dict) -> None:
        self._move(path, "sent", answer)

    def refuse(self, path: Path, answer: dict) -> None:
        """An entry the register refused for what it is (not for being unreachable): set aside, never retried."""
        self._move(path, "refused", answer)

    def sent_entries(self) -> list:
        out = []
        for p in sorted((self.dir / "sent").glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out

    def refused(self) -> list:
        out = []
        for p in sorted((self.dir / "refused").glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out

    def drop(self, entry_id: str) -> bool:
        """A pending entry withdrawn before it was ever sent: gone, as if never written."""
        for p in self.dir.glob("*.json"):
            if p.name.endswith(f"-{entry_id[:12]}.json"):
                p.unlink()
                return True
        return False

    def _move(self, path: Path, folder: str, answer: dict) -> None:
        dest = self.dir / folder
        dest.mkdir(exist_ok=True)
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        doc["answer"] = answer
        (dest / Path(path).name).write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
        Path(path).unlink()


# ---- the client ----

class RegisterError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status


class RegisterClient:
    """The calls a machine makes; every one carries the machine's signature over its body."""

    def __init__(self, base_url: str, key: SigningKey, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.timeout = timeout

    def _get(self, path: str) -> dict:
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415
        try:
            with urllib.request.urlopen(self.base_url + path, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            raise RegisterError(e.code, (e.read().decode("utf-8", "replace") or e.reason)[:400]) from None

    def public_key(self) -> dict:
        """The register's key id, public PEM and whether it is a development key; pin it at enrolment."""
        return self._get("/api/public-key")

    def model(self, serial: int) -> dict:
        return self._get(f"/api/models/{int(serial)}")

    def withdrawn(self) -> list:
        return list(self._get("/api/withdrawn").get("withdrawn") or [])

    def _post(self, path: str, body: dict) -> dict:
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415
        data = canonical_bytes(body)
        req = urllib.request.Request(self.base_url + path, data=data, method="POST", headers={
            "Content-Type": "application/json", "X-Machine-Key-Id": self.key.key_id,
            "X-Machine-Signature": self.key.sign(data)})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            raise RegisterError(e.code, (e.read().decode("utf-8", "replace") or e.reason)[:400]) from None

    def enrol(self, invite: str, machine_name: str) -> dict:
        return self._post("/api/enrol", {"invite": invite, "machine_name": machine_name,
                                         "public_key_pem": self.key.public_pem(), "nonce": secrets.token_hex(8)})

    def publish(self, signed: dict) -> dict:
        return self._post("/api/records", signed)

    def withdraw(self, serial: int, reason: str, nonce: str | None = None) -> dict:
        """An outbox entry carries its nonce from the day it was written, so a retry is the same body, not a replay."""
        return self._post("/api/withdraw", {"serial": serial, "reason": reason, "nonce": nonce or secrets.token_hex(8)})

    def release(self, generation_id: str, serial: int, fingerprint: list, nonce: str | None = None,
                manifest: bytes | str | None = None) -> dict:
        """A released output's fingerprint into the index under its model's bucket, with its Content Credentials
        manifest store when the file carries one, so a stripped copy can be led back to it."""
        body = {"generation_id": generation_id, "serial": serial, "fingerprint": fingerprint,
                "nonce": nonce or secrets.token_hex(8)}
        if manifest:
            # An outbox entry already holds it as base64 text.
            body["manifest"] = manifest if isinstance(manifest, str) else base64.b64encode(manifest).decode()
        return self._post("/api/outputs", body)

    def remove_output(self, generation_id: str, nonce: str | None = None) -> dict:
        """A released piece its maker took down; only the machine that released it may say so."""
        return self._post("/api/outputs/remove", {"generation_id": generation_id, "nonce": nonce or secrets.token_hex(8)})

    def claim(self, record_sha256: str, track_id: str, fingerprint: list | None = None, declaration: str = "",
              nonce: str | None = None, phases: list | None = None) -> dict:
        """A registered model's training track, claimed on the account's declaration; a recording may train many models.
        With `phases` the fingerprint is their first, so it need not be sent twice."""
        body = {"record_sha256": record_sha256, "track_id": track_id, "declaration": declaration, "nonce": nonce or secrets.token_hex(8)}
        if fingerprint is not None:
            body["fingerprint"] = fingerprint
        if phases is not None:
            body["phases"] = phases
        return self._post("/api/claims", body)

    def check_claim(self, phases: list) -> dict:
        """Whether a recording already trains a registered model (grade none, mine or other) and the serial of the
        released output it is, if any."""
        return self._post("/api/claims/check", {"phases": phases, "nonce": secrets.token_hex(8)})

    def verify(self, phases: list, watermark_id: int | None = None) -> dict:
        """Anyone's question: the four-phase fingerprint and the codeword when it read. No key."""
        return self._post_public("/api/verify", {"phases": phases, "watermark_id": watermark_id})

    def _post_public(self, path: str, body: dict) -> dict:
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415
        req = urllib.request.Request(self.base_url + path, data=canonical_bytes(body), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=max(self.timeout, 60)) as r:
                return json.loads(r.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            raise RegisterError(e.code, (e.read().decode("utf-8", "replace") or e.reason)[:400]) from None

    def log_append(self, entry: dict, nonce: str | None = None) -> dict:
        """A warden's entry in the signed log: a revoke, admit or withhold naming its sitting; returns its leaf and index."""
        return self._post("/log/append", {"entry": entry, "nonce": nonce or secrets.token_hex(8)})

    def served(self, serial: int, space_url: str, state: str, nonce: str | None = None) -> dict:
        return self._post("/api/served", {"serial": serial, "space_url": space_url, "state": state, "nonce": nonce or secrets.token_hex(8)})

    def sync(self, outbox: Outbox, register_public_pem: str | None = None, on_answer=None) -> list:
        """Every pending entry, in order; stops at the first the register refuses so order is kept.
        With the register's pinned key, a publish answer whose countersignature is not that key's is refused
        before it is marked sent; `on_answer(entry, answer)` runs for each accepted answer."""
        done = []
        for path, entry in outbox.pending():
            kind, body = entry["kind"], entry["body"]
            try:
                answer = {"publish": self.publish, "withdraw": lambda b: self.withdraw(**b),
                          "served": lambda b: self.served(**b), "release": lambda b: self.release(**b),
                          "remove_output": lambda b: self.remove_output(**b),
                          "claim": lambda b: self.claim(**b)}[kind](body)
            except RegisterError as e:
                # Not enrolled, a replay of another body, too many calls this minute, or the register itself failing:
                # stop and keep order.
                # A refusal of this entry for what it is (wrong account, no such serial, a record it will not take)
                # is set aside so the entries behind it are not held for ever.
                if e.status in (401, 409, 429) or e.status >= 500:
                    raise
                outbox.refuse(path, {"status": e.status, "message": str(e)})
                continue
            if kind == "publish" and register_public_pem is not None:
                if not countersignature_valid(answer.get("countersignature") or {}, body["record_sha256"], register_public_pem):
                    raise RegisterError(502, "the register's countersignature is not the pinned key's; nothing was marked sent")
            if on_answer is not None:
                on_answer(entry, answer)
            outbox.mark_sent(path, answer)
            done.append((entry["id"], answer))
        return done
