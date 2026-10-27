"""The register's web face: signed calls for machines, public reads for everyone, the signed append-only log.
Settings by environment: REGISTER_DATA_DIR (store, mirror), REGISTER_KEY_FILE (the
register's private key; REGISTER_DEV_KEY=1 lets a missing one be made, for development
only), REGISTER_LOG_KEY_FILE and REGISTER_LOG_DEV_KEY (the same for the log's own key),
REGISTER_PUBLIC_URL.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from importlib import resources
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from khaos_attribution import registry as R
from khaos_attribution import translog as T
from khaos_attribution.watermark import decode_codeword
from register_service import softbinding
from register_service.index import Index, phases_of
from register_service.store import Store, now_utc
from register_service.translog import EntryRefused, LogBlocked, export as export_log

MAX_BODY = 8_000_000   # a long output's four-phase fingerprint runs to a few megabytes
MANIFEST_MAX_BYTES = 1_000_000   # a piece's own manifest store is kilobytes
CLAIM_MAX_BODY = 24_000_000   # a twelve-minute track's four phases, with room
DEFAULT_PORT = 8912   # 8900 is the Engine Room's
STATIC_CACHE = {"Cache-Control": "public, max-age=86400"}
log = logging.getLogger("register_service")


def settings() -> dict:
    data = Path(os.environ.get("REGISTER_DATA_DIR", "./register-data")).resolve()
    key_file = Path(os.environ.get("REGISTER_KEY_FILE", str(data / "register-key.pem")))
    dev_key = os.environ.get("REGISTER_DEV_KEY") == "1"
    return {"data": data, "key_file": key_file, "dev_key": dev_key,
            "log_key_file": Path(os.environ.get("REGISTER_LOG_KEY_FILE", str(data / "log-key.pem"))),
            # The log key has its own ceremony, so its development setting is its own; unset, it follows the register's.
            "log_dev_key": os.environ.get("REGISTER_LOG_DEV_KEY", "1" if dev_key else "0") == "1",
            "log_public_delay_days": float(os.environ.get("REGISTER_LOG_PUBLIC_DELAY_DAYS", "7")),
            "log_witnesses": [w.strip() for w in os.environ.get("REGISTER_LOG_WITNESSES", PUBLIC_MIRROR_RAW).split(",") if w.strip()],
            # A development key reachable from outside, by decision: the deploy sets it beside REGISTER_DEV_KEY.
            "dev_key_public": os.environ.get("REGISTER_DEV_KEY_PUBLIC") == "1",
            "public_url": os.environ.get("REGISTER_PUBLIC_URL", f"http://127.0.0.1:{DEFAULT_PORT}").rstrip("/")}


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
PUBLIC_MIRROR_RAW = "https://raw.githubusercontent.com/guild-register/ledger/main"   # the public mirror the footer links


def check_dev_key_exposure(cfg: dict, host: str) -> None:
    """A development key serves on loopback, and behind a proxy only when the deploy says so: binding loopback
    keeps nothing private once a proxy fronts it, so the public address is what is checked."""
    if not (cfg["dev_key"] or cfg.get("log_dev_key")):
        return
    if host not in LOOPBACK_HOSTS:
        raise SystemExit("a development key serves on loopback only")
    public_host = urlsplit(cfg["public_url"]).hostname or ""
    if public_host not in LOOPBACK_HOSTS and not cfg["dev_key_public"]:
        raise SystemExit(f"a development key is reachable at {cfg['public_url']} through a proxy; set "
                         "REGISTER_DEV_KEY_PUBLIC=1 to serve it there knowingly, or point REGISTER_PUBLIC_URL at loopback")


def dev_marker(cfg: dict) -> Path:
    return cfg["key_file"].with_suffix(".dev")


def _load_key(key_file: Path, dev: bool, what: str, setting: str) -> R.SigningKey:
    """A key ever served as a development key is refused once that setting is gone; none is made without it."""
    marker = key_file.with_suffix(".dev")
    if key_file.is_file():
        key = R.SigningKey.load(key_file)
        if dev:
            marker.write_text(key.key_id + "\n", encoding="utf-8")
        elif marker.is_file() and marker.read_text(encoding="utf-8").strip() == key.key_id:
            raise SystemExit(f"the key at {key_file} is the development key for the {what} ({key.key_id}); "
                             "the Guild's key is placed by the ceremony (deploy/key-ceremony.md)")
        return key
    if not dev:
        raise SystemExit(f"no {what} at {key_file}: the Guardians' key is made in the ceremony, never here "
                         f"({setting}=1 makes a development key)")
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key = R.SigningKey.load_or_create(key_file)
    marker.write_text(key.key_id + "\n", encoding="utf-8")
    return key


def load_register_key(cfg: dict) -> R.SigningKey:
    """The countersigning key."""
    return _load_key(cfg["key_file"], cfg["dev_key"], "register key", "REGISTER_DEV_KEY")


def log_key_settings(cfg: dict) -> tuple:
    return cfg.get("log_key_file") or cfg["data"] / "log-key.pem", cfg.get("log_dev_key", cfg["dev_key"])


def load_log_key(cfg: dict) -> R.SigningKey:
    """The log's own key, which signs tree heads and nothing else."""
    path, dev = log_key_settings(cfg)
    return _load_key(path, dev, "log key", "REGISTER_LOG_DEV_KEY")


PUBLIC_DELAY_MAX_DAYS = 7   # the verifiers allow this much, plus slack, for a body not yet in the public mirror


def log_places(cfg: dict) -> tuple:
    """(the full export, the delayed public export, the delay in days, every published copy that witnesses the log)."""
    mirror, public = cfg["data"] / "mirror", cfg["data"] / "mirror-public"
    delay = min(max(float(cfg.get("log_public_delay_days", PUBLIC_DELAY_MAX_DAYS)), 0.0), PUBLIC_DELAY_MAX_DAYS)
    witnesses = [mirror, public, cfg["data"].parent / "ledger-mirror", *cfg.get("log_witnesses", [])]
    return mirror, public, delay, witnesses


def export_all(cfg: dict, log, key: R.SigningKey, dev: bool) -> bool:
    mirror, public, delay, _ = log_places(cfg)
    return export_log(log, mirror, log_key_doc(key, dev), public=public, delay_days=delay)


def log_key_doc(key: R.SigningKey, dev: bool) -> dict:
    committed = T.committed_key()
    return {"log_id": T.LOG_ID, "key_id": key.key_id, "public_key_pem": key.public_pem(), "development": bool(dev),
            "committed": bool(committed and committed.key_id == key.key_id),
            "signed_bytes": "RFC 8785 JSON of {log_id, root_sha256, timestamp, tree_size, version}; Ed25519"}


def public_view(record: dict) -> dict:
    """The record as the artist chose to show it: model alone, with tracks, or with writers' shares."""
    view = record.get("public_view", "model")
    out = json.loads(json.dumps(record))
    ts = out.get("training_set", {})
    tracks = ts.get("tracks", [])
    if view == "model":
        ts["tracks"] = []
        ts["track_count"] = len(tracks)
    elif view == "tracks":
        for t in tracks:
            t.pop("writers", None)
    return out


class RateLimit:
    """A token bucket per client address: the public doors take a bounded number of calls a minute each."""

    def __init__(self, per_minute: int):
        self.per_minute, self._buckets, self._lock = per_minute, {}, threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(client, (float(self.per_minute), now))
            tokens = min(float(self.per_minute), tokens + (now - last) * self.per_minute / 60.0)
            if tokens < 1.0:
                self._buckets[client] = (tokens, now)
                return False
            self._buckets[client] = (tokens - 1.0, now)
            if len(self._buckets) > 10000:
                stale = [k for k, (_, t) in self._buckets.items() if now - t > 600]
                for k in stale:
                    del self._buckets[k]
            return True


PUBLIC_LIMITS = {"/api/verify": 20, "/api/enrol": 6}
# The log's public reads: a verifier walks a few files per entry; a crawler is bounded.
LOG_READ_LIMIT = 240
SIGNED_LIMIT = 120
# A verify key's own lane: far above the public bound, still a bound, per key.
KEYED_VERIFY_LIMIT = 600
# A Publish checks and claims every training track at once; a large catalogue must fit in a minute.
CLAIM_LIMITS = {"/api/claims": 600, "/api/claims/check": 600}


def create_app(cfg: dict | None = None) -> FastAPI:
    cfg = cfg or settings()
    store = Store(cfg["data"] / "register.sqlite3")
    index = Index(cfg["data"] / "index.sqlite3")
    key = load_register_key(cfg)
    log_key, log_dev = load_log_key(cfg), log_key_settings(cfg)[1]
    store.log.signer = log_key
    mirror = cfg["data"] / "mirror"
    (mirror / "models").mkdir(parents=True, exist_ok=True)
    started, once = cfg["data"] / "log-started", cfg["data"] / "log-open-unwitnessed"
    store.log_open(log_places(cfg)[3], unwitnessed=once.is_file(), started=started.is_file())
    if not store.log.blocked:
        # Outside the database on purpose: a database that lost its log cannot also forget that one was started.
        if store.log.size():
            started.touch()
        once.unlink(missing_ok=True)
    if store.log.blocked:
        log.error("the log is blocked: %s", store.log.blocked)
    # No schema served either: the public doors are documented in deploy/README.md, not enumerated to callers.
    app = FastAPI(title="Guild Register", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.index, app.state.key, app.state.cfg = store, index, key, cfg

    async def signed_body(request: Request, enrolling: bool = False, limit: int = MAX_BODY) -> tuple:
        raw = await request.body()
        if len(raw) > limit:
            raise HTTPException(413, "too large")
        try:
            body = json.loads(raw)
        except ValueError:
            raise HTTPException(400, "not JSON") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "the body must be a JSON object")
        if R.canonical_bytes(body) != raw:
            raise HTTPException(400, "the body must be canonical JSON (sorted keys, no spaces)")
        key_id = request.headers.get("X-Machine-Key-Id", "")
        sig = request.headers.get("X-Machine-Signature", "")
        if enrolling:
            pem = body.get("public_key_pem", "")
            try:
                if R.key_id_of(R.public_raw(R.load_public_pem(pem))) != key_id:
                    raise HTTPException(401, "the key id is not the public key's")
            except HTTPException:
                raise
            except Exception:  # noqa: BLE001 — any malformed key is the same refusal
                raise HTTPException(400, "public_key_pem is not an Ed25519 public key") from None
            machine = None
        else:
            machine = store.machine(key_id)
            if not machine:
                raise HTTPException(401, "this machine is not enrolled")
            pem = machine["public_pem"]
        if not R.verify(pem, raw, sig):
            raise HTTPException(401, "the signature is not this machine's over this body")
        return body, machine, key_id, pem

    def write_mirror(serial: int) -> None:
        rec = store.record(serial)
        doc = {"serial": serial, "state": rec["state"], "record": public_view(rec["signed"]["record"]),
               "record_sha256": rec["record_sha256"], "machine_key_id": rec["machine_key_id"],
               "machine_public_pem": rec.get("machine_public_pem"),
               "countersignature": rec["countersignature"], "served": rec["served"],
               "withdrawn_at": rec["withdrawn_at"], "withdraw_reason": rec["withdraw_reason"],
               "log_state": "unresolved" if serial in store.unresolved_serials() else None}
        tmp = mirror / "models" / f"{serial}.json.new"
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, mirror / "models" / f"{serial}.json")
        # The mirror is rsynced on a timer; a reader must never meet a half-written ledger.
        new = mirror / "ledger.jsonl.new"
        with open(new, "w", encoding="utf-8") as fh:
            for e in store.ledger():
                fh.write(json.dumps(e, sort_keys=True) + "\n")
        os.replace(new, mirror / "ledger.jsonl")

    def log_mirror_quietly() -> None:
        try:
            export_all(cfg, store.log, log_key, log_dev)
        except Exception as exc:  # noqa: BLE001
            log.error("log mirror not written: %s", exc)

    def mirror_quietly(serial: int) -> None:
        # The record is in the ledger; a mirror that cannot be written is logged and written on the next change.
        try:
            write_mirror(serial)
        except Exception as exc:  # noqa: BLE001
            log.error("mirror not written for %s: %s", serial, exc)
        log_mirror_quietly()

    log_mirror_quietly()

    limits = {path: RateLimit(n) for path, n in {**PUBLIC_LIMITS, **CLAIM_LIMITS}.items()}
    log_read_limit = RateLimit(LOG_READ_LIMIT)
    lookup_limits = {path: RateLimit(n) for path, n in softbinding.LIMITS.items()}
    signed_limit = RateLimit(SIGNED_LIMIT)
    from register_service.verify_keys import Allowed  # noqa: PLC0415
    verify_key_allowed = Allowed(cfg)
    keyed_verify_limit = RateLimit(KEYED_VERIFY_LIMIT)

    @app.middleware("http")
    async def rate_limited(request: Request, call_next):
        # Caddy appends the real address to whatever the client sent, so the LAST entry is the one to trust.
        client = (request.headers.get("x-forwarded-for", "").split(",")[-1].strip() or (request.client.host if request.client else "?"))
        path = request.url.path
        lookup = softbinding.limit_key(path)
        if lookup is not None:
            bucket = lookup_limits[lookup]
            if path.endswith("/byContent") and verify_key_allowed(request.headers.get("x-khaos-verify-key")):
                client, bucket = "key:" + request.headers["x-khaos-verify-key"].strip(), keyed_verify_limit
            if not bucket.allow(client):
                return JSONResponse({"detail": "too many requests from this address; try again in a minute"}, status_code=429,
                                    headers={"Retry-After": "60"})
        elif request.method in ("GET", "HEAD") and path.startswith(("/log/", "/mirror/log/")):
            if not log_read_limit.allow(client):
                return JSONResponse({"detail": "too many requests from this address; try again in a minute"}, status_code=429,
                                    headers={"Retry-After": "60"})
        elif request.method == "POST":
            bucket = limits.get(path) or (signed_limit if path.startswith("/api/") else None)
            if path == "/api/verify" and verify_key_allowed(request.headers.get("x-khaos-verify-key")):
                client, bucket = "key:" + request.headers["x-khaos-verify-key"].strip(), keyed_verify_limit
            if bucket is not None and not bucket.allow(client):
                return JSONResponse({"detail": "too many requests from this address; try again in a minute"}, status_code=429,
                                    headers={"Retry-After": "60"})
        return await call_next(request)

    @app.post("/api/enrol")
    async def enrol(request: Request):
        body, _, key_id, pem = await signed_body(request, enrolling=True)
        try:
            return store.enrol(str(body.get("invite", "")), key_id, str(body.get("machine_name", ""))[:80], pem)
        except PermissionError as e:
            raise HTTPException(403, str(e)) from None

    @app.post("/api/records")
    async def records(request: Request):
        signed, machine, key_id, pem = await signed_body(request)
        presented = signed.pop("countersignature", None)
        problems = R.signed_problems(signed, pem)
        if problems:
            raise HTTPException(422, "; ".join(problems))
        record = signed["record"]
        problems = R.record_problems(record)
        if problems:
            raise HTTPException(422, "; ".join(problems))
        if record["model"]["machine_key_id"] != key_id:
            raise HTTPException(422, "the record names another machine as its maker")
        # A countersignature this register made, presented again after a recovery, gets its reserved serial back.
        if not (isinstance(presented, dict) and R.countersignature_valid(presented, signed["record_sha256"], key.public_pem())):
            presented = None
        serial, counter, already = store.register(signed, key_id, machine["account_id"],
                                                  lambda n, at: R.countersign(signed, n, key, at or now_utc()), presented)
        if not already:
            mirror_quietly(serial)
        return {"serial": serial, "countersignature": counter, "already": already,
                "page": f"{cfg['public_url']}/models/{serial}"}

    def serial_of(body: dict) -> int:
        try:
            return int(body.get("serial"))
        except (TypeError, ValueError):
            raise HTTPException(422, "serial must be an integer") from None

    def once(body: dict, key_id: str, work):
        """The nonce is an idempotency key: the same signed body again gets its first answer, never a second effect."""
        digest = R.sha256_hex(R.canonical_bytes(body))
        try:
            earlier = store.nonce_answer(key_id, str(body.get("nonce", "")), digest)
        except PermissionError as e:
            raise HTTPException(409, str(e)) from None
        if earlier is not None:
            return earlier
        out = work()
        store.record_nonce(key_id, str(body.get("nonce")), digest, out)
        return out

    @app.post("/api/withdraw")
    async def withdraw(request: Request):
        body, _, key_id, _ = await signed_body(request)

        def work():
            try:
                out = store.withdraw(serial_of(body), key_id, str(body.get("reason", "")))
            except KeyError as e:
                raise HTTPException(404, e.args[0]) from None
            except PermissionError as e:
                raise HTTPException(403, str(e)) from None
            index.release_claims(out["serial"])
            index.withdraw_model(out["serial"])
            mirror_quietly(out["serial"])
            return out
        return await run_in_threadpool(once, body, key_id, work)

    @app.post("/api/served")
    async def served(request: Request):
        body, _, key_id, _ = await signed_body(request)
        state = str(body.get("state", ""))
        if state not in ("serving", "removed"):
            raise HTTPException(422, "state is serving or removed")

        def work():
            rec = store.record(serial_of(body))
            if rec and rec["state"] == "withdrawn" and state == "serving":
                raise HTTPException(410, "the model is withdrawn; a Space does not serve it")
            review = bool(rec and state == "serving" and rec["serial"] in store.unresolved_serials())
            try:
                out = store.served(serial_of(body), key_id, str(body.get("space_url", ""))[:200], state)
            except KeyError as e:
                raise HTTPException(404, e.args[0]) from None
            except PermissionError as e:
                raise HTTPException(403, str(e)) from None
            mirror_quietly(out["serial"])
            return {**out, "under_review": True} if review else out
        answer = await run_in_threadpool(once, body, key_id, work)
        # Under review is not withdrawn: the report is taken, and the Space shows the model as under review.
        return JSONResponse(answer, status_code=202) if answer.get("under_review") else answer


    @app.post("/api/outputs")
    async def outputs(request: Request):
        """A released output into the index, under its model's bucket: by the model's own account, or by a Space serving it."""
        body, machine, key_id, _ = await signed_body(request)

        def work():
            rec = store.record(serial_of(body))
            if not rec:
                raise HTTPException(404, "no such model")
            if rec["state"] == "withdrawn":
                raise HTTPException(410, "the model is withdrawn; its outputs are not released into the index")
            if machine["role"] == "workshop" and rec["account_id"] != machine["account_id"]:
                raise HTTPException(403, "a Workshop releases outputs of its own account's models only")
            if machine["role"] == "space" and not any(s["key_id"] == key_id and s["state"] == "serving" for s in rec["served"]):
                raise HTTPException(403, "a Space releases outputs of the models it reports serving")
            gid = str(body.get("generation_id") or "").strip()
            if not gid or len(gid) > 120:
                raise HTTPException(422, "generation_id is a short string")
            manifest = manifest_bytes(body.get("manifest"))    # a bad one is not kept; the release still stands
            try:
                out = index.release(gid, rec["serial"], int(rec["signed"]["record"]["model"]["watermark_payload"]),
                                    rec["signed"]["record"]["artist_id"], body.get("fingerprint"), key_id, machine["role"])
            except ValueError as e:
                raise HTTPException(422, str(e)) from None
            kept = manifest is not None and index.keep_manifest(gid, manifest, key_id)
            return {**{k: out[k] for k in ("generation_id", "serial", "payload", "released_at", "landmarks", "already")},
                    "manifest_kept": kept}
        return await run_in_threadpool(once, body, key_id, work)

    @app.post("/api/outputs/remove")
    async def outputs_remove(request: Request):
        """A released piece its maker took down: by the machine that released it. Its fingerprint stays so a copy
        found later is still named; the answer says removed, and the lookup no longer offers it."""
        body, machine, key_id, _ = await signed_body(request)

        def work():
            gid = str(body.get("generation_id") or "").strip()
            out = index.output(gid) if gid and len(gid) <= 120 else None
            if out is None:
                raise HTTPException(404, "no such released piece")
            if out["released_by"] != key_id:
                raise HTTPException(403, "only the machine that released a piece removes it")
            out = index.remove_output(gid, int(out["serial"]))
            return {"generation_id": gid, "serial": out["serial"], "removed_at": out["removed_at"]}
        return await run_in_threadpool(once, body, key_id, work)

    def manifest_bytes(value) -> bytes | None:
        """A released output's Content Credentials manifest store (base64 of its JUMBF); None when none was sent or it
        is not one."""
        if value is None:
            return None
        import base64  # noqa: PLC0415
        import binascii  # noqa: PLC0415
        try:
            data = base64.b64decode(str(value), validate=True)
        except (binascii.Error, ValueError):
            return None
        return data if len(data) <= MANIFEST_MAX_BYTES and data[4:8] == b"jumb" else None

    def workshop_only(machine: dict) -> None:
        if machine["role"] != "workshop":
            raise HTTPException(403, "only a Workshop claims or checks training tracks")

    def query_phases_in(body: dict) -> list:
        """A body's phases through the index's one bounded parser; a bad body is a 422 in the parser's words."""
        try:
            return phases_of(body.get("phases"))
        except ValueError as e:
            raise HTTPException(422, str(e)) from None

    @app.post("/api/claims")
    async def claims(request: Request):
        """A registered model's training track, claimed on its account's declaration by the record it was registered under."""
        body, machine, key_id, _ = await signed_body(request, limit=CLAIM_MAX_BODY)
        workshop_only(machine)

        def work():
            serial = store.serial_of_record(str(body.get("record_sha256") or ""))
            rec = store.record(serial) if serial is not None else None
            if not rec:
                raise HTTPException(404, "no model is registered under that record")
            if rec["account_id"] != machine["account_id"]:
                raise HTTPException(403, "a Workshop claims tracks for its own account's models only")
            if rec["state"] == "withdrawn":
                raise HTTPException(410, "the model is withdrawn")
            track_id = str(body.get("track_id") or "").strip()
            declaration = str(body.get("declaration") or "").strip()
            trained = {str(t.get("track_id")) for t in (rec["signed"]["record"].get("training_set") or {}).get("tracks") or []}
            if track_id not in trained:
                raise HTTPException(422, "a claim names one of the training tracks the model's own record lists")
            if not declaration or len(declaration) > 64:
                raise HTTPException(422, "declaration names the declaration the artist made")
            phases = body.get("phases")
            if phases is not None and not (isinstance(phases, list) and all(isinstance(p, list) for p in phases) and len(phases) <= 8):
                raise HTTPException(422, "phases is a short list of fingerprint lists")
            fingerprint = body.get("fingerprint") or (phases[0] if phases else None)
            try:
                out = index.claim(rec["account_id"], rec["serial"], track_id, fingerprint, declaration, key_id, phases)
            except ValueError as e:
                raise HTTPException(422, str(e)) from None
            return {"serial": rec["serial"], "track_id": track_id, **out}
        # The fingerprint matching is heavy: off the event loop, so verify and enrol stay answered meanwhile.
        return await run_in_threadpool(once, body, key_id, work)

    @app.post("/api/claims/check")
    async def claims_check(request: Request):
        """Whether a recording already trains a registered model (nobody's, this account's, or also another account's)
        and whether it is a released output: the Workshop's one question at import."""
        body, machine, _, _ = await signed_body(request)
        workshop_only(machine)
        return await run_in_threadpool(index.check_claim, query_phases_in(body), machine["account_id"])

    @app.get("/api/models/{serial}/claims")
    def model_claims(serial: int):
        rec = store.record(serial)
        if not rec:
            raise HTTPException(404, "no such model")
        tracks = len((rec["signed"]["record"].get("training_set") or {}).get("tracks") or [])
        return {"serial": serial, "tracks": tracks, **index.claims_of(serial)}

    @app.post("/api/verify")
    async def verify(request: Request):
        """Anyone's question, no key: a four-phase fingerprint and, when it read, the watermark codeword."""
        raw = await request.body()
        if len(raw) > MAX_BODY:
            raise HTTPException(413, "too large")
        try:
            body = json.loads(raw)
        except ValueError:
            raise HTTPException(400, "not JSON") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "the body must be a JSON object")
        phases = query_phases_in(body)
        payload = None
        codeword = body.get("watermark_id")
        if codeword is not None:
            try:
                decoded = decode_codeword(int(codeword))
            except (TypeError, ValueError):
                decoded = None
            if decoded is None:
                raise HTTPException(422, "watermark_id is not a codeword")
            payload = decoded[0]
        answer = await run_in_threadpool(index.verify, phases, payload)
        if answer["grade"] == "output":
            rec = store.record(int(answer["output"]["serial"]))
            answer["model"] = _model_summary(rec) if rec else None
            answer["removed_at"] = answer["output"].get("removed_at")
        elif answer["grade"] == "bucket":
            answer["models"] = [_model_summary(store.record(s)) for s in store.serials_in_bucket(payload)]
        return answer

    def _declared_line(d: dict) -> str:
        layers = {"composition": "the songs", "recording": "the recordings", "performers": "the performers",
                  "voices": "other voices", "society": "the collecting society"}
        named = ", ".join(layers.get(x, x) for x in d.get("layers") or [])
        return f"{d.get('declared', 0)} of {d.get('of', 0)} tracks, the artist confirming each of: {named}"

    def _claims_line(c: dict, total: int) -> str:
        from register_service import pages as P  # noqa: PLC0415
        line = f"{c['claimed']} of {total} claimed on the artist's declaration"
        if c["shared"]:
            line += f" · {c['shared']} also train{'s' if c['shared'] == 1 else ''} another account's registered model"
        return P.esc(line)

    def _model_summary(rec: dict) -> dict:
        r = rec["signed"]["record"]
        return {"serial": rec["serial"], "artist_id": r["artist_id"], "artist_name": r["artist_name"], "run_id": r["model"]["run_id"],
                "state": rec["state"], "page": f"{cfg['public_url']}/models/{rec['serial']}"}

    @app.get("/api/public-key")
    def public_key():
        return {"register_key_id": key.key_id, "public_key_pem": key.public_pem(), "development": bool(cfg["dev_key"])}

    @app.get("/api/models/{serial}")
    def model_json(serial: int):
        path = mirror / "models" / f"{serial}.json"
        if not path.is_file():
            raise HTTPException(404, "no such model")
        return Response(path.read_bytes(), media_type="application/json")

    @app.get("/api/withdrawn")
    def withdrawn():
        """`withdrawn`: models a Space must not serve. `unresolved`: models under review because a recovery lost an entry
        after their admission and the wardens have not answered it; a Space keeps them and shows them as under review."""
        return {"withdrawn": store.withdrawn_serials(), "unresolved": store.unresolved_serials()}

    @app.get("/api/ledger")
    def ledger(since: int = 0):
        return PlainTextResponse("\n".join(json.dumps(e, sort_keys=True) for e in store.ledger(since)) + "\n",
                                 media_type="application/x-ndjson")

    @app.api_route("/models/{serial}", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def model_page(serial: int):
        from register_service import pages as P  # noqa: PLC0415
        rec = store.record(serial)
        if not rec:
            return HTMLResponse(P.page("No such model · Guild Register", P.hero("Guild Register", "No such model.",
                                f"No model is registered as No. {serial}.", []), dev=bool(cfg["dev_key"]),
                                path=f"/models/{serial}", base=cfg["public_url"],
                                extra_head='<meta name="robots" content="noindex">'), status_code=404)
        r = public_view(rec["signed"]["record"])
        m, ts = r["model"], r["training_set"]
        tracks = ts.get("tracks") or []
        unresolved = serial in store.unresolved_serials()
        facts = [P.state_chip("under review" if unresolved and rec["state"] != "withdrawn" else rec["state"]),
                 f"registered {P.esc(P.day(rec['published_at']))}", P.esc(m["base_model"]),
                 P.count(len(tracks) or ts.get("track_count", 0), "track")]
        note = ("What the artist's Workshop attested, bound to hashes the artist's own machine can reproduce. The register cannot "
                "see the audio and does not verify the claim; it makes it checkable.")
        if unresolved:
            note = ("Under review: an entry after this model's admission was lost when the register was recovered, and the wardens "
                    "have not yet answered it. Until they confirm it or strike it, whether it was struck is unknown. ") + note
        body = P.hero(f"Register No. {serial}", r["artist_name"], note, facts, P.seal_for(serial, rec["signed"]["record"]))
        engine_id = f" <span class=\"mono\">{P.esc(m['base_model_id'])}</span>" if m.get("base_model_id") else ""
        kv = [("Base model", f"{P.esc(m['base_model'])}{engine_id} <span class=\"muted\">· {P.esc(m['base_model_licence'])} licence</span>"),
              ("Adapter", f'<span class="mono">{P.esc(m["adapter_sha256"])}</span>'),
              ("Model card", f'<span class="mono">{P.esc(m["card_sha256"])}</span>'),
              ("Dataset hash", f'<span class="mono">{P.esc(r["provenance"]["dataset_hash"])}</span>'),
              ("Watermark bucket", P.esc(m["watermark_payload"])),
              ("Countersigned by", f'<span class="mono">{P.esc(rec["countersignature"]["register_key_id"])}</span>'),
              ("Training tracks", _claims_line(index.claims_of(serial), len((rec["signed"]["record"].get("training_set") or {}).get("tracks") or []))),
              *([("Trained on", P.esc("; ".join(ts["data_kinds"])))] if ts.get("data_kinds") else []),
              *([("Declared", P.esc(_declared_line(ts["declaration"])))] if ts.get("declaration") else []),
              ("Registered", P.esc(rec["published_at"]))]
        body += P.section("The record", "every value as the artist's Workshop attested it")
        body += P.table("200px minmax(0, 1fr)", ["", ""], [[(f'<span class="stage-kv-label">{P.esc(k)}</span>', ""), (v, "wrap")] for k, v in kv])
        if tracks:
            rows = []
            for i, t in enumerate(tracks, 1):
                writers = ", ".join(f"{P.esc(w['name'])} {w['share']:.0%}" for w in t.get("writers", [])) or '<span class="muted">—</span>'
                rows.append([(str(i), "num"), P.esc(t.get("title") or t["track_id"]), (writers, "muted")])
            body += P.section("Trained on", P.count(len(tracks), "track"))
            body += P.table("44px minmax(0, 1fr) minmax(0, 1fr)", ["#", "Track", "Writers"], rows)
        else:
            body += P.section("Trained on", "the artist shows the model alone")
            body += f'<div class="empty">{P.count(ts.get("track_count", 0), "track")}; the artist chose not to list them.</div>'
        if rec["served"]:
            body += P.section("Served at")
            body += P.table("minmax(0, 1fr) 120px 160px", ["Listening Space", "State", "Since"],
                            [[(f'<a href="{P.esc(s_["space_url"])}">{P.esc(s_["space_url"])}</a>', "wrap"), (P.esc(s_["state"]), "muted"), (P.esc(P.day(s_["at"])), "muted")]
                             for s_ in rec["served"]])
        body += P.log_section(store.log.of_record(rec["record_sha256"]), store.log.struck())
        body += f'<div class="stage-pill-row"><a class="k-btn k-btn--pill" href="/api/models/{serial}">The record as JSON</a></div>'
        n_tracks = len(tracks) or ts.get("track_count", 0)
        withdrawn = " It has since been withdrawn." if rec["state"] == "withdrawn" else ""
        description = (f"Register No. {serial}: {r['artist_name']}'s fine-tuned model on {m['base_model']}, trained on "
                       f"{P.count(n_tracks, 'track')} and registered {P.day(rec['published_at'])}.{withdrawn} The record as the "
                       "artist's Workshop attested it, countersigned by the Guild Register.")
        # A withdrawn record stays readable as the ledger's history, but search engines are asked to drop it.
        noindex = '<meta name="robots" content="noindex">' if rec["state"] == "withdrawn" else ""
        return P.page(f"Register No. {serial} · {r['artist_name']} · Guild Register", body, dev=bool(cfg["dev_key"]),
                      description=description, path=f"/models/{serial}", base=cfg["public_url"], extra_head=noindex)

    @app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def home():
        from register_service import pages as P  # noqa: PLC0415
        serials = store.serials()
        rows = []
        for serial in serials[-100:][::-1]:
            rec = store.record(serial)
            if not rec:
                continue
            r = rec["signed"]["record"]
            rows.append((f"/models/{serial}", [(str(serial), "num"), P.esc(r["artist_name"]), (P.esc(P.day(rec["published_at"])), "muted"),
                                                (P.state_chip(rec["state"]), "")]))
        para = ("The public ledger of registered fine-tuned models: what each was trained on, as its artist attested, "
                "countersigned by the register and never edited.")
        facts = [P.count(len(serials), "model") + " registered", "ledger unbroken" if store.ledger_status()["ok"] else "ledger BROKEN",
                 "serials from 65,536"]
        body = P.hero("", "The Guild Register", para, facts)
        body += P.section("Registered models", "newest first")
        body += (P.table("80px minmax(0, 1fr) 120px 120px", ["No.", "Artist", "Registered", "State"], rows) if rows
                 else '<div class="empty">No model is registered yet.</div>')
        return P.page("The Guild Register · the public ledger of fine-tuned music models", body, dev=bool(cfg["dev_key"]),
                      base=cfg["public_url"], extra_head=P.site_json_ld(cfg["public_url"]))

    @app.api_route("/robots.txt", methods=["GET", "HEAD"])
    def robots():
        from register_service import pages as P  # noqa: PLC0415
        return PlainTextResponse(P.robots_txt(cfg["public_url"]))

    @app.api_route("/sitemap.xml", methods=["GET", "HEAD"])
    def sitemap():
        from register_service import pages as P  # noqa: PLC0415
        records = [(serial, store.record(serial) or {}) for serial in store.serials()]
        models = [(serial, P.day(rec.get("published_at"))) for serial, rec in records if rec.get("state") != "withdrawn"]
        return Response(P.sitemap_xml(cfg["public_url"], models), media_type="application/xml")

    @app.api_route("/favicon.svg", methods=["GET", "HEAD"])
    def favicon_svg():
        from register_service import pages as P  # noqa: PLC0415
        return Response(P.FAVICON_SVG, media_type="image/svg+xml", headers=STATIC_CACHE)

    def static_png(name: str) -> Response:
        return Response(resources.files("register_service").joinpath("static", name).read_bytes(), media_type="image/png",
                        headers=STATIC_CACHE)

    # Browsers ask for /favicon.ico whatever the page names; a PNG under that name is read the same.
    app.add_api_route("/favicon.ico", lambda: static_png("icon.png"), methods=["GET", "HEAD"])
    app.add_api_route("/apple-touch-icon.png", lambda: static_png("icon.png"), methods=["GET", "HEAD"])
    app.add_api_route("/card.png", lambda: static_png("card.png"), methods=["GET", "HEAD"])

    # ---- the signed append-only log ----

    def cached(request: Request, doc, max_age: int) -> Response:
        """JSON with an ETag over its bytes; a matching If-None-Match is a 304."""
        data = (json.dumps(doc, sort_keys=True) + "\n").encode("utf-8")
        etag = '"' + R.sha256_hex(data)[:32] + '"'
        headers = {"ETag": etag, "Cache-Control": f"public, max-age={max_age}" + (", immutable" if max_age >= 86400 else "")}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return Response(data, media_type="application/json", headers=headers)

    def size_param(value, name: str) -> int:
        try:
            n = int(value)
        except (TypeError, ValueError):
            raise HTTPException(422, f"{name} must be an integer") from None
        if n < 0:
            raise HTTPException(422, f"{name} must not be negative")
        return n

    @app.exception_handler(LogBlocked)
    async def log_blocked(request: Request, exc: LogBlocked):
        return JSONResponse({"detail": f"the register is not taking changes: {exc}"}, status_code=503)

    @app.post("/log/append")
    async def log_append(request: Request):
        """A warden's entry, signed by the warden's enrolled machine: a strike, an admit or a withhold, naming the sitting."""
        body, machine, key_id, _ = await signed_body(request)
        if machine["role"] != "warden":
            raise HTTPException(403, "only a warden's machine appends to the log")
        entry = body.get("entry")
        if not isinstance(entry, dict):
            raise HTTPException(422, "entry must be an object")

        def work():
            try:
                out = store.log_append(entry, key_id)
            except KeyError as e:
                raise HTTPException(404, e.args[0]) from None
            except (EntryRefused, ValueError) as e:
                raise HTTPException(422, str(e)) from None
            log_mirror_quietly()
            return out
        return await run_in_threadpool(once, body, key_id, work)

    @app.api_route("/log/sth", methods=["GET", "HEAD"])
    def log_sth(request: Request):
        head = store.log.head()
        if head is None:
            raise HTTPException(404, "the log is empty")
        return cached(request, head, 10)

    @app.api_route("/log/sth/{size}", methods=["GET", "HEAD"])
    def log_sth_at(request: Request, size: int):
        head = store.log.head(size)
        if head is None:
            raise HTTPException(404, "no head of that size")
        # The newest size is signed again daily and after a key change; only older sizes never change.
        return cached(request, head, 10 if size >= store.log.size() else 31536000)

    @app.api_route("/log/proof/{leaf_hash}", methods=["GET", "HEAD"])
    def log_proof(request: Request, leaf_hash: str, size: int | None = None):
        if not T.is_hex256(leaf_hash):
            raise HTTPException(422, "a leaf hash is 64 lowercase hex characters")
        if size is not None and store.log.head(size_param(size, "size")) is None:
            raise HTTPException(404, "no head of that size")
        try:
            proof = store.log.inclusion(leaf_hash, size)
        except KeyError as e:
            raise HTTPException(404, e.args[0]) from None
        return cached(request, proof, 31536000 if size is not None and size < store.log.size() else 10)

    @app.api_route("/log/consistency", methods=["GET", "HEAD"])
    def log_consistency(request: Request):
        params = request.query_params
        a, b = size_param(params.get("from"), "from"), size_param(params.get("to"), "to")
        if not 0 < a <= b or store.log.head(a) is None or store.log.head(b) is None:
            raise HTTPException(404, "from and to must be sizes of signed heads, from no larger than to")
        return cached(request, store.log.consistency(a, b), 31536000)

    @app.api_route("/log/entries", methods=["GET", "HEAD"])
    def log_entries(request: Request):
        params = request.query_params
        start = size_param(params.get("start", 0), "start")
        end = size_param(params.get("end", start + 100), "end")
        if end <= start:
            raise HTTPException(422, "end must be larger than start")
        from register_service.translog import PAGE_MAX  # noqa: PLC0415
        return cached(request, {"start": start, "entries": store.log.entries(start, min(end, start + PAGE_MAX))}, 60)

    @app.api_route("/log/key", methods=["GET", "HEAD"])
    def log_key_route(request: Request):
        return cached(request, log_key_doc(log_key, log_dev), 300)

    @app.api_route("/log/anchors", methods=["GET", "HEAD"])
    def log_anchors(request: Request):
        from register_service.translog import receipt_b64  # noqa: PLC0415
        rows = [{**{k: a[k] for k in ("tree_size", "root_sha256", "backend", "state", "stamped_at", "completed_at")},
                 "receipt_base64": receipt_b64(a["receipt"])} for a in store.log.anchors()]
        return cached(request, {"anchors": rows}, 300)

    @app.api_route("/mirror/log/{name:path}", methods=["GET", "HEAD"])
    def log_mirror_file(name: str):
        """The static export, served as the mirror holds it, so the page and the command read the same files."""
        base = (mirror / "log").resolve()
        path = (base / name).resolve()
        if base not in path.parents or not path.is_file() or path.suffix not in (".json", ".ots"):
            raise HTTPException(404, "no such file")
        media = "application/json" if path.suffix == ".json" else "application/octet-stream"
        return Response(path.read_bytes(), media_type=media, headers={"Cache-Control": "public, max-age=10"})

    @app.api_route("/log/verify.js", methods=["GET", "HEAD"])
    def log_verify_js():
        return Response(resources.files("register_service").joinpath("static", "log-verify.js").read_bytes(),
                        media_type="text/javascript", headers={"Cache-Control": "public, max-age=3600"})

    @app.api_route("/verify-entry", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def verify_entry_page():
        from register_service import pages as P  # noqa: PLC0415
        return P.page("Verify an entry · Guild Register", P.verify_entry_body(T.committed_key(), log_key_doc(log_key, log_dev)),
                      dev=bool(cfg["dev_key"]), path="/verify-entry", base=cfg["public_url"],
                      description="Check that an entry is in the Guild Register's signed append-only log, in your browser, "
                                  "from the log's own files and its public key.")

    @app.get("/health")
    def health():
        return JSONResponse({"ok": True, "ledger_ok": store.ledger_status()["ok"], "models": len(store.serials()), "outputs": index.count(),
                             "log_blocked": store.log.blocked,
                             "development": bool(cfg["dev_key"])})

    softbinding.add_routes(app, index, cfg["public_url"])
    return app
