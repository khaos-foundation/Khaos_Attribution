"""Content Credentials (C2PA) for Khaos outputs: a signed, time-stamped record inside the audio file saying it is
AI-generated, by which model, under which licence and with which watermark. Each installation signs with its own
identity: a local issuer certificate in the operating company's name and a signer certificate under it, both
development certificates that C2PA checkers show as untrusted. Needs the `credentials` extra (c2pa-python).
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import shutil
import stat
import socket
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

# The IPTC term C2PA uses for media made by a trained model.
DIGITAL_SOURCE_TYPE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
KHAOS_ASSERTION = "dev.khaos.provenance"
# Written beside the current label while verifiers that read only it are in use; read when it is the only one.
LEGACY_ASSERTION = "org.khaos.provenance"
ASSERTION_VERSION = 1
# The watermark's soft-binding algorithm name. No c2pa.soft-binding assertion is written: through the SDK's JSON
# route its value is stored as an array, not the byte string the specification defines, and a signed file keeps it.
WATERMARK_ALG = "dev.khaos.audioseal-bucket.1"
# A public RFC 3161 time-stamping service; KHAOS_TSA_URL overrides it, "off" signs without a timestamp.
DEFAULT_TSA_URL = "http://timestamp.digicert.com"
MIME_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".m4a": "audio/mp4"}
IDENTITY_FILES = ("issuer.pem", "issuer.key", "signer.pem", "signer.key")
SIGNER_DAYS = 825
ISSUER_DAYS = 3650
RENEW_DAYS = 30
SIGNED_STATES = ("Valid", "Trusted")
# The SDK's codes that tell an expiry from a change. A certificate past its date fails the SDK's check though the file
# is as it was signed, time-stamped or not: the SDK holds no trust list for the time-stamping service.
UNTRUSTED = "signingCredential.untrusted"
EXPIRED = "signingCredential.expired"
SIGNATURE_VALIDATED = "claimSignature.validated"
CONTENT_MATCHES = ("assertion.dataHash.match", "assertion.bmffHash.match")
# A time-stamping service that does not answer within this is skipped for TSA_RETRY_S, not waited on per file.
TSA_PROBE_S = 2.0
TSA_RETRY_S = 600.0
# Record fields that travel inside the file; the prompt and anything about a listener never do.
RECORD_FIELDS = ("artist_id", "artist_name", "adapter_version", "adapter_hash", "base_model", "base_model_id",
                 "base_model_licence", "engine_commit", "generation_id", "watermark_id", "register_serial", "timestamp")
# Compared with the record a room signed from. A field the manifest does not carry was not signed by that version and
# is not compared; these must always be there.
AGREEMENT_FIELDS = ("artist_id", "artist_name", "adapter_version", "adapter_hash", "base_model", "base_model_id",
                    "base_model_licence", "engine_commit", "generation_id", "watermark_id", "timestamp")
AGREEMENT_REQUIRED = ("artist_id", "adapter_hash")


class CredentialsUnavailable(RuntimeError):
    """The credentials extra is not installed, or the identity cannot be made."""


class CredentialsBroken(RuntimeError):
    """The file carries credentials whose signature no longer holds; it is never re-signed silently."""


def available() -> bool:
    try:
        import c2pa  # noqa: F401, PLC0415
        from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


@dataclass
class Identity:
    """One installation's signing identity: the chain (signer then issuer, PEM) and the signer's key."""
    chain_pem: str
    key: object
    signer_id: str
    issuer_pem: str


def _name(organisation: str, product: str, role: str, issuer_key):
    """Named for the company and product; the installation is told apart by its own issuer key, nothing else."""
    from cryptography import x509  # noqa: PLC0415
    from cryptography.x509.oid import NameOID  # noqa: PLC0415
    installation = hashlib.sha256(_public(issuer_key)).hexdigest()[:16]
    return x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, organisation),
                      x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, product),
                      x509.NameAttribute(NameOID.COMMON_NAME, f"{product} {role} {installation}")])


def _certificate(subject, subject_key, issuer, issuer_key, *, ca: bool, days: int):
    from cryptography import x509  # noqa: PLC0415
    from cryptography.x509.oid import ExtendedKeyUsageOID  # noqa: PLC0415
    now = datetime.datetime.now(datetime.timezone.utc)
    usage = dict(content_commitment=False, key_encipherment=False, data_encipherment=False, key_agreement=False,
                 encipher_only=False, decipher_only=False)
    b = (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(subject_key.public_key())
         .serial_number(x509.random_serial_number())
         .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(days=days))
         .add_extension(x509.BasicConstraints(ca=ca, path_length=0 if ca else None), critical=True)
         .add_extension(x509.KeyUsage(digital_signature=not ca, key_cert_sign=ca, crl_sign=ca, **usage), critical=True)
         .add_extension(x509.SubjectKeyIdentifier.from_public_key(subject_key.public_key()), critical=False)
         .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_key.public_key()), critical=False))
    if not ca:
        b = b.add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
    return b.sign(issuer_key, None)


def _write(path: Path, data: bytes, *, private: bool) -> None:
    """Write through a temporary file in the same folder, so a reader never sees half a key or certificate."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o600 if private else 0o644)
        _replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _replace(src, dst) -> None:
    """os.replace, retried briefly on Windows where a file being read cannot be replaced."""
    deadline = time.monotonic() + 5.0
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if sys.platform != "win32" or time.monotonic() > deadline:
                raise
            time.sleep(0.05)


_THREAD_LOCK = threading.Lock()


@contextlib.contextmanager
def _identity_lock(directory: Path):
    """One maker of an identity at a time, across threads and processes."""
    with _THREAD_LOCK, open(directory / ".lock", "a+b") as fh:
        if sys.platform == "win32":
            import msvcrt  # noqa: PLC0415
            fh.seek(0)
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # noqa: PLC0415
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _loaded(paths: dict):
    """The four files as (issuer, issuer_key, signer, signer_key), or None when any is missing or unreadable."""
    from cryptography import x509  # noqa: PLC0415
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
    try:
        return (x509.load_pem_x509_certificate(paths["issuer.pem"].read_bytes()),
                serialization.load_pem_private_key(paths["issuer.key"].read_bytes(), None),
                x509.load_pem_x509_certificate(paths["signer.pem"].read_bytes()),
                serialization.load_pem_private_key(paths["signer.key"].read_bytes(), None))
    except Exception:  # noqa: BLE001 — a missing, partial or foreign file means the identity is remade
        return None


def _public(key) -> bytes:
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _reusable_signer(loaded) -> bool:
    """A signer key that matches its in-date certificate signed by the issuer's key, whatever the names say: a rename
    that stopped half-way reissues its certificate rather than minting a new key."""
    if loaded is None:
        return False
    _, issuer_key, signer, signer_key = loaded
    soon = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=RENEW_DAYS)
    try:
        issuer_key.public_key().verify(signer.signature, signer.tbs_certificate_bytes)
    except Exception:  # noqa: BLE001 — not signed by this issuer's key
        return False
    return signer.public_key().public_bytes_raw() == _public(signer_key) and signer.not_valid_after_utc > soon


def _sound(loaded) -> tuple[bool, bool]:
    """(issuer usable, signer usable): each key matches its certificate, the signer is the issuer's and in date."""
    if loaded is None:
        return False, False
    issuer, issuer_key, signer, signer_key = loaded
    soon = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=RENEW_DAYS)
    issuer_ok = (issuer.public_key().public_bytes_raw() == _public(issuer_key) and issuer.not_valid_after_utc > soon)
    if not issuer_ok:
        return False, False
    try:
        signer.verify_directly_issued_by(issuer)
    except Exception:  # noqa: BLE001 — any failure means this signer is not the issuer's
        return True, False
    return True, signer.public_key().public_bytes_raw() == _public(signer_key) and signer.not_valid_after_utc > soon


def _named(loaded, organisation: str, product: str) -> bool:
    """Whether both certificates carry the names this installation's identity should have."""
    issuer, issuer_key, signer, _ = loaded
    return (issuer.subject == _name(organisation, product, "issuer", issuer_key)
            and signer.subject == _name(organisation, product, "signer", issuer_key)
            and signer.issuer == issuer.subject)


def _archive(d: Path, names: tuple[str, ...]) -> None:
    """Keep the certificates about to be replaced; keys are never replaced by a rename."""
    present = [n for n in names if (d / n).exists()]
    if not present:
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    where = d / "archive" / stamp
    where.mkdir(parents=True, exist_ok=True)
    for n in present:
        shutil.copy2(d / n, where / n)


def load_identity(directory: Path | str, *, product: str, organisation: str) -> Identity:
    """This installation's identity, made once under `directory` (keys readable by the owner only) and reused.
    A signer near expiry or not the issuer's is remade; certificates under another name are reissued over the same
    keys, the old ones archived."""
    if not available():
        raise CredentialsUnavailable("Content Credentials need the credentials extra (c2pa-python)")
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
    from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: PLC0415
    d = Path(directory)
    paths = {name: d / name for name in IDENTITY_FILES}
    loaded = _loaded(paths)
    if _sound(loaded) != (True, True) or not _named(loaded, organisation, product):
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        with _identity_lock(d):
            loaded = _loaded(paths)
            issuer_ok, signer_ok = _sound(loaded)
            pem = serialization.Encoding.PEM
            pkcs8 = (pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
            reissued = False
            if issuer_ok:
                issuer, issuer_key = loaded[0], loaded[1]
                name = _name(organisation, product, "issuer", issuer_key)
                if issuer.subject != name:
                    _archive(d, ("issuer.pem", "signer.pem"))
                    issuer = _certificate(name, issuer_key, name, issuer_key, ca=True, days=ISSUER_DAYS)
                    _write(paths["issuer.pem"], issuer.public_bytes(pem), private=False)
                    reissued = True
            else:
                issuer_key = ed25519.Ed25519PrivateKey.generate()
                name = _name(organisation, product, "issuer", issuer_key)
                issuer = _certificate(name, issuer_key, name, issuer_key, ca=True, days=ISSUER_DAYS)
                _write(paths["issuer.key"], issuer_key.private_bytes(*pkcs8), private=True)
                _write(paths["issuer.pem"], issuer.public_bytes(pem), private=False)
            signer_name = _name(organisation, product, "signer", issuer_key)
            left = (issuer.not_valid_after_utc - datetime.datetime.now(datetime.timezone.utc)).days
            if issuer_ok and not signer_ok and _reusable_signer(loaded):
                _archive(d, ("signer.pem",))
                signer = _certificate(signer_name, loaded[3], issuer.subject, issuer_key,
                                      ca=False, days=min(SIGNER_DAYS, left))
                _write(paths["signer.pem"], signer.public_bytes(pem), private=False)
            elif not (issuer_ok and signer_ok):
                signer_key = ed25519.Ed25519PrivateKey.generate()
                signer = _certificate(signer_name, signer_key, issuer.subject, issuer_key,
                                      ca=False, days=min(SIGNER_DAYS, left))
                _write(paths["signer.key"], signer_key.private_bytes(*pkcs8), private=True)
                _write(paths["signer.pem"], signer.public_bytes(pem), private=False)
            elif reissued or loaded[2].subject != signer_name:
                if not reissued:
                    _archive(d, ("signer.pem",))
                signer = _certificate(signer_name, loaded[3], issuer.subject, issuer_key,
                                      ca=False, days=min(SIGNER_DAYS, left))
                _write(paths["signer.pem"], signer.public_bytes(pem), private=False)
            loaded = _loaded(paths)
    issuer, _, signer, key = loaded
    issuer_pem = issuer.public_bytes(serialization.Encoding.PEM).decode()
    return Identity(chain_pem=signer.public_bytes(serialization.Encoding.PEM).decode() + issuer_pem, key=key,
                    signer_id=hashlib.sha256(_public(key)).hexdigest()[:16], issuer_pem=issuer_pem)


def manifest(record: dict, *, product: str, version: str, title: str, mime: str, note: str | None = None) -> dict:
    """The C2PA manifest for one output: created by a trained model, and Khaos's own record of which one."""
    khaos = {"ai_generated": True, "assertion_version": ASSERTION_VERSION,
             **{k: record[k] for k in RECORD_FIELDS if record.get(k) is not None}}
    if note:
        khaos["note"] = note
    assertions = [
        {"label": "c2pa.actions", "data": {"actions": [{
            "action": "c2pa.created", "digitalSourceType": DIGITAL_SOURCE_TYPE,
            "softwareAgent": {"name": product, "version": version}}]}},
        {"label": KHAOS_ASSERTION, "data": khaos},
        {"label": LEGACY_ASSERTION, "data": khaos},
    ]
    return {"claim_generator_info": [{"name": product, "version": version}], "title": title, "format": mime,
            "assertions": assertions}


def tsa_url() -> str | None:
    url = os.environ.get("KHAOS_TSA_URL", DEFAULT_TSA_URL).strip()
    return None if url.lower() in ("", "off", "none", "0", "false") else url


_TSA_DOWN: dict[str, float] = {}


def _tsa_reachable(url: str) -> bool:
    """A quick connect to the service; a refusal is remembered for TSA_RETRY_S so blocked machines sign at once."""
    if time.monotonic() < _TSA_DOWN.get(url, 0.0):
        return False
    parts = urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        socket.create_connection((parts.hostname, port), timeout=TSA_PROBE_S).close()
    except OSError:
        _TSA_DOWN[url] = time.monotonic() + TSA_RETRY_S
        return False
    return True


def sign_file(path: Path | str, record: dict, identity: Identity, *, product: str, version: str,
              note: str | None = None, timestamp_url: str | None = "default") -> dict:
    """Embed a signed manifest in the audio file in place; time-stamped when the service answers, else signed
    without one. Returns {signed, timestamped, signer}; raises when the file cannot be signed."""
    import c2pa  # noqa: PLC0415
    path = Path(path)
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"{path.suffix} files cannot carry Content Credentials here")
    doc = json.dumps(manifest(record, product=product, version=version, title=path.name, mime=mime, note=note))
    url = tsa_url() if timestamp_url == "default" else timestamp_url
    mode = stat.S_IMODE(path.stat().st_mode)
    if url and not _tsa_reachable(url):
        url = None
    last = None
    for attempt_url in ([url, None] if url else [None]):
        signer = c2pa.Signer.from_callback(lambda data: identity.key.sign(data), c2pa.C2paSigningAlg.ED25519,
                                           identity.chain_pem, attempt_url)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".c2pa")
        os.close(fd)
        try:
            with open(path, "rb") as src, open(tmp, "wb") as dst:
                c2pa.Builder.from_json(doc).sign(signer, mime, src, dst)
            os.chmod(tmp, mode)    # the temp file is made private; the signed file keeps the original's mode
            _replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 — a time-stamping service that fails leaves the file unstamped
            last = exc
            if attempt_url:
                _TSA_DOWN[attempt_url] = time.monotonic() + TSA_RETRY_S    # a failed stamp, not only a closed port
            continue
        finally:
            signer.close()
            Path(tmp).unlink(missing_ok=True)
        if attempt_url:
            _TSA_DOWN.pop(attempt_url, None)
        return {"signed": True, "timestamped": attempt_url is not None, "signer": identity.signer_id}
    raise RuntimeError(f"the Content Credentials could not be signed: {last}")


MANIFEST_STORE_MAX_BYTES = 1_000_000


def manifest_store(path: Path | str) -> bytes | None:
    """The C2PA manifest store a signed WAV carries in its own RIFF chunk, as the bytes a soft-binding lookup serves;
    None for another format, an unsigned file, or one larger than the register keeps."""
    import struct  # noqa: PLC0415
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
            if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return None
            while True:
                chunk = fh.read(8)
                if len(chunk) < 8:
                    return None
                cid, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
                if cid == b"C2PA":
                    if size > MANIFEST_STORE_MAX_BYTES:
                        return None
                    data = fh.read(size)
                    return data if len(data) == size and data[4:8] == b"jumb" else None
                fh.seek(size + (size % 2), 1)
    except OSError:
        return None


def holds(got: dict) -> bool:
    """Whether what read_file returned is a signature over the file as it is: the SDK says so, or its one objection
    is the signer certificate's date."""
    return got.get("state") in SIGNED_STATES or bool(got.get("expired"))


def is_signed(path: Path | str) -> bool:
    """True when the file carries Content Credentials whose signature still holds."""
    return holds(read_file(path))


def read_file(path: Path | str) -> dict:
    """What a file's Content Credentials say: {present, state, expired, ai_generated, khaos, signer, timestamped,
    trusted, checked}. `state` is the SDK's verdict (Valid, Invalid, Trusted) and `expired` says it is Invalid only for
    the signer certificate's date; `checked` is False when the file could not be searched, `error` set if it failed."""
    out = {"present": False, "state": None, "expired": False, "ai_generated": False, "khaos": None, "signer": None,
           "timestamped": False, "trusted": False, "checked": True}
    if not available():
        out.update(checked=False, reason="the credentials extra is not installed")
        return out
    import c2pa  # noqa: PLC0415
    try:
        reader = c2pa.Reader(str(path))
    except c2pa.C2paError.ManifestNotFound as exc:
        out["reason"] = str(exc)[:200]
        return out
    except c2pa.C2paError.NotSupported as exc:
        out.update(checked=False, reason=str(exc)[:200])
        return out
    except Exception as exc:  # noqa: BLE001 — unreadable, or credentials the SDK cannot parse
        out.update(error=str(exc)[:200], reason=str(exc)[:200])
        return out
    try:
        # Summarised into a copy: what a summary read before it raised is never returned.
        return _summarise(reader, dict(out))
    except Exception as exc:  # noqa: BLE001 — a manifest the SDK cannot summarise reads as present but not intact
        out.update(present=True, state="Invalid", error=str(exc)[:200], reason=str(exc)[:200])
        return out
    finally:
        reader.close()


def _expired_only(validation: dict) -> bool:
    """True when the SDK's one objection, beside an issuer in no trust list, is the signer certificate's date, and it
    confirmed the claim's signature and the content hash: the file is as it was signed."""
    active = validation.get("activeManifest") or {}
    failed = {r.get("code") for r in active.get("failure", [])}
    passed = {r.get("code") for r in active.get("success", [])}
    return (EXPIRED in failed and failed <= {EXPIRED, UNTRUSTED} and not validation.get("ingredientDeltas")
            and SIGNATURE_VALIDATED in passed and any(code in passed for code in CONTENT_MATCHES))


def _summarise(reader, out: dict) -> dict:
    active = reader.get_active_manifest() or {}
    validation = reader.get_validation_results() or {}
    results = validation.get("activeManifest") or {}
    codes = {r.get("code") for kind in ("success", "informational", "failure") for r in results.get(kind, [])}
    out.update(present=bool(active), state=reader.get_validation_state())
    out["expired"] = bool(active) and out["state"] not in SIGNED_STATES and _expired_only(validation)
    found = {}
    for a in active.get("assertions", []):
        data = a.get("data") or {}
        if a.get("label", "").startswith("c2pa.actions"):
            out["ai_generated"] = out["ai_generated"] or any(
                act.get("digitalSourceType") == DIGITAL_SOURCE_TYPE for act in data.get("actions", []))
        elif a.get("label") in (KHAOS_ASSERTION, LEGACY_ASSERTION):
            found[a["label"]] = data
    out["khaos"] = found.get(KHAOS_ASSERTION) or found.get(LEGACY_ASSERTION)
    info = active.get("signature_info") or {}
    out["signer"] = info.get("common_name") or info.get("issuer")
    out["timestamped"] = "timeStamp.validated" in codes
    out["trusted"] = out["state"] == "Trusted"
    return out


def condition(path: Path | str) -> str:
    """What signing may do with a file: leave it ("intact", which an unchanged file stays after its signer
    certificate expires), sign it ("absent"), refuse it ("broken": credentials whose signature no longer holds, never
    re-signed silently) or nothing ("unsupported")."""
    if not available():
        raise CredentialsUnavailable("Content Credentials need the credentials extra (c2pa-python)")
    if Path(path).suffix.lower() not in MIME_TYPES:
        return "unsupported"
    got = read_file(path)
    if holds(got):
        return "intact"
    if got["present"] or got.get("error"):
        return "broken"
    return "absent" if got["checked"] else "unsupported"


def agrees(path: Path | str, record: dict) -> bool:
    """True when the file's signature holds and its Khaos assertion says what `record` says, field by field."""
    got = read_file(path)
    khaos = got.get("khaos")
    if not holds(got) or not khaos:
        return False
    if any(khaos.get(k) is None for k in AGREEMENT_REQUIRED):
        return False
    return all(khaos[k] == record.get(k) for k in AGREEMENT_FIELDS if k in khaos)
