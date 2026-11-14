"""A signer certificate that runs out says nothing about the file: an unchanged file still holds and is never signed
again, the verifier says the certificate expired, and a file changed after signing stays broken whatever the date."""

import dataclasses
import datetime
import math
import struct
import time
import wave

import pytest

from khaos_attribution import credentials as C
from khaos_attribution import verify as V
from khaos_attribution import watermark as W

pytestmark = pytest.mark.skipif(not C.available(), reason="the credentials extra is not installed")
RECORD = {"artist_id": "art", "artist_name": "Art", "adapter_version": "run_x", "adapter_hash": "abc",
          "generation_id": "job/a.wav", "watermark_id": 59368, "timestamp": "2026-10-15T10:00:00"}
LIFE_S = 3


def _wav(path, seconds=1.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * i / 16000)))
                               for i in range(int(16000 * seconds))))
    return path


def _short_lived(directory):
    """An installation's identity whose signer certificate lasts seconds, and the moment it runs out."""
    from cryptography.hazmat.primitives import serialization
    ident = C.load_identity(directory, product="Khaos Workshop", organisation="Khaos Test Ltd")
    issuer, issuer_key, signer, signer_key = C._loaded({n: directory / n for n in C.IDENTITY_FILES})
    brief = C._certificate(signer.subject, signer_key, issuer.subject, issuer_key, ca=False, days=LIFE_S / 86400)
    pem = brief.public_bytes(serialization.Encoding.PEM).decode()
    return dataclasses.replace(ident, chain_pem=pem + ident.issuer_pem), brief.not_valid_after_utc


def _change(wav):
    data = bytearray(wav.read_bytes())
    data[60] ^= 0xFF   # inside the audio samples, after the RIFF header
    wav.write_bytes(bytes(data))


@pytest.fixture(scope="module")
def lapsed(tmp_path_factory):
    """(kept, changed): two files signed without a timestamp by a certificate that has since expired, one of them
    changed after it was signed."""
    d = tmp_path_factory.mktemp("lapsed")
    ident, ends = _short_lived(d / "id")
    kept, changed = _wav(d / "kept.wav"), _wav(d / "changed.wav")
    for wav in (kept, changed):
        C.sign_file(wav, RECORD, ident, product="Khaos Workshop", version="0.1.0", timestamp_url=None)
    in_date = C.read_file(kept)["state"]
    _change(changed)
    time.sleep(max(0.0, (ends - datetime.datetime.now(datetime.timezone.utc)).total_seconds()) + 1.1)
    assert in_date in C.SIGNED_STATES
    return kept, changed


@pytest.fixture
def no_detector(monkeypatch):
    monkeypatch.setattr(W.Watermarker, "available", staticmethod(lambda: False))


@pytest.fixture
def detector(monkeypatch):
    """A stand-in detector that hears the record's own codeword."""
    monkeypatch.setattr(W.Watermarker, "available", staticmethod(lambda: True))
    monkeypatch.setattr(W.Watermarker, "detect_array", staticmethod(lambda audio, sr: (0.97, 59368, False)))


def test_an_unchanged_file_still_holds_after_its_signer_certificate_expires(lapsed):
    """Guards the rooms: "intact" is the one condition they neither refuse nor sign, and they then ask agrees()."""
    kept, _ = lapsed
    assert C.condition(kept) == "intact" and C.is_signed(kept)
    assert C.agrees(kept, RECORD) and not C.agrees(kept, {**RECORD, "adapter_hash": "def"})
    got = C.read_file(kept)
    assert got["state"] == "Invalid" and got["expired"] is True and got["timestamped"] is False


def test_a_file_changed_after_signing_is_broken_whatever_the_certificates_date(lapsed):
    _, changed = lapsed
    got = C.read_file(changed)
    assert got["state"] == "Invalid" and got["expired"] is False
    assert C.condition(changed) == "broken" and not C.is_signed(changed) and not C.agrees(changed, RECORD)


def _results(failure=("signingCredential.untrusted", "signingCredential.expired"),
             success=("claimSignature.validated", "assertion.hashedURI.match", "assertion.dataHash.match"),
             informational=("signingCredential.ocsp.skipped",), **more):
    """The SDK's validation results, shaped as it returns them for a Khaos file past its certificate's date."""
    lists = {"failure": failure, "success": success, "informational": informational}
    return {"activeManifest": {k: [{"code": c} for c in v] for k, v in lists.items()}, **more}


def test_only_the_certificates_date_reads_as_expired():
    assert C._expired_only(_results())
    assert C._expired_only(_results(success=("claimSignature.validated", "assertion.bmffHash.match")))
    stamped = _results(success=("timeStamp.validated", "claimSignature.validated", "assertion.dataHash.match"),
                       informational=("timeStamp.untrusted",))
    assert C._expired_only(stamped)


@pytest.mark.parametrize("results", [
    _results(failure=("signingCredential.untrusted",)),
    _results(failure=("signingCredential.untrusted", "signingCredential.expired", "assertion.dataHash.mismatch")),
    _results(failure=("signingCredential.untrusted", "signingCredential.expired", "assertion.hashedURI.mismatch")),
    _results(failure=("signingCredential.untrusted", "signingCredential.expired", "claimSignature.mismatch")),
    _results(failure=("signingCredential.untrusted", "signingCredential.expired", "signingCredential.revoked")),
    _results(failure=("signingCredential.untrusted", "signingCredential.expired", "timeStamp.mismatch")),
    _results(success=("claimSignature.validated", "assertion.hashedURI.match")),
    _results(success=("assertion.dataHash.match",)),
    _results(ingredientDeltas=[{"validationDeltas": {"failure": [{"code": "assertion.dataHash.mismatch"}]}}]),
    {},
])
def test_any_other_objection_is_not_an_expiry(results):
    """Guards the rule's edges: every failure beside the date, a hash or signature the SDK did not confirm, and an
    ingredient's own results all leave the file broken."""
    assert not C._expired_only(results)


def test_the_verifier_says_the_certificate_expired_never_that_the_file_changed(lapsed, no_detector):
    kept, _ = lapsed
    r = V.verify_file(kept)
    assert r["verdict"] == "expired" and r["status"] == "INCONCLUSIVE"
    assert "certificate is past its date and the file carries no timestamp" in r["message"]
    assert "changed after it was signed" not in r["message"] and "says the audio is AI-generated" in r["message"]
    creds = r["content_credentials"]
    assert creds["intact"] and creds["expired"] and creds["state"] == "Invalid"
    assert creds["record"]["artist_id"] == "art"
    assert "signing certificate expired" in V._line(r) and "BROKEN" not in V._line(r)


def test_an_expired_record_the_watermark_agrees_with_is_detected_as_a_signed_one_is(lapsed, detector):
    kept, _ = lapsed
    r = V.verify_file(kept)
    assert r["verdict"] == "expired" and r["watermark"]["matches_record"] is True and r["status"] == "DETECTED"


def test_a_changed_file_reads_as_tampered_after_the_certificate_expires(lapsed, detector):
    _, changed = lapsed
    r = V.verify_file(changed)
    assert r["verdict"] == "tampered" and r["status"] == "INCONCLUSIVE"
    assert "changed after it was signed" in r["message"] and not r["content_credentials"]["intact"]


def test_a_time_stamped_file_is_not_told_it_carries_no_timestamp():
    said = V.message("expired", {"timestamped": True})
    assert "certificate is past its date" in said and "no timestamp" not in said and "since" not in said
    assert V.VERDICTS["expired"] == said


def test_every_verdict_is_one_of_six():
    """Guards the rooms and the agents' tool, which name the verdict kinds: a new one is adopted there too."""
    assert set(V.VERDICTS) == {"confirmed", "signed", "expired", "tampered", "marked", "none"}


def test_an_in_date_signature_is_signed_not_expired(tmp_path, no_detector):
    ident = C.load_identity(tmp_path / "id", product="Khaos Workshop", organisation="Khaos Test Ltd")
    wav = _wav(tmp_path / "a.wav")
    C.sign_file(wav, RECORD, ident, product="Khaos Workshop", version="0.1.0", timestamp_url=None)
    r = V.verify_file(wav)
    assert r["verdict"] == "signed" and r["content_credentials"]["expired"] is False
