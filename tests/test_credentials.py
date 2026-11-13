"""Content Credentials: one identity per installation, a signed manifest that says AI-generated and carries Khaos's
record (never the prompt), and a reader that sees tampering."""

import math
import os
import stat
import struct
import wave

import pytest

from khaos_attribution import credentials as C

pytestmark = pytest.mark.skipif(not C.available(), reason="the credentials extra is not installed")
RECORD = {"artist_id": "art", "artist_name": "Art", "adapter_version": "run_x", "adapter_hash": "abc",
          "base_model": "ACE-Step 1.5 (turbo)", "base_model_licence": "MIT", "generation_id": "job/a.wav",
          "watermark_id": 59368, "prompt": "never inside the file", "timestamp": "2026-10-15T10:00:00"}


def _wav(path, seconds=1.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * i / 16000)))
                               for i in range(int(16000 * seconds))))
    return path


def test_one_identity_per_installation_kept_private(tmp_path):
    a = C.load_identity(tmp_path / "id", product="Khaos Workshop", organisation="Khaos Test Ltd")
    b = C.load_identity(tmp_path / "id", product="Khaos Workshop", organisation="Khaos Test Ltd")
    assert a.signer_id == b.signer_id and a.chain_pem == b.chain_pem
    for name in ("issuer.key", "signer.key"):
        assert stat.S_IMODE(os.stat(tmp_path / "id" / name).st_mode) == 0o600
    assert "BEGIN CERTIFICATE" in a.chain_pem and a.chain_pem.count("BEGIN CERTIFICATE") == 2


def test_a_signed_output_says_ai_generated_and_carries_the_record_but_not_the_prompt(tmp_path):
    ident = C.load_identity(tmp_path / "id", product="Khaos Workshop", organisation="Khaos Test Ltd")
    wav = _wav(tmp_path / "a.wav")
    out = C.sign_file(wav, RECORD, ident, product="Khaos Workshop", version="0.1.0", timestamp_url=None)
    assert out == {"signed": True, "timestamped": False, "signer": ident.signer_id}
    got = C.read_file(wav)
    assert got["present"] and got["state"] in ("Valid", "Trusted") and got["ai_generated"]
    assert got["khaos"]["ai_generated"] is True and got["khaos"]["watermark_id"] == 59368
    assert got["khaos"]["base_model_licence"] == "MIT" and "prompt" not in got["khaos"]
    assert b"never inside the file" not in wav.read_bytes()


def test_tampered_audio_breaks_the_signature(tmp_path):
    ident = C.load_identity(tmp_path / "id", product="Khaos Workshop", organisation="Khaos Test Ltd")
    wav = _wav(tmp_path / "a.wav")
    C.sign_file(wav, RECORD, ident, product="Khaos Workshop", version="0.1.0", timestamp_url=None)
    data = bytearray(wav.read_bytes())
    data[60] ^= 0xFF   # inside the audio samples, after the RIFF header
    wav.write_bytes(bytes(data))
    assert C.read_file(wav)["state"] not in ("Valid", "Trusted")


def test_a_file_without_credentials_says_so(tmp_path):
    got = C.read_file(_wav(tmp_path / "plain.wav"))
    assert got["present"] is False and got["ai_generated"] is False


def test_the_manifest_names_the_trained_model_source_type():
    m = C.manifest(RECORD, product="Khaos Workshop", version="0.1.0", title="a.wav", mime="audio/wav")
    action = m["assertions"][0]["data"]["actions"][0]
    assert action["action"] == "c2pa.created" and action["digitalSourceType"] == C.DIGITAL_SOURCE_TYPE
    assert set(m["assertions"][1]["data"]) <= set(C.RECORD_FIELDS) | {"ai_generated", "assertion_version"}


def test_concurrent_first_use_makes_one_coherent_identity(tmp_path):
    """Two makers at once must leave one issuer whose signer it issued, never a crossed chain."""
    from concurrent.futures import ThreadPoolExecutor
    from cryptography import x509
    for trial in range(5):
        d = tmp_path / f"id{trial}"
        with ThreadPoolExecutor(4) as pool:
            ids = list(pool.map(lambda _: C.load_identity(d, product="Khaos Workshop", organisation="Khaos Test Ltd"), range(4)))
        assert len({i.signer_id for i in ids}) == 1
        x509.load_pem_x509_certificate((d / "signer.pem").read_bytes()).verify_directly_issued_by(
            x509.load_pem_x509_certificate((d / "issuer.pem").read_bytes()))


def test_a_signer_that_is_not_the_issuers_is_remade(tmp_path):
    a = C.load_identity(tmp_path / "a", product="Khaos Workshop", organisation="Khaos Test Ltd")
    C.load_identity(tmp_path / "b", product="Khaos Workshop", organisation="Khaos Test Ltd")
    for name in ("signer.pem", "signer.key"):
        (tmp_path / "a" / name).write_bytes((tmp_path / "b" / name).read_bytes())
    again = C.load_identity(tmp_path / "a", product="Khaos Workshop", organisation="Khaos Test Ltd")
    assert again.issuer_pem == a.issuer_pem and again.signer_id != a.signer_id
    from cryptography import x509
    signer, issuer = x509.load_pem_x509_certificates(again.chain_pem.encode())
    signer.verify_directly_issued_by(issuer)
    wav = _wav(tmp_path / "x.wav")
    C.sign_file(wav, RECORD, again, product="Khaos Workshop", version="0.1.0", timestamp_url=None)
    assert C.is_signed(wav)


def test_an_unreachable_timestamp_service_does_not_hold_the_signature(tmp_path, monkeypatch):
    import time
    monkeypatch.setattr(C, "TSA_PROBE_S", 0.5)
    ident = C.load_identity(tmp_path / "id", product="Khaos Workshop", organisation="Khaos Test Ltd")
    t = time.monotonic()
    for n in range(3):
        out = C.sign_file(_wav(tmp_path / f"{n}.wav"), RECORD, ident, product="Khaos Workshop", version="0.1.0",
                          timestamp_url="http://10.255.255.1:9")
        assert out["signed"] and not out["timestamped"]
    assert time.monotonic() - t < 5


def test_is_signed_is_false_for_a_broken_signature(tmp_path):
    ident = C.load_identity(tmp_path / "id", product="Khaos Workshop", organisation="Khaos Test Ltd")
    wav = _wav(tmp_path / "a.wav")
    C.sign_file(wav, RECORD, ident, product="Khaos Workshop", version="0.1.0", timestamp_url=None)
    assert C.is_signed(wav)
    data = bytearray(wav.read_bytes())
    data[60] ^= 0xFF
    wav.write_bytes(bytes(data))
    assert not C.is_signed(wav) and not C.is_signed(_wav(tmp_path / "plain.wav"))


def test_a_signed_file_keeps_its_permissions_and_off_is_read_in_any_case(tmp_path, monkeypatch):
    import os
    import stat
    import wave
    from khaos_attribution import credentials as C
    wav = tmp_path / "p.wav"
    with wave.open(str(wav), "wb") as fh:
        fh.setnchannels(1); fh.setsampwidth(2); fh.setframerate(8000); fh.writeframes(b"\x00\x00" * 8000)
    os.chmod(wav, 0o644)
    ident = C.load_identity(tmp_path / "id", product="Khaos Test", organisation="Khaos Test Ltd")
    C.sign_file(wav, {"generation_id": "g", "watermark_id": 59368}, ident, product="Khaos Test", version="0", timestamp_url=None)
    assert stat.S_IMODE(wav.stat().st_mode) == 0o644
    for value in ("off", "OFF", "none", ""):
        monkeypatch.setenv("KHAOS_TSA_URL", value)
        assert C.tsa_url() is None, value


def _identity(tmp_path, organisation="Khaos Test Ltd"):
    return C.load_identity(tmp_path / "id", product="Khaos Workshop", organisation=organisation)


def _sign(tmp_path, ident, name="a.wav", record=RECORD):
    wav = _wav(tmp_path / name)
    C.sign_file(wav, record, ident, product="Khaos Workshop", version="0.1.0", timestamp_url=None)
    return wav


def _assertions(path):
    import c2pa
    reader = c2pa.Reader(str(path))
    try:
        return {a["label"]: a.get("data") for a in reader.get_active_manifest()["assertions"]}
    finally:
        reader.close()


def test_both_labels_are_written_and_the_current_one_is_read(tmp_path):
    got = _assertions(_sign(tmp_path, _identity(tmp_path)))
    assert got[C.KHAOS_ASSERTION] == got[C.LEGACY_ASSERTION]
    assert got[C.KHAOS_ASSERTION]["watermark_id"] == 59368


def test_a_file_with_only_the_old_label_still_reads(tmp_path, monkeypatch):
    real = C.manifest

    def old(record, **kw):
        m = real(record, **kw)
        m["assertions"] = [a for a in m["assertions"] if a["label"] != C.KHAOS_ASSERTION]
        return m
    monkeypatch.setattr(C, "manifest", old)
    wav = _sign(tmp_path, _identity(tmp_path))
    assert C.KHAOS_ASSERTION not in _assertions(wav)
    assert C.read_file(wav)["khaos"]["artist_id"] == "art"


def test_the_watermark_travels_as_a_soft_binding(tmp_path):
    import base64
    got = _assertions(_sign(tmp_path, _identity(tmp_path)))[C.SOFT_BINDING]
    assert got["alg"] == C.WATERMARK_ALG
    assert int.from_bytes(base64.b64decode(got["blocks"][0]["value"]), "big") == 59368
    unmarked = C.manifest({**RECORD, "watermark_id": None}, product="p", version="0", title="t", mime="audio/wav")
    assert C.SOFT_BINDING not in [a["label"] for a in unmarked["assertions"]]


@pytest.mark.parametrize("fmt", ["FLAC", "MP3"])
def test_flac_and_mp3_are_signed_and_read_back(tmp_path, fmt):
    sf = pytest.importorskip("soundfile")
    audio, sr = sf.read(str(_wav(tmp_path / "src.wav")), dtype="float32")
    out = tmp_path / f"x.{fmt.lower()}"
    sf.write(str(out), audio, sr, format=fmt)
    plain = C.read_file(out)
    assert plain["present"] is False and plain["checked"] and "error" not in plain
    C.sign_file(out, RECORD, _identity(tmp_path), product="Khaos Workshop", version="0", timestamp_url=None)
    got = C.read_file(out)
    assert got["state"] in C.SIGNED_STATES and got["khaos"]["watermark_id"] == 59368 and got["ai_generated"]
    assert len(sf.read(str(out))[0]) == len(audio)


def test_a_format_the_sdk_cannot_open_is_not_checked(tmp_path):
    sf = pytest.importorskip("soundfile")
    audio, sr = sf.read(str(_wav(tmp_path / "src.wav")), dtype="float32")
    sf.write(str(tmp_path / "x.ogg"), audio, sr, format="OGG", subtype="VORBIS")
    got = C.read_file(tmp_path / "x.ogg")
    assert got["checked"] is False and got["present"] is False and "error" not in got
    assert C.condition(tmp_path / "x.ogg") == "unsupported"


def test_the_condition_says_whether_signing_may_touch_a_file(tmp_path):
    ident = _identity(tmp_path)
    assert C.condition(_wav(tmp_path / "plain.wav")) == "absent"
    wav = _sign(tmp_path, ident)
    assert C.condition(wav) == "intact"
    data = bytearray(wav.read_bytes())
    data[60] ^= 0xFF
    wav.write_bytes(bytes(data))
    assert C.condition(wav) == "broken"


def test_the_manifest_agrees_with_its_record_and_only_its_record(tmp_path):
    wav = _sign(tmp_path, _identity(tmp_path))
    assert C.agrees(wav, RECORD)
    assert not C.agrees(wav, {**RECORD, "adapter_hash": "def"})
    assert not C.agrees(wav, {**RECORD, "watermark_id": 1})
    assert C.agrees(wav, {**RECORD, "prompt": "the sidecar keeps it", "register_serial": 65537})
    data = bytearray(wav.read_bytes())
    data[60] ^= 0xFF
    wav.write_bytes(bytes(data))
    assert not C.agrees(wav, RECORD)


def test_a_field_an_older_version_never_signed_is_not_compared(tmp_path):
    older = {k: v for k, v in RECORD.items() if k != "generation_id"}
    wav = _sign(tmp_path, _identity(tmp_path), record=older)
    assert C.agrees(wav, RECORD)
    bare = _sign(tmp_path, _identity(tmp_path), name="b.wav", record={"generation_id": "g", "watermark_id": 59368})
    assert not C.agrees(bare, {"generation_id": "g", "watermark_id": 59368})


def test_a_new_company_name_reissues_the_certificates_over_the_same_keys(tmp_path):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    before = _identity(tmp_path, "Old Name Ltd")
    keys = {n: (tmp_path / "id" / n).read_bytes() for n in ("issuer.key", "signer.key")}
    old_issuer = (tmp_path / "id" / "issuer.pem").read_bytes()
    signed_before = _sign(tmp_path, before)
    after = _identity(tmp_path, "KHAOS FOUNDATION LIMITED")
    assert after.signer_id == before.signer_id and after.chain_pem != before.chain_pem
    assert {n: (tmp_path / "id" / n).read_bytes() for n in keys} == keys
    signer, issuer = x509.load_pem_x509_certificates(after.chain_pem.encode())
    signer.verify_directly_issued_by(issuer)
    for cert in (signer, issuer):
        assert cert.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)[0].value == "KHAOS FOUNDATION LIMITED"
    kept = list((tmp_path / "id" / "archive").glob("*/issuer.pem"))
    assert len(kept) == 1 and kept[0].read_bytes() == old_issuer
    assert C.is_signed(signed_before) and C.is_signed(_sign(tmp_path, after, name="b.wav"))
    assert _identity(tmp_path, "KHAOS FOUNDATION LIMITED").chain_pem == after.chain_pem
    assert len(list((tmp_path / "id" / "archive").iterdir())) == 1
