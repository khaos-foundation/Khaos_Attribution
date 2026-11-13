"""The verifier: origin is confirmed only by an index match, a signature only says the record is intact, a
watermark alone is never proof, and a negative is never proof a person made it."""

import json
import math
import shutil
import struct
import subprocess
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from khaos_attribution import credentials as C
from khaos_attribution import verify as V
from khaos_attribution import watermark as W

needs_c2pa = pytest.mark.skipif(not C.available(), reason="the credentials extra is not installed")
RECORD = {"artist_id": "art", "artist_name": "Art", "adapter_version": "run_x", "generation_id": "job/a.wav",
          "watermark_id": 59368, "timestamp": "2026-10-15T10:00:00"}


def _wav(path, seconds=3.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"".join(struct.pack("<h", int(5000 * math.sin(2 * math.pi * (220 + 200 * (i // 2205 % 5)) * i / 22050)))
                               for i in range(int(22050 * seconds))))
    return path


def _signed(tmp_path, name="a.wav"):
    ident = C.load_identity(tmp_path / "id", product="Khaos Workshop", organisation="Khaos Test Ltd")
    wav = _wav(tmp_path / name)
    C.sign_file(wav, RECORD, ident, product="Khaos Workshop", version="0", timestamp_url=None)
    return wav


@pytest.fixture
def no_detector(monkeypatch):
    monkeypatch.setattr(W.Watermarker, "available", staticmethod(lambda: False))


@pytest.fixture
def detector(monkeypatch):
    """A stand-in detector that hears the given codeword."""
    heard = {"codeword": 59368, "probability": 0.97}
    monkeypatch.setattr(W.Watermarker, "available", staticmethod(lambda: True))
    monkeypatch.setattr(W.Watermarker, "detect_array",
                        staticmethod(lambda audio, sr: (heard["probability"], heard["codeword"], False)))
    return heard


def test_an_unmarked_file_is_not_called_human(tmp_path, no_detector):
    r = V.verify_file(_wav(tmp_path / "plain.wav"))
    assert r["verdict"] == "none" and "not proof a person made it" in r["message"]
    assert r["watermark"]["checked"] is False and r["register"] is None


@needs_c2pa
def test_an_intact_signature_is_signed_not_confirmed(tmp_path, no_detector):
    r = V.verify_file(_signed(tmp_path))
    assert r["verdict"] == "signed" and "does not prove" in r["message"]
    assert r["content_credentials"]["ai_generated"] and r["content_credentials"]["record"]["artist_id"] == "art"


@needs_c2pa
def test_a_changed_file_reads_as_tampered(tmp_path, no_detector):
    wav = _signed(tmp_path)
    data = bytearray(wav.read_bytes())
    data[60] ^= 0xFF
    wav.write_bytes(bytes(data))
    assert V.verify_file(wav)["verdict"] == "tampered"


def test_a_watermark_alone_is_marked_and_checked_against_the_record(tmp_path, detector):
    r = V.verify_file(_wav(tmp_path / "plain.wav"))
    assert r["verdict"] == "marked" and "does not prove" in r["message"]
    assert r["watermark"]["watermark_id"] == 59368 and r["watermark"]["payload"] == W.decode_codeword(59368)[0]


@needs_c2pa
def test_a_watermark_that_disagrees_with_the_record_is_said(tmp_path, detector):
    detector["codeword"] = W.encode_payload(1001)
    r = V.verify_file(_signed(tmp_path, "b.wav"))
    assert r["verdict"] == "signed" and r["watermark"]["matches_record"] is False
    assert "says the audio is AI-generated" in r["message"]


class _Register(BaseHTTPRequestHandler):
    answer = {"grade": "output", "output": {"serial": 65536}, "model": {"artist_name": "Art", "page": "https://r/models/65536"}}
    seen = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, self.headers.get("X-Khaos-Verify-Key"), body))
        out = json.dumps(type(self).answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def register():
    server = HTTPServer(("127.0.0.1", 0), _Register)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Register.seen = []
    _Register.answer = {"grade": "output", "output": {"serial": 65536}, "model": {"artist_name": "Art", "page": "https://r/models/65536"}}
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_only_an_index_match_confirms_origin(tmp_path, register, detector):
    r = V.verify_file(_wav(tmp_path / "plain.wav"), register_url=register, register_key="press-key")
    assert r["verdict"] == "confirmed" and r["register"]["model"]["artist_name"] == "Art"
    path, key, body = _Register.seen[-1]
    assert path == "/api/verify" and key == "press-key" and body["watermark_id"] == 59368 and body["phases"]
    _Register.answer = {"grade": "bucket", "payload": 1}
    assert V.verify_file(_wav(tmp_path / "plain.wav"), register_url=register)["verdict"] == "marked"


@needs_c2pa
def test_a_register_that_cannot_answer_is_said_and_confirms_nothing(tmp_path, no_detector):
    r = V.verify_file(_signed(tmp_path), register_url="http://127.0.0.1:9")
    assert r["verdict"] == "signed" and r["register"]["grade"] is None and r["register"]["error"]


@pytest.mark.parametrize("fmt,subtype", [("OGG", "VORBIS"), ("MP3", "MPEG_LAYER_III"), ("FLAC", "PCM_16")])
def test_compressed_formats_are_read(tmp_path, no_detector, fmt, subtype):
    sf = pytest.importorskip("soundfile")
    audio, sr = sf.read(str(_wav(tmp_path / "src.wav")), dtype="float32")
    out = tmp_path / f"x.{fmt.lower()}"
    sf.write(str(out), audio, sr, format=fmt, subtype=subtype)
    assert V.verify_file(out)["verdict"] == "none"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_m4a_is_read_through_ffmpeg(tmp_path, no_detector):
    out = tmp_path / "x.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(_wav(tmp_path / "src.wav")), "-c:a", "aac", str(out)],
                   check=True)
    audio, sr = V.load_audio(out)
    assert sr > 0 and len(audio) > sr
    assert V.verify_file(out)["verdict"] == "none"


@needs_c2pa
def test_the_command_prints_the_verdict_and_fails_on_unreadable_files(tmp_path, capsys, no_detector):
    assert V.main([str(_signed(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "claims to come from Khaos" in out and "Content Credentials: intact, AI-generated" in out
    (tmp_path / "junk.wav").write_bytes(b"not audio")
    assert V.main([str(tmp_path / "junk.wav"), "--json"]) == 2
    assert "error" in json.loads(capsys.readouterr().out)


def test_nothing_longer_than_the_cap_is_decoded(tmp_path, no_detector, monkeypatch):
    monkeypatch.setattr(V, "MAX_VERIFY_SECONDS", 2)
    with pytest.raises(V.AudioUnreadable, match="longer than"):
        V.verify_file(_wav(tmp_path / "long.wav", seconds=3.0))
    assert V.audio_seconds(_wav(tmp_path / "short.wav", seconds=1.0)) == pytest.approx(1.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_ffmpeg_never_decodes_past_the_cap(tmp_path, monkeypatch):
    out = tmp_path / "x.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(_wav(tmp_path / "src.wav", seconds=3.0)), "-c:a", "aac", str(out)],
                   check=True)
    monkeypatch.setattr(V, "audio_seconds", lambda p: None)   # a header that cannot be read up front
    with pytest.raises(V.AudioUnreadable, match="longer than"):
        V.load_audio(out, max_seconds=1)


def test_a_register_answer_that_is_not_an_object_is_an_error(tmp_path, register, no_detector):
    _Register.answer = ["not", "an", "object"]
    try:
        r = V.verify_file(_wav(tmp_path / "plain.wav"), register_url=register)
    finally:
        _Register.answer = {"grade": "output", "output": {"serial": 65536}, "model": {"artist_name": "Art"}}
    assert r["verdict"] == "none" and r["register"]["grade"] is None and "not a JSON object" in r["register"]["error"]


def test_one_bad_file_never_stops_the_batch(tmp_path, capsys, no_detector, monkeypatch):
    real = V.verify_file
    monkeypatch.setattr(V, "verify_file", lambda f, **k: (_ for _ in ()).throw(RuntimeError("boom")) if f.name == "bad.wav" else real(f, **k))
    good, bad = _wav(tmp_path / "good.wav"), _wav(tmp_path / "bad.wav")
    assert V.main([str(bad), str(good)]) == 2
    out = capsys.readouterr().out
    assert "bad.wav: boom" in out and "good.wav: No Khaos mark found" in out


def _plain(path, frames, channels=1, level=0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(struct.pack("<h", level) * frames * channels)
    return path


@pytest.mark.parametrize("frames, channels", [(220, 1), (22050 * 3, 1), (22050 * 3, 2)])
def test_a_clip_too_short_or_silent_is_none_and_the_register_is_not_asked(tmp_path, no_detector, frames, channels):
    """Ten milliseconds, and three silent seconds in mono and stereo: nothing to fingerprint, said so, never a guess."""
    r = V.verify_file(_plain(tmp_path / "quiet.wav", frames, channels), register_url="http://127.0.0.1:9")
    assert r["verdict"] == "none" and "not proof a person made it" in r["message"]
    assert r["register"] == {"grade": None, "url": "http://127.0.0.1:9", "error": "too short or too quiet to fingerprint"}


def test_a_file_with_no_audio_is_never_handed_to_the_detector(tmp_path, detector):
    """The stand-in detector hears a mark in anything; a file of no frames is not listened to at all."""
    r = V.verify_file(_plain(tmp_path / "empty.wav", 0))
    assert r["verdict"] == "none" and r["watermark"] == {"checked": False, "reason": "the file holds no audio"}


@pytest.fixture
def silence(detector):
    detector["probability"] = 0.02
    return detector


@needs_c2pa
def test_an_intact_record_the_watermark_agrees_with_is_detected(tmp_path, detector):
    r = V.verify_file(_signed(tmp_path))
    assert r["verdict"] == "signed" and r["watermark"]["matches_record"] is True and r["status"] == "DETECTED"


@needs_c2pa
def test_a_disagreeing_watermark_or_a_broken_record_is_inconclusive(tmp_path, detector):
    detector["codeword"] = W.encode_payload(1001)
    assert V.verify_file(_signed(tmp_path, "b.wav"))["status"] == "INCONCLUSIVE"
    detector["codeword"] = 59368
    wav = _signed(tmp_path, "c.wav")
    data = bytearray(wav.read_bytes())
    data[60] ^= 0xFF
    wav.write_bytes(bytes(data))
    r = V.verify_file(wav)
    assert r["verdict"] == "tampered" and r["status"] == "INCONCLUSIVE"


def test_a_watermark_alone_is_inconclusive(tmp_path, detector):
    r = V.verify_file(_wav(tmp_path / "plain.wav"))
    assert r["verdict"] == "marked" and r["status"] == "INCONCLUSIVE"


@needs_c2pa
def test_nothing_found_after_every_check_is_not_detected(tmp_path, silence):
    r = V.verify_file(_wav(tmp_path / "plain.wav"))
    assert r["verdict"] == "none" and r["status"] == "NOT_DETECTED"


def test_nothing_found_without_every_check_is_not_a_negative(tmp_path, no_detector):
    assert V.verify_file(_wav(tmp_path / "plain.wav"))["status"] == "INCONCLUSIVE"


@needs_c2pa
def test_a_format_credentials_cannot_live_in_is_not_supported(tmp_path, silence):
    sf = pytest.importorskip("soundfile")
    audio, sr = sf.read(str(_wav(tmp_path / "src.wav")), dtype="float32")
    sf.write(str(tmp_path / "x.ogg"), audio, sr, format="OGG", subtype="VORBIS")
    r = V.verify_file(tmp_path / "x.ogg")
    assert r["verdict"] == "none" and r["status"] == "NOT_SUPPORTED"


@needs_c2pa
def test_a_reader_that_fails_unexpectedly_is_an_error(tmp_path, silence, monkeypatch):
    import c2pa
    monkeypatch.setattr(c2pa, "Reader", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    r = V.verify_file(_wav(tmp_path / "plain.wav"))
    assert r["status"] == "ERROR" and r["content_credentials"]["error"] == "boom"


def test_a_file_that_cannot_be_read_is_an_error_in_the_json(tmp_path, capsys, no_detector):
    (tmp_path / "junk.wav").write_bytes(b"not audio")
    assert V.main([str(tmp_path / "junk.wav"), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "ERROR"


def test_every_status_word_is_one_of_five():
    assert set(V.STATUSES) == {"DETECTED", "NOT_DETECTED", "INCONCLUSIVE", "NOT_SUPPORTED", "ERROR"}
