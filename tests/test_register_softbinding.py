"""The C2PA soft-binding lookup on the register: a released piece's manifest kept beside its fingerprint, found again
from a clip of its audio or its fingerprint, and served as the manifest store; a watermark only ever narrows."""
import base64
import io
import json
import sys

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
sf = pytest.importorskip("soundfile")
c2pa = pytest.importorskip("c2pa")

from khaos_attribution import credentials as C  # noqa: E402
from khaos_attribution.fingerprint import fingerprint_array, query_phases  # noqa: E402
from khaos_attribution.watermark import encode_payload  # noqa: E402
from register_service import softbinding as SB  # noqa: E402
from tests.test_register_index import _clip, _music, register  # noqa: E402, F401
from tests.test_register_service import _machine, _signed  # noqa: E402


def _signed_wav(tmp_path, audio, sr) -> bytes:
    """A real signed WAV's manifest store, as a Workshop or a Space would send it at release."""
    wav = tmp_path / "piece.wav"
    sf.write(str(wav), audio, sr, subtype="PCM_16")
    ident = C.load_identity(tmp_path / "identity", product="Khaos Test", organisation="Khaos Test Ltd")
    assert C.sign_file(wav, {"generation_id": "gen-1", "watermark_id": 59368, "artist_name": "Artist X"}, ident,
                       product="Khaos Test", version="0", timestamp_url=None)["signed"]
    store = C.manifest_store(wav)
    assert store and store[4:8] == b"jumb"
    return store


def _wav_bytes(audio, sr) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _released(register, tmp_path, with_manifest=True):
    key, client = _machine(register, "artist-one")
    signed = _signed(key)
    serial = client.publish(signed)["serial"]
    audio, sr = _music(1)
    manifest = _signed_wav(tmp_path, audio, sr) if with_manifest else None
    out = client.release("gen-1", serial, fingerprint_array(audio, sr), manifest=manifest)
    return signed, serial, audio, sr, manifest, out, client


def test_the_lookup_lists_its_algorithms(register):
    _, _, http = register
    got = http.get(f"{SB.PREFIX}/services/supportedAlgorithms").json()
    assert got == {"watermarks": [{"alg": SB.WATERMARK_ALG}], "fingerprints": [{"alg": SB.FINGERPRINT_ALG}]}


def test_a_clip_of_a_released_piece_leads_back_to_its_manifest(register, tmp_path):
    _, _, http = register
    signed, serial, audio, sr, manifest, out, _ = _released(register, tmp_path)
    assert out["manifest_kept"] is True
    clip = _wav_bytes(_clip(audio, sr, 3.0, 5.0), sr)
    got = http.post(f"{SB.PREFIX}/matches/byContent", content=clip, headers={"Content-Type": "audio/wav"}).json()
    assert [m["manifestId"] for m in got["matches"]] == ["gen-1"]
    assert got["matches"][0]["endpoint"] == "https://register.example" + SB.PREFIX
    assert 0 < got["matches"][0]["similarityScore"] <= 100
    served = http.get(f"{SB.PREFIX}/manifests/gen-1")
    assert served.status_code == 200 and served.headers["content-type"] == "application/c2pa"
    assert served.content == manifest
    reader = c2pa.Reader("application/c2pa", io.BytesIO(served.content))
    assert "gen-1" in reader.json(), "the served bytes are a manifest store the C2PA library reads"


def test_a_fingerprint_binding_finds_it_and_a_watermark_alone_never_does(register, tmp_path):
    _, _, http = register
    signed, serial, audio, sr, *_ = _released(register, tmp_path)
    value = base64.b64encode(json.dumps(query_phases(_clip(audio, sr, 2.0, 6.0), sr)).encode()).decode()
    got = http.post(f"{SB.PREFIX}/matches/byBinding", json={"alg": SB.FINGERPRINT_ALG, "value": value}).json()
    assert [m["manifestId"] for m in got["matches"]] == ["gen-1"]
    short = base64.b64encode(json.dumps([[[1, 0], [2, 50]]]).encode()).decode()
    assert http.get(f"{SB.PREFIX}/matches/byBinding", params={"alg": SB.FINGERPRINT_ALG, "value": short}).json() == {"matches": []}
    assert http.get(f"{SB.PREFIX}/matches/byBinding", params={"alg": SB.FINGERPRINT_ALG, "value": "@@"}).status_code == 400
    refused = http.get(f"{SB.PREFIX}/matches/byBinding", params={"alg": SB.WATERMARK_ALG, "value": "59368"})
    assert refused.status_code == 400 and "group of models" in refused.json()["detail"]
    assert http.get(f"{SB.PREFIX}/matches/byBinding", params={"alg": "other.alg.1", "value": short}).status_code == 400


def test_the_watermark_hint_narrows_the_search(register, tmp_path):
    _, _, http = register
    signed, serial, audio, sr, *_ = _released(register, tmp_path)
    clip = _wav_bytes(_clip(audio, sr, 3.0, 5.0), sr)
    own = encode_payload(signed["record"]["model"]["watermark_payload"])
    other = encode_payload((signed["record"]["model"]["watermark_payload"] + 1) % 2048)
    hint = lambda cw: {"hintAlg": SB.WATERMARK_ALG, "hintValue": str(cw)}  # noqa: E731
    url = f"{SB.PREFIX}/matches/byContent"
    head = {"Content-Type": "audio/wav"}
    assert [m["manifestId"] for m in http.post(url, params=hint(own), content=clip, headers=head).json()["matches"]] == ["gen-1"]
    assert http.post(url, params=hint(other), content=clip, headers=head).json()["matches"] == []
    assert http.post(url, params={"hintAlg": SB.WATERMARK_ALG, "hintValue": "12"}, content=clip, headers=head).status_code == 400


def test_a_piece_released_without_its_manifest_is_not_offered(register, tmp_path):
    _, _, http = register
    *_, audio, sr, manifest, out, _ = _released(register, tmp_path, with_manifest=False)
    assert out["manifest_kept"] is False
    clip = _wav_bytes(_clip(audio, sr, 3.0, 5.0), sr)
    assert http.post(f"{SB.PREFIX}/matches/byContent", content=clip, headers={"Content-Type": "audio/wav"}).json() == {"matches": []}
    assert http.get(f"{SB.PREFIX}/manifests/gen-1").status_code == 404


def test_only_the_releasing_machine_sets_a_manifest_and_only_a_manifest_store_is_taken(register, tmp_path):
    app, _, _ = register
    signed, serial, audio, sr, manifest, out, client = _released(register, tmp_path)
    assert app.state.index.keep_manifest("gen-1", b"\x00\x00\x00\x10jumbother", "another-key") is False
    assert app.state.index.manifest("gen-1") == manifest
    out = client.release("gen-2", serial, fingerprint_array(*_music(2)), manifest=b"not a manifest store")
    assert out["manifest_kept"] is False and out["already"] is False, "a bad manifest is not kept; the release stands"


def test_content_lookups_are_rate_limited_and_refuse_what_is_not_audio(register):
    _, _, http = register
    url = f"{SB.PREFIX}/matches/byContent"
    assert http.post(url, content=b"x", headers={"Content-Type": "text/plain"}).status_code == 415
    codes = [http.post(url, content=b"", headers={"Content-Type": "audio/wav", "X-Forwarded-For": "203.0.113.9"}).status_code
             for _ in range(SB.LIMITS[url] + 1)]
    assert codes[-1] == 429 and 429 not in codes[:-1]
    assert http.get(f"{SB.PREFIX}/manifests/{'x' * 200}").status_code == 400


def test_a_withdrawn_models_pieces_are_not_offered(register, tmp_path):
    _, _, http = register
    signed, serial, audio, sr, manifest, out, client = _released(register, tmp_path)
    client.withdraw(serial, "the artist withdrew it")
    clip = _wav_bytes(_clip(audio, sr, 3.0, 5.0), sr)
    assert http.post(f"{SB.PREFIX}/matches/byContent", content=clip, headers={"Content-Type": "audio/wav"}).json() == {"matches": []}
    assert http.get(f"{SB.PREFIX}/manifests/gen-1").status_code == 410


def test_a_small_file_describing_huge_audio_is_refused_before_it_is_decoded(register, tmp_path):
    """A few megabytes can claim hours of many-channel audio; the register refuses it rather than run out of memory."""
    import numpy as np
    _, _, http = register
    buf = io.BytesIO()
    sf.write(buf, np.zeros((384000, 8), dtype=np.float32), 384000, format="FLAC")
    got = http.post(f"{SB.PREFIX}/matches/byContent", content=buf.getvalue(), headers={"Content-Type": "audio/flac"})
    assert got.status_code == 415 and "96 kHz" in got.json()["detail"]
    assert http.post(f"{SB.PREFIX}/matches/byContent", content=b"x" * (SB.CONTENT_MAX_BYTES + 1),
                     headers={"Content-Type": "audio/wav"}).status_code == 413


def test_a_piece_its_maker_removed_is_named_but_no_longer_offered(register, tmp_path):
    """Removal keeps the fingerprint, so a copy found later is still named, with the fact; the lookup stops offering."""
    _, _, http = register
    signed, serial, audio, sr, manifest, out, client = _released(register, tmp_path)
    clip = _wav_bytes(_clip(audio, sr, 3.0, 5.0), sr)
    assert http.post(f"{SB.PREFIX}/matches/byContent", content=clip, headers={"Content-Type": "audio/wav"}).json()["matches"]
    _, stranger = _machine(register, "artist-two")
    with pytest.raises(Exception, match="released"):
        stranger.remove_output("gen-1")
    gone = client.remove_output("gen-1")
    assert gone["serial"] == serial and gone["removed_at"]
    assert client.remove_output("gen-1")["removed_at"] == gone["removed_at"], "removing twice is once"
    answer = client.verify(query_phases(_clip(audio, sr, 3.0, 5.0), sr), None)
    assert answer["grade"] == "output" and answer["removed_at"] == gone["removed_at"]
    assert http.post(f"{SB.PREFIX}/matches/byContent", content=clip, headers={"Content-Type": "audio/wav"}).json() == {"matches": []}
    assert http.get(f"{SB.PREFIX}/manifests/gen-1").status_code == 410


def test_a_query_is_bounded_in_phases_and_landmarks(register, tmp_path):
    """No body may cost more matching than a real query: at most eight phases, each at most the landmark cap."""
    _, _, http = register
    _released(register, tmp_path)
    too_many = [[[1, 0]]] * 9
    assert http.post("/api/verify", json={"phases": too_many}).status_code == 422
    value = base64.b64encode(json.dumps(too_many).encode()).decode()
    assert http.post(f"{SB.PREFIX}/matches/byBinding", json={"alg": SB.FINGERPRINT_ALG, "value": value}).status_code == 400
