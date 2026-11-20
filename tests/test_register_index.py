"""The fingerprint index beside the register: outputs released under their model's bucket by the owning Workshop or a
serving Space, and verify in three grades on real audio through the transforms the bar passed."""
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
sf = pytest.importorskip("soundfile")
from fastapi.testclient import TestClient  # noqa: E402

from khaos_attribution import registry as R  # noqa: E402
from khaos_attribution.fingerprint import fingerprint_array, query_phases  # noqa: E402
from khaos_attribution.watermark import derive_payload, encode_payload  # noqa: E402
from register_service.app import create_app  # noqa: E402
from tests.test_register_service import _Transport, _machine, _signed  # noqa: E402


@pytest.fixture
def register(tmp_path, monkeypatch):
    cfg = {"log_enable": True, "data": tmp_path / "data", "key_file": tmp_path / "data" / "k.pem", "dev_key": True, "public_url": "https://register.example"}
    app = create_app(cfg)
    app.state.store.create_account("artist-one")
    app.state.store.create_account("artist-two")
    http = TestClient(app)

    def public(self, path, body):
        r = http.post(path, content=R.canonical_bytes(body), headers={"Content-Type": "application/json"})
        if r.status_code >= 400:
            raise R.RegisterError(r.status_code, r.text)
        return r.json()
    monkeypatch.setattr(R.RegisterClient, "_post_public", public)
    return app, app.state.store, http


def _music(seed: int, seconds: float = 12.0, sr: int = 22050):
    """A deterministic piece with real spectral structure: a few notes with harmonics and a beat, not a sine."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    out = np.zeros_like(t)
    notes = rng.uniform(110, 880, size=8)
    for i, f in enumerate(notes):
        start, end = i * seconds / 8, (i + 1) * seconds / 8
        env = ((t >= start) & (t < end)).astype(float)
        for k, a in ((1, 1.0), (2, 0.5), (3, 0.25), (5, 0.1)):
            out += a * env * np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28))
    beat = (np.mod(t, 0.5) < 0.02).astype(float) * rng.normal(0, 1, size=t.shape)
    out = out / np.max(np.abs(out)) * 0.8 + 0.1 * beat
    return out.astype(np.float32), sr


def _clip(audio, sr, start_s, seconds):
    a, b = int(start_s * sr), int((start_s + seconds) * sr)
    return audio[a:b]


def test_release_is_by_the_owning_workshop_or_a_serving_space_and_verify_names_the_output(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    _, stranger = _machine(register, "artist-two")
    _, space = _machine(register, "artist-two", role="space")
    signed = _signed(key)
    serial = client.publish(signed)["serial"]
    audio, sr = _music(1)
    fps = fingerprint_array(audio, sr)
    with pytest.raises(R.RegisterError, match="own account"):
        stranger.release("gen-1", serial, fps)
    with pytest.raises(R.RegisterError, match="reports serving"):
        space.release("gen-1", serial, fps)
    out = client.release("gen-1", serial, fps)
    assert out["serial"] == serial and out["landmarks"] == len(fps) and out["already"] is False
    assert client.release("gen-1", serial, fps)["already"] is True
    space.served(serial, "https://space.example", "serving")
    assert space.release("gen-2", serial, fingerprint_array(*_music(2)))["already"] is False
    assert http.get("/health").json()["outputs"] == 2
    # verify: a clip off the grid, re-encoded, with the watermark read → the exact output
    codeword = encode_payload(signed["record"]["model"]["watermark_payload"])
    clip = _clip(audio, sr, 3.0, 5.0)
    answer = client.verify(query_phases(clip, sr), codeword)
    assert answer["grade"] == "output" and answer["output"]["generation_id"] == "gen-1" and answer["searched"] == "bucket"
    assert answer["model"]["serial"] == serial and answer["model"]["artist_name"] == "Artist X"
    # no watermark: the whole index, same answer
    answer = client.verify(query_phases(clip, sr), None)
    assert answer["grade"] == "output" and answer["output"]["generation_id"] == "gen-1" and answer["searched"] == "all"
    # the watermark read but the audio is not in the index: the bucket's models
    answer = client.verify(query_phases(_music(9)[0], sr), codeword)
    assert answer["grade"] == "bucket" and [m["serial"] for m in answer["models"]] == [serial]
    # nothing read, nothing matched
    assert client.verify(query_phases(_music(9)[0], sr), None)["grade"] == "none"
    with pytest.raises(R.RegisterError, match="codeword"):
        client.verify(query_phases(clip, sr), "not-a-codeword")


def test_verify_survives_mp3_when_ffmpeg_is_here(register, tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("no ffmpeg")
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    audio, sr = _music(3, seconds=20)
    client.release("gen-3", serial, fingerprint_array(audio, sr))
    wav, mp3, back = tmp_path / "a.wav", tmp_path / "a.mp3", tmp_path / "b.wav"
    sf.write(str(wav), audio, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "4.37", "-t", "8", "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "128k", str(mp3)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), str(back)], check=True)
    q, qsr = sf.read(str(back), dtype="float32", always_2d=True)
    answer = client.verify(query_phases(q[:, 0], qsr), None)
    assert answer["grade"] == "output" and answer["output"]["generation_id"] == "gen-3"
    from khaos_attribution.fingerprint import query_phases_file
    assert client.verify(query_phases_file(back), None)["output"]["generation_id"] == "gen-3"


def test_a_withdrawn_models_outputs_are_not_released(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    client.withdraw(serial, "x")
    with pytest.raises(R.RegisterError, match="withdrawn") as e:
        client.release("gen-w", serial, fingerprint_array(*_music(4)))
    assert e.value.status == 410, "a refusal of the entry for what it is: set aside by sync, never a jam"
    box = R.Outbox(store.path.parent / "box")
    box.append("release", {"generation_id": "gen-w2", "serial": serial, "fingerprint": fingerprint_array(*_music(5)), "nonce": "a1b2c3d4e5f60718"})
    assert client.sync(box) == [] and len(box.refused()) == 1 and box.pending() == []


def _tracks(*ids):
    return {"rights_sha256": "f" * 64, "tracks": [{"track_id": t, "title": t, "duration_sec": 20.0, "writers": []} for t in ids]}


def _fp(audio, sr):
    return [list(l) for l in fingerprint_array(audio, sr)]


def _ph(audio, sr):
    return [[list(l) for l in p] for p in query_phases(audio, sr)]


def test_a_recording_may_train_many_models_and_every_page_says_when_it_is_shared(register, tmp_path):
    """Claims are a record, not a lock: another account's claim on the same recording is taken and linked, both models'
    pages say so, a long piece holding a claimed track is found either way round, a claim names one of the record's
    own tracks, withdrawal ends a model's claims and their links, and only a Workshop claims or checks."""
    app, store, http = register
    key, one = _machine(register, "artist-one")
    key2, two = _machine(register, "artist-two")
    _, space = _machine(register, "artist-two", role="space")
    first = _signed(key)
    serial = one.publish(first)["serial"]
    audio, sr = _music(1, seconds=20.0)
    fps = _fp(audio, sr)
    assert one.check_claim(_ph(audio, sr))["grade"] == "none"
    with pytest.raises(R.RegisterError, match="training tracks the model's own record lists"):
        one.claim(first["record_sha256"], "not-in-the-record", fps, "khaos.declaration/1")
    out = one.claim(first["record_sha256"], "t1", fps, "khaos.declaration/1", phases=_ph(audio, sr))
    assert out["claimed"] is True and out["already"] is False and out["serial"] == serial and out["shared_with_accounts"] == 0
    assert one.claim(first["record_sha256"], "t1", fps, "khaos.declaration/1")["already"] is True
    mine = one.check_claim(_ph(audio, sr))
    assert mine["grade"] == "mine" and mine["serial"] == serial and mine["track_id"] == "t1" and mine["released_output"] is None
    theirs = two.check_claim(_ph(_clip(audio, sr, 2.0, 15.0), sr))
    assert theirs["grade"] == "other" and "serial" not in theirs and "track_id" not in theirs
    with pytest.raises(R.RegisterError, match="only a Workshop"):
        space.check_claim(_ph(audio, sr))
    second = _signed(key2, training_set=_tracks("x1", "x2", "x3"))
    serial2 = two.publish(second)["serial"]
    shared = two.claim(second["record_sha256"], "x1", fps, "khaos.declaration/1")
    assert shared["claimed"] is True and shared["shared_with_accounts"] == 1, "another account trains on it too, and it is said"
    assert two.claim(second["record_sha256"], "x2", _fp(*_music(5, 20.0)), "khaos.declaration/1")["shared_with_accounts"] == 0
    assert http.get(f"/api/models/{serial2}/claims").json() == {"serial": serial2, "tracks": 3, "claimed": 2, "shared": 1}
    assert http.get(f"/api/models/{serial}/claims").json() == {"serial": serial, "tracks": 1, "claimed": 1, "shared": 1}
    page = __import__("html").unescape(http.get(f"/models/{serial}").text)
    assert "1 of 1 claimed on the artist's declaration · 1 also trains another account's registered model" in page
    medley = np.concatenate([_music(7, 60.0)[0], audio, _music(8, 60.0)[0]])
    both = one.check_claim(_ph(medley, sr))
    assert both["grade"] == "other" and both["also_mine"] is True, "a claimed track inside a long piece is found, and both claims are said"
    box = R.Outbox(tmp_path / "box")
    box.append("claim", {"record_sha256": second["record_sha256"], "track_id": "x3", "fingerprint": _fp(*_music(6, 20.0)),
                         "declaration": "khaos.declaration/1", "nonce": "claim-nonce-1"})
    assert len(two.sync(box)) == 1 and not box.refused()
    with pytest.raises(R.RegisterError, match="own account"):
        two.claim(first["record_sha256"], "t1", fps, "khaos.declaration/1")
    one.withdraw(serial, "consent withdrawn")
    assert http.get(f"/api/models/{serial2}/claims").json()["shared"] == 0, "the link ends with the withdrawn model's claim"
    assert one.check_claim(_ph(audio, sr))["grade"] == "other"


def test_the_index_answers_readers_while_another_thread_claims(tmp_path):
    """Requests run on the loop and in the threadpool: every claim lands and no reader errors while they are written."""
    import random
    import threading
    from register_service.index import Index
    ix = Index(tmp_path / "i.sqlite3")

    def fp(seed, n=5000):
        r = random.Random(seed)
        return [[r.randrange(1 << 30), i * 10] for i in range(n)]
    errors, stop = [], threading.Event()

    def writer():
        for k in range(20):
            try:
                ix.claim(1, k, f"t{k}", fp(k), "d", "key")
                ix.release(f"g{k}", k, 1, "a", fp(1000 + k), "key", "workshop")
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

    def reader():
        q = [[tuple(x) for x in fp(3)[:1500]]]
        while not stop.is_set():
            try:
                ix.check_claim(q, 2), ix.claims_of(3), ix.verify(q, None)
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
    readers = [threading.Thread(target=reader) for _ in range(3)]
    for t in readers:
        t.start()
    w = threading.Thread(target=writer)
    w.start(); w.join(); stop.set()
    for t in readers:
        t.join()
    assert errors == [] and ix.db.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 20


def test_a_claim_sent_as_phases_alone_stores_the_first_phase_and_a_twelve_minute_track_fits(register):
    """The fingerprint is the first phase, so it is not sent twice; a long track's four phases fit the claim allowance."""
    from register_service.app import CLAIM_MAX_BODY
    app, store, http = register
    key, one = _machine(register, "artist-one")
    first = _signed(key)
    one.publish(first)
    audio, sr = _music(3, seconds=20.0)
    out = one.claim(first["record_sha256"], "t1", declaration="khaos.declaration/1", phases=_ph(audio, sr))
    assert out["claimed"] is True
    assert app.state.index.db.execute("SELECT landmarks FROM claims").fetchone()[0] == len(_fp(audio, sr))
    per_second = len(R.canonical_bytes({"phases": _ph(audio, sr)})) / 20.0
    assert per_second * 720 * 2 < CLAIM_MAX_BODY, "denser music than this test piece still fits a twelve-minute track"


def _near_copies(tmp_path, a_sha, b_sha, shared=120):
    """An index holding gen-a and gen-b of one model, near-copies sharing `shared` of a's 200 landmarks."""
    from register_service.index import Index
    ix = Index(tmp_path / "i.sqlite3")
    a = [[1000 + i, 100 * i] for i in range(200)]
    b = a[:shared] + [[5000 + i, 100 * i] for i in range(shared, 200)]
    ix.release("gen-a", 7, 1, "artist", a, "k", "space", sha256=a_sha)
    ix.release("gen-b", 7, 1, "artist", b, "k", "space", sha256=b_sha)
    return ix, [a]


def test_near_copies_are_told_apart_by_the_file_hash_they_were_released_with(tmp_path):
    """Two releases too alike for dominance: the query's file hash names its own piece, with a count of the other."""
    ix, phases = _near_copies(tmp_path, "a" * 64, "b" * 64)
    answer = ix.verify(phases, None, "a" * 64)
    assert answer["grade"] == "output" and answer["output"]["generation_id"] == "gen-a" and answer["near_copies"] == 1
    trailing = ix.verify(phases, None, "b" * 64)
    assert trailing["output"]["generation_id"] == "gen-b" and trailing["votes"] == 120 and trailing["runner_up"] == 200
    assert "sha256" not in answer["output"] and "sha256" not in ix.output("gen-a")


def test_near_copies_stay_unnamed_without_a_unique_hash(tmp_path):
    """No hash, a hash of neither, releases without hashes, or two releases with one hash: refused as before."""
    ix, phases = _near_copies(tmp_path, "a" * 64, "b" * 64)
    assert ix.verify(phases, None)["grade"] == "none"
    assert ix.verify(phases, None, "c" * 64)["grade"] == "none"
    ix, phases = _near_copies(tmp_path / "unhashed", None, None)
    assert ix.verify(phases, None, "a" * 64)["grade"] == "none"
    ix, phases = _near_copies(tmp_path / "same", "a" * 64, "a" * 64)
    assert ix.verify(phases, None, "a" * 64)["grade"] == "none"
    assert ix.verify(phases, 1, "a" * 64) == {"grade": "bucket", "payload": 1, "searched": "bucket"}


def test_a_dominant_match_is_unchanged_by_a_hash(tmp_path):
    """When the winner doubles the runner-up the hash is not read and the answer is today's."""
    ix, phases = _near_copies(tmp_path, "a" * 64, "b" * 64, shared=40)
    answer = ix.verify(phases, None, "b" * 64)
    assert answer["output"]["generation_id"] == "gen-a" and answer["near_copies"] == 0


def test_a_release_hash_is_64_lowercase_hex(tmp_path):
    """A malformed hash is refused at release, never stored."""
    from register_service.index import Index
    ix = Index(tmp_path / "i.sqlite3")
    for bad in ("A" * 64, "a" * 63, 7):
        with pytest.raises(ValueError, match="sha256"):
            ix.release("gen-x", 7, 1, "artist", [[1, 0]], "k", "space", sha256=bad)
    assert ix.output("gen-x") is None


def test_the_register_names_the_near_copy_whose_file_was_released_and_never_answers_a_hash(register):
    """Through the client: two near-copy pieces one sample apart, each file's hash names its own release."""
    import hashlib
    _, _, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    audio, sr = _music(1)
    twin = audio.copy()
    twin[-1] += 1e-4
    files = {}
    for gid, a in (("gen-a", audio), ("gen-b", twin)):
        files[gid] = hashlib.sha256(a.tobytes()).hexdigest()
        client.release(gid, serial, fingerprint_array(a, sr), sha256=files[gid])
    phases = query_phases(audio, sr)
    assert client.verify(phases, None)["grade"] == "none"
    for gid, sha in files.items():
        answer = client.verify(phases, None, sha256=sha)
        assert answer["grade"] == "output" and answer["output"]["generation_id"] == gid and answer["near_copies"] == 1
        assert sha not in json.dumps(answer)
    r = http.post("/api/verify", json={"phases": phases, "sha256": "not-a-hash"})
    assert r.status_code == 422 and "sha256" in r.text


def test_an_index_file_from_before_the_hash_gains_its_column_in_place(tmp_path):
    """An index written before release hashes opens with the column added, its pieces kept and unhashed."""
    import sqlite3
    from register_service.index import Index
    path = tmp_path / "i.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE outputs (generation_id TEXT PRIMARY KEY, serial INTEGER NOT NULL, payload INTEGER NOT NULL,"
               " artist_id TEXT NOT NULL, released_at TEXT NOT NULL, released_by TEXT NOT NULL, kind TEXT NOT NULL,"
               " landmarks INTEGER NOT NULL, withdrawn INTEGER NOT NULL DEFAULT 0, removed_at TEXT)")
    db.execute("INSERT INTO outputs (generation_id, serial, payload, artist_id, released_at, released_by, kind, landmarks)"
               " VALUES ('old', 7, 1, 'artist', 'then', 'k', 'space', 1)")
    db.commit()
    db.close()
    ix = Index(path)
    assert ix.output("old")["generation_id"] == "old"
    assert ix.db.execute("SELECT sha256 FROM outputs WHERE generation_id = 'old'").fetchone()[0] is None
    ix.release("new", 7, 1, "artist", [[1, 0]], "k", "space", sha256="a" * 64)
    assert ix.db.execute("SELECT sha256 FROM outputs WHERE generation_id = 'new'").fetchone()[0] == "a" * 64


def test_a_malformed_hash_at_release_is_dropped_and_the_release_stands(register):
    """A bad hash from a machine costs the piece its tie-break, never its place in the index."""
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    audio, sr = _music(1)
    assert client.release("gen-x", serial, fingerprint_array(audio, sr), sha256="NOT-HEX")["already"] is False
    assert client.verify(query_phases(audio, sr), None)["output"]["generation_id"] == "gen-x"
