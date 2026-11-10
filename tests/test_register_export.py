"""The static export writes what a new leaf adds and nothing else, its files are the files a whole export writes,
byte for byte, after any sequence of writes, and every proof it publishes verifies against a head it publishes."""
import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
pytest.importorskip("rfc8785")

import register_service.translog as RT  # noqa: E402
from khaos_attribution import registry as R  # noqa: E402
from khaos_attribution import translog as T  # noqa: E402
from khaos_attribution.translog import verify as V  # noqa: E402
from register_service.store import Store, now_utc  # noqa: E402

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.now = START

    def rfc3339(self) -> str:
        return self.now.strftime("%Y-%m-%dT%H:%M:%SZ")

    def pass_(self, hours: float) -> None:
        self.now += timedelta(hours=hours)


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(T, "now_rfc3339", c.rfc3339)
    return c


def _store(tmp_path) -> Store:
    store = Store(tmp_path / "register.sqlite3")
    store.log.signer = R.SigningKey.generate()
    with store._tx():
        store.log.remember_key_locked()
        store.log.genesis_locked()
    store.create_account("artist")
    return store


def _key_doc(store) -> dict:
    return {"log_id": T.LOG_ID, "key_id": store.log.signer.key_id, "public_key_pem": store.log.signer.public_pem()}


def _publish(store, name: str) -> str:
    sha = hashlib.sha256(name.encode()).hexdigest()
    signed = {"record_sha256": sha, "record": {"artist_id": name, "public_view": "model", "model": {"watermark_payload": 40}}}
    key = R.SigningKey.generate()
    store.register(signed, "k", 1, lambda n, at: R.countersign(signed, n, key, at or now_utc()))
    return sha


def _files(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file() and not p.name.startswith(".")}


def _same_as_a_whole_export(store, tmp_path, clock, mirror: Path, public: Path, tag: str) -> None:
    whole, whole_public = tmp_path / f"whole-{tag}", tmp_path / f"whole-public-{tag}"
    assert RT.export(store.log, whole, _key_doc(store), public=whole_public, delay_days=7, now=clock.now, full=True)
    for mine, theirs in ((mirror, whole), (public, whole_public)):
        a, b = _files(mine / "log"), _files(theirs / "log")
        assert sorted(a) == sorted(b), sorted(set(a) ^ set(b))[:6]
        assert [k for k in a if a[k] != b[k]] == []


@pytest.mark.parametrize("seed", range(6))
def test_after_any_sequence_of_writes_the_files_are_a_whole_exports_files(tmp_path, clock, seed):
    """Admissions, strikes, withholds, heartbeats, a key change and days passing, exported after some and not others."""
    rng = random.Random(seed)
    store = _store(tmp_path)
    mirror, public = tmp_path / "mirror", tmp_path / "mirror-public"
    admitted, kinds = [], set()
    for step in range(70):
        clock.pass_(rng.choice((0.5, 6, 30, 80)))
        roll = rng.random()
        if roll < 0.5 or not admitted:
            admitted.append(_publish(store, f"model-{seed}-{step}"))
            kinds.add("admit")
        elif roll < 0.65:
            sha = rng.choice(admitted)
            for leaf, state in store.log.admit_states(sha).items():
                if state != "struck":
                    with store._tx():
                        store.log.revoke_locked(leaf, "struck")
                    kinds.add("revoke")
        elif roll < 0.78:
            leaf = store.log.admit_leaf_of(rng.choice(admitted))
            try:
                store.log_withhold(leaf, "a private name")
                kinds.add("withhold")
            except RT.EntryRefused:
                pass
        elif roll < 0.93:
            with store._tx():
                store.log.heartbeat_locked()
        else:
            store.log.signer = R.SigningKey.generate()
            with store._tx():
                store.log.remember_key_locked()
                store.log.resign_locked()
            kinds.add("key")
        if rng.random() < 0.6:
            assert RT.export(store.log, mirror, _key_doc(store), public=public, delay_days=7, now=clock.now)
            _same_as_a_whole_export(store, tmp_path, clock, mirror, public, f"{seed}-{step}")
    assert {"admit", "revoke", "withhold"} <= kinds
    assert RT.export(store.log, mirror, _key_doc(store), public=public, delay_days=7, now=clock.now)
    _same_as_a_whole_export(store, tmp_path, clock, mirror, public, f"{seed}-end")


def _writes(monkeypatch) -> list:
    seen, original = [], RT._write

    def counting(path, data):
        seen.append(path)
        original(path, data)
    monkeypatch.setattr(RT, "_write", counting)
    return seen


def test_one_more_leaf_writes_that_leafs_files_whatever_the_logs_size(tmp_path, clock, monkeypatch):
    store = _store(tmp_path)
    mirror = tmp_path / "mirror"
    counts = []
    for size in (10, 80):
        while store.log.size() < size:
            _publish(store, f"model-{store.log.size()}")
        assert RT.export(store.log, mirror, _key_doc(store))
        sha = _publish(store, f"one-more-{size}")
        seen = _writes(monkeypatch)
        assert RT.export(store.log, mirror, _key_doc(store))
        monkeypatch.undo()
        monkeypatch.setattr(T, "now_rfc3339", clock.rfc3339)
        leaf = store.log.admit_leaf_of(sha)
        assert mirror / "log" / "proofs" / f"{leaf}.json" in seen and mirror / "log" / "entries" / f"{leaf}.json" in seen
        counts.append(len(seen))
    assert counts[0] == counts[1] <= 12, counts


def test_every_published_proof_verifies_against_a_published_head_under_the_current_key(tmp_path, clock):
    """A proof stays as written while the log grows, so it names the head it was made for; after a key change that
    head is one the new key signed."""
    store = _store(tmp_path)
    mirror = tmp_path / "mirror"
    shas = [_publish(store, f"model-{i}") for i in range(9)]
    assert RT.export(store.log, mirror, _key_doc(store))
    first = (mirror / "log" / "proofs" / f"{store.log.admit_leaf_of(shas[0])}.json").read_bytes()
    shas += [_publish(store, f"later-{i}") for i in range(5)]
    assert RT.export(store.log, mirror, _key_doc(store))
    assert (mirror / "log" / "proofs" / f"{store.log.admit_leaf_of(shas[0])}.json").read_bytes() == first
    store.log.signer = R.SigningKey.generate()
    with store._tx():
        store.log.remember_key_locked()
        store.log.resign_locked()
    assert RT.export(store.log, mirror, _key_doc(store))
    pem = store.log.signer.public_pem()
    for sha in shas:
        out = V.prove(sha, V.Mirror(str(mirror)), pem)
        assert out["entry"]["record_sha256"] == sha and out["key_id"] == store.log.signer.key_id
    heads = sorted(int(p.stem) for p in (mirror / "log" / "sth").glob("[0-9]*.json"))
    for a, b in zip(heads, heads[1:]):
        doc = json.loads((mirror / "log" / "consistency" / f"{a}.json").read_text())
        assert (doc["first"], doc["second"]) == (a, b)
        old, new = (json.loads((mirror / "log" / "sth" / f"{k}.json").read_text()) for k in (a, b))
        T.verify_consistency(a, b, bytes.fromhex(old["root_sha256"]), bytes.fromhex(new["root_sha256"]), T.unhex(doc["proof"]))


def test_files_changed_behind_the_exports_back_are_written_again_whole(tmp_path, clock):
    """The export trusts its own record only while the mirror's newest head is the one it wrote."""
    store = _store(tmp_path)
    mirror = tmp_path / "mirror"
    sha = _publish(store, "one")
    assert RT.export(store.log, mirror, _key_doc(store))
    proof = mirror / "log" / "proofs" / f"{store.log.admit_leaf_of(sha)}.json"
    good = proof.read_bytes()
    proof.write_text("{}")
    latest = mirror / "log" / "sth" / "latest.json"
    latest.write_text(json.dumps({**json.loads(latest.read_text()), "tree_size": 1}))
    _publish(store, "two")
    assert RT.export(store.log, mirror, _key_doc(store))
    assert proof.read_bytes() == good
    proof.write_text("{}")
    assert RT.export(store.log, mirror, _key_doc(store), full=True)
    assert proof.read_bytes() == good, "the whole export is the repair"


def test_a_publish_is_registered_and_mirrored_off_the_event_loop(tmp_path):
    import asyncio

    from fastapi.testclient import TestClient

    from register_service.app import create_app
    from tests.test_register_service import _machine, _signed
    cfg = {"data": tmp_path / "data", "key_file": tmp_path / "data" / "register-key.pem", "dev_key": True, "public_url": "https://register.example"}
    app = create_app(cfg)
    store = app.state.store
    store.create_account("artist-one")
    on_loop, register, export = [], store.register, RT._write_log

    def _loop_running() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    def watched_register(*a, **k):
        on_loop.append(("register", _loop_running()))
        return register(*a, **k)

    def watched_export(*a, **k):
        on_loop.append(("export", _loop_running()))
        return export(*a, **k)
    store.register, RT._write_log = watched_register, watched_export
    try:
        key, client = _machine((app, store, TestClient(app)), "artist-one")
        client.publish(_signed(key))
    finally:
        RT._write_log = export
    assert {name for name, _ in on_loop} == {"register", "export"} and not any(running for _, running in on_loop), on_loop
