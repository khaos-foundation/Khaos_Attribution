"""The static export writes what a new leaf adds and nothing else, its files are the files a whole export writes,
byte for byte, after any sequence of writes, and every proof it publishes verifies against a head it publishes."""
import hashlib
import json
import random
import shutil
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
    seen, original = [], RT.write_whole

    def counting(path, data):
        seen.append(path)
        original(path, data)
    monkeypatch.setattr(RT, "write_whole", counting)
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


class _Died(Exception):
    """The export stopped here, as a crash would stop it."""


def _export_dying_before(store, mirror, public, clock, monkeypatch, k: int, full: bool = False) -> bool:
    """One export that dies before its k-th change to the mirror's files; True when it finished with fewer."""
    made, write, remove = [0], RT.write_whole, RT._remove

    def step():
        if made[0] == k:
            raise _Died
        made[0] += 1
    with monkeypatch.context() as m:
        m.setattr(RT, "write_whole", lambda path, data: (step(), write(path, data))[1])
        m.setattr(RT, "_remove", lambda path: (step(), remove(path))[1])
        try:
            RT.export(store.log, mirror, _key_doc(store), public=public, delay_days=7, now=clock.now, full=full)
        except _Died:
            return False
    return True


def _every_leaf_proves(where: Path, pem: str) -> int:
    """Each leaf the copy's newest head holds, proved from the copy's own files; a copy with no head yet holds none."""
    if not (where / "log" / "sth" / "latest.json").is_file():
        return 0
    n = json.loads((where / "log" / "sth" / "latest.json").read_text())["tree_size"]
    leaves = json.loads((where / "log" / "hashes.json").read_text())["leaves"][:n]
    for leaf in leaves:
        V.prove(leaf, V.Mirror(str(where)), pem)
    return len(leaves)


def _crash_everywhere(store, tmp_path, clock, monkeypatch, mirror, public, full: bool = False) -> int:
    """The export about to run, killed before each change it makes in turn, each time from the same files and the
    same record of the last export; after every death both copies still prove every leaf their head holds."""
    keep = tmp_path / "keep"
    shutil.rmtree(keep, ignore_errors=True)
    for name, where in (("mirror", mirror), ("public", public)):
        if where.exists():
            shutil.copytree(where, keep / name)
    record = [tuple(r) for r in store.db.execute("SELECT * FROM log_exports").fetchall()]
    pem, deaths = store.log.signer.public_pem(), 0
    while True:
        for name, where in (("mirror", mirror), ("public", public)):
            shutil.rmtree(where, ignore_errors=True)
            if (keep / name).exists():
                shutil.copytree(keep / name, where)
        store.db.execute("DELETE FROM log_exports")
        store.db.executemany("INSERT INTO log_exports VALUES (?, ?, ?, ?, ?)", record)
        finished = _export_dying_before(store, mirror, public, clock, monkeypatch, deaths, full)
        for where in (mirror, public):
            _every_leaf_proves(where, pem)
        if finished:
            return deaths
        deaths += 1


def test_a_crash_right_after_a_withhold_leaves_a_copy_that_still_verifies(tmp_path, clock, monkeypatch):
    """The withheld body is not taken away before the head that proves its withhold is filed."""
    store = _store(tmp_path)
    mirror, public = tmp_path / "mirror", tmp_path / "mirror-public"
    shas = [_publish(store, f"model-{i}") for i in range(4)]
    clock.pass_(24 * 30)
    assert RT.export(store.log, mirror, _key_doc(store), public=public, delay_days=7, now=clock.now)
    store.log_withhold(store.log.admit_leaf_of(shas[1]), "a private name")
    assert _crash_everywhere(store, tmp_path, clock, monkeypatch, mirror, public) >= 8
    leaf = store.log.admit_leaf_of(shas[1])
    for where in (mirror, public):
        assert not (where / "log" / "entries" / f"{leaf}.json").exists(), "the finished export serves no withheld body"
        assert json.loads((where / "log" / "bodies.json").read_text())["bodies"][store.log.leaf(leaf)["idx"]] is None


@pytest.mark.parametrize("seed", range(3))
def test_an_export_killed_at_any_write_leaves_copies_that_verify(tmp_path, clock, monkeypatch, seed):
    """Random writes and days passing; every export is killed before each of its writes in turn, and what is on disk
    after each death proves every leaf of the head it holds. Exports are never further apart than the two days the
    verifiers allow a delayed copy beyond its week, as the daily heartbeat keeps them."""
    rng = random.Random(1000 + seed)
    store = _store(tmp_path)
    mirror, public = tmp_path / "mirror", tmp_path / "mirror-public"
    admitted, deaths = [], 0
    for step in range(14):
        clock.pass_(rng.choice((6, 20, 40)))
        roll = rng.random()
        if roll < 0.45 or not admitted:
            admitted.append(_publish(store, f"model-{seed}-{step}"))
        elif roll < 0.6:
            for leaf, state in store.log.admit_states(rng.choice(admitted)).items():
                if state != "struck":
                    with store._tx():
                        store.log.revoke_locked(leaf, "struck")
        elif roll < 0.85:
            try:
                store.log_withhold(store.log.admit_leaf_of(rng.choice(admitted)), "a private name")
            except RT.EntryRefused:
                pass
        else:
            with store._tx():
                store.log.heartbeat_locked()
        deaths += _crash_everywhere(store, tmp_path, clock, monkeypatch, mirror, public)
        _same_as_a_whole_export(store, tmp_path, clock, mirror, public, f"crash-{seed}-{step}")
    assert deaths > 80
    assert json.loads((public / "log" / "bodies.json").read_text())["pending_from"] not in (None, 0), "some bodies became public"


def _withheld_inside_the_delay(tmp_path, clock) -> tuple:
    """A body still inside the delay on the public copy, withheld before the delay passes it."""
    store = _store(tmp_path)
    mirror, public = tmp_path / "mirror", tmp_path / "mirror-public"
    for i in range(3):
        _publish(store, f"model-{i}")
    clock.pass_(24 * 30)
    assert RT.export(store.log, mirror, _key_doc(store), public=public, delay_days=7, now=clock.now)
    late = store.log.admit_leaf_of(_publish(store, "late"))
    clock.pass_(24 * 6.5)
    assert RT.export(store.log, mirror, _key_doc(store), public=public, delay_days=7, now=clock.now)
    assert json.loads((public / "log" / "bodies.json").read_text())["bodies"][store.log.leaf(late)["idx"]] is None
    clock.pass_(2)
    store.log_withhold(late, "a private name")
    return store, mirror, public


def test_a_crash_anywhere_as_the_delay_passes_a_body_withheld_before_it_was_public(tmp_path, clock, monkeypatch):
    """Exports a day apart: until the new head is filed, the body stays explained by the delay the old head knew."""
    store, mirror, public = _withheld_inside_the_delay(tmp_path, clock)
    clock.pass_(24)
    assert _crash_everywhere(store, tmp_path, clock, monkeypatch, mirror, public) >= 8


def test_whatever_a_crashed_export_left_log_export_makes_the_public_copy_verify(tmp_path, clock, monkeypatch):
    """The corner the write order cannot close: exports more than two days apart, killed between the new head and the
    last bodies file, leave the public copy failing; log-export's whole export, run after a crash, repairs every one."""
    store, mirror, public = _withheld_inside_the_delay(tmp_path, clock)
    clock.pass_(24 * 3)
    with store._tx():
        store.log.heartbeat_locked()
    clock.pass_(1)
    keep = tmp_path / "keep-corner"
    for name, where in (("mirror", mirror), ("public", public)):
        shutil.copytree(where, keep / name)
    record = [tuple(r) for r in store.db.execute("SELECT * FROM log_exports").fetchall()]
    pem, failing, k = store.log.signer.public_pem(), [], 0
    while True:
        for name, where in (("mirror", mirror), ("public", public)):
            shutil.rmtree(where)
            shutil.copytree(keep / name, where)
        store.db.execute("DELETE FROM log_exports")
        store.db.executemany("INSERT INTO log_exports VALUES (?, ?, ?, ?, ?)", record)
        finished = _export_dying_before(store, mirror, public, clock, monkeypatch, k)
        try:
            _every_leaf_proves(public, pem)
        except T.LogError:
            failing.append(k)
            assert RT.export(store.log, mirror, _key_doc(store), public=public, delay_days=7, now=clock.now, full=True)
            assert _every_leaf_proves(public, pem) == _every_leaf_proves(mirror, pem) == store.log.size()
        if finished:
            break
        k += 1
    assert failing, "the corner is reached"


def test_a_lost_leaf_the_old_head_knew_as_held_back_is_still_owed_the_delay(tmp_path):
    """A leaf a recovery put back without its body: nothing to keep, but the old head's delay still explains it."""
    root = tmp_path / "log"
    root.mkdir()
    (root / "bodies.json").write_text(json.dumps({"tree_size": 3, "pending_from": 2, "bodies": ["{}", "{}", None]}))
    snap = {"leaves": ["l0", "l1", "l2", "l3"], "kinds": ["genesis", "admit", "unavailable", "withhold"],
            "refers": [None, None, None, "l2"], "raw": [b"{}", b"{}", b"", b"{}"]}
    shown = [b"{}", b"{}", None, b"{}"]
    assert RT._owed_to_the_old_head(root, snap, shown, None, 3) == ({}, [2])
    assert RT._owed_to_the_old_head(root, snap, shown, {"tree_size": 3, "pending_from": 2}, 3) == ({}, [2])


def test_the_first_export_after_a_recovery_killed_anywhere_leaves_copies_that_verify(tmp_path, clock, monkeypatch):
    """The database and the full export restored from before an admit the public copy still held back: the leaf comes
    back without its body, and log-recover's whole export, killed before each write, never leaves it unexplained."""
    store = _store(tmp_path)
    mirror, public = tmp_path / "mirror", tmp_path / "mirror-public"
    for i in range(3):
        _publish(store, f"model-{i}")
    clock.pass_(24 * 30)
    assert RT.export(store.log, mirror, _key_doc(store), public=public, delay_days=7, now=clock.now)
    store.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(store.path, tmp_path / "backup.sqlite3")
    shutil.copytree(mirror, tmp_path / "mirror-backup")
    _publish(store, "late")
    assert RT.export(store.log, mirror, _key_doc(store), public=public, delay_days=7, now=clock.now)
    signer = store.log.signer
    for p in tmp_path.glob("register.sqlite3*"):
        p.unlink()
    shutil.copy(tmp_path / "backup.sqlite3", tmp_path / "register.sqlite3")
    shutil.rmtree(mirror)
    shutil.copytree(tmp_path / "mirror-backup", mirror)
    store = Store(tmp_path / "register.sqlite3")
    store.log.signer = signer
    clock.pass_(24 * 8)
    ok, lines = RT.recover(store, [mirror, public], signer.public_pem())
    assert ok and any("without its body" in x for x in lines), lines
    assert _crash_everywhere(store, tmp_path, clock, monkeypatch, mirror, public, full=True) >= 8


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
