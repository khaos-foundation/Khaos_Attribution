"""The log's export in tiles of 4: a full tile's bytes never change, an export with nothing new writes nothing, a
tile seals only below the delay edge it is published beside, a stop after any file leaves copies both readers
verify (moving from the old layout, a withhold inside a sealed tile, a delay-held leaf withheld), the readers fall
back to the old layout only where the tiles are absent, and khaos-log-verify's tile cache never outlives a check."""
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
pytest.importorskip("rfc8785")

import register_service.translog as RT  # noqa: E402
from khaos_attribution import translog as T  # noqa: E402
from khaos_attribution.translog import verify as V  # noqa: E402
from tests import log_files as LF  # noqa: E402
from tests.test_register_export import _Died, _key_doc, _publish, _store, clock  # noqa: E402,F401
from tests.test_register_log import run_page_verifier  # noqa: E402


@pytest.fixture(autouse=True)
def four(monkeypatch):
    monkeypatch.setattr(T, "TILE", 4)


def _export(store, clock, mirror, public, delay=7, full=False):
    assert RT.export(store.log, mirror, _key_doc(store), public=public, delay_days=delay, now=clock.now, full=full)


def _snapshot(*places) -> dict:
    # The lock file is the writers' own, not the export's.
    return {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for place in places for p in sorted(Path(place).rglob("*"))
            if p.is_file() and not p.name.startswith(".")}


def _stops(store, clock, monkeypatch, tmp_path, mirror, public, tag: str, delay=7, full=False) -> list:
    """The export about to run, stopped after each file it writes or removes in turn; a copy of both places at each stop."""
    keep = tmp_path / f"keep-{tag}"
    for name, where in (("mirror", mirror), ("public", public)):
        if where.exists():
            shutil.copytree(where, keep / name)
    record = [tuple(r) for r in store.db.execute("SELECT * FROM log_exports").fetchall()]
    stops, k = [], 0
    while True:
        for name, where in (("mirror", mirror), ("public", public)):
            shutil.rmtree(where, ignore_errors=True)
            if (keep / name).exists():
                shutil.copytree(keep / name, where)
        store.db.execute("DELETE FROM log_exports")
        store.db.executemany("INSERT INTO log_exports VALUES (?, ?, ?, ?, ?)", record)
        made, write, remove = [0], RT.write_whole, RT._remove

        def step():
            if made[0] == k:
                raise _Died
            made[0] += 1
        finished = True
        with monkeypatch.context() as m:
            m.setattr(RT, "write_whole", lambda path, data: (step(), write(path, data))[1])
            m.setattr(RT, "_remove", lambda path: (step(), remove(path))[1])
            try:
                RT.export(store.log, mirror, _key_doc(store), public=public, delay_days=delay, now=clock.now, full=full)
            except _Died:
                finished = False
        stop = tmp_path / f"stop-{tag}-{k}"
        for where in (mirror, public):
            shutil.copytree(where, stop / where.name)
        stops.append(stop)
        if finished:
            return stops
        k += 1


def _both_readers_verify(tmp_path, stops: list, pem: str) -> int:
    """Every leaf each stopped copy's head holds proves in Python; the page proves the first and last leaf of each."""
    cases = []
    for stop in stops:
        for where in sorted(stop.iterdir()):
            if not (where / "log" / "sth" / "latest.json").is_file():
                continue
            n = json.loads((where / "log" / "sth" / "latest.json").read_text())["tree_size"]
            leaves = V.read_leaves(V.Mirror(str(where)), n)
            for leaf in leaves:
                V.prove(leaf, V.Mirror(str(where)), pem)
            cases += [(where, leaves[0], pem), (where, leaves[-1], pem)]
    for (where, leaf, _), shown in zip(cases, run_page_verifier(tmp_path, cases)):
        # A head older than two days is said first; the proof is what counts here.
        assert any(x.startswith("ok|Proved") for x in shown) and not any(x.startswith("bad|Not proved") for x in shown), (str(where), shown)
    return len(cases)


def _format_three(store, root: Path) -> None:
    """The layout before tiles, as its export wrote it, and its record."""
    snap = RT.snapshot(store.log)
    leaves, n = snap["leaves"], len(snap["leaves"])
    for i, leaf in enumerate(leaves):
        RT.write_whole(root / "proofs" / f"{leaf}.json", RT._json({"leaf_hash": leaf, "leaf_index": i, "tree_size": n,
                                                                    "audit_path": T.hexes(snap["tree"].inclusion(i, n))}))
    RT.write_whole(root / "hashes.json", RT._json({"tree_size": n, "leaves": leaves}))
    RT.write_whole(root / "bodies.json", RT._json({"tree_size": n, "pending_from": None,
                                                   "bodies": [b.decode() if b is not None else None for b in snap["bodies"]]}))
    for head in snap["heads"]:
        RT.write_whole(root / "sth" / f"{head['tree_size']}.json", RT._json(head))
    RT.write_whole(root / "sth" / "latest.json", RT._json(snap["heads"][-1]))
    RT.write_whole(root / "key.json", RT._json(_key_doc(store)))
    with store._tx():
        store.db.execute("INSERT OR REPLACE INTO log_exports VALUES (?, 3, ?, ?, NULL)", (str(root.resolve()), store.log.signer.key_id, n))


def test_a_full_tile_never_changes_and_a_sealed_body_tile_changes_only_for_a_withhold(tmp_path, clock):
    store = _store(tmp_path)
    mirror, public = tmp_path / "mirror", tmp_path / "mirror-public"
    shas = [_publish(store, f"model-{i}") for i in range(9)]
    clock.pass_(24 * 30)
    _export(store, clock, mirror, public)
    tiles = {n: (mirror / "log" / n).read_bytes() for n in ("hashes/0.json", "hashes/1.json", "bodies/0.json", "bodies/1.json")}
    for i in range(5):
        _publish(store, f"later-{i}")
        clock.pass_(24 * 9)
        _export(store, clock, mirror, public)
    assert {n: (mirror / "log" / n).read_bytes() for n in tiles} == tiles
    leaf = store.log.admit_leaf_of(shas[0])
    store.log_withhold(leaf, "a private name")
    _export(store, clock, mirror, public)
    assert (mirror / "log" / "hashes/0.json").read_bytes() == tiles["hashes/0.json"]
    index = store.log.leaf(leaf)["idx"]
    assert index < 4 and json.loads((mirror / "log" / "bodies/0.json").read_text())["bodies"][index] is None
    assert (mirror / "log" / "bodies/1.json").read_bytes() == tiles["bodies/1.json"]


def test_an_export_with_nothing_new_writes_no_file_and_an_anchor_still_reaches_both_places(tmp_path, clock):
    store = _store(tmp_path)
    mirror, public = tmp_path / "mirror", tmp_path / "mirror-public"
    for i in range(6):
        _publish(store, f"model-{i}")
    _export(store, clock, mirror, public)
    before = _snapshot(mirror, public)
    _export(store, clock, mirror, public)
    assert _snapshot(mirror, public) == before
    head = store.log.head()
    with store._tx():
        store.db.execute("INSERT INTO log_anchors (tree_size, root_sha256, backend, stamped_at, receipt, state)"
                         " VALUES (?, ?, 'test', ?, ?, 'pending')", (head["tree_size"], head["root_sha256"], head["timestamp"], b"receipt"))
    _export(store, clock, mirror, public)
    for where in (mirror, public):
        assert json.loads((where / "log" / "anchors.json").read_text())[0]["tree_size"] == head["tree_size"]
    with store._tx():
        store.db.execute("UPDATE log_anchors SET state = 'complete', receipt = ?", (b"done",))
    RT.export_anchors(store.log, mirror, public)
    for where in (mirror, public):
        assert (where / "log" / "anchors" / f"{head['tree_size']}.ots").read_bytes() == b"done"


def test_moving_from_the_old_layout_verifies_at_every_stop(tmp_path, clock, monkeypatch):
    store = _store(tmp_path)
    mirror, public = tmp_path / "mirror", tmp_path / "mirror-public"
    for i in range(6):
        _publish(store, f"model-{i}")
    clock.pass_(24 * 30)
    for where in (mirror, public):
        _format_three(store, where / "log")
    for i in range(5):
        _publish(store, f"after-{i}")
    stops = _stops(store, clock, monkeypatch, tmp_path, mirror, public, "move")
    assert _both_readers_verify(tmp_path, stops, store.log.signer.public_pem()) > 20
    for where in (mirror, public):
        assert not (where / "log" / "hashes.json").exists() and not (where / "log" / "bodies.json").exists()
        assert (where / "log" / "hashes" / "tail.json").is_file() and (where / "log" / "bodies" / "0.json").is_file()


def test_a_withhold_inside_a_sealed_tile_verifies_at_every_stop(tmp_path, clock, monkeypatch):
    store = _store(tmp_path)
    mirror, public = tmp_path / "mirror", tmp_path / "mirror-public"
    shas = [_publish(store, f"model-{i}") for i in range(10)]
    clock.pass_(24 * 30)
    _export(store, clock, mirror, public)
    leaf = store.log.admit_leaf_of(shas[1])
    index = store.log.leaf(leaf)["idx"]
    assert index < json.loads((public / "log" / "bodies" / "tail.json").read_text())["first"], "the leaf is in a sealed tile"
    store.log_withhold(leaf, "a private name")
    stops = _stops(store, clock, monkeypatch, tmp_path, mirror, public, "withhold")
    _both_readers_verify(tmp_path, stops, store.log.signer.public_pem())
    for where in (mirror, public):
        assert json.loads((where / "log" / "bodies" / f"{index // 4}.json").read_text())["bodies"][index % 4] is None


def test_a_delay_held_leaf_withheld_seals_its_tile_only_beside_the_head_that_proves_it(tmp_path, clock, monkeypatch):
    """A leaf the old head knew only as held back by the delay, withheld before the delay passes it."""
    store = _store(tmp_path)
    mirror, public = tmp_path / "mirror", tmp_path / "mirror-public"
    for i in range(3):
        _publish(store, f"model-{i}")
    clock.pass_(24 * 30)
    _export(store, clock, mirror, public)
    late = [store.log.admit_leaf_of(_publish(store, f"late-{i}")) for i in range(4)]
    clock.pass_(24 * 6.5)
    _export(store, clock, mirror, public)
    index = store.log.leaf(late[1])["idx"]
    assert LF.bodies(public / "log")[index] is None and index // 4 == 1
    clock.pass_(2)
    store.log_withhold(late[1], "a private name")
    clock.pass_(24)
    stops = _stops(store, clock, monkeypatch, tmp_path, mirror, public, "late")
    _both_readers_verify(tmp_path, stops, store.log.signer.public_pem())
    old_size = store.log.size() - 1
    for stop in stops:
        log = stop / "mirror-public" / "log"
        if json.loads((log / "sth" / "latest.json").read_text())["tree_size"] == old_size:
            assert not (log / "bodies" / "1.json").exists(), "the tile seals only beside the head that proves the withhold"
    assert json.loads((public / "log" / "bodies" / "1.json").read_text())["bodies"][index % 4] is None


def test_the_sealed_tiles_never_move_back_and_a_whole_export_leaves_no_tile_past_its_tail(tmp_path, clock):
    """Delay 0, then 7: the tiles sealed stay, a withhold still reaches one, and a whole export keeps none past its tail."""
    store = _store(tmp_path)
    mirror, public = tmp_path / "mirror", tmp_path / "mirror-public"
    shas = [_publish(store, f"model-{i}") for i in range(10)]
    _export(store, clock, mirror, public, delay=0)
    sealed = json.loads((public / "log" / "bodies" / "tail.json").read_text())["first"]
    assert sealed == 8
    _publish(store, "next")
    _export(store, clock, mirror, public, delay=7)
    tail = json.loads((public / "log" / "bodies" / "tail.json").read_text())
    assert tail["first"] == sealed and tail["pending_from"] < sealed
    leaf = store.log.admit_leaf_of(shas[5])
    store.log_withhold(leaf, "a private name")
    _export(store, clock, mirror, public, delay=7)
    index = store.log.leaf(leaf)["idx"]
    assert json.loads((public / "log" / "bodies" / f"{index // 4}.json").read_text())["bodies"][index % 4] is None
    pem = store.log.signer.public_pem()
    V.load_log(V.Mirror(str(public)), pem)
    _export(store, clock, mirror, public, delay=7, full=True)
    first = json.loads((public / "log" / "bodies" / "tail.json").read_text())["first"]
    assert first < sealed and [p.stem for p in (public / "log" / "bodies").glob("*.json")
                               if p.stem.isdigit() and int(p.stem) >= first // 4] == []
    for leaf_hex in LF.leaves(public / "log"):
        V.prove(leaf_hex, V.Mirror(str(public)), pem)


def test_the_old_layout_is_read_only_where_the_tiles_are_absent(tmp_path, clock):
    store = _store(tmp_path)
    mirror, public = tmp_path / "mirror", tmp_path / "mirror-public"
    for i in range(5):
        _publish(store, f"model-{i}")
    _export(store, clock, mirror, public)
    pem = store.log.signer.public_pem()
    old = tmp_path / "old"
    _format_three(store, old / "log")
    assert V.load_log(V.Mirror(str(old)), pem)["latest"] == V.load_log(V.Mirror(str(mirror)), pem)["latest"]
    torn = tmp_path / "torn"
    shutil.copytree(mirror, torn)
    (torn / "log" / "bodies" / "tail.json").unlink()
    _format_three(store, torn / "log")
    with pytest.raises(V.Unreadable, match="no body tail"):
        V.load_log(V.Mirror(str(torn)), pem)
    bad = tmp_path / "bad"
    shutil.copytree(mirror, bad)
    (bad / "log" / "hashes" / "tail.json").write_text("{not json")
    with pytest.raises(V.Unreadable):
        V.load_log(V.Mirror(str(bad)), pem)
    shown = run_page_verifier(tmp_path, [(old, LF.leaves(mirror / "log")[0], pem), (torn, LF.leaves(mirror / "log")[0], pem)])
    assert any(x.startswith("ok|Proved") for x in shown[0]) and shown[1][-1] == "bad|Not proved: the mirror has hash tiles but no body tail"


def test_a_cached_tile_that_fails_is_read_again_and_a_withheld_body_leaves_the_cache(tmp_path, clock):
    store = _store(tmp_path)
    mirror, public = tmp_path / "mirror", tmp_path / "mirror-public"
    shas = [_publish(store, f"model-{i}") for i in range(9)]
    _export(store, clock, mirror, public)
    pem, key_id = store.log.signer.public_pem(), store.log.signer.key_id
    cache = tmp_path / "cache"

    def load():
        return V.load_log(V.CachedMirror(V.Mirror(str(mirror)), cache, key_id), pem)
    load()
    kept = cache / key_id / "4"
    assert (kept / "hashes" / "0.json").is_file() and (kept / "bodies" / "1.json").is_file()
    (kept / "hashes" / "0.json").write_text("{not json")
    load()
    doc = json.loads((kept / "hashes" / "1.json").read_text())
    doc["leaves"][0] = "00" * 32
    (kept / "hashes" / "1.json").write_text(json.dumps(doc))
    load()
    assert json.loads((kept / "hashes" / "1.json").read_text()) == json.loads((mirror / "log" / "hashes" / "1.json").read_text())
    leaf = store.log.admit_leaf_of(shas[0])
    index = store.log.leaf(leaf)["idx"]
    store.log_withhold(leaf, "a private name")
    _export(store, clock, mirror, public)
    log = load()
    assert log["entries"][index] is None and leaf in log["withheld"]
    assert json.loads((kept / "bodies" / f"{index // 4}.json").read_text())["bodies"][index % 4] is None
    broken = tmp_path / "broken"
    shutil.copytree(mirror, broken)
    LF.edit_bodies(broken / "log", lambda t: t.replace('"serial":', '"serial":1'))
    with pytest.raises(T.LogError):
        V.load_log(V.CachedMirror(V.Mirror(str(broken)), tmp_path / "cache2", key_id), pem)


def test_the_cache_lives_where_it_is_told(monkeypatch, tmp_path):
    monkeypatch.setenv("KHAOS_LOG_CACHE", str(tmp_path / "elsewhere"))
    assert V.cache_base() == tmp_path / "elsewhere"
    monkeypatch.delenv("KHAOS_LOG_CACHE")
    assert V.cache_base() == Path.home() / ".khaos" / "log-cache"


def test_an_unchanged_file_is_answered_by_its_tag(tmp_path):
    import register_service.app as A
    from fastapi.testclient import TestClient
    cfg = {"log_enable": True, "data": tmp_path / "data", "key_file": tmp_path / "data" / "register-key.pem", "dev_key": True,
           "public_url": "https://register.example"}
    http = TestClient(A.create_app(cfg))
    first = http.get("/mirror/log/hashes/tail.json")
    assert first.status_code == 200 and first.headers["cache-control"] == "public, max-age=10" and first.headers["etag"]
    again = http.get("/mirror/log/hashes/tail.json", headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304 and not again.content and again.headers["etag"] == first.headers["etag"]
    assert http.get("/mirror/log/hashes/tail.json", headers={"If-None-Match": '"other"'}).status_code == 200
