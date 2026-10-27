"""The log against a lying mirror, a restored database, overlapping exports, a key switch and a quiet week: what a
verifier may conclude only from proved data, and what the register refuses rather than fork."""
import json
import shutil
import sqlite3
import sys
import threading

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
pytest.importorskip("rfc8785")

from fastapi.testclient import TestClient  # noqa: E402

from khaos_attribution import registry as R  # noqa: E402
from khaos_attribution import translog as T  # noqa: E402
from khaos_attribution.translog import verify as V  # noqa: E402
from register_service.app import create_app  # noqa: E402
from tests.test_register_service import _machine, _signed, _Transport, register  # noqa: E402,F401


def _mirror(app):
    return app.state.cfg["data"] / "mirror"


def _two(register):
    key, client = _machine(register, "artist-one")
    s1, s2 = _signed(key), _signed(key, run_id="run_2")
    return key, client, s1, client.publish(s1)["serial"], s2, client.publish(s2)["serial"]


def _prove(target, where, pem, state=None):
    return V.prove(target, V.Mirror(str(where)), pem, state)


def _copy(app, tmp_path, name="lie"):
    dst = tmp_path / name
    shutil.copytree(_mirror(app), dst)
    return dst


def _drop_body(where, leaf):
    """A mirror that leaves a body out, everywhere it could be read from."""
    (where / "log" / "entries" / f"{leaf}.json").unlink(missing_ok=True)
    bundle = where / "log" / "bodies.json"
    if bundle.is_file():
        doc = json.loads(bundle.read_text())
        hashes = json.loads((where / "log" / "hashes.json").read_text())["leaves"]
        doc["bodies"][hashes.index(leaf)] = None
        bundle.write_text(json.dumps(doc))


# ---- 1. what is proved is the requested record's admit, and a missing body needs a proved withhold ----

def test_a_lying_mirror_cannot_prove_another_records_admit_or_a_never_registered_record(register, tmp_path):
    app, store, http = register
    _, _, s1, _, s2, _ = _two(register)
    pem = store.log.signer.public_pem()
    lie = _copy(app, tmp_path)
    records = lie / "log" / "records.json"
    leaf2 = store.log.admit_leaf_of(s2["record_sha256"])
    fake = "ab" * 32
    if records.is_file():
        doc = json.loads(records.read_text())
        doc[s1["record_sha256"]] = doc[s2["record_sha256"]]
        doc[fake] = doc[s2["record_sha256"]]
        records.write_text(json.dumps(doc))
    hint = lie / "log" / "struck.json"
    hint.write_text(json.dumps({}))
    out = _prove(s1["record_sha256"], lie, pem)
    assert out["entry"]["kind"] == "admit" and out["entry"]["record_sha256"] == s1["record_sha256"]
    _drop_body(lie, leaf2)
    with pytest.raises((V.NotFound, T.LogError)):
        _prove(fake, lie, pem)


def test_a_body_missing_without_a_proved_withhold_fails(register, tmp_path):
    app, store, http = register
    _, _, s1, _, _, _ = _two(register)
    lie = _copy(app, tmp_path)
    leaf = store.log.admit_leaf_of(s1["record_sha256"])
    _drop_body(lie, leaf)
    with pytest.raises(T.LogError, match="withhold"):
        _prove(leaf, lie, store.log.signer.public_pem())
    store.log_withhold(leaf, "a private name")
    _export(app)
    out = _prove(leaf, _mirror(app), store.log.signer.public_pem())
    assert out["entry"] is None and out["withheld"]


def _export(app):
    from register_service.app import log_key_doc
    from register_service.translog import export
    export(app.state.store.log, _mirror(app), log_key_doc(app.state.store.log.signer, True))


# ---- 2. strikes come from proved entries, never from an editable hint ----

def test_a_strike_is_found_from_proved_entries_even_when_the_admit_is_withheld_or_the_hint_is_edited(register, tmp_path):
    app, store, http = register
    _, _, s1, _, _, _ = _two(register)
    _, warden = _machine(register, "artist-two", role="warden")
    leaf = store.log.admit_leaf_of(s1["record_sha256"])
    pem = store.log.signer.public_pem()
    warden.log_append({"kind": "revoke", "revokes": leaf, "reason": "consent withdrawn", "sitting": "S1"})
    warden.log_append({"kind": "withhold", "withholds": leaf, "reason": "privacy", "sitting": "S1"})
    out = _prove(leaf, _mirror(app), pem)
    assert out["struck"] and out["struck"]["reason"] == "consent withdrawn"
    assert leaf in json.loads((_mirror(app) / "log" / "struck.json").read_text()), "the hint is never filtered by withholding"
    lie = _copy(app, tmp_path)
    (lie / "log" / "struck.json").write_text("{}")
    assert _prove(leaf, lie, pem)["struck"]["reason"] == "consent withdrawn"


def test_a_withheld_revoke_still_strikes_with_its_reason_withheld(register):
    app, store, http = register
    _, _, s1, _, _, _ = _two(register)
    _, warden = _machine(register, "artist-two", role="warden")
    leaf = store.log.admit_leaf_of(s1["record_sha256"])
    revoke = warden.log_append({"kind": "revoke", "revokes": leaf, "reason": "a private matter", "sitting": "S1"})
    warden.log_append({"kind": "withhold", "withholds": revoke["leaf_hash"], "reason": "privacy", "sitting": "S1"})
    out = _prove(s1["record_sha256"], _mirror(app), store.log.signer.public_pem())
    assert out["struck"] and out["struck"]["reason"] is None and out["struck"]["reason_withheld"]


# ---- 3. a database behind its own published log refuses to append ----

def _restore(app, cfg, tmp_path, client, key):
    store = app.state.store
    db = cfg["data"] / "register.sqlite3"
    store.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(db, tmp_path / "backup.sqlite3")
    client.publish(_signed(key, run_id="run_3"))
    for p in db.parent.glob("register.sqlite3*"):
        p.unlink()
    shutil.copy(tmp_path / "backup.sqlite3", db)
    return create_app(cfg)


def test_a_restored_database_behind_the_mirror_refuses_to_append_and_check_says_why(register, tmp_path, monkeypatch, capsys):
    app, store, http = register
    key, client, s1, _, _, _ = _two(register)
    cfg = app.state.cfg
    mirror_head = None
    app2 = _restore(app, cfg, tmp_path, client, key)
    mirror_head = json.loads((_mirror(app) / "log" / "sth" / "latest.json").read_text())
    assert app2.state.store.log.size() < mirror_head["tree_size"]
    client._post = _Transport(TestClient(app2), key)
    with pytest.raises(R.RegisterError) as e:
        client.publish(_signed(key, run_id="run_4"))
    assert e.value.status == 503 and "behind" in str(e.value)
    assert json.loads((_mirror(app) / "log" / "sth" / "latest.json").read_text()) == mirror_head, "the mirror is not overwritten"
    from register_service.__main__ import main
    monkeypatch.setenv("REGISTER_DATA_DIR", str(cfg["data"]))
    monkeypatch.setenv("REGISTER_DEV_KEY", "1")
    assert main(["check"]) == 1 and "behind" in capsys.readouterr().out


def test_log_recover_replays_the_missing_entries_when_they_reproduce_the_signed_root(register, tmp_path, monkeypatch, capsys):
    app, store, http = register
    key, client, s1, _, _, _ = _two(register)
    cfg = app.state.cfg
    app2 = _restore(app, cfg, tmp_path, client, key)
    head = json.loads((_mirror(app) / "log" / "sth" / "latest.json").read_text())
    from register_service.__main__ import main
    monkeypatch.setenv("REGISTER_DATA_DIR", str(cfg["data"]))
    monkeypatch.setenv("REGISTER_DEV_KEY", "1")
    assert main(["log-recover"]) == 0
    assert "not in the database" in capsys.readouterr().out, "a record only the log still holds is named"
    app3 = create_app(cfg)
    st = app3.state.store
    assert st.log.size() == head["tree_size"] and st.log.head()["root_sha256"] == head["root_sha256"]
    client._post = _Transport(TestClient(app3), key)
    client.publish(_signed(key, run_id="run_5"))
    assert st.log.problems(st.log.signer.public_pem()) == []
    del app2


def test_log_recover_puts_back_a_leaf_whose_published_body_is_gone_and_marks_it_lost(register, tmp_path, monkeypatch, capsys):
    app, store, http = register
    key, client, s1, _, _, _ = _two(register)
    cfg = app.state.cfg
    _restore(app, cfg, tmp_path, client, key)
    hashes = json.loads((_mirror(app) / "log" / "hashes.json").read_text())["leaves"]
    for place in (_mirror(app), cfg["data"] / "mirror-public"):
        _drop_body(place, hashes[-1])
    from register_service.__main__ import main
    monkeypatch.setenv("REGISTER_DATA_DIR", str(cfg["data"]))
    monkeypatch.setenv("REGISTER_DEV_KEY", "1")
    assert main(["log-recover"]) == 0 and "without its body" in capsys.readouterr().out
    st = create_app(cfg).state.store
    last = st.log.entries(st.log.size() - 1, st.log.size())[0]["entry"]
    assert last["kind"] == "withhold" and last["withholds"] == hashes[-1]


# ---- 4. one standing admit per record ----

def test_a_warden_cannot_admit_a_withdrawn_or_already_admitted_record_and_withdraw_strikes_every_admit(register):
    app, store, http = register
    _, client, s1, ser1, s2, ser2 = _two(register)
    _, warden = _machine(register, "artist-two", role="warden")
    for sha in (s1["record_sha256"],):
        with pytest.raises(R.RegisterError) as e:
            warden.log_append({"kind": "admit", "record_sha256": sha, "sitting": "S1"})
        assert e.value.status == 422
    leaf = store.log.admit_leaf_of(s1["record_sha256"])
    warden.log_append({"kind": "revoke", "revokes": leaf, "reason": "pending a re-examination", "sitting": "S1"})
    again = warden.log_append({"kind": "admit", "record_sha256": s1["record_sha256"], "sitting": "S2"})
    client.withdraw(ser1, "the artist withdrew")
    struck = store.log.struck()
    assert again["leaf_hash"] in struck and leaf in struck
    client.withdraw(ser2, "gone")
    with pytest.raises(R.RegisterError) as e:
        warden.log_append({"kind": "admit", "record_sha256": s2["record_sha256"], "sitting": "S3"})
    assert e.value.status == 422 and "withdrawn" in str(e.value)


# ---- 5. overlapping exports never move the mirror backwards ----

def test_overlapping_exports_never_move_latest_backwards(register):
    import register_service.translog as RT
    app, store, http = register
    key, client = _machine(register, "artist-one")
    client.publish(_signed(key))
    original, gate, reached = RT._write, threading.Event(), threading.Event()

    def slow(path, data):
        if path.name == "latest.json" and threading.current_thread().name == "A":
            reached.set()
            gate.wait(5)
        original(path, data)
    RT._write = slow
    try:
        from register_service.app import log_key_doc
        a = threading.Thread(target=RT.export, args=(store.log, _mirror(app), log_key_doc(store.log.signer, True)), name="A")
        a.start()
        reached.wait(5)
        worker = threading.Thread(target=lambda: client.publish(_signed(key, run_id="run_2")))
        worker.start()
        worker.join(0.5)
        gate.set()
        a.join(5)
        worker.join(5)
    finally:
        RT._write = original
    assert json.loads((_mirror(app) / "log" / "sth" / "latest.json").read_text())["tree_size"] == store.log.size()


# ---- 6. a quiet log is re-signed daily and an old head is flagged ----

def test_the_daily_job_signs_a_fresh_head_and_the_verifier_flags_an_old_one(register, monkeypatch, capsys):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    s1 = _signed(key)
    client.publish(s1)
    from register_service import anchor
    from register_service.__main__ import refresh_head
    before = store.log.head()
    old = T.sign_head(before["tree_size"], bytes.fromhex(before["root_sha256"]), store.log.signer, timestamp="2026-01-01T00:00:00Z")
    store.db.execute("UPDATE log_heads SET timestamp = ?, signature = ? WHERE tree_size = ?",
                     (old["timestamp"], old["signature"], before["tree_size"]))
    _export(app)
    out = _prove(s1["record_sha256"], _mirror(app), store.log.signer.public_pem())
    assert out["stale_hours"] > 48 and out["frozen"] and any("may be frozen" in w for w in out["warnings"])
    refresh_head(app.state.cfg, store)
    fresh = store.log.head()
    assert fresh["tree_size"] == before["tree_size"] + 1 and fresh["timestamp"] > old["timestamp"]
    assert anchor is not None
    out = _prove(s1["record_sha256"], _mirror(app), store.log.signer.public_pem())
    assert out["warnings"] == []


# ---- 7. a key switch does not strand a remembered head; the newest size is never immutable ----

def test_a_head_remembered_before_a_key_switch_is_still_checked_after_it(register, tmp_path):
    app, store, http = register
    key, client, s1, _, _, _ = _two(register)
    dev_pem = store.log.signer.public_pem()
    _prove(s1["record_sha256"], _mirror(app), dev_pem)
    seen = {k: json.loads((_mirror(app) / "log" / "sth" / "1.json").read_text())[k] for k in ("tree_size", "root_sha256")}
    (app.state.cfg["data"] / "log-key.pem").unlink()
    (app.state.cfg["data"] / "log-key.dev").unlink()
    guild = R.SigningKey.load_or_create(app.state.cfg["data"] / "log-key.pem")
    app2 = create_app({**app.state.cfg, "log_dev_key": False})
    client._post = _Transport(TestClient(app2), key)
    client.publish(_signed(key, run_id="run_3"))
    out = _prove(s1["record_sha256"], _mirror(app), guild.public_pem(), seen)
    assert out["consistent_since"] == seen["tree_size"]
    forged = dict(seen, root_sha256="0" * 64)
    with pytest.raises(T.LogError, match="rewritten"):
        _prove(s1["record_sha256"], _mirror(app), guild.public_pem(), forged)


def test_the_newest_size_head_is_never_served_as_immutable(register):
    app, store, http = register
    _two(register)
    n = store.log.size()
    assert "immutable" not in http.get(f"/log/sth/{n}").headers["cache-control"]
    assert "immutable" in http.get(f"/log/sth/{n - 1}").headers["cache-control"]


# ---- 8. old rows with a blank reason ----

def test_genesis_takes_a_withdrawn_record_whose_reason_is_blank(register, tmp_path):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    client.withdraw(serial, "   ")
    db = sqlite3.connect(str(store.path))
    for table in ("log_leaves", "log_bodies", "log_nodes", "log_heads"):
        db.execute(f"DROP TABLE {table}")
    db.commit()
    db.close()
    shutil.rmtree(_mirror(app) / "log")
    shutil.rmtree(app.state.cfg["data"] / "mirror-public")
    (app.state.cfg["data"] / "log-started").unlink()
    again = create_app(app.state.cfg).state.store
    kinds = [e["entry"]["kind"] for e in again.log.entries(0, 10)]
    assert kinds == ["genesis", "admit", "revoke"]


# ---- 9. who appended ----

def test_a_warden_entry_names_the_machine_that_appended_it_and_the_ledger_has_the_row(register):
    app, store, http = register
    _, _, s1, _, _, _ = _two(register)
    wkey, warden = _machine(register, "artist-two", role="warden")
    leaf = store.log.admit_leaf_of(s1["record_sha256"])
    out = warden.log_append({"kind": "revoke", "revokes": leaf, "reason": "r", "sitting": "S1"})
    entry = store.log.entries(out["index"], out["index"] + 1)[0]["entry"]
    assert entry["appended_by"] == wkey.key_id
    row = store.ledger()[-1]
    assert row["kind"] == "log-append" and row["key_id"] == wkey.key_id and row["detail"]["leaf_hash"] == out["leaf_hash"]


# ---- 10. new bodies reach the public mirror only after the grace period ----

def test_new_bodies_wait_out_the_grace_period_before_the_public_mirror_and_a_withhold_inside_it_never_lands(register, tmp_path):
    app, store, http = register
    _, _, s1, _, _, _ = _two(register)
    public = app.state.cfg["data"] / "mirror-public"
    leaf = store.log.admit_leaf_of(s1["record_sha256"])
    pem = store.log.signer.public_pem()
    assert (public / "log" / "sth" / "latest.json").is_file()
    assert not (public / "log" / "entries" / f"{leaf}.json").exists(), "a new body waits"
    out = _prove(leaf, public, pem)
    assert out["entry"] is None and out["pending"] and any("not yet public" in w for w in out["warnings"]) and out["held_back_after"] == 1
    store.log_withhold(leaf, "a private name")
    from register_service.app import log_key_doc
    from register_service.translog import export
    export(store.log, _mirror(app), log_key_doc(store.log.signer, True), public=public, delay_days=0)
    assert not (public / "log" / "entries" / f"{leaf}.json").exists(), "withheld inside the grace period, never public"
    genesis = json.loads((public / "log" / "hashes.json").read_text())["leaves"][0]
    assert (public / "log" / "entries" / f"{genesis}.json").exists(), "after the grace period the body is public"
    lie = tmp_path / "lie"
    shutil.copytree(public, lie)
    doc = json.loads((lie / "log" / "bodies.json").read_text())
    doc["pending_from"], doc["bodies"][0] = 0, None
    (lie / "log" / "bodies.json").write_text(json.dumps(doc))
    head1 = json.loads((lie / "log" / "sth" / "1.json").read_text())
    old = T.sign_head(1, bytes.fromhex(head1["root_sha256"]), store.log.signer, timestamp="2026-01-01T00:00:00Z")
    (lie / "log" / "sth" / "1.json").write_text(json.dumps(old))
    with pytest.raises(T.LogError, match="publication delay"):
        _prove(genesis, lie, pem)


def test_the_browser_verifier_holds_the_same_line_against_a_lying_or_stale_mirror(register, tmp_path):
    """The page: a dropped body fails, a strike survives an emptied hint and a withheld admit, an old head is flagged."""
    from tests.test_register_log import run_page_verifier
    app, store, http = register
    _, _, s1, _, s2, _ = _two(register)
    _, warden = _machine(register, "artist-two", role="warden")
    pem = store.log.signer.public_pem()
    leaf1, leaf2 = store.log.admit_leaf_of(s1["record_sha256"]), store.log.admit_leaf_of(s2["record_sha256"])
    dropped = _copy(app, tmp_path, "dropped")
    _drop_body(dropped, leaf2)
    warden.log_append({"kind": "revoke", "revokes": leaf1, "reason": "consent withdrawn", "sitting": "S1"})
    warden.log_append({"kind": "withhold", "withholds": leaf1, "reason": "privacy", "sitting": "S1"})
    hint = _copy(app, tmp_path, "hint")
    (hint / "log" / "struck.json").write_text("{}")
    stale = _copy(app, tmp_path, "stale")
    latest = json.loads((stale / "log" / "sth" / "latest.json").read_text())
    old = T.sign_head(latest["tree_size"], bytes.fromhex(latest["root_sha256"]), store.log.signer, timestamp="2026-01-01T00:00:00Z")
    for name in ("latest.json", f"{latest['tree_size']}.json"):
        (stale / "log" / "sth" / name).write_text(json.dumps(old))
    drop, never, struck, aged = run_page_verifier(tmp_path, [
        (dropped, leaf2, pem), (_mirror(app), "ab" * 32, pem), (hint, leaf1, pem), (stale, s2["record_sha256"], pem)])
    assert drop[0].startswith("bad|Not proved") and "withhold" in drop[0]
    assert never[0].startswith("bad|Not proved: no admitted entry")
    assert struck[0].startswith("ok|Proved") and "bad|Struck: consent withdrawn (S1)" in struck
    assert aged[0].startswith("bad|Not current") and any(x.startswith("ok|Proved") for x in aged) and "may be frozen" in aged[0]


def test_log_recover_refuses_entries_that_do_not_make_the_signed_root(register, tmp_path, monkeypatch, capsys):
    """A published copy whose last entry was swapped for another, body and hash alike, is not replayed."""
    app, store, http = register
    key, client, s1, _, _, _ = _two(register)
    cfg = app.state.cfg
    app2 = _restore(app, cfg, tmp_path, client, key)
    size_before = app2.state.store.log.size()
    forged = T.make_entry("admit", serial=99999, record_sha256="cd" * 32)
    body = T.jcs(forged)
    for place in (_mirror(app), cfg["data"] / "mirror-public"):
        hashes_file, bundle_file = place / "log" / "hashes.json", place / "log" / "bodies.json"
        hashes, bundle = json.loads(hashes_file.read_text()), json.loads(bundle_file.read_text())
        hashes["leaves"][-1] = T.leaf_hash(body).hex()
        bundle["bodies"][-1] = body.decode()
        hashes_file.write_text(json.dumps(hashes))
        bundle_file.write_text(json.dumps(bundle))
    from register_service.__main__ import main
    monkeypatch.setenv("REGISTER_DATA_DIR", str(cfg["data"]))
    monkeypatch.setenv("REGISTER_DEV_KEY", "1")
    assert main(["log-recover"]) == 1 and "INCIDENT" in capsys.readouterr().out
    assert create_app(cfg).state.store.log.size() == size_before
