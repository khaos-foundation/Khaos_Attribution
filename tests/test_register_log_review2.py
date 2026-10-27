"""The log on a quiet week, on a new server and after a recovery: a heartbeat keeps freshness in the tree, the delay
runs from the head that first held an entry, a database no published copy vouches for appends nothing, and a
recovered log keeps its serials."""
import json
import shutil
import sys
import threading
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
pytest.importorskip("rfc8785")

from fastapi.testclient import TestClient  # noqa: E402

from khaos_attribution import registry as R  # noqa: E402
from khaos_attribution import translog as T  # noqa: E402
from khaos_attribution.translog import verify as V  # noqa: E402
from register_service.app import create_app, export_all, log_places  # noqa: E402
from tests.test_register_service import _machine, _signed, _Transport, register  # noqa: E402,F401


def _age_every_head(store, days):
    """Every head re-signed as if made `days` ago, so the delay and freshness can be seen without waiting."""
    old = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for r in store.db.execute("SELECT id, tree_size, root_sha256 FROM log_heads").fetchall():
        h = T.sign_head(r["tree_size"], bytes.fromhex(r["root_sha256"]), store.log.signer, timestamp=old)
        store.db.execute("UPDATE log_heads SET timestamp = ?, signature = ? WHERE id = ?", (old, h["signature"], r["id"]))


def _struck_last(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    signed = _signed(key)
    client.publish(signed)
    _, warden = _machine(register, "artist-two", role="warden")
    leaf = store.log.admit_leaf_of(signed["record_sha256"])
    warden.log_append({"kind": "revoke", "revokes": leaf, "reason": "struck by the Guild", "sitting": "S9"})
    return signed, leaf


# ---- N1: a heartbeat entry, one head per size, the delay from the head that first held the entry ----

def test_a_quiet_logs_strike_reaches_the_public_mirror_after_the_delay(register):
    from register_service.__main__ import refresh_head
    app, store, http = register
    signed, leaf = _struck_last(register)
    _age_every_head(store, 30)
    refresh_head(app.state.cfg, store)
    sizes = [h["tree_size"] for h in store.log.all_heads()]
    assert len(sizes) == len(set(sizes)), "every size has exactly one head"
    beat = store.log.entries(store.log.size() - 1, store.log.size())[0]["entry"]
    assert beat["kind"] == "heartbeat" and set(beat) == {"version", "kind", "time", "salt"}
    public = app.state.cfg["data"] / "mirror-public"
    out = V.prove(leaf, V.Mirror(str(public)), store.log.signer.public_pem())
    assert out["struck"] and out["struck"]["reason"] == "struck by the Guild" and out["held_back_after"] == 0


def test_a_mirror_cannot_hold_back_an_old_strike_by_calling_it_delayed(register, tmp_path):
    from register_service.__main__ import refresh_head
    app, store, http = register
    signed, leaf = _struck_last(register)
    _age_every_head(store, 30)
    refresh_head(app.state.cfg, store)
    lie = tmp_path / "lie"
    shutil.copytree(app.state.cfg["data"] / "mirror-public", lie)
    doc = json.loads((lie / "log" / "bodies.json").read_text())
    revoke_index = json.loads((lie / "log" / "hashes.json").read_text())["leaves"].index(store.log.struck()[leaf]["leaf_hash"])
    doc["pending_from"], doc["bodies"][revoke_index] = revoke_index, None
    (lie / "log" / "bodies.json").write_text(json.dumps(doc))
    with pytest.raises(T.LogError, match="publication delay"):
        V.prove(leaf, V.Mirror(str(lie)), store.log.signer.public_pem())


def test_held_back_entries_after_the_target_exit_with_their_own_code(register, tmp_path, capsys):
    """Inside the delay a strike may be among the held-back entries: proved, but not the plain all-clear."""
    app, store, http = register
    key, client = _machine(register, "artist-one")
    s1 = _signed(key)
    client.publish(s1)
    client.publish(_signed(key, run_id="run_2"))
    pem = tmp_path / "log.pub"
    pem.write_text(store.log.signer.public_pem())
    public = app.state.cfg["data"] / "mirror-public"
    genesis = store.log.entries(0, 1)[0]["leaf_hash"]
    rc = V.main([genesis, "--mirror", str(public), "--key", str(pem), "--state", "none"])
    assert rc == V.HELD_BACK == 4 and "not yet public" in capsys.readouterr().out
    assert V.main([genesis, "--mirror", str(app.state.cfg["data"] / "mirror"), "--key", str(pem), "--state", "none"]) == V.PROVED


def test_a_key_switch_appends_a_heartbeat_rather_than_a_second_head_of_one_size(register):
    app, store, http = register
    _struck_last(register)
    (app.state.cfg["data"] / "log-key.pem").unlink()
    (app.state.cfg["data"] / "log-key.dev").unlink()
    guild = R.SigningKey.load_or_create(app.state.cfg["data"] / "log-key.pem")
    st = create_app({**app.state.cfg, "log_dev_key": False}).state.store
    sizes = [h["tree_size"] for h in st.log.all_heads()]
    assert len(sizes) == len(set(sizes)) and st.log.head()["key_id"] == guild.key_id
    assert st.log.entries(st.log.size() - 1, st.log.size())[0]["entry"]["kind"] == "heartbeat"


# ---- N2: no readable published copy, no appends ----

def _restore_elsewhere(app, tmp_path, client, key, witnesses):
    """The backup put on a new server: the database from before the last publish, no mirror folders."""
    cfg = app.state.cfg
    db = cfg["data"] / "register.sqlite3"
    app.state.store.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(db, tmp_path / "backup.sqlite3")
    client.publish(_signed(key, run_id="run_3"))
    published = tmp_path / "published"
    shutil.copytree(cfg["data"] / "mirror-public", published)
    for p in db.parent.glob("register.sqlite3*"):
        p.unlink()
    for d in ("mirror", "mirror-public"):
        shutil.rmtree(cfg["data"] / d)
    shutil.copy(tmp_path / "backup.sqlite3", db)
    return create_app({**cfg, "log_witnesses": [str(w) if w != "published" else str(published) for w in witnesses]})


def test_a_restore_onto_a_new_server_is_blocked_by_the_public_copy(register, tmp_path):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    client.publish(_signed(key))
    app2 = _restore_elsewhere(app, tmp_path, client, key, ["published"])
    assert app2.state.store.log.blocked and "this database is behind the log published" in app2.state.store.log.blocked


def test_the_public_repository_is_a_witness_by_default(monkeypatch):
    from register_service.app import PUBLIC_MIRROR_RAW, settings
    monkeypatch.delenv("REGISTER_LOG_WITNESSES", raising=False)
    assert settings()["log_witnesses"] == [PUBLIC_MIRROR_RAW] == ["https://raw.githubusercontent.com/guild-register/ledger/main"]


def test_no_readable_published_copy_fails_closed_unless_the_log_is_new(register, tmp_path):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    client.publish(_signed(key))
    app2 = _restore_elsewhere(app, tmp_path, client, key, ["http://127.0.0.1:9/ledger"])
    blocked = app2.state.store.log.blocked
    assert blocked and "no published copy" in blocked
    client._post = _Transport(TestClient(app2), key)
    with pytest.raises(R.RegisterError) as e:
        client.publish(_signed(key, run_id="run_4"))
    assert e.value.status == 503
    fresh = tmp_path / "fresh"
    first = create_app({"data": fresh, "key_file": fresh / "register-key.pem", "dev_key": True, "public_url": "x",
                        "log_witnesses": [str(tmp_path / "empty-repo")]})
    assert first.state.store.log.blocked is None and first.state.store.log.size() == 1, "the very first start opens"
    cut_off = tmp_path / "cut-off"
    unsure = create_app({"data": cut_off, "key_file": cut_off / "register-key.pem", "dev_key": True, "public_url": "x",
                         "log_witnesses": ["http://127.0.0.1:9/ledger"]})
    assert unsure.state.store.log.blocked, "a first start that cannot read the public copy cannot know it is the first"
    override = create_app({"data": cut_off, "key_file": cut_off / "register-key.pem", "dev_key": True, "public_url": "x",
                           "log_witnesses": ["http://127.0.0.1:9/ledger"], "log_unwitnessed": True})
    assert override.state.store.log.blocked is None


# ---- N3: lost entries whose bodies were never published are replayed as leaves ----

def test_recover_replays_a_lost_entry_whose_body_is_withheld_or_held_back(register, tmp_path):
    from register_service.translog import recover
    app, store, http = register
    cfg = app.state.cfg
    key, client = _machine(register, "artist-one")
    client.publish(_signed(key))
    _, warden = _machine(register, "artist-two", role="warden")
    store.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(cfg["data"] / "register.sqlite3", tmp_path / "backup.sqlite3")
    s2 = _signed(key, run_id="run_2")
    client.publish(s2)
    warden.log_append({"kind": "withhold", "withholds": store.log.admit_leaf_of(s2["record_sha256"]), "reason": "privacy",
                       "sitting": "S"})
    head = store.log.head()
    for p in cfg["data"].glob("register.sqlite3*"):
        p.unlink()
    shutil.copy(tmp_path / "backup.sqlite3", cfg["data"] / "register.sqlite3")
    st = create_app(cfg).state.store
    ok, lines = recover(st, log_places(cfg)[3], st.log.signer.public_pem())
    assert ok, lines
    assert st.log.size() == head["tree_size"] and st.log.tree().root().hex() == head["root_sha256"]
    assert st.log.problems(st.log.signer.public_pem()) == [] and any("without its body" in x for x in lines)


# ---- N4: a republished record keeps the serial the log admitted it under ----

def test_a_record_republished_after_recovery_keeps_its_serial_and_adds_no_admit(register, tmp_path):
    from register_service.translog import recover
    app, store, http = register
    cfg = app.state.cfg
    key, client = _machine(register, "artist-one")
    client.publish(_signed(key))
    store.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(cfg["data"] / "register.sqlite3", tmp_path / "backup.sqlite3")
    s2 = _signed(key, run_id="run_2")
    first = client.publish(s2)
    for p in cfg["data"].glob("register.sqlite3*"):
        p.unlink()
    shutil.copy(tmp_path / "backup.sqlite3", cfg["data"] / "register.sqlite3")
    st = create_app(cfg).state.store
    assert recover(st, log_places(cfg)[3], st.log.signer.public_pem())[0]
    app3 = create_app(cfg)
    size = app3.state.store.log.size()
    client._post = _Transport(TestClient(app3), key)
    again = client.publish(s2)
    assert again["serial"] == first["serial"] and again["countersignature"] == first["countersignature"]
    assert app3.state.store.log.size() == size and len(app3.state.store.log.standing_admits(s2["record_sha256"])) == 1


# ---- N5: an export that cannot get the lock in time is skipped, not waited on ----

def test_an_export_waits_a_bounded_time_for_the_lock_then_skips(register, monkeypatch, caplog):
    import register_service.translog as RT
    app, store, http = register
    monkeypatch.setattr(RT, "EXPORT_LOCK_TIMEOUT", 0.2)
    held = threading.Event()
    release = threading.Event()

    def hold():
        with RT._export_lock(app.state.cfg["data"] / "mirror"):
            held.set()
            release.wait(5)
    t = threading.Thread(target=hold)
    t.start()
    held.wait(5)
    try:
        assert export_all(app.state.cfg, store.log, store.log.signer, True) is False
    finally:
        release.set()
        t.join(5)
    assert any("skipped" in r.message for r in caplog.records)
    assert export_all(app.state.cfg, store.log, store.log.signer, True) is True


def test_the_page_says_the_same_about_held_back_entries_and_a_quiet_logs_strike(register, tmp_path):
    from register_service.__main__ import refresh_head
    from tests.test_register_log import run_page_verifier
    app, store, http = register
    public = app.state.cfg["data"] / "mirror-public"
    pem = store.log.signer.public_pem()
    genesis = store.log.entries(0, 1)[0]["leaf_hash"]
    signed, leaf = _struck_last(register)
    fresh = tmp_path / "fresh"
    shutil.copytree(public, fresh)
    _age_every_head(store, 30)
    refresh_head(app.state.cfg, store)
    held, quiet = run_page_verifier(tmp_path, [(fresh, genesis, pem), (public, leaf, pem)])
    assert held[0].startswith("ok|Proved") and any("newer entries are not yet public" in x for x in held)
    assert quiet[0].startswith("ok|Proved") and "bad|Struck: struck by the Guild (S9)" in quiet
    assert not any("not yet public" in x for x in quiet)
