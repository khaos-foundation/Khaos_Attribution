"""A strike lost in a recovery stays an open question until the wardens answer it, a frozen copy is never an
all-clear, and only the log's own keys, a one-shot marker and a started-log marker decide whether the register opens."""
import json
import shutil
import sys

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


def _vrun(target, where, pem, tmp_path, capsys):
    key = tmp_path / "k.pub"
    key.write_text(pem)
    rc = V.main([target, "--mirror", str(where), "--key", str(key), "--state", "none"])
    return rc, capsys.readouterr().out


def _lose_a_strike(register, tmp_path):
    """A strike made after the backup, the full export lost, only the delayed public copy left: recovered blind."""
    from register_service.translog import recover
    app, store, http = register
    cfg = app.state.cfg
    key, client = _machine(register, "artist-one")
    signed = _signed(key)
    serial = client.publish(signed)["serial"]
    _, warden = _machine(register, "artist-two", role="warden")
    store.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(cfg["data"] / "register.sqlite3", tmp_path / "backup.sqlite3")
    leaf = store.log.admit_leaf_of(signed["record_sha256"])
    warden.log_append({"kind": "revoke", "revokes": leaf, "reason": "struck by the Guild", "sitting": "S9"})
    for p in cfg["data"].glob("register.sqlite3*"):
        p.unlink()
    shutil.rmtree(cfg["data"] / "mirror")
    shutil.copy(tmp_path / "backup.sqlite3", cfg["data"] / "register.sqlite3")
    st = create_app(cfg).state.store
    ok, lines = recover(st, log_places(cfg)[3], st.log.signer.public_pem())
    assert ok, lines
    export_all(cfg, st.log, st.log.signer, True)
    return app, cfg, st, signed, serial, leaf, warden


# ---- 1. a lost entry after an admit leaves it unresolved ----

def test_a_strike_lost_in_recovery_is_never_a_clean_proof(register, tmp_path, capsys):
    app, cfg, st, signed, serial, leaf, _ = _lose_a_strike(register, tmp_path)
    last = st.log.entries(st.log.size() - 1, st.log.size())[0]["entry"]
    assert last["kind"] == "withhold" and last["lost"] is True
    rc, out = _vrun(leaf, cfg["data"] / "mirror", st.log.signer.public_pem(), tmp_path, capsys)
    assert rc == V.HELD_BACK and "lost in a recovery; whether it struck this model is unknown" in out
    from tests.test_register_log import run_page_verifier
    page, = run_page_verifier(tmp_path, [(cfg["data"] / "mirror", leaf, st.log.signer.public_pem())])
    assert page[0].startswith("ok|Proved") and any("lost in a recovery; whether it struck this model is unknown" in x for x in page)


def _lost_leaf(store):
    return store.db.execute("SELECT leaf_hash FROM log_leaves WHERE kind = 'unavailable'").fetchone()["leaf_hash"]


def test_an_unresolved_model_is_under_review_until_a_warden_answers_the_lost_leaf(register, tmp_path, capsys):
    """Unresolved is its own list, never withdrawn; a confirm names the lost leaf it answers and needs a sitting."""
    app, cfg, st, signed, serial, leaf, warden = _lose_a_strike(register, tmp_path)
    app2 = create_app(cfg)
    http = TestClient(app2)
    store = app2.state.store
    lost = _lost_leaf(store)
    assert store.log.standing_admits(signed["record_sha256"]) == []
    listed = http.get("/api/withdrawn").json()
    assert serial in listed["unresolved"] and serial not in listed["withdrawn"]
    assert "under review" in http.get(f"/models/{serial}").text
    warden._post = _Transport(http, warden.key)
    with pytest.raises(R.RegisterError) as e:
        warden.log_append({"kind": "confirm", "answers": lost})
    assert e.value.status == 422, "a confirmation names its sitting"
    with pytest.raises(R.RegisterError) as e:
        warden.log_append({"kind": "confirm", "confirms": leaf, "sitting": "S10"})
    assert e.value.status == 422, "a confirmation names the lost leaf it answers"
    workshop_key, workshop = _machine((app2, store, http), "artist-one")
    with pytest.raises(R.RegisterError) as e:
        workshop.log_append({"kind": "confirm", "answers": lost, "sitting": "S10"})
    assert e.value.status == 403
    warden.log_append({"kind": "confirm", "answers": lost, "sitting": "S10"})
    assert store.log.standing_admits(signed["record_sha256"]) == [leaf]
    assert serial not in http.get("/api/withdrawn").json()["unresolved"]
    rc, out = _vrun(leaf, cfg["data"] / "mirror", store.log.signer.public_pem(), tmp_path, capsys)
    assert rc == V.PROVED, out


def test_a_revoke_resolves_the_lost_strike_as_struck(register, tmp_path, capsys):
    app, cfg, st, signed, serial, leaf, warden = _lose_a_strike(register, tmp_path)
    app2 = create_app(cfg)
    http = TestClient(app2)
    warden._post = _Transport(http, warden.key)
    warden.log_append({"kind": "revoke", "revokes": leaf, "reason": "struck again after the recovery", "sitting": "S10"})
    rc, out = _vrun(leaf, cfg["data"] / "mirror", app2.state.store.log.signer.public_pem(), tmp_path, capsys)
    assert rc == V.PROVED and "STRUCK: struck again after the recovery" in out


# ---- 2. a frozen copy has its own verdict ----

def test_a_copy_older_than_two_days_exits_frozen_even_for_a_struck_model(register, tmp_path, capsys):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    signed = _signed(key)
    client.publish(signed)
    _, warden = _machine(register, "artist-two", role="warden")
    leaf = store.log.admit_leaf_of(signed["record_sha256"])
    warden.log_append({"kind": "revoke", "revokes": leaf, "reason": "struck", "sitting": "S"})
    frozen = tmp_path / "frozen"
    shutil.copytree(app.state.cfg["data"] / "mirror", frozen)
    latest = json.loads((frozen / "log" / "sth" / "latest.json").read_text())
    old = T.sign_head(latest["tree_size"], bytes.fromhex(latest["root_sha256"]), store.log.signer, timestamp="2026-01-01T00:00:00Z")
    for name in ("latest.json", f"{latest['tree_size']}.json"):
        (frozen / "log" / "sth" / name).write_text(json.dumps(old))
    rc, out = _vrun(leaf, frozen, store.log.signer.public_pem(), tmp_path, capsys)
    assert rc == V.FROZEN == 5 and "this copy may be frozen" in out
    from tests.test_register_log import run_page_verifier
    page, = run_page_verifier(tmp_path, [(frozen, leaf, store.log.signer.public_pem())])
    assert page[0].startswith("bad|Not current") and "this copy may be frozen" in page[0]


# ---- 3. only the key that signed a size may speak for it ----

def test_a_head_signed_by_a_retired_key_cannot_close_the_register(register, tmp_path):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    client.publish(_signed(key))
    dev = store.log.signer
    cfg = app.state.cfg
    (cfg["data"] / "log-key.pem").unlink()
    (cfg["data"] / "log-key.dev").unlink()
    R.SigningKey.load_or_create(cfg["data"] / "log-key.pem")
    assert create_app({**cfg, "log_dev_key": False}).state.store.log.blocked is None
    evil = tmp_path / "evil"
    (evil / "log" / "sth").mkdir(parents=True)
    (evil / "log" / "sth" / "latest.json").write_text(json.dumps(T.sign_head(999, b"\x11" * 32, dev)))
    st = create_app({**cfg, "log_dev_key": False, "log_witnesses": [str(evil)]}).state.store
    assert st.log.blocked is None, "a retired key's head beyond its own sizes is not a witness"


def test_the_log_key_switch_destroys_the_retired_development_key():
    text = open("deploy/register-log-key.sh").read()
    assert "shred -u" in text and 'mv "$KEY" "$RETIRED.pem"' in text


# ---- 4. the unwitnessed open is one shot, and only for a new log ----

def test_the_unwitnessed_open_is_a_marker_used_once_and_refused_for_a_log_with_entries(tmp_path):
    cut_off = tmp_path / "cut-off"
    cfg = {"data": cut_off, "key_file": cut_off / "register-key.pem", "dev_key": True, "public_url": "x",
           "log_witnesses": ["http://127.0.0.1:9/ledger"]}
    cut_off.mkdir()
    marker = cut_off / "log-open-unwitnessed"
    marker.write_text("")
    first = create_app(cfg).state.store
    assert first.log.blocked is None and first.log.size() == 1 and not marker.exists(), "used once, then gone"
    shutil.rmtree(cut_off / "mirror")
    shutil.rmtree(cut_off / "mirror-public")
    marker.write_text("")
    again = create_app(cfg).state.store
    assert again.log.blocked and "has entries" in again.log.blocked


# ---- 5. once started, "nothing published" no longer opens an empty log ----

def test_an_empty_log_on_a_started_register_is_blocked_even_when_nothing_answers(register, tmp_path):
    import sqlite3
    app, store, http = register
    cfg = app.state.cfg
    assert (cfg["data"] / "log-started").is_file()
    db = sqlite3.connect(str(store.path))
    for table in ("log_leaves", "log_bodies", "log_nodes", "log_heads"):
        db.execute(f"DELETE FROM {table}")
    db.commit()
    db.close()
    for d in ("mirror", "mirror-public"):
        shutil.rmtree(cfg["data"] / d)
    st = create_app(cfg).state.store
    assert st.log.blocked and "started" in st.log.blocked and st.log.size() == 0


def test_backups_carry_the_started_marker():
    assert "log-started" in open("deploy/register-backup.sh").read()
    assert "log-started" in open("deploy/register-restore.sh").read()
