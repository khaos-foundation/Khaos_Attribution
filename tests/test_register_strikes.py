"""A warden's strike takes a model out of every door a Space reads, until a warden admits it again; an account's own
withdrawal is final. The mirror's model files follow the log, and a release racing a withdrawal is never offered."""
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
from register_service.app import create_app, log_places  # noqa: E402
from tests.test_register_service import _machine, _signed, _Transport, register  # noqa: E402,F401

FP = [[h, h * 10] for h in range(1, 40)]


def _published(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    signed = _signed(key)
    serial = client.publish(signed)["serial"]
    _, space = _machine(register, "artist-two", role="space")
    space.served(serial, "https://space.example", "serving")
    client.release("gen-1", serial, FP)
    _, warden = _machine(register, "artist-two", role="warden")
    return key, client, space, warden, signed, serial


def _strike(store, warden, signed, reason="consent withdrawn"):
    leaf = store.log.admit_leaf_of_latest(signed["record_sha256"])
    return warden.log_append({"kind": "revoke", "revokes": leaf, "reason": reason, "sitting": "S1"})


def test_a_wardens_strike_takes_the_model_out_of_every_door(register):
    app, store, http = register
    key, client, space, warden, signed, serial = _published(register)
    _strike(store, warden, signed)
    listed = http.get("/api/withdrawn").json()
    assert serial in listed["withdrawn"] and listed["struck"] == [serial] and serial not in listed["unresolved"]
    assert not app.state.index.offered("gen-1"), "a struck model's released piece is not offered"
    for call in (lambda: space.served(serial, "https://space.example", "serving"), lambda: client.release("gen-2", serial, FP)):
        with pytest.raises(R.RegisterError) as e:
            call()
        assert e.value.status == 410
    payload = signed["record"]["model"]["watermark_payload"]
    assert serial not in store.serials_in_bucket(payload)
    assert http.get(f"/api/models/{serial}").json()["state"] == "struck"
    on_disk = json.loads((app.state.cfg["data"] / "mirror" / "models" / f"{serial}.json").read_text())
    assert on_disk["state"] == "struck", "the public mirror follows the strike"
    assert "struck" in http.get(f"/models/{serial}").text


def test_a_struck_model_a_warden_admits_again_is_served_again(register):
    app, store, http = register
    key, client, space, warden, signed, serial = _published(register)
    _strike(store, warden, signed)
    warden.log_append({"kind": "admit", "record_sha256": signed["record_sha256"], "sitting": "S2"})
    listed = http.get("/api/withdrawn").json()
    assert serial not in listed["withdrawn"] and listed["struck"] == []
    assert app.state.index.offered("gen-1")
    assert space.served(serial, "https://space.example", "serving")["state"] == "serving"
    assert http.get(f"/api/models/{serial}").json()["state"] == "registered"


def test_an_accounts_withdrawal_is_final_through_a_restart_and_a_wardens_admit(register):
    """Withdrawal strikes every admit too, so the log alone cannot tell it from a warden's strike: the state decides."""
    app, store, http = register
    key, client, space, warden, signed, serial = _published(register)
    client.withdraw(serial, "the artist withdrew")
    app2 = create_app(app.state.cfg)
    http2 = TestClient(app2)
    assert app2.state.store.record(serial)["state"] == "withdrawn"
    warden._post = _Transport(http2, warden.key)
    with pytest.raises(R.RegisterError) as e:
        warden.log_append({"kind": "admit", "record_sha256": signed["record_sha256"], "sitting": "S2"})
    assert e.value.status == 422 and "withdrawn" in str(e.value)
    assert app2.state.store.record(serial)["state"] == "withdrawn" and not app2.state.index.offered("gen-1")
    listed = http2.get("/api/withdrawn").json()
    assert listed["withdrawn"] == [serial] and listed["struck"] == []


def test_striking_a_model_under_review_withdraws_it(register, tmp_path):
    from tests.test_register_log_lost_strikes import _lose_a_strike
    app, cfg, st, signed, serial, leaf, warden = _lose_a_strike(register, tmp_path)
    app2 = create_app(cfg)
    http = TestClient(app2)
    assert serial in http.get("/api/withdrawn").json()["unresolved"]
    warden._post = _Transport(http, warden.key)
    warden.log_append({"kind": "revoke", "revokes": leaf, "reason": "struck again", "sitting": "S10"})
    listed = http.get("/api/withdrawn").json()
    assert serial in listed["withdrawn"] and serial not in listed["unresolved"]


def test_a_strike_the_recovery_puts_back_takes_effect_at_the_next_start(register, tmp_path):
    from register_service.translog import recover
    app, store, http = register
    cfg = app.state.cfg
    key, client, space, warden, signed, serial = _published(register)
    store.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(cfg["data"] / "register.sqlite3", tmp_path / "backup.sqlite3")
    _strike(store, warden, signed)
    for p in cfg["data"].glob("register.sqlite3*"):
        p.unlink()
    shutil.copy(tmp_path / "backup.sqlite3", cfg["data"] / "register.sqlite3")
    st = create_app(cfg).state.store
    assert st.log.blocked
    ok, lines = recover(st, log_places(cfg)[3], st.log.signer.public_pem())
    assert ok, lines
    app3 = create_app(cfg)
    assert app3.state.store.record(serial)["state"] == "struck"
    assert serial in TestClient(app3).get("/api/withdrawn").json()["withdrawn"]
    assert not app3.state.index.offered("gen-1")


def test_an_account_withdrawal_the_recovery_puts_back_is_final(register, tmp_path):
    from register_service.translog import recover
    app, store, http = register
    cfg = app.state.cfg
    key, client, space, warden, signed, serial = _published(register)
    store.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(cfg["data"] / "register.sqlite3", tmp_path / "backup.sqlite3")
    client.withdraw(serial, "the artist withdrew")
    for p in cfg["data"].glob("register.sqlite3*"):
        p.unlink()
    shutil.copy(tmp_path / "backup.sqlite3", cfg["data"] / "register.sqlite3")
    st = create_app(cfg).state.store
    ok, lines = recover(st, log_places(cfg)[3], st.log.signer.public_pem())
    assert ok, lines
    app3 = create_app(cfg)
    assert app3.state.store.record(serial)["state"] == "withdrawn"
    assert app3.state.store.ledger_ok()


def test_the_model_json_answers_from_the_store_and_a_missing_file_is_written_at_start(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    path = app.state.cfg["data"] / "mirror" / "models" / f"{serial}.json"
    path.unlink()
    doc = http.get(f"/api/models/{serial}")
    assert doc.status_code == 200 and doc.json()["serial"] == serial
    create_app(app.state.cfg)
    assert json.loads(path.read_text())["serial"] == serial


def test_a_release_that_lands_after_a_withdrawal_is_not_offered(register, monkeypatch):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    index = app.state.index
    real = index.release

    def withdrawn_meanwhile(*a, **k):
        # The withdraw route runs between this release's read of the record and its insert.
        store.withdraw(serial, key.key_id, "the artist withdrew")
        index.withdraw_model(serial)
        return real(*a, **k)

    monkeypatch.setattr(index, "release", withdrawn_meanwhile)
    client.release("gen-late", serial, FP)
    assert not index.offered("gen-late")
