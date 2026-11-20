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
from tests import log_files as LF  # noqa: E402

pytestmark = pytest.mark.usefixtures("tile_widths")

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


def _interleaved(store, monkeypatch, after):
    """store.log_append that runs ``after`` between its commit and the app's own next step."""
    real = store.log_append

    def append_then(entry, key_id):
        out = real(entry, key_id)
        monkeypatch.setattr(store, "log_append", real)
        after()
        return out
    monkeypatch.setattr(store, "log_append", append_then)


def test_a_withdrawal_landing_between_a_readmit_and_its_offer_keeps_the_pieces_out(register, monkeypatch):
    app, store, http = register
    key, client, space, warden, signed, serial = _published(register)
    _strike(store, warden, signed)
    _interleaved(store, monkeypatch, lambda: store.withdraw(serial, key.key_id, "the artist withdrew"))
    warden.log_append({"kind": "admit", "record_sha256": signed["record_sha256"], "sitting": "S2"})
    assert store.state_of(serial) == "withdrawn" and not app.state.index.offered("gen-1")


def test_a_readmit_landing_between_a_strike_and_its_offer_leaves_the_pieces_offered(register, monkeypatch):
    app, store, http = register
    key, client, space, warden, signed, serial = _published(register)
    _interleaved(store, monkeypatch, lambda: store.log_append(
        {"kind": "admit", "record_sha256": signed["record_sha256"], "sitting": "S2"}, warden.key.key_id))
    _strike(store, warden, signed)
    assert store.state_of(serial) == "registered" and app.state.index.offered("gen-1")


def test_withdrawing_a_struck_model_is_a_log_entry_no_admit_undoes(register, tmp_path, capsys):
    from khaos_attribution.translog import verify as V
    from tests.test_register_log import run_page_verifier
    app, store, http = register
    key, client, space, warden, signed, serial = _published(register)
    _strike(store, warden, signed)
    client.withdraw(serial, "the artist withdrew")
    last = store.log.entries(store.log.size() - 1, store.log.size())[0]["entry"]
    assert last["kind"] == "withdraw" and last["record_sha256"] == signed["record_sha256"] and "reason" not in last
    with pytest.raises(R.RegisterError) as e:
        warden.log_append({"kind": "admit", "record_sha256": signed["record_sha256"], "sitting": "S2"})
    assert e.value.status == 422
    leaf = store.log.admit_leaf_of(signed["record_sha256"])
    pem = tmp_path / "k.pub"
    pem.write_text(store.log.signer.public_pem())
    mirror = app.state.cfg["data"] / "mirror"
    assert V.main([leaf, "--mirror", str(mirror), "--key", str(pem), "--state", "none"]) == V.PROVED
    assert "WITHDRAWN by the account that published it" in capsys.readouterr().out
    page, = run_page_verifier(tmp_path, [(mirror, leaf, store.log.signer.public_pem())])
    assert "bad|Withdrawn by the account that published it." in page
    assert "Withdrawn by the account that published it" in http.get(f"/models/{serial}").text


def test_a_struck_model_withdrawn_after_the_backup_comes_back_withdrawn(register, tmp_path):
    from register_service.translog import recover
    app, store, http = register
    cfg = app.state.cfg
    key, client, space, warden, signed, serial = _published(register)
    _strike(store, warden, signed)
    store.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(cfg["data"] / "register.sqlite3", tmp_path / "backup.sqlite3")
    client.withdraw(serial, "the artist withdrew")
    for p in cfg["data"].glob("register.sqlite3*"):
        p.unlink()
    shutil.copy(tmp_path / "backup.sqlite3", cfg["data"] / "register.sqlite3")
    st = create_app(cfg).state.store
    assert st.record(serial)["state"] == "struck"
    ok, lines = recover(st, log_places(cfg)[3], st.log.signer.public_pem())
    assert ok, lines
    app3 = create_app(cfg)
    store3 = app3.state.store
    assert store3.record(serial)["state"] == "withdrawn"
    warden._post = _Transport(TestClient(app3), warden.key)
    with pytest.raises(R.RegisterError) as e:
        warden.log_append({"kind": "admit", "record_sha256": signed["record_sha256"], "sitting": "S2"})
    assert e.value.status == 422 and store3.ledger_ok()


def test_the_start_sets_every_offer_from_the_store_both_ways(register):
    app, store, http = register
    key, client, space, warden, signed, serial = _published(register)
    app.state.index.withdraw_model(serial)
    assert not app.state.index.offered("gen-1")
    assert create_app(app.state.cfg).state.index.offered("gen-1"), "a re-admit cut short before its offer is mended at start"


def test_a_move_the_start_takes_from_the_log_has_a_ledger_row(register, tmp_path):
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
    assert recover(st, log_places(cfg)[3], st.log.signer.public_pem())[0]
    store3 = create_app(cfg).state.store
    rows = [r for r in store3.ledger() if r["kind"] == "state"]
    assert rows and rows[-1]["serial"] == serial and rows[-1]["detail"] == {"state": "struck", "restored_from_log": True}
    assert store3.ledger_ok()


def _lose_a_strike_with_a_piece(register, tmp_path):
    """A released piece, a strike after the backup, and a recovery from the public copy alone: the model under review."""
    from register_service.translog import recover
    app, store, http = register
    cfg = app.state.cfg
    key, client, space, warden, signed, serial = _published(register)
    store.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(cfg["data"] / "register.sqlite3", tmp_path / "backup.sqlite3")
    _strike(store, warden, signed)
    for p in cfg["data"].glob("register.sqlite3*"):
        p.unlink()
    shutil.rmtree(cfg["data"] / "mirror")
    shutil.copy(tmp_path / "backup.sqlite3", cfg["data"] / "register.sqlite3")
    st = create_app(cfg).state.store
    assert recover(st, log_places(cfg)[3], st.log.signer.public_pem())[0]
    return cfg, signed, serial, warden


def test_a_model_under_review_keeps_its_pieces_out_until_a_warden_answers(register, tmp_path):
    cfg, signed, serial, warden = _lose_a_strike_with_a_piece(register, tmp_path)
    app2 = create_app(cfg)
    http = TestClient(app2)
    assert serial in http.get("/api/withdrawn").json()["unresolved"]
    assert not app2.state.index.offered("gen-1"), "the start never restores a model under review"
    lost = app2.state.store.db.execute("SELECT leaf_hash FROM log_leaves WHERE kind = 'unavailable'").fetchone()["leaf_hash"]
    warden._post = _Transport(http, warden.key)
    warden.log_append({"kind": "confirm", "answers": lost, "sitting": "S10"})
    assert serial not in http.get("/api/withdrawn").json()["unresolved"]
    assert app2.state.index.offered("gen-1"), "the wardens' answer brings the pieces back"


def test_a_record_put_back_from_the_log_takes_its_offer_from_the_log(register, tmp_path):
    from register_service.translog import recover
    app, store, http = register
    cfg = app.state.cfg
    key, client = _machine(register, "artist-one")
    _, warden = _machine(register, "artist-two", role="warden")
    store.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(cfg["data"] / "register.sqlite3", tmp_path / "before-publish.sqlite3")
    signed = _signed(key)
    serial = client.publish(signed)["serial"]
    client.release("gen-1", serial, FP)
    app.state.index.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(cfg["data"] / "index.sqlite3", tmp_path / "index-before-strike.sqlite3")
    _strike(store, warden, signed)
    for name, backup in (("register", "before-publish"), ("index", "index-before-strike")):
        for p in cfg["data"].glob(f"{name}.sqlite3*"):
            p.unlink()
        shutil.copy(tmp_path / f"{backup}.sqlite3", cfg["data"] / f"{name}.sqlite3")
    st = create_app(cfg).state.store
    assert recover(st, log_places(cfg)[3], st.log.signer.public_pem())[0]
    app3 = create_app(cfg)
    assert app3.state.index.offered("gen-1"), "the index, older than the strike, knows nothing of it yet"
    client._post = _Transport(TestClient(app3), key)
    client.publish(signed)
    assert app3.state.store.state_of(serial) == "struck" and not app3.state.index.offered("gen-1")


def test_every_reader_names_an_accounts_withdrawal_as_one(register, tmp_path, capsys):
    from khaos_attribution.translog import verify as V
    from tests.test_register_log import run_page_verifier
    app, store, http = register
    key, client = _machine(register, "artist-one")
    signed = _signed(key)
    serial = client.publish(signed)["serial"]
    client.withdraw(serial, "asked to")
    leaf = store.log.admit_leaf_of(signed["record_sha256"])
    pem = tmp_path / "k.pub"
    pem.write_text(store.log.signer.public_pem())
    mirror = app.state.cfg["data"] / "mirror"
    assert V.main([leaf, "--mirror", str(mirror), "--key", str(pem), "--state", "none"]) == V.PROVED
    out = capsys.readouterr().out
    assert "WITHDRAWN by the account that published it: asked to" in out and "STRUCK" not in out
    page, = run_page_verifier(tmp_path, [(mirror, leaf, store.log.signer.public_pem())])
    assert "bad|Withdrawn by the account that published it: asked to" in page
    assert "Withdrawn by the account that published it: asked to" in http.get(f"/models/{serial}").text


def test_a_withdraw_entry_reaches_the_public_copy_without_the_delay(register):
    app, store, http = register
    key, client, space, warden, signed, serial = _published(register)
    _strike(store, warden, signed)
    client.withdraw(serial, "the artist withdrew")
    public = app.state.cfg["data"] / "mirror-public" / "log"
    entry = json.loads(LF.bodies(public)[store.log.size() - 1])
    assert entry["kind"] == "withdraw" and LF.pending_from(public) is not None
