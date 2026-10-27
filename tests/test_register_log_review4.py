"""After a recovery that lost an admit: its serial is reserved and returned only to the record that holds this
register's countersignature for it, one lost entry never takes the catalogue offline, and a warden answers a lost
leaf once for every model it could have touched."""
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


def _lose_an_admit(register, tmp_path):
    """Two records published; the second's admit is lost with the full export, only the delayed public copy left."""
    from register_service.translog import recover
    app, store, http = register
    cfg = app.state.cfg
    store.create_account("artist-three")
    k1, c1 = _machine(register, "artist-one")
    s1 = _signed(k1)
    c1.publish(s1)
    k3, c3 = _machine(register, "artist-three")
    _, warden = _machine(register, "artist-two", role="warden")
    store.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(cfg["data"] / "register.sqlite3", tmp_path / "backup.sqlite3")
    s2 = _signed(k1, run_id="run_2")
    first = c1.publish(s2)
    for p in cfg["data"].glob("register.sqlite3*"):
        p.unlink()
    shutil.rmtree(cfg["data"] / "mirror")
    shutil.copy(tmp_path / "backup.sqlite3", cfg["data"] / "register.sqlite3")
    st = create_app(cfg).state.store
    ok, lines = recover(st, log_places(cfg)[3], st.log.signer.public_pem())
    assert ok, lines
    app3 = create_app(cfg)
    http3 = TestClient(app3)
    for client, key in ((c1, k1), (c3, k3), (warden, warden.key)):
        client._post = _Transport(http3, key)
    return app3, http3, s1, s2, first, (k1, c1), (k3, c3), warden


# ---- 1. a lost admit's serial is reserved, and returned to the record its countersignature names ----

def test_a_lost_admits_serial_is_never_given_to_another_record(register, tmp_path):
    app3, http3, s1, s2, first, (k1, c1), (k3, c3), warden = _lose_an_admit(register, tmp_path)
    other = c3.publish(_signed(k3, run_id="run_9"))
    assert other["serial"] != first["serial"] and other["serial"] > first["serial"]


def test_the_lost_record_gets_its_serial_back_with_its_countersignature_and_waits_on_the_wardens(register, tmp_path):
    app3, http3, s1, s2, first, (k1, c1), (k3, c3), warden = _lose_an_admit(register, tmp_path)
    again = c1.publish(s2, countersignature=first["countersignature"])
    assert again["serial"] == first["serial"] and again["countersignature"] == first["countersignature"]
    store = app3.state.store
    assert store.log.standing_admits(s2["record_sha256"]) == [] and again["serial"] in store.unresolved_serials()
    admit = store.log.admitted(s2["record_sha256"])
    assert admit["restores"]["serial"] == first["serial"] and admit["restores"]["linked_by"] == "countersigned serial"
    assert store.reserved_serials() == [], "the reservation is returned"


def test_a_republish_without_or_with_a_forged_countersignature_gets_a_new_serial(register, tmp_path):
    app3, http3, s1, s2, first, (k1, c1), (k3, c3), warden = _lose_an_admit(register, tmp_path)
    forged = dict(first["countersignature"], register_signature="00" * 64)
    again = c1.publish(s2, countersignature=forged)
    assert again["serial"] != first["serial"]
    assert app3.state.store.reserved_serials() == [first["serial"]], "the reservation stays for the real holder"


# ---- 2. one lost entry never takes the catalogue offline ----

def test_unresolved_models_stay_out_of_withdrawn_and_serving_them_is_accepted_as_under_review(register, tmp_path):
    app3, http3, s1, s2, first, (k1, c1), (k3, c3), warden = _lose_an_admit(register, tmp_path)
    store = app3.state.store
    serial1 = store.serial_of_record(s1["record_sha256"])
    listed = http3.get("/api/withdrawn").json()
    assert serial1 in listed["unresolved"] and listed["withdrawn"] == []
    _, space = _machine((app3, store, http3), "artist-one", role="space")
    answer = http3.post("/api/served", content=R.canonical_bytes(body := {"serial": serial1, "space_url": "https://s.example",
                                                                             "state": "serving", "nonce": "nonce-under-0001"}),
                        headers={"X-Machine-Key-Id": space.key.key_id, "X-Machine-Signature": space.key.sign(R.canonical_bytes(body))})
    assert answer.status_code == 202 and answer.json()["under_review"] is True


def test_one_confirm_naming_the_lost_leaf_resolves_every_model_it_could_have_touched(register, tmp_path):
    app3, http3, s1, s2, first, (k1, c1), (k3, c3), warden = _lose_an_admit(register, tmp_path)
    store = app3.state.store
    lost = store.db.execute("SELECT leaf_hash FROM log_leaves WHERE kind = 'unavailable'").fetchone()["leaf_hash"]
    standing = store.log.admit_leaf_of(s1["record_sha256"])
    warden.log_append({"kind": "confirm", "answers": lost, "sitting": "S10"})
    assert store.log.standing_admits(s1["record_sha256"]) == [standing] and store.unresolved_serials() == []
    with pytest.raises(R.RegisterError) as e:
        warden.log_append({"kind": "confirm", "answers": lost, "confirms": standing, "sitting": "S11"})
    assert e.value.status == 422, "a confirm of an admit that is not unresolved is refused"
    with pytest.raises(R.RegisterError) as e:
        warden.log_append({"kind": "confirm", "answers": standing, "sitting": "S11"})
    assert e.value.status == 422, "a confirm answers a lost leaf"
