"""A record coming back after a recovery never comes back standing on its own word: its admit waits on every lost
leaf still open; reservations live in the log so a later restore keeps them; an earlier register key still counts."""
import io
import contextlib
import shutil
import sys

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
pytest.importorskip("rfc8785")

from fastapi.testclient import TestClient  # noqa: E402

from khaos_attribution import registry as R  # noqa: E402
from khaos_attribution.translog import verify as V  # noqa: E402
from register_service.app import create_app, export_all, log_places  # noqa: E402
from tests.test_register_service import _machine, _signed, _Transport, register  # noqa: E402,F401


def _vrun(target, where, pem, tmp_path):
    key = tmp_path / "k.pub"
    key.write_text(pem)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = V.main([target, "--mirror", str(where), "--key", str(key), "--state", "none"])
    return rc, buf.getvalue()


def _restore_and_recover(cfg, backup):
    from register_service.translog import recover
    for p in cfg["data"].glob("register.sqlite3*"):
        p.unlink()
    shutil.rmtree(cfg["data"] / "mirror", ignore_errors=True)
    shutil.copy(backup, cfg["data"] / "register.sqlite3")
    st = create_app(cfg).state.store
    ok, lines = recover(st, log_places(cfg)[3], st.log.signer.public_pem())
    assert ok, lines
    return create_app(cfg)


def _struck_then_lost(register, tmp_path):
    """s2 published and struck by the Guild after the backup; the full export and database lost; recovered blind."""
    app, store, http = register
    cfg = app.state.cfg
    store.create_account("artist-three")
    k1, c1 = _machine(register, "artist-one")
    c1.publish(_signed(k1))
    k3, c3 = _machine(register, "artist-three")
    _, warden = _machine(register, "artist-two", role="warden")
    store.db.execute("PRAGMA wal_checkpoint(FULL)")
    backup = tmp_path / "backup.sqlite3"
    shutil.copy(cfg["data"] / "register.sqlite3", backup)
    s2 = _signed(k1, run_id="run_2")
    first = c1.publish(s2)
    warden.log_append({"kind": "revoke", "revokes": store.log.admit_leaf_of(s2["record_sha256"]), "reason": "Guild strike",
                       "sitting": "S1"})
    app3 = _restore_and_recover(cfg, backup)
    http3 = TestClient(app3)
    for client, key in ((c1, k1), (c3, k3), (warden, warden.key)):
        client._post = _Transport(http3, key)
    return app3, cfg, backup, s2, first, (k1, c1), (k3, c3), warden


# ---- a republished record starts unresolved against every open lost leaf ----

@pytest.mark.parametrize("with_countersignature", [True, False])
def test_a_record_republished_after_a_recovery_is_unresolved_until_a_warden_answers(register, tmp_path, with_countersignature):
    app3, cfg, backup, s2, first, (k1, c1), _, warden = _struck_then_lost(register, tmp_path)
    again = c1.publish(s2, first["countersignature"] if with_countersignature else None)
    store = app3.state.store
    assert store.log.standing_admits(s2["record_sha256"]) == [] and again["serial"] in store.unresolved_serials()
    admit = store.log.admitted(s2["record_sha256"])
    assert admit["pending_lost"], "the admit names the lost leaves it waits on"
    export_all(cfg, store.log, store.log.signer, True)
    rc, out = _vrun(s2["record_sha256"], cfg["data"] / "mirror", store.log.signer.public_pem(), tmp_path)
    assert rc == V.HELD_BACK and "lost in a recovery" in out
    from tests.test_register_log import run_page_verifier
    page, = run_page_verifier(tmp_path, [(cfg["data"] / "mirror", s2["record_sha256"], store.log.signer.public_pem())])
    assert any("lost in a recovery" in x for x in page), page
    for lost in admit["pending_lost"]:
        warden.log_append({"kind": "confirm", "answers": lost, "confirms": store.log.admit_leaf_of_latest(s2["record_sha256"]),
                           "sitting": "S2"})
    assert store.log.standing_admits(s2["record_sha256"]) != []
    export_all(cfg, store.log, store.log.signer, True)
    rc, out = _vrun(s2["record_sha256"], cfg["data"] / "mirror", store.log.signer.public_pem(), tmp_path)
    assert rc == V.PROVED, out


def test_a_confirm_made_before_the_republish_does_not_answer_for_it(register, tmp_path):
    app3, cfg, backup, s2, first, (k1, c1), _, warden = _struck_then_lost(register, tmp_path)
    store = app3.state.store
    lost = [r["leaf_hash"] for r in store.db.execute("SELECT leaf_hash FROM log_leaves WHERE kind = 'unavailable'")]
    for leaf in lost[:-1]:
        warden.log_append({"kind": "confirm", "answers": leaf, "sitting": "S2"})
    c1.publish(s2, first["countersignature"])
    admit = store.log.admitted(s2["record_sha256"])
    assert admit["pending_lost"] == lost[-1:], "only the loss still open is waited on"
    assert store.log.standing_admits(s2["record_sha256"]) == []


# ---- earlier register keys count, reservations survive a second restore ----

def test_a_countersignature_from_before_a_register_key_switch_wins_back_its_serial(register, tmp_path):
    app3, cfg, backup, s2, first, (k1, c1), _, warden = _struck_then_lost(register, tmp_path)
    (cfg["data"] / "register-key.pem").unlink()
    (cfg["data"] / "register-key.dev").unlink()
    R.SigningKey.load_or_create(cfg["data"] / "register-key.pem")
    app4 = create_app({**cfg, "dev_key": False, "log_dev_key": True})
    c1._post = _Transport(TestClient(app4), k1)
    again = c1.publish(s2, first["countersignature"])
    assert again["serial"] == first["serial"]


def test_reservations_live_in_the_log_and_survive_a_second_restore_from_an_older_backup(register, tmp_path):
    app3, cfg, backup, s2, first, (k1, c1), (k3, c3), warden = _struck_then_lost(register, tmp_path)
    reserved = app3.state.store.reserved_serials()
    assert first["serial"] in reserved
    c3.publish(_signed(k3, run_id="run_7"))
    app5 = _restore_and_recover(cfg, backup)
    again = app5.state.store.reserved_serials()
    assert set(reserved) <= set(again), "rebuilt from the recovery's own entries"
    c3._post = _Transport(TestClient(app5), k3)
    assert c3.publish(_signed(k3, run_id="run_8"))["serial"] not in again
    c1._post = _Transport(TestClient(app5), k1)
    assert c1.publish(s2, first["countersignature"])["serial"] == first["serial"]


def test_a_warden_admit_cannot_make_an_unresolved_record_standing(register, tmp_path):
    """A plain admit is refused while the record waits on a loss, naming the confirm to use; it never skips the answer."""
    app3, cfg, backup, s2, first, (k1, c1), _, warden = _struck_then_lost(register, tmp_path)
    store = app3.state.store
    s1_sha = store.db.execute("SELECT record_sha256 FROM records ORDER BY serial LIMIT 1").fetchone()["record_sha256"]
    assert store.log.admit_states(s1_sha) and "unresolved" in store.log.admit_states(s1_sha).values()
    with pytest.raises(R.RegisterError) as e:
        warden.log_append({"kind": "admit", "record_sha256": s1_sha, "sitting": "S3"})
    assert e.value.status == 422 and '"kind":"confirm","answers"' in str(e.value).replace("\\", "")
    assert store.log.standing_admits(s1_sha) == []


def test_every_warden_admit_waits_on_the_losses_still_open(register, tmp_path):
    app3, cfg, backup, s2, first, (k1, c1), _, warden = _struck_then_lost(register, tmp_path)
    store = app3.state.store
    s1_sha = store.db.execute("SELECT record_sha256 FROM records ORDER BY serial LIMIT 1").fetchone()["record_sha256"]
    admit = store.log.admit_leaf_of(s1_sha)
    warden.log_append({"kind": "revoke", "revokes": admit, "reason": "re-examined", "sitting": "S3"})
    out = warden.log_append({"kind": "admit", "record_sha256": s1_sha, "sitting": "S4"})
    entry = store.log.entries(out["index"], out["index"] + 1)[0]["entry"]
    assert entry["pending_lost"] == store.log.open_lost_global() != []
    assert store.log.standing_admits(s1_sha) == []
