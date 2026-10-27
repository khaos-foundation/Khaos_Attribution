"""The register's signed append-only log end to end: publish and withdraw append, wardens alone strike, withheld bodies
leave every proof standing, the static mirror alone verifies, the genesis migration, anchors and the public reads."""
import json
import shutil
import sqlite3
import sys

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
pytest.importorskip("rfc8785")

from khaos_attribution import registry as R  # noqa: E402
from khaos_attribution import translog as T  # noqa: E402
from khaos_attribution.translog import verify as V  # noqa: E402
from register_service.app import create_app  # noqa: E402
from tests.test_register_service import _machine, _signed, register  # noqa: E402,F401


def _mirror(app):
    return app.state.cfg["data"] / "mirror"


def _log_pem(app, tmp_path):
    path = tmp_path / "log-key.pub"
    path.write_text(app.state.store.log.signer.public_pem(), encoding="ascii")
    return path


def _publish(register, account="artist-one", **over):
    key, client = _machine(register, account)
    signed = _signed(key, **over)
    return key, client, signed, client.publish(signed)["serial"]


def test_a_publish_appends_an_admit_whose_proof_verifies_against_the_signed_head_offline(register):
    app, store, http = register
    _, _, signed, serial = _publish(register)
    key_doc = http.get("/log/key").json()
    head = http.get("/log/sth").json()
    T.verify_head(head, key_doc["public_key_pem"])
    assert key_doc["key_id"] != app.state.key.key_id, "the log signs with its own key, never the countersigning key"
    leaf = store.log.admit_leaf_of(signed["record_sha256"])
    proof = http.get(f"/log/proof/{leaf}").json()
    T.verify_inclusion(bytes.fromhex(leaf), proof["leaf_index"], head["tree_size"], T.unhex(proof["audit_path"]),
                       bytes.fromhex(head["root_sha256"]))
    entries = http.get("/log/entries", params={"start": 0, "end": 10}).json()["entries"]
    assert [e["entry"]["kind"] for e in entries] == ["genesis", "admit"]
    admit = entries[1]["entry"]
    assert admit["serial"] == serial and admit["record_sha256"] == signed["record_sha256"] and len(admit["salt"]) == 32
    assert T.entry_leaf_hex(admit) == leaf and admit["countersignature"]["register_key_id"] == app.state.key.key_id


def test_append_is_refused_without_a_wardens_signature_and_to_workshop_and_space_keys(register):
    app, store, http = register
    _, client, signed, _ = _publish(register)
    leaf = store.log.admit_leaf_of(signed["record_sha256"])
    strike = {"entry": {"kind": "revoke", "revokes": leaf, "reason": "consent withdrawn", "sitting": "First sitting"},
              "nonce": "nonce-strike-0001"}
    unsigned = http.post("/log/append", content=R.canonical_bytes(strike), headers={"Content-Type": "application/json"})
    assert unsigned.status_code == 401
    stranger = R.SigningKey.generate()
    data = R.canonical_bytes(strike)
    forged = http.post("/log/append", content=data, headers={"X-Machine-Key-Id": stranger.key_id,
                                                             "X-Machine-Signature": stranger.sign(data)})
    assert forged.status_code == 401
    _, space = _machine(register, "artist-one", role="space")
    for machine in (client, space):
        with pytest.raises(R.RegisterError) as e:
            machine._post("/log/append", strike)
        assert e.value.status == 403
    _, warden = _machine(register, "artist-two", role="warden")
    with pytest.raises(R.RegisterError) as e:
        warden._post("/log/append", {**strike, "entry": {**strike["entry"], "sitting": ""}, "nonce": "nonce-strike-0002"})
    assert e.value.status == 422, "a warden's entry names its sitting"
    out = warden._post("/log/append", strike)
    assert out["tree_size"] == store.log.size() and store.log.struck()[leaf]["reason"] == "consent withdrawn"


def test_a_revocation_appends_and_never_changes_the_original_leaf(register):
    app, store, http = register
    _, client, signed, serial = _publish(register)
    leaf = store.log.admit_leaf_of(signed["record_sha256"])
    before = (store.log.leaf(leaf), store.log.body(leaf), store.log.head())
    client.withdraw(serial, "the artist asked")
    assert (store.log.leaf(leaf), store.log.body(leaf)) == before[:2], "the admitted leaf is untouched"
    after = store.log.head()
    assert after["tree_size"] == before[2]["tree_size"] + 1
    T.verify_consistency(before[2]["tree_size"], after["tree_size"], bytes.fromhex(before[2]["root_sha256"]),
                         bytes.fromhex(after["root_sha256"]), T.unhex(http.get("/log/consistency", params={
                             "from": before[2]["tree_size"], "to": after["tree_size"]}).json()["proof"]))
    revoke = store.log.entries(after["tree_size"] - 1, after["tree_size"])[0]["entry"]
    assert revoke["kind"] == "revoke" and revoke["revokes"] == leaf and revoke["reason"] == "the artist asked"
    page = http.get(f"/models/{serial}").text
    assert '<span class="struck">Admitted</span>' in page and "the artist asked" in page
    assert store.log.problems(store.log.signer.public_pem()) == []


def test_a_withheld_body_leaves_every_proof_and_head_valid_and_is_itself_an_entry(register, tmp_path):
    app, store, http = register
    _, _, signed, _ = _publish(register)
    leaf = store.log.admit_leaf_of(signed["record_sha256"])
    heads_before = store.log.heads()
    proof_before = store.log.inclusion(leaf)
    out = store.log_withhold(leaf, "a person's name in the record")
    assert store.log.heads()[: len(heads_before)] == heads_before, "no head changes"
    T.verify_inclusion(bytes.fromhex(leaf), proof_before["leaf_index"], proof_before["tree_size"],
                       T.unhex(proof_before["audit_path"]), bytes.fromhex(heads_before[-1]["root_sha256"]))
    head = store.log.head()
    proof = store.log.inclusion(leaf)
    T.verify_inclusion(bytes.fromhex(leaf), proof["leaf_index"], head["tree_size"], T.unhex(proof["audit_path"]),
                       bytes.fromhex(head["root_sha256"]))
    shown = store.log.entries(0, 10)
    assert [e["withheld"] for e in shown] == [False, True, False]
    assert shown[1]["entry"] is None and shown[2]["leaf_hash"] == out["leaf_hash"]
    withhold = shown[2]["entry"]
    assert withhold["kind"] == "withhold" and withhold["withholds"] == leaf and "serial" not in withhold
    from register_service.app import log_key_doc
    from register_service.translog import export
    export(store.log, _mirror(app), log_key_doc(store.log.signer, True))
    assert not (_mirror(app) / "log" / "entries" / f"{leaf}.json").exists()
    rc = V.main([leaf, "--mirror", str(_mirror(app)), "--key", str(_log_pem(app, tmp_path)), "--state", "none"])
    assert rc == V.PROVED
    with pytest.raises(Exception):
        store.log_withhold(leaf, "again")


def test_verification_succeeds_against_the_static_mirror_alone_and_fails_when_it_is_altered(register, tmp_path, capsys):
    """The export copied to a folder, the register gone: the command proves inclusion and consistency from files."""
    app, store, http = register
    _, client, signed, serial = _publish(register)
    copy = tmp_path / "copy"
    shutil.copytree(_mirror(app), copy)
    pem = _log_pem(app, tmp_path)
    state = tmp_path / "state.json"
    assert V.main([signed["record_sha256"], "--mirror", str(copy), "--key", str(pem), "--state", str(state)]) == V.PROVED
    assert "PROVED" in capsys.readouterr().out
    record_file = tmp_path / "signed.json"
    record_file.write_text(json.dumps(signed), encoding="utf-8")
    assert V.main([str(record_file), "--mirror", str(copy), "--key", str(pem), "--state", str(state)]) == V.PROVED
    client.withdraw(serial, "x")
    capsys.readouterr()
    shutil.rmtree(copy)
    shutil.copytree(_mirror(app), copy)
    assert V.main([str(record_file), "--mirror", str(copy), "--key", str(pem), "--state", str(state), "--json"]) == V.PROVED
    out = json.loads(capsys.readouterr().out)
    assert out["struck"]["reason"] == "x" and out["consistent_since"] == 2
    leaf = store.log.admit_leaf_of(signed["record_sha256"])
    body = copy / "log" / "entries" / f"{leaf}.json"
    body.write_bytes(body.read_bytes().replace(b'"serial":', b'"serial":1'))
    assert V.main([leaf, "--mirror", str(copy), "--key", str(pem), "--state", "none"]) == V.FAILED
    other = R.SigningKey.generate().public_pem()
    (tmp_path / "other.pub").write_text(other, encoding="ascii")
    assert V.main([leaf, "--mirror", str(copy), "--key", str(tmp_path / "other.pub"), "--state", "none"]) == V.FAILED
    assert V.main(["0" * 64, "--mirror", str(copy), "--key", str(pem), "--state", "none"]) == V.NOT_FOUND
    assert V.main([leaf, "--mirror", str(tmp_path / "nowhere"), "--key", str(pem), "--state", "none"]) == V.UNREADABLE


def test_a_mirror_that_rewrites_history_fails_the_consistency_check_against_the_head_seen_before(register, tmp_path):
    app, store, http = register
    _, _, signed, _ = _publish(register)
    pem = _log_pem(app, tmp_path)
    state = tmp_path / "state.json"
    assert V.main([signed["record_sha256"], "--mirror", str(_mirror(app)), "--key", str(pem), "--state", str(state)]) == V.PROVED
    seen = json.loads(state.read_text())[str(_mirror(app))]
    forged = dict(seen, root_sha256="0" * 64)
    state.write_text(json.dumps({str(_mirror(app)): forged}))
    assert V.main([signed["record_sha256"], "--mirror", str(_mirror(app)), "--key", str(pem), "--state", str(state)]) == V.FAILED


def test_the_genesis_migration_carries_the_old_ledger_tip_and_every_existing_record(register, tmp_path):
    """A database from before the log: one genesis entry, then each record admitted at its own time, a withdrawn one struck."""
    app, store, http = register
    _, client, signed, serial = _publish(register)
    _, client2, signed2, serial2 = _publish(register, account="artist-two", run_id="run_2")
    client2.withdraw(serial2, "withdrawn before the log")
    copy = tmp_path / "old" / "register.sqlite3"
    copy.parent.mkdir()
    src = sqlite3.connect(str(store.path))
    dst = sqlite3.connect(str(copy))
    src.backup(dst)
    src.close()
    for table in ("log_leaves", "log_bodies", "log_nodes", "log_heads", "log_anchors"):
        dst.execute(f"DROP TABLE {table}")
    dst.commit()
    tip = dst.execute("SELECT entry_sha256 FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()[0]
    length = dst.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]
    published = dict(dst.execute("SELECT serial, published_at FROM records").fetchall())
    dst.close()
    cfg = {**app.state.cfg, "data": copy.parent, "key_file": copy.parent / "register-key.pem"}
    migrated = create_app(cfg).state.store
    entries = [e["entry"] for e in migrated.log.entries(0, 10)]
    assert [e["kind"] for e in entries] == ["genesis", "admit", "admit", "revoke"]
    assert entries[0]["ledger_tip_sha256"] == tip and entries[0]["ledger_length"] == length
    assert [e["serial"] for e in entries[1:3]] == [serial, serial2] and entries[1]["time"] == published[serial]
    assert entries[3]["revokes"] == migrated.log.admit_leaf_of(signed2["record_sha256"])
    assert migrated.log.problems(migrated.log.signer.public_pem()) == []
    assert migrated.ledger_ok(), "the old ledger stays as it was"
    again = create_app(cfg).state.store
    assert again.log.size() == 4, "genesis runs once"


def test_the_log_key_is_its_own_and_a_development_one_is_refused_without_its_setting(tmp_path):
    from register_service.app import load_log_key
    cfg = {"data": tmp_path, "key_file": tmp_path / "register-key.pem", "dev_key": True, "log_dev_key": True,
           "log_key_file": tmp_path / "log-key.pem", "public_url": "x"}
    dev = load_log_key(cfg)
    assert (tmp_path / "log-key.dev").read_text().strip() == dev.key_id
    with pytest.raises(SystemExit, match="development key"):
        load_log_key({**cfg, "log_dev_key": False})
    with pytest.raises(SystemExit, match="ceremony"):
        load_log_key({**cfg, "log_dev_key": False, "log_key_file": tmp_path / "none.pem"})


def test_anchors_stamp_the_newest_root_once_and_upgrade_into_the_mirror(register):
    """A fake timestamper stands in for OpenTimestamps: no test touches the network."""
    from register_service import anchor

    class Fake(anchor.Timestamper):
        name = "fake"

        def __init__(self):
            self.ready = False

        def stamp(self, root):
            return b"pending:" + root

        def upgrade(self, receipt):
            return receipt.replace(b"pending:", b"complete:") if self.ready else None

    app, store, http = register
    _publish(register)
    fake = Fake()
    first = anchor.stamp_latest(store, fake)
    assert first["state"] == "pending" and anchor.stamp_latest(store, fake) is None, "one stamp per head"
    assert anchor.upgrade_pending(store, fake) == []
    fake.ready = True
    assert anchor.upgrade_pending(store, fake) == [first["tree_size"]]
    rows = http.get("/log/anchors").json()["anchors"]
    assert rows[0]["state"] == "complete" and rows[0]["root_sha256"] == store.log.head()["root_sha256"]
    from register_service.translog import export_anchors
    export_anchors(store.log, _mirror(app))
    assert (_mirror(app) / "log" / "anchors" / f"{first['tree_size']}.ots").read_bytes().startswith(b"complete:")


def test_the_public_reads_cache_with_etags_and_are_bounded_per_address(register):
    app, store, http = register
    _publish(register)
    r = http.get("/log/sth/1")
    assert r.status_code == 200 and "immutable" in r.headers["cache-control"]
    assert http.get("/log/sth/1", headers={"If-None-Match": r.headers["etag"]}).status_code == 304
    assert http.get("/log/sth/99").status_code == 404 and http.get("/log/proof/xyz").status_code == 422
    first = http.get("/log/entries", params={"start": 1, "end": 2}).json()["entries"][0]["leaf_hash"]
    at_two = http.get(f"/log/proof/{first}", params={"size": 2})
    assert at_two.status_code == 200 and at_two.json()["tree_size"] == 2 and "immutable" in at_two.headers["cache-control"]
    assert http.get(f"/log/proof/{first}", params={"size": 1}).status_code == 404, "a leaf is not in a head from before it"
    assert store.log.leaf(first)["idx"] == 1
    assert http.get("/log/consistency", params={"from": 2, "to": 1}).status_code == 404
    assert http.get("/mirror/log/sth/latest.json").json() == http.get("/log/sth").json()
    assert http.get("/mirror/log/../register.sqlite3").status_code == 404
    from register_service.app import LOG_READ_LIMIT
    codes = [http.get("/log/sth", headers={"X-Forwarded-For": "198.51.100.7"}).status_code for _ in range(LOG_READ_LIMIT + 2)]
    assert codes[-1] == 429


def test_the_verify_page_runs_the_browser_verifier_from_its_own_file(register):
    """The site's policy forbids inline script; the verifier is a file and the page names the key it checks against."""
    app, store, http = register
    page = http.get("/verify-entry").text
    assert '<script src="/log/verify.js" defer></script>' in page and "<script>" not in page
    assert "No log key is committed" in page or "Pinned log key" in page
    js = http.get("/log/verify.js")
    assert js.status_code == 200 and "Ed25519" in js.text and "verifyConsistency" in js.text


def test_a_new_log_key_signs_the_current_tree_afresh_and_proves_every_earlier_entry(register, tmp_path):
    """The ceremony's switch: the development key's heads stay as history; the Guild's key signs the whole tree at start."""
    app, store, http = register
    _, _, signed, _ = _publish(register)
    size = store.log.size()
    dev_head, before = store.log.head(), len(store.log.all_heads())
    (app.state.cfg["data"] / "log-key.pem").unlink()
    (app.state.cfg["data"] / "log-key.dev").unlink()
    guild = R.SigningKey.load_or_create(app.state.cfg["data"] / "log-key.pem")
    restarted = create_app({**app.state.cfg, "log_dev_key": False}).state.store
    head = restarted.log.head()
    assert head["key_id"] == guild.key_id and head["tree_size"] == size and head["root_sha256"] == dev_head["root_sha256"]
    assert len(restarted.log.all_heads()) == before + 1 and restarted.log.head(1)["key_id"] == dev_head["key_id"]
    assert restarted.log.problems(guild.public_pem()) == []
    (tmp_path / "guild.pub").write_text(guild.public_pem(), encoding="ascii")
    assert V.main([signed["record_sha256"], "--mirror", str(_mirror(app)), "--key", str(tmp_path / "guild.pub"),
                   "--state", "none"]) == V.PROVED


def test_the_commands_withhold_check_the_whole_log_and_record_the_log_key_never_a_development_one(register, monkeypatch, capsys):
    from register_service.__main__ import main
    app, store, http = register
    _, _, signed, _ = _publish(register)
    data = app.state.cfg["data"]
    monkeypatch.setenv("REGISTER_DATA_DIR", str(data))
    monkeypatch.setenv("REGISTER_DEV_KEY", "1")
    leaf = store.log.admit_leaf_of(signed["record_sha256"])
    assert main(["log-withhold", leaf, "a name that should not be public"]) == 0
    assert not (data / "mirror" / "log" / "entries" / f"{leaf}.json").exists()
    assert main(["check"]) == 0 and "log whole" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="REGISTER_LOG_DEV_KEY"):
        main(["key", "--log", "--present", "A, B"])
    store.db.execute("UPDATE log_bodies SET body = CAST('{}' AS BLOB) WHERE leaf_hash = ?", (leaf,))
    assert main(["check"]) == 1 and "does not hash to its leaf" in capsys.readouterr().out


def test_the_browser_verifier_proves_the_same_mirror_and_refuses_a_changed_body_or_another_key(register, tmp_path):
    """The page's JavaScript under Node, with Node's WebCrypto: the same answers as the command."""
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    app, store, http = register
    _, _, signed, _ = _publish(register)
    _, client2, signed2, serial2 = _publish(register, account="artist-two", run_id="run_2")
    client2.withdraw(serial2, "asked to")
    entry = store.log.entries(2, 3)[0]["entry"]
    info = tmp_path / "info.json"
    info.write_text(json.dumps({"records": [signed["record_sha256"]], "struck_record": signed2["record_sha256"], "entry": entry,
                                "leaf": store.log.admit_leaf_of(signed["record_sha256"])}))
    from pathlib import Path
    js = Path("src/register_service/static/log-verify.js").resolve()
    run = subprocess.run([node, "tests/js/log_verify_harness.mjs", str(js), str(_mirror(app)), str(info)], capture_output=True,
                         text=True, timeout=60, check=True)
    first, struck, pasted, changed, other = [json.loads(line) for line in run.stdout.strip().splitlines()]
    assert first[0].startswith("ok|Proved: the admit entry for No. 65536") and not any("cannot check Ed25519" in x for x in first)
    assert any(x == "bad|Struck: asked to" for x in struck)
    assert pasted[0].startswith("ok|Proved: the admit entry for No. 65537") and "pasted, hashed here" in pasted[1]
    assert changed == ["bad|Not proved: the entry's body does not hash to its leaf"]
    assert other[0].startswith("bad|Not proved: the head is signed by key")


def test_the_opentimestamps_backend_round_trips_a_receipt_with_its_calendars_stood_in_for(monkeypatch):
    """stamp then upgrade through the library's own types; the calendars are local stand-ins, never the network."""
    pytest.importorskip("opentimestamps")
    from opentimestamps import calendar
    from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
    from opentimestamps.core.op import OpAppend, OpSHA256
    from opentimestamps.core.timestamp import Timestamp
    from register_service import anchor

    confirmed = {}

    def submit(self, digest, timeout=None):
        stamp = Timestamp(digest)
        inner = stamp.ops.add(OpAppend(b"\x01" * 16)).ops.add(OpSHA256())
        inner.attestations.add(PendingAttestation("https://alice.btc.calendar.opentimestamps.org"))
        confirmed[inner.msg] = False
        return stamp

    def get_timestamp(self, commitment, timeout=None):
        if not confirmed.get(commitment):
            raise calendar.CommitmentNotFoundError("pending")
        done = Timestamp(commitment)
        done.attestations.add(BitcoinBlockHeaderAttestation(900000))
        return done

    monkeypatch.setattr(calendar.RemoteCalendar, "submit", submit)
    monkeypatch.setattr(calendar.RemoteCalendar, "get_timestamp", get_timestamp)
    ots = anchor.OpenTimestamps(calendars=("https://a.example", "https://b.example"))
    receipt = ots.stamp(bytes(range(32)))
    assert receipt.startswith(b"\x00OpenTimestamps") and ots.upgrade(receipt) is None
    for k in confirmed:
        confirmed[k] = True
    upgraded = ots.upgrade(receipt)
    assert upgraded is not None and upgraded != receipt
    attestations = [a for _, a in anchor._deserialise(upgraded).timestamp.all_attestations()]
    assert any(isinstance(a, BitcoinBlockHeaderAttestation) and a.height == 900000 for a in attestations)
