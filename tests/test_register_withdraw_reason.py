"""An account's reason for withdrawing a model reaches no public copy but the log's revoke entry, where a withhold
takes it out of view: not the ledger, not the mirror's ledger.jsonl, not the model's JSON."""
import json
import sys

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
pytest.importorskip("rfc8785")

from tests.test_register_service import _machine, _signed, register  # noqa: E402,F401

WORDS = "a reason only the account and the wardens should read"


def test_the_reason_is_only_in_the_log_and_a_withhold_takes_it_out(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    signed = _signed(key)
    serial = client.publish(signed)["serial"]
    client.withdraw(serial, WORDS)
    mirror = app.state.cfg["data"] / "mirror"
    assert WORDS not in json.dumps(store.ledger())
    assert WORDS not in (mirror / "ledger.jsonl").read_text()
    assert WORDS not in http.get(f"/api/models/{serial}").text
    assert WORDS not in (mirror / "models" / f"{serial}.json").read_text()
    revoke = [e for e in store.log.of_record(signed["record_sha256"]) if e["kind"] == "revoke"][0]
    assert revoke["entry"]["reason"] == WORDS
    store.log_withhold(revoke["leaf_hash"], "a private matter")
    assert store.log.entries(revoke["idx"], revoke["idx"] + 1)[0]["entry"] is None
    assert WORDS not in http.get(f"/models/{serial}").text
