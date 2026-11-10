"""The register's contract: a derived bucket, a machine key, a record that is whole, a countersignature
that covers what it says, and an outbox that keeps order and never doubles."""
import json
import sys

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")

from khaos_attribution import registry as R  # noqa: E402
from khaos_attribution.watermark import (  # noqa: E402
    DERIVED_PAYLOADS, PAYLOAD_SPACE, RESERVED_PAYLOADS, decode_codeword, derive_payload, derive_watermark_id,
)


def _record(**over):
    rec = {
        "record_version": R.RECORD_VERSION, "artist_id": "artist_x", "artist_name": "Artist X", "artist_mark": "AX",
        "consent": {"statement_sha256": "a" * 64, "scope": {"generate": "platform_users", "commercial_outputs": False}},
        "model": {"run_id": "run_1", "adapter_sha256": "b" * 64, "card_sha256": "c" * 64, "card_schema_version": "0.25.1",
                  "watermark_payload": derive_payload("k1", "run_1"), "base_model": "ACE-Step 1.5", "base_model_commit": "ca1e85f",
                  "base_model_licence": "MIT", "workshop_commit": "abc1234", "engine_commit": "ca1e85f",
                  "trained_at_utc": "2026-10-10T10:00:00Z", "machine_key_id": "k1"},
        "provenance": {"dataset_hash": "khaos.dataset_hash/2:" + "d" * 64, "config_hash": "e" * 64, "seed": 42},
        "training_set": {"rights_sha256": "f" * 64, "tracks": [{"track_id": "t1", "title": "One", "duration_sec": 200.0, "writers": [{"name": "W", "share": 1.0}]}]},
        "measured": {"heldout": None, "scorecard": True, "abstention": None, "calibration": None},
        "public_view": "tracks", "published_at_utc": "2026-10-10T10:05:00Z",
    }
    rec.update(over)
    return rec


def test_the_derived_payload_is_a_bucket_outside_the_reserved_range_and_the_same_every_time():
    seen = {derive_payload(f"key{i}", f"run{j}") for i in range(40) for j in range(40)}
    assert all(p in DERIVED_PAYLOADS for p in seen) and not any(p in RESERVED_PAYLOADS for p in seen)
    assert len(seen) > 1000, "a hash spreads over the buckets"
    assert derive_payload("k", "r") == derive_payload("k", "r") and derive_payload("k", "r") != derive_payload("k", "s")
    codeword = derive_watermark_id("k", "r")
    assert decode_codeword(codeword)[0] == derive_payload("k", "r") and DERIVED_PAYLOADS.stop == PAYLOAD_SPACE


def test_a_record_holding_a_number_that_is_not_finite_is_refused_wherever_it_sits():
    """NaN and the infinities are not JSON: a record carrying one could not be read back by anything but Python."""
    key = R.SigningKey.generate()
    for value in (float("nan"), float("inf"), float("-inf")):
        bad = _record()
        bad["training_set"]["tracks"][0]["writers"][0]["share"] = value
        assert [p for p in R.record_problems(bad) if "not finite" in p] == ["training_set.tracks[0].writers[0].share is not finite"]
        deep = _record()
        deep["measured"]["heldout"] = {"loss": [0.1, value]}
        assert "measured.heldout.loss[1] is not finite" in R.record_problems(deep)
        with pytest.raises(ValueError, match="not finite"):
            R.sign_record(bad, key)
    assert R.record_problems(_record()) == []


def test_a_machine_key_is_made_once_and_its_id_is_the_public_keys(tmp_path):
    k1 = R.SigningKey.load_or_create(tmp_path / R.MACHINE_KEY_FILE)
    k2 = R.SigningKey.load_or_create(tmp_path / R.MACHINE_KEY_FILE)
    assert k1.key_id == k2.key_id and len(k1.key_id) == 16
    assert (tmp_path / R.MACHINE_KEY_PUBLIC_FILE).read_text(encoding="utf-8") == k1.public_pem()
    sig = k1.sign(b"hello")
    assert R.verify(k1.public_pem(), b"hello", sig) and not R.verify(k1.public_pem(), b"hullo", sig)
    assert not R.verify(R.SigningKey.generate().public_pem(), b"hello", sig)


def test_a_record_is_refused_in_words_until_it_is_whole():
    assert R.record_problems(_record()) == []
    assert any("public_view" in p for p in R.record_problems(_record(public_view="all")))
    bad = _record(); bad["model"]["watermark_payload"] = 5
    assert any("derived payload" in p for p in R.record_problems(bad))
    bad = _record(); bad["training_set"]["tracks"] = []
    assert any("tracks" in p for p in R.record_problems(bad))
    bad = _record(public_view="writers")
    assert any("writers_agreed" in p for p in R.record_problems(bad))
    bad["training_set"]["writers_agreed"] = True
    assert R.record_problems(bad) == []
    with pytest.raises(ValueError, match="public_view"):
        R.sign_record(_record(public_view="all"), R.SigningKey.generate())


def test_signing_and_countersigning_cover_exactly_what_they_say():
    machine, register = R.SigningKey.generate(), R.SigningKey.generate()
    signed = R.sign_record(_record(), machine)
    assert R.signed_problems(signed, machine.public_pem()) == []
    tampered = json.loads(json.dumps(signed)); tampered["record"]["artist_name"] = "Someone else"
    assert any("record_sha256" in p for p in R.signed_problems(tampered, machine.public_pem()))
    assert any("machine_key_id" in p for p in R.signed_problems(signed, register.public_pem()))
    counter = R.countersign(signed, R.FIRST_SERIAL, register, "2026-10-10T10:06:00Z")
    assert R.countersignature_valid(counter, signed["record_sha256"], register.public_pem())
    assert not R.countersignature_valid({**counter, "serial": counter["serial"] + 1}, signed["record_sha256"], register.public_pem())
    assert not R.countersignature_valid(counter, "0" * 64, register.public_pem())
    assert not R.countersignature_valid(counter, signed["record_sha256"], machine.public_pem())
    with pytest.raises(ValueError):
        R.countersign(signed, 2048, register, "2026-10-10T10:06:00Z")


def test_the_outbox_keeps_order_never_doubles_and_moves_what_was_taken(tmp_path):
    box = R.Outbox(tmp_path)
    a = box.append("publish", {"n": 1}); b = box.append("withdraw", {"serial": 70000, "reason": "x"})
    assert box.append("publish", {"n": 1}) == a and len(box.pending()) == 2
    kinds = [e["kind"] for _, e in box.pending()]
    assert kinds == ["publish", "withdraw"]
    path, entry = box.pending()[0]
    box.mark_sent(path, {"serial": 65536})
    assert [e["id"] for _, e in box.pending()] == [b]
    sent = json.loads(next((tmp_path / R.OUTBOX_DIR / "sent").glob("*.json")).read_text(encoding="utf-8"))
    assert sent["answer"] == {"serial": 65536} and sent["id"] == a


def test_the_client_signs_every_body_and_syncs_in_order(monkeypatch, tmp_path):
    key = R.SigningKey.generate()
    client = R.RegisterClient("https://register.example", key)
    calls = []

    def fake_post(path, body):
        calls.append((path, body))
        if path == "/api/records":
            return {"serial": R.FIRST_SERIAL + len(calls)}
        return {"ok": True}
    monkeypatch.setattr(client, "_post", fake_post)
    box = R.Outbox(tmp_path)
    box.append("publish", {"signed_version": R.SIGNED_VERSION, "record": {}, "record_sha256": "0" * 64, "machine_key_id": key.key_id, "machine_signature": ""})
    box.append("served", {"serial": 65537, "space_url": "https://space.example", "state": "serving"})
    done = client.sync(box)
    assert [c[0] for c in calls] == ["/api/records", "/api/served"] and len(done) == 2 and not box.pending()


def test_sync_refuses_a_countersignature_that_is_not_the_pinned_keys(monkeypatch, tmp_path):
    key, register, impostor = R.SigningKey.generate(), R.SigningKey.generate(), R.SigningKey.generate()
    client = R.RegisterClient("https://register.example", key)
    signed = R.sign_record(_record(), key)
    monkeypatch.setattr(client, "_post", lambda path, body: {"serial": R.FIRST_SERIAL, "countersignature": R.countersign(signed, R.FIRST_SERIAL, impostor, "2026-10-10T00:00:00Z")})
    box = R.Outbox(tmp_path); box.append("publish", signed)
    with pytest.raises(R.RegisterError, match="pinned"):
        client.sync(box, register_public_pem=register.public_pem())
    assert len(box.pending()) == 1, "an unverified answer leaves the entry pending"
    monkeypatch.setattr(client, "_post", lambda path, body: {"serial": R.FIRST_SERIAL, "countersignature": R.countersign(signed, R.FIRST_SERIAL, register, "2026-10-10T00:00:00Z")})
    seen = []
    client.sync(box, register_public_pem=register.public_pem(), on_answer=lambda e, a: seen.append(a["serial"]))
    assert seen == [R.FIRST_SERIAL] and not box.pending()


def test_the_real_client_speaks_http_to_a_live_register(tmp_path):
    """The client's own urllib call: headers, canonical bytes, error mapping, against the service on a socket."""
    pytest.importorskip("fastapi"); pytest.importorskip("uvicorn")
    import socket, threading, time, urllib.request
    import uvicorn
    sys.path.insert(0, "src")
    from register_service.app import create_app
    app = create_app({"data": tmp_path / "d", "key_file": tmp_path / "d" / "k.pem", "dev_key": True, "public_url": "http://x"})
    app.state.store.create_account("a")
    invite = app.state.store.create_invite("a")
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0)); port = s_.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    key = R.SigningKey.generate()
    client = R.RegisterClient(f"http://127.0.0.1:{port}", key)
    with pytest.raises(R.RegisterError) as e:
        client.withdraw(65536, "x")
    assert e.value.status == 401 and "not enrolled" in str(e.value)
    assert client.enrol(invite, "box")["account"] == "a"
    rec = _record(); rec["model"]["machine_key_id"] = key.key_id
    rec["model"]["watermark_payload"] = derive_payload(key.key_id, "run_1")
    assert client.publish(R.sign_record(rec, key))["serial"] == R.FIRST_SERIAL
    server.should_exit = True


def test_an_outbox_entry_found_in_both_pending_and_sent_is_finished_not_resent(tmp_path):
    """A crash between the copy into sent/ and the unlink would otherwise resend the entry forever."""
    from khaos_attribution.registry import Outbox
    box = Outbox(tmp_path)
    box.append("served", {"serial": 1, "nonce": "0123456789ab"})
    (path, entry), = box.pending()
    box._move(path, "sent", {"ok": True})
    path.write_text(json.dumps(entry), encoding="utf-8")    # the unlink that never happened
    assert box.pending() == [] and not path.exists()
    assert [e["kind"] for e in box.sent_entries()] == ["served"]
