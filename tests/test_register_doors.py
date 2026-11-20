"""Each signed door takes only the roles it is for; a Space's address is an https origin; a caller is identified
before its body is read; the decoder and the rate limits answer 'later' instead of queueing; the import check
takes what a claim takes; /health says the version; the operator's withhold is in the ledger."""
import ipaddress
import json
import sys
import threading

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")

from khaos_attribution import registry as R  # noqa: E402
from tests.test_register_service import _machine, _signed, _Transport, register  # noqa: E402,F401

FP = [[h, h * 10] for h in range(1, 40)]


def _published(register):
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    return key, client, serial


@pytest.mark.parametrize("role", ["space", "warden"])
def test_only_a_workshop_publishes_or_withdraws(register, role):
    app, store, http = register
    key, client, serial = _published(register)
    other_key, other = _machine(register, "artist-one", role=role)
    signed = R.sign_record(_signed(other_key)["record"], other_key)
    with pytest.raises(R.RegisterError) as e:
        other.publish(signed)
    assert e.value.status == 403
    with pytest.raises(R.RegisterError) as e:
        other.withdraw(serial, "not a workshop's to withdraw")
    assert e.value.status == 403 and store.record(serial)["state"] == "registered"


def test_a_wardens_machine_releases_no_piece(register):
    key, client, serial = _published(register)
    _, warden = _machine(register, "artist-one", role="warden")
    with pytest.raises(R.RegisterError) as e:
        warden.release("gen-w", serial, FP)
    assert e.value.status == 403


@pytest.mark.parametrize("url,kept", [
    ("https://space.example", "https://space.example"),
    ("https://space.example/", "https://space.example"),
    ("https://Space.Example:8443/artists/x?y=1", "https://space.example:8443"),
    ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
    ("http://localhost:8000/", "http://localhost:8000"),
])
def test_a_spaces_address_is_kept_as_its_origin(register, url, kept):
    app, store, http = register
    key, client, serial = _published(register)
    _, space = _machine(register, "artist-two", role="space")
    assert space.served(serial, url, "serving")["space_url"] == kept
    assert http.get(f"/api/models/{serial}").json()["served"][0]["space_url"] == kept


@pytest.mark.parametrize("url", ["javascript:alert(1)", "http://space.example", "ftp://space.example", "https://", "", "space.example"])
def test_an_address_that_is_not_an_https_origin_is_refused(register, url):
    key, client, serial = _published(register)
    _, space = _machine(register, "artist-two", role="space")
    with pytest.raises(R.RegisterError) as e:
        space.served(serial, url, "serving")
    assert e.value.status == 422 and "public_url" in str(e.value)


def test_a_spaces_older_address_row_is_its_origin_and_a_later_report_moves_it(register, tmp_path):
    from register_service.store import Store
    app, store, http = register
    key, client, serial = _published(register)
    space_key, space = _machine(register, "artist-two", role="space")
    for url, at in (("https://Space.Example/artists/x", "2026-01-01T00:00:00Z"), ("https://space.example/", "2026-01-02T00:00:00Z"),
                    ("http://space.example", "2026-01-03T00:00:00Z")):
        store.db.execute("INSERT INTO served (serial, space_url, state, at, key_id) VALUES (?, ?, 'serving', ?, ?)",
                         (serial, url, at, space_key.key_id))
    reopened = Store(store.path)
    rows = {r["space_url"]: r["at"] for r in reopened.record(serial)["served"]}
    assert rows == {"https://space.example": "2026-01-02T00:00:00Z", "http://space.example": "2026-01-03T00:00:00Z"}
    space.served(serial, "https://space.example/", "removed")
    states = {r["space_url"]: r["state"] for r in store.record(serial)["served"]}
    assert states["https://space.example"] == "removed"


def test_an_unknown_caller_is_refused_before_its_body_is_read(register, monkeypatch):
    app, store, http = register
    import register_service.app as A
    parsed = []
    real = A.json.loads
    monkeypatch.setattr(A.json, "loads", lambda raw, *a, **k: parsed.append(len(raw)) or real(raw, *a, **k))
    key = R.SigningKey.generate()
    data = b"{" + b" " * 1000 + b"}"
    r = http.post("/api/records", content=data, headers={"X-Machine-Key-Id": key.key_id, "X-Machine-Signature": key.sign(data)})
    assert r.status_code == 401 and parsed == []


def test_a_busy_decoder_answers_later_instead_of_queueing(register):
    from register_service import softbinding as SB
    app, store, http = register
    answers = []
    # A decoder that queues would wait on the permits this test holds, so the request runs where it can time out.
    ask = threading.Thread(target=lambda: answers.append(http.post(
        f"{SB.PREFIX}/matches/byContent", content=b"RIFF" + b"\0" * 64, headers={"Content-Type": "audio/wav"})), daemon=True)
    assert SB._decoding.acquire(blocking=False) and SB._decoding.acquire(blocking=False)
    try:
        ask.start()
        ask.join(timeout=10)
        assert answers, "the lookup queued behind a busy decoder instead of answering"
        assert answers[0].status_code == 429 and answers[0].headers["retry-after"]
    finally:
        SB._decoding.release()
        SB._decoding.release()
        ask.join(timeout=10)


def test_one_ipv6_host_has_one_bucket_however_many_addresses_it_takes(register):
    from register_service.app import PUBLIC_LIMITS
    app, store, http = register
    bad = {"phases": [[[1, 2]]]}
    net = ipaddress.IPv6Network("2001:db8:1:2::/64")
    codes = [http.post("/api/verify", json=bad, headers={"X-Forwarded-For": str(net[i + 1])}).status_code
             for i in range(PUBLIC_LIMITS["/api/verify"] + 3)]
    assert codes[-1] == 429


def test_an_ipv4_host_written_as_ipv6_shares_its_bucket():
    from register_service.app import client_of
    assert client_of("::ffff:203.0.113.5") == client_of("203.0.113.5") == "203.0.113.5"
    assert client_of("2001:db8:1:2::9") == client_of("2001:db8:1:2:ffff::1") == "2001:db8:1:2::/64"


def test_the_limiter_holds_a_bounded_table_and_never_scans_it_per_call():
    from register_service.app import RateLimit
    limit = RateLimit(10, max_clients=100)
    for i in range(1000):
        assert limit.allow(f"10.0.{i // 256}.{i % 256}", now=float(i))
    assert len(limit._buckets) <= 100


def test_the_import_check_takes_a_body_as_large_as_a_claim(register):
    from register_service.app import CLAIM_MAX_BODY, MAX_BODY
    app, store, http = register
    key, client = _machine(register, "artist-one")
    phases = [[[h % 33554432, h] for h in range(i * 160_000, (i + 1) * 160_000)] for i in range(4)]
    body = {"phases": phases, "nonce": "n" * 16}
    assert MAX_BODY < len(R.canonical_bytes(body)) < CLAIM_MAX_BODY
    assert client.check_claim(phases)["grade"] == "none"


def test_health_says_which_version_of_the_register_answers(register):
    import khaos_attribution
    app, store, http = register
    assert http.get("/health").json()["version"] == khaos_attribution.__version__


def test_the_operators_withhold_has_a_ledger_row(register):
    app, store, http = register
    key, client, serial = _published(register)
    leaf = store.log.admit_leaf_of(store.record(serial)["record_sha256"])
    out = store.log_withhold(leaf, "a private matter")
    row = store.ledger()[-1]
    assert row["kind"] == "log-withhold" and row["detail"] == {"withholds": leaf, "leaf_hash": out["leaf_hash"], "index": out["index"]}
    assert "private" not in json.dumps(row) and store.ledger_ok()
