"""The register end to end: enrolment by invite, a numbered countersigned record, no doubles, withdrawal by
the owning account only, where it is served, the three public views, and an unbroken ledger."""
import json
import sys

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from khaos_attribution import registry as R  # noqa: E402
from khaos_attribution.watermark import derive_payload  # noqa: E402
from register_service.app import create_app, public_view  # noqa: E402
from register_service.store import Store  # noqa: E402
from tests.test_registry import _record  # noqa: E402


class _Transport:
    """The client's one HTTP call, routed into the test app."""

    def __init__(self, client: TestClient, key: R.SigningKey):
        self.client, self.key = client, key

    def __call__(self, path, body):
        data = R.canonical_bytes(body)
        r = self.client.post(path, content=data, headers={"Content-Type": "application/json", "X-Machine-Key-Id": self.key.key_id,
                                                          "X-Machine-Signature": self.key.sign(data)})
        if r.status_code >= 400:
            raise R.RegisterError(r.status_code, r.text)
        return r.json()


@pytest.fixture
def register(tmp_path):
    cfg = {"data": tmp_path / "data", "key_file": tmp_path / "data" / "register-key.pem", "dev_key": True, "public_url": "https://register.example"}
    app = create_app(cfg)
    store: Store = app.state.store
    store.create_account("artist-one")
    store.create_account("artist-two")
    return app, store, TestClient(app)


def _machine(register, account, role="workshop"):
    app, store, http = register
    key = R.SigningKey.generate()
    client = R.RegisterClient("https://register.example", key)
    client._post = _Transport(http, key)
    client.enrol(store.create_invite(account, role), "a machine")
    return key, client


def _signed(key, **over):
    rec = _record(**over)
    rec["model"]["machine_key_id"] = key.key_id
    rec["model"]["watermark_payload"] = derive_payload(key.key_id, rec["model"]["run_id"])
    return R.sign_record(rec, key)


def test_the_register_needs_a_dev_or_real_key_and_never_makes_one_in_production(tmp_path):
    with pytest.raises(SystemExit, match="ceremony"):
        create_app({"data": tmp_path, "key_file": tmp_path / "none.pem", "dev_key": False, "public_url": "x"})


def test_a_key_once_served_as_the_development_key_is_refused_without_the_setting(tmp_path):
    """Deleting REGISTER_DEV_KEY by hand never promotes the development key; a ceremony key in its place serves."""
    from register_service.app import load_register_key
    cfg = {"data": tmp_path, "key_file": tmp_path / "register-key.pem", "dev_key": True, "public_url": "x"}
    dev = load_register_key(cfg)
    assert (tmp_path / "register-key.dev").read_text().strip() == dev.key_id
    with pytest.raises(SystemExit, match="development key"):
        load_register_key({**cfg, "dev_key": False})
    (tmp_path / "register-key.pem").unlink()
    ceremony = R.SigningKey.load_or_create(tmp_path / "register-key.pem")
    assert load_register_key({**cfg, "dev_key": False}).key_id == ceremony.key_id != dev.key_id


def test_enrolment_takes_an_invite_once_and_an_unenrolled_machine_cannot_publish(register):
    app, store, http = register
    key = R.SigningKey.generate()
    client = R.RegisterClient("x", key); client._post = _Transport(http, key)
    with pytest.raises(R.RegisterError, match="not enrolled"):
        client.publish(_signed(key))
    code = store.create_invite("artist-one")
    assert client.enrol(code, "laptop")["account"] == "artist-one"
    other = R.SigningKey.generate(); c2 = R.RegisterClient("x", other); c2._post = _Transport(http, other)
    with pytest.raises(R.RegisterError, match="already used"):
        c2.enrol(code, "another")


def test_a_record_gets_a_serial_from_65536_a_valid_countersignature_and_the_same_serial_twice(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    signed = _signed(key)
    first = client.publish(signed)
    assert first["serial"] == R.FIRST_SERIAL and first["already"] is False
    pem = http.get("/api/public-key").json()["public_key_pem"]
    assert R.countersignature_valid(first["countersignature"], signed["record_sha256"], pem)
    again = client.publish(signed)
    assert again["serial"] == R.FIRST_SERIAL and again["already"] is True
    second = client.publish(_signed(key, published_at_utc="2026-10-11T00:00:00Z"))
    assert second["serial"] == R.FIRST_SERIAL + 1
    page = http.get(f"/models/{R.FIRST_SERIAL}")
    assert page.status_code == 200 and "Register No. 65536" in page.text and "One" in page.text
    doc = http.get(f"/api/models/{R.FIRST_SERIAL}").json()
    assert doc["serial"] == R.FIRST_SERIAL and doc["state"] == "registered" and doc["record"]["training_set"]["tracks"][0].get("writers") is None


def test_a_record_naming_another_machine_or_signed_by_another_key_is_refused(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    other, _ = _machine(register, "artist-two")
    signed = _signed(other)
    with pytest.raises(R.RegisterError, match="signature|key"):
        client.publish(signed)
    rec = _record(); rec["model"]["machine_key_id"] = other.key_id
    with pytest.raises(R.RegisterError, match="another machine"):
        client.publish(R.sign_record(rec, key))


def test_withdrawal_is_the_owning_accounts_alone_and_only_a_space_reports_serving(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    _, stranger = _machine(register, "artist-two")
    _, space = _machine(register, "artist-two", role="space")
    serial = client.publish(_signed(key))["serial"]
    with pytest.raises(R.RegisterError, match="account"):
        stranger.withdraw(serial, "not mine")
    with pytest.raises(R.RegisterError, match="only an enrolled Space"):
        stranger.served(serial, "https://evil.example", "serving")
    with pytest.raises(R.RegisterError, match="only an enrolled Space"):
        client.served(serial, "https://evil.example", "serving")
    assert space.served(serial, "https://space.example", "serving")["state"] == "serving"
    out = client.withdraw(serial, "the artist withdrew consent")
    assert out["state"] == "withdrawn" and http.get("/api/withdrawn").json()["withdrawn"] == [serial]
    doc = http.get(f"/api/models/{serial}").json()
    assert doc["state"] == "withdrawn" and doc["served"][0]["space_url"] == "https://space.example"
    assert "withdrawn" in http.get(f"/models/{serial}").text
    assert doc["machine_public_pem"].startswith("-----BEGIN PUBLIC KEY-----"), "a reader of the mirror can check the machine's signature"


def test_a_captured_body_cannot_be_replayed(register):
    """A served or withdraw body carries a nonce the register takes once: the same signed bytes again have no second effect."""
    app, store, http = register
    key, client = _machine(register, "artist-one")
    _, space = _machine(register, "artist-one", role="space")
    serial = client.publish(_signed(key))["serial"]
    captured = {}

    real = space._post

    def capture(path, body):
        captured[path] = body
        return real(path, body)
    space._post = capture
    assert space.served(serial, "https://space.example", "serving")["state"] == "serving"
    assert space.served(serial, "https://space.example", "removed")["state"] == "removed"
    replayed = real("/api/served", captured["/api/served"])
    assert replayed["state"] == "removed", "the captured bytes get their own first answer back and change nothing"
    assert http.get(f"/api/models/{serial}").json()["served"][0]["state"] == "removed"


def test_a_record_that_passes_the_check_never_breaks_its_page(register):
    """Every shape the page and the views read is checked before a serial is spent."""
    app, store, http = register
    key, client = _machine(register, "artist-one")
    for breaker in (lambda r: r["training_set"].__setitem__("tracks", ["t1"]),
                    lambda r: r.__setitem__("provenance", {}),
                    lambda r: r["training_set"]["tracks"][0]["writers"].__setitem__(0, {"name": "W"}),
                    lambda r: r.__setitem__("model", "x")):
        rec = _record(); rec["model"]["machine_key_id"] = key.key_id
        rec["model"]["watermark_payload"] = derive_payload(key.key_id, "run_1")
        breaker(rec)
        with pytest.raises(ValueError):
            R.sign_record(rec, key)
    # malformed but signed bodies are refusals in words, never 500s
    for path, body, status in (("/api/withdraw", {"serial": "abc", "reason": "x", "nonce": "0123456789ab"}, 422),
                               ("/api/served", {"serial": 65536, "space_url": "x", "state": "sideways", "nonce": "0123456789ac"}, 422)):
        data = R.canonical_bytes(body)
        r = http.post(path, content=data, headers={"Content-Type": "application/json", "X-Machine-Key-Id": key.key_id, "X-Machine-Signature": key.sign(data)})
        assert r.status_code == status, (path, r.status_code, r.text)
    data = b"[1]"
    r = http.post("/api/records", content=data, headers={"Content-Type": "application/json", "X-Machine-Key-Id": key.key_id, "X-Machine-Signature": key.sign(data)})
    assert r.status_code == 400


def test_the_three_public_views_show_what_the_artist_chose():
    rec = _record(public_view="model")
    v = public_view(rec)["training_set"]
    assert v["tracks"] == [] and v["track_count"] == 1
    v = public_view(_record(public_view="tracks"))["training_set"]
    assert v["tracks"][0]["title"] == "One" and "writers" not in v["tracks"][0]
    full = _record(public_view="writers"); full["training_set"]["writers_agreed"] = True
    assert public_view(full)["training_set"]["tracks"][0]["writers"][0]["name"] == "W"


def test_the_ledger_chains_every_entry_and_a_changed_entry_breaks_it(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    client.withdraw(serial, "x")
    kinds = [e["kind"] for e in store.ledger()]
    assert kinds == ["account", "account", "enrol", "publish", "withdraw"] and store.ledger_ok()
    assert "machine" not in store.ledger()[2]["detail"], "the public ledger names the key and the account, never the machine's name"
    assert http.get("/api/public-key").json()["development"] is True and "development key" in http.get("/").text
    assert http.get("/health").json()["ledger_ok"] is True
    lines = http.get("/api/ledger").text.strip().splitlines()
    assert len(lines) == 5 and json.loads(lines[-1])["kind"] == "withdraw"
    store.db.execute("UPDATE ledger SET detail_json = '{\"reason\": \"y\"}' WHERE kind = 'withdraw'")
    assert not store.ledger_ok()
    assert (app.state.cfg["data"] / "mirror" / "models" / f"{serial}.json").is_file()
    assert (app.state.cfg["data"] / "mirror" / "ledger.jsonl").read_text(encoding="utf-8").count("\n") == 5


def test_the_same_signed_body_again_gets_its_first_answer_and_a_refused_one_does_not_burn_its_nonce(register):
    """The nonce is an idempotency key: a lost answer is recovered by sending the same bytes; a 404 can be retried."""
    app, store, http = register
    key, client = _machine(register, "artist-one")
    _, space = _machine(register, "artist-one", role="space")
    serial = client.publish(_signed(key))["serial"]
    body = {"serial": serial, "space_url": "https://space.example", "state": "serving", "nonce": "nonce-0001-abcd"}
    first = space._post("/api/served", body)
    again = space._post("/api/served", body)
    assert again == first, "the same body is answered the same, with no second effect"
    assert len(http.get(f"/api/models/{serial}").json()["served"]) == 1
    with pytest.raises(R.RegisterError, match="replay"):
        space._post("/api/served", {**body, "state": "removed"})
    missing = {"serial": 70000, "reason": "x", "nonce": "nonce-0002-abcd"}
    for _ in range(2):
        with pytest.raises(R.RegisterError) as e:
            client._post("/api/withdraw", missing)
        assert e.value.status == 404, "a refusal is the same refusal twice, never a replay"


def test_sync_sets_a_refused_entry_aside_and_goes_on(register, tmp_path):
    """A withdrawal the register refuses for what it is does not hold the publishes behind it."""
    app, store, http = register
    key, client = _machine(register, "artist-one")
    box = R.Outbox(tmp_path)
    box.append("withdraw", {"serial": 70000, "reason": "never issued", "nonce": "0123456789ab"})
    box.append("publish", _signed(key))
    done = client.sync(box)
    assert [a["serial"] for _, a in done] == [R.FIRST_SERIAL] and not box.pending()
    refused = box.refused()
    assert len(refused) == 1 and refused[0]["answer"]["status"] == 404
    assert box.drop("nothing") is False
    eid = box.append("served", {"serial": R.FIRST_SERIAL, "space_url": "x", "state": "serving", "nonce": "0123456789ac"})
    assert box.drop(eid) is True and not box.pending()


def test_sync_waits_on_too_many_calls_and_never_sets_the_entry_aside(register, tmp_path):
    """A 429 is the register asking for time: the entry stays pending for the next sync, never refused."""
    key, client = _machine(register, "artist-one")
    box = R.Outbox(tmp_path)
    box.append("publish", _signed(key))

    def busy(path, body):
        raise R.RegisterError(429, "too many requests from this address; try again in a minute")
    client._post = busy
    with pytest.raises(R.RegisterError):
        client.sync(box)
    assert len(box.pending()) == 1 and not box.refused()


def test_the_ceremony_makes_a_key_once_and_records_it_in_the_ledger_never_a_development_one(tmp_path, monkeypatch, capsys):
    """make-key refuses to overwrite; key records the serving key's id, its public hash and who was present, and
    refuses while the service runs on a development key."""
    from register_service.__main__ import main
    from register_service.store import Store
    key_path = tmp_path / "ceremony" / "register-key.pem"
    assert main(["make-key", str(key_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("key id ") and "BEGIN PUBLIC KEY" in out and key_path.is_file() and key_path.with_suffix(".pub").is_file()
    with pytest.raises(SystemExit, match="made once"):
        main(["make-key", str(key_path)])
    monkeypatch.setenv("REGISTER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REGISTER_KEY_FILE", str(key_path))
    monkeypatch.setenv("REGISTER_DEV_KEY", "1")
    with pytest.raises(SystemExit, match="development key"):
        main(["key", "--present", "A, B"])
    monkeypatch.delenv("REGISTER_DEV_KEY")
    assert main(["key", "--present", "Ava Guardian, Ben Guardian", "--note", "first ceremony"]) == 0
    entries = Store(tmp_path / "data" / "register.sqlite3").ledger()
    assert entries[-1]["kind"] == "key" and entries[-1]["detail"]["present"] == ["Ava Guardian", "Ben Guardian"]
    assert len(entries[-1]["detail"]["public_key_sha256"]) == 64 and entries[-1]["key_id"] == out.split()[2]


def test_the_public_doors_take_a_bounded_number_of_calls_a_minute_per_address(register):
    """verify and enrol are rate-limited per client address; a signed call has its own, higher bound; the limit
    answers 429 with Retry-After and refills with time."""
    from register_service.app import PUBLIC_LIMITS, RateLimit
    app, store, http = register
    bad = {"phases": [[[1, 2]]]}
    codes = [http.post("/api/verify", json=bad).status_code for _ in range(PUBLIC_LIMITS["/api/verify"] + 3)]
    assert codes[: PUBLIC_LIMITS["/api/verify"]] == [200] * PUBLIC_LIMITS["/api/verify"] and codes[-1] == 429
    r = http.post("/api/verify", json=bad)
    assert r.status_code == 429 and r.headers["retry-after"] == "60"
    assert http.get("/api/withdrawn").status_code == 200, "reads are never limited"
    assert http.post("/api/verify", json=bad, headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 200, "another address, its own bucket"
    # The proxy appends the real address last; a forged first entry does not buy a fresh bucket.
    forged = [http.post("/api/verify", json=bad, headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.9"}).status_code for i in range(30)]
    assert 429 in forged
    bucket = RateLimit(6)
    assert all(bucket.allow("a", now=0.0) for _ in range(6)) and not bucket.allow("a", now=0.0)
    assert bucket.allow("a", now=10.0), "refills at the rate: one token every ten seconds at six a minute"


def test_a_verify_key_holder_is_never_rate_limited_and_only_its_hash_is_kept(register):
    """Authorities, researchers and media get an unlimited, free lane; a wrong or revoked key gets the public one."""
    from register_service import verify_keys
    from register_service.app import PUBLIC_LIMITS
    app, store, http = register
    cfg = {"data": store.path.parent}
    key = verify_keys.issue(cfg, "A newsroom", "media")
    assert key not in verify_keys._path(cfg).read_text()
    bad = {"phases": [[[1, 2]]]}
    n = PUBLIC_LIMITS["/api/verify"] + 5
    assert {http.post("/api/verify", json=bad, headers={"X-Khaos-Verify-Key": key}).status_code for _ in range(n)} == {200}
    wrong = [http.post("/api/verify", json=bad, headers={"X-Khaos-Verify-Key": key + "x"}).status_code for _ in range(n)]
    assert 429 in wrong
    assert verify_keys.revoke(cfg, "A newsroom") == 1
    with pytest.raises(ValueError):
        verify_keys.issue(cfg, "Someone", "anyone")


def test_the_model_page_states_what_trained_the_model_and_how_it_was_declared(register):
    """The training-data statement in a record is checked by the contract and shown on the public page."""
    from khaos_attribution import registry as RG
    app, store, http = register
    key, client = _machine(register, "artist-one")
    rec = _record()
    rec["training_set"]["declaration"] = {"version": "khaos.declaration/2", "layers": ["composition", "recording", "performers",
                                          "voices", "society"], "declared": 3, "of": 3}
    rec["training_set"]["data_kinds"] = ["sections of the full mix, each with a written description"]
    assert RG.record_problems(rec) == []
    rec["model"]["machine_key_id"] = key.key_id
    rec["model"]["watermark_payload"] = derive_payload(key.key_id, rec["model"]["run_id"])
    serial = client.publish(R.sign_record(rec, key))["serial"]
    page = http.get(f"/models/{serial}").text
    assert "sections of the full mix, each with a written description" in page
    assert "3 of 3 tracks, the artist confirming each of: the songs, the recordings, the performers" in page
    bad = _record()
    for wrong in ({"layers": [], "declared": 1, "of": 1}, {"layers": ["composition"], "declared": 5, "of": 3},
                  {"layers": ["composition"], "declared": -1, "of": 3}):
        probe = _record()
        probe["training_set"]["declaration"] = wrong
        assert any("declaration" in p for p in RG.record_problems(probe)), wrong
    bad["training_set"]["declaration"] = {"layers": "all", "declared": 1, "of": 1}
    bad["training_set"]["data_kinds"] = ["", 3]
    problems = RG.record_problems(bad)
    assert any("declaration" in p for p in problems) and any("data_kinds" in p for p in problems)


def test_the_bucket_is_a_column_filled_for_older_records_and_withdrawn_ones_are_not_served(register, tmp_path):
    """A bucket lookup reads an indexed column; a file from before the column is filled on open; a withdrawn model
    is neither in its bucket nor accepted as serving."""
    app, store, http = register
    key, client = _machine(register, "artist-one")
    signed = _signed(key)
    serial = client.publish(signed)["serial"]
    payload = signed["record"]["model"]["watermark_payload"]
    assert store.serials_in_bucket(payload) == [serial]
    store.db.execute("UPDATE records SET payload = NULL")
    assert store.serials_in_bucket(payload) == []
    from register_service.store import Store
    again = Store(store.path)
    assert again.serials_in_bucket(payload) == [serial], "the column is filled on open"
    _, space = _machine(register, "artist-two", role="space")
    client.withdraw(serial, "x")
    assert store.serials_in_bucket(payload) == []
    with pytest.raises(R.RegisterError, match="withdrawn"):
        space.served(serial, "https://space.example", "serving")
    assert space.served(serial, "https://space.example", "removed")["state"] == "removed"


def test_a_failed_write_leaves_nothing_behind(register, monkeypatch):
    """Every store write is one transaction: a failure half-way undoes its earlier statements."""
    app, store, http = register
    from register_service import store as S
    original = S.Store._ledger

    def broken(self, *a, **k):
        raise RuntimeError("the ledger write failed")
    monkeypatch.setattr(S.Store, "_ledger", broken)
    before = len(store.db.execute("SELECT * FROM accounts").fetchall())
    with pytest.raises(RuntimeError):
        store.create_account("half")
    assert len(store.db.execute("SELECT * FROM accounts").fetchall()) == before
    monkeypatch.setattr(S.Store, "_ledger", original)
    assert store.ledger_ok()


def test_the_public_pages_read_the_ledger_check_at_most_once_a_minute(register):
    _, store, http = register
    first = store.ledger_status()
    store.db.execute("UPDATE ledger SET detail_json = '{\"x\": 1}' WHERE seq = 1")
    assert store.ledger_status()["checked_at_utc"] == first["checked_at_utc"], "within the minute, the last reading"
    assert store.ledger_ok() is False, "the real check is always whole"
    assert store.ledger_status(max_age_s=0)["ok"] is False


def test_the_schema_and_docs_are_not_served(register):
    """The public doors are documented in the deploy notes, never enumerated to a caller."""
    _, _, http = register
    assert http.get("/openapi.json").status_code == 404 and http.get("/docs").status_code == 404
    assert http.get("/health").status_code == 200


def test_a_development_key_serves_behind_a_proxy_only_by_decision():
    """Loopback alone keeps nothing private once a proxy fronts it: the public address decides."""
    from register_service.app import check_dev_key_exposure
    loop = {"dev_key": True, "dev_key_public": False, "public_url": "http://127.0.0.1:8912"}
    check_dev_key_exposure(loop, "127.0.0.1")
    with pytest.raises(SystemExit, match="loopback only"):
        check_dev_key_exposure(loop, "0.0.0.0")
    fronted = {**loop, "public_url": "https://register.example"}
    with pytest.raises(SystemExit, match="REGISTER_DEV_KEY_PUBLIC"):
        check_dev_key_exposure(fronted, "127.0.0.1")
    check_dev_key_exposure({**fronted, "dev_key_public": True}, "127.0.0.1")
    check_dev_key_exposure({**fronted, "dev_key": False}, "0.0.0.0")


def test_search_engines_and_link_previews_find_a_description_a_card_and_every_model(register):
    """The pages carry a description, a canonical address and a share card; robots.txt and the sitemap name every model."""
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    home = http.get("/").text
    assert '<meta name="description"' in home and '<link rel="canonical" href="https://register.example/">' in home
    assert '<meta property="og:image" content="https://register.example/card.png">' in home and "application/ld+json" in home
    page = http.get(f"/models/{serial}").text
    assert f'<link rel="canonical" href="https://register.example/models/{serial}">' in page
    assert f'<meta name="description" content="Register No. {serial}: ' in page and "fine-tuned model on" in page
    robots = http.get("/robots.txt")
    assert robots.status_code == 200 and "Disallow: /api/" in robots.text and "Sitemap: https://register.example/sitemap.xml" in robots.text
    sitemap = http.get("/sitemap.xml")
    assert sitemap.headers["content-type"].startswith("application/xml")
    assert f"<loc>https://register.example/models/{serial}</loc>" in sitemap.text
    for path, kind in (("/card.png", "image/png"), ("/favicon.ico", "image/png"), ("/apple-touch-icon.png", "image/png"),
                       ("/favicon.svg", "image/svg+xml")):
        answer = http.get(path)
        assert answer.status_code == 200 and answer.headers["content-type"].startswith(kind) and answer.content


def test_link_checkers_asking_with_head_get_an_answer_not_a_405(register):
    """A HEAD request to a public page answers 200, as a GET would."""
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    for path in ("/", f"/models/{serial}", "/robots.txt", "/sitemap.xml", "/card.png"):
        assert http.head(path).status_code == 200, path
