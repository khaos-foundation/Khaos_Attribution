"""The public reads are bounded: an address gets a number of them a minute, the ledger and the sitemap come in
pages, counts are counted by the database, the ledger's check reads only what is new, and two writes at once never
leave a torn ledger in the mirror."""
import json
import threading

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("fastapi")

import register_service.store as S  # noqa: E402
from register_service import app as A  # noqa: E402
from tests.test_register_service import _machine, _signed, register  # noqa: E402,F401


def _published(register, n: int) -> list:
    key, client = _machine(register, "artist-one")
    return [client.publish(_signed(key, run_id=f"run_{i}"))["serial"] for i in range(n)]


@pytest.mark.parametrize("path", ["/", "/models/65536", "/sitemap.xml", "/api/ledger", "/api/withdrawn", "/health",
                                  "/api/models/65536"])
def test_every_public_read_is_bounded_per_address(register, path, monkeypatch):
    app, store, http = register
    _published(register, 1)
    monkeypatch.setattr(A, "READ_LIMIT", 5)
    http = type(http)(A.create_app(app.state.cfg))
    who = {"X-Forwarded-For": "198.51.100.9"}
    codes = [http.get(path, headers=who).status_code for _ in range(7)]
    assert codes[:5] == [200] * 5 and codes[5:] == [429, 429], codes
    assert http.get(path, headers={"X-Forwarded-For": "198.51.100.10"}).status_code == 200, "another address, its own bucket"


def test_the_ledger_comes_in_pages_and_the_first_page_is_the_answer_it_always_gave(register, monkeypatch):
    app, store, http = register
    _published(register, 4)
    rows = store.ledger()
    whole = http.get("/api/ledger")
    assert [json.loads(x) for x in whole.text.strip().splitlines()] == rows and "link" not in whole.headers
    monkeypatch.setattr(A, "LEDGER_PAGE", 3)
    http = type(http)(A.create_app(app.state.cfg))
    seen, since = [], 0
    while True:
        r = http.get("/api/ledger", params={"since": since})
        page = [json.loads(x) for x in r.text.strip().splitlines() if x]
        assert len(page) <= 3
        seen += page
        if "link" not in r.headers:
            break
        since = page[-1]["seq"]
        assert r.headers["link"] == f'</api/ledger?since={since}>; rel="next"'
    assert seen == rows and len(rows) > 6


def test_the_sitemap_comes_in_pages_and_robots_names_each(register, monkeypatch):
    app, store, http = register
    serials = _published(register, 5)
    one = http.get("/sitemap.xml").text
    assert all(f"/models/{s}<" in one for s in serials) and "<loc>https://register.example/</loc>" in one
    monkeypatch.setattr(A, "SITEMAP_PAGE", 2)
    http = type(http)(A.create_app(app.state.cfg))
    pages = [http.get("/sitemap.xml", params={"page": n}).text for n in (1, 2, 3)]
    assert http.get("/sitemap.xml").text == pages[0] and "<loc>https://register.example/</loc>" in pages[0]
    found = [[s for s in serials if f"/models/{s}<" in page] for page in pages]
    assert found == [serials[:2], serials[2:4], serials[4:]]
    assert http.get("/sitemap.xml", params={"page": 4}).status_code == 404
    robots = http.get("/robots.txt").text
    assert "Sitemap: https://register.example/sitemap.xml\n" in robots
    assert "Sitemap: https://register.example/sitemap.xml?page=2\n" in robots and "?page=3\n" in robots and "?page=4" not in robots


def test_the_home_page_health_and_sitemap_never_read_every_record(register):
    """Counts come from the database and a page reads the rows it shows: no list of every serial, no record per model."""
    app, store, http = register
    _published(register, 6)
    assert http.get("/health").json()["models"] == 6 and "6 models registered" in http.get("/").text
    assert store.count() == 6 and store.latest_serials(2) == store.serials()[-2:][::-1]
    http.get("/")   # the ledger's first whole check is the start's, not a page's
    for path in ("/health", "/sitemap.xml"):
        store.serials = lambda: (_ for _ in ()).throw(AssertionError("every serial was listed"))
        store.record = lambda serial: (_ for _ in ()).throw(AssertionError("a record was read for a list"))
        try:
            assert http.get(path).status_code == 200
        finally:
            del store.serials, store.record
    store.serials = lambda: (_ for _ in ()).throw(AssertionError("every serial was listed"))
    try:
        assert http.get("/").status_code == 200
    finally:
        del store.serials


def test_the_ledgers_check_reads_only_the_rows_since_the_last_one(register, monkeypatch):
    app, store, http = register
    _published(register, 5)
    hashed = []
    real = S.sha256_hex
    monkeypatch.setattr(S, "sha256_hex", lambda data: hashed.append(1) or real(data))
    assert store.ledger_status(max_age_s=0)["ok"]
    whole = len(hashed)
    assert whole == len(store.ledger())
    store.create_account("another")
    store.create_account("and-another")
    del hashed[:]
    assert store.ledger_status(max_age_s=0)["ok"]
    assert len(hashed) == 3 < whole, "the two new rows and the row they follow"
    store.db.execute("UPDATE ledger SET detail_json = '{\"account\": \"x\"}' WHERE seq = (SELECT MAX(seq) FROM ledger)")
    assert store.ledger_status(max_age_s=0)["ok"] is False, "a row that does not chain from the last reading is found"


def test_a_row_changed_behind_the_last_reading_is_found_by_the_whole_check(register, monkeypatch):
    """The whole chain is read at start, once a day and by the operator's check; between those a page reads what is new."""
    app, store, http = register
    _published(register, 3)
    assert store.ledger_status(max_age_s=0)["ok"]
    store.db.execute("UPDATE ledger SET detail_json = '{\"x\": 1}' WHERE seq = 1")
    assert store.ledger_ok() is False, "the operator's check is always whole"
    assert store.ledger_status(max_age_s=0)["ok"] is True
    monkeypatch.setattr(S, "LEDGER_WHOLE_CHECK_S", 0.0)
    assert store.ledger_status(max_age_s=0)["ok"] is False
    assert S.Store(store.path).ledger_status()["ok"] is False, "a start reads the whole chain"


def test_two_writes_at_once_leave_a_whole_ledger_in_the_mirror(register, monkeypatch):
    """One writer at a time and a temporary file of its own: a reader of the mirror never meets a torn ledger, not
    even while a second write is half-way through its own copy."""
    import os
    app, store, http = register
    key, client = _machine(register, "artist-one")
    client.publish(_signed(key, run_id="run_0"))
    ledger_file = app.state.cfg["data"] / "mirror" / "ledger.jsonl"
    at_replace, go_replace, mid_write, go_write = (threading.Event() for _ in range(4))
    real_replace, real_ledger = os.replace, store.ledger

    def held_replace(src, dst):
        if str(dst) == str(ledger_file) and threading.current_thread().name == "first":
            at_replace.set()
            go_replace.wait(10)
        real_replace(src, dst)

    def held_ledger(since_seq=0, limit=None):
        rows = real_ledger(since_seq) if limit is None else real_ledger(since_seq, limit)
        if threading.current_thread().name != "second" or since_seq or limit is not None:
            return rows

        def half_way():
            yield from rows[:1]
            mid_write.set()
            go_write.wait(10)
            yield from rows[1:]
        return half_way()
    monkeypatch.setattr(os, "replace", held_replace)
    store.ledger = held_ledger
    try:
        from anyio import to_thread  # noqa: F401 — the publishes below run in its worker threads
        names = iter(("first", "second"))
        real_register = store.register

        def named_register(*a, **k):
            threading.current_thread().name = next(names)
            return real_register(*a, **k)
        store.register = named_register
        first = threading.Thread(target=lambda: client.publish(_signed(key, run_id="run_a")))
        first.start()
        assert at_replace.wait(10)
        second = threading.Thread(target=lambda: client.publish(_signed(key, run_id="run_b")))
        second.start()
        mid_write.wait(1.5)
        go_replace.set()
        first.join(10)
        lines = ledger_file.read_text(encoding="utf-8").splitlines()
        whole = real_ledger()
        assert [json.loads(x) for x in lines] == whole[:len(lines)] and len(lines) >= len(whole) - 1, "a torn ledger was served"
        go_write.set()
        second.join(10)
    finally:
        go_replace.set()
        go_write.set()
        del store.ledger, store.register
    assert [json.loads(x) for x in ledger_file.read_text(encoding="utf-8").splitlines()] == real_ledger()
