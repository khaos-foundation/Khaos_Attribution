"""The public reads are bounded: an address gets a number of them a minute, the ledger and the sitemap come in
pages, counts are counted by the database, the ledger's check reads only what is new, and two writes at once never
leave a torn ledger in the mirror."""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("fastapi")

import register_service.store as S  # noqa: E402
import register_service.translog as RT  # noqa: E402
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
    """The whole chain is read at start, once an hour and by the operator's check; between those a page reads what is new."""
    app, store, http = register
    _published(register, 3)
    assert store.ledger_status(max_age_s=0)["ok"]
    store.db.execute("UPDATE ledger SET detail_json = '{\"x\": 1}' WHERE seq = 1")
    assert store.ledger_ok() is False, "the operator's check is always whole"
    assert store.ledger_status(max_age_s=0)["ok"] is True
    clock = [S.time.monotonic()]
    monkeypatch.setattr(S.time, "monotonic", lambda: clock[0])
    clock[0] += 3540
    assert store.ledger_status()["ok"] is True, "inside the hour a page reads only what is new"
    clock[0] += 120
    assert store.ledger_status()["ok"] is False, "an altered row is said within the hour"
    assert S.Store(store.path).ledger_status()["ok"] is False, "a start reads the whole chain"


def _lines(path) -> list:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_a_write_appends_its_rows_to_the_mirrors_ledger_and_a_ledger_that_is_not_the_databases_is_written_whole(register, monkeypatch):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    client.publish(_signed(key, run_id="run_0"))
    ledger_file = app.state.cfg["data"] / "mirror" / "ledger.jsonl"
    before, inode = ledger_file.read_bytes(), ledger_file.stat().st_ino
    whole = []
    monkeypatch.setattr(A, "write_whole", lambda path, data, real=A.write_whole: (whole.append(path.name), real(path, data))[1])
    client.publish(_signed(key, run_id="run_1"))
    after = ledger_file.read_bytes()
    assert ledger_file.stat().st_ino == inode and after.startswith(before) and len(after) > len(before), "the new rows were appended"
    assert "ledger.jsonl" not in whole and _lines(ledger_file) == store.ledger()
    # A last line that is not the database's row of that number: the file is not this database's, and is replaced.
    rows = after.decode().splitlines()
    ledger_file.write_text("\n".join(rows[:-1] + [rows[-1].replace('"publish"', '"publish "')]) + "\n")
    client.publish(_signed(key, run_id="run_2"))
    assert "ledger.jsonl" in whole and _lines(ledger_file) == store.ledger()
    # A line cut short by a crash is not built on.
    ledger_file.write_bytes(ledger_file.read_bytes()[:-9])
    client.publish(_signed(key, run_id="run_3"))
    assert _lines(ledger_file) == store.ledger()
    ledger_file.write_text("not the ledger\n")
    A.write_ledger(store, ledger_file, whole=True)
    assert _lines(ledger_file) == store.ledger()


def test_two_writes_at_once_leave_a_whole_ledger_in_the_mirror(register):
    """One writer at a time: the second reads where the ledger ends only after the first has added its rows, so no
    row is filed twice and none is torn."""
    app, store, http = register
    key, client = _machine(register, "artist-one")
    client.publish(_signed(key, run_id="run_0"))
    ledger_file = app.state.cfg["data"] / "mirror" / "ledger.jsonl"
    reading, go = threading.Event(), threading.Event()
    real_ledger, real_register, names = store.ledger, store.register, iter(("first", "second"))

    def held_ledger(since_seq=0, limit=None):
        rows = real_ledger(since_seq, limit)
        if threading.current_thread().name == "first" and since_seq and limit is None:
            # The first writer knows where the file ends and what it will add; it is held before adding it.
            reading.set()
            go.wait(10)
        return rows

    def named_register(*a, **k):
        threading.current_thread().name = next(names)
        return real_register(*a, **k)
    store.ledger, store.register = held_ledger, named_register
    try:
        first = threading.Thread(target=lambda: client.publish(_signed(key, run_id="run_a")))
        first.start()
        assert reading.wait(10)
        second = threading.Thread(target=lambda: client.publish(_signed(key, run_id="run_b")))
        second.start()
        second.join(1.5)
        go.set()
        first.join(10)
        second.join(10)
    finally:
        go.set()
        del store.ledger, store.register
    assert _lines(ledger_file) == real_ledger(), "a row was filed twice, or not at all"


def _ledger_one_row_behind(tmp_path) -> tuple:
    store = S.Store(tmp_path / "data" / "register.sqlite3")
    for i in range(3):
        store.create_account(f"account-{i}")
    path = tmp_path / "data" / "mirror" / "ledger.jsonl"
    A.write_ledger(store, path, whole=True)
    store.create_account("account-new")
    return store, path


def _append_with_a_whole_write_landing_after_its_read(store, path, start_whole) -> None:
    """The service's append, with the operator's whole write started once the append has read what to add."""
    real = store.ledger

    def racing(since_seq=0, limit=None):
        rows = real(since_seq, limit)
        if since_seq and limit is None:
            start_whole()
        return rows
    store.ledger = racing
    try:
        A.write_ledger(store, path)
    finally:
        del store.ledger


def test_a_whole_write_waits_for_an_append_under_way_in_this_process(tmp_path):
    store, path = _ledger_one_row_behind(tmp_path)
    whole = []

    def start_whole():
        whole.append(threading.Thread(target=lambda: A.write_ledger(S.Store(store.path), path, whole=True)))
        whole[0].start()
        whole[0].join(1.0)
    _append_with_a_whole_write_landing_after_its_read(store, path, start_whole)
    whole[0].join(30)
    assert [r["seq"] for r in _lines(path)] == [r["seq"] for r in store.ledger()], "a row was filed twice"
    store.create_account("account-later")
    A.write_ledger(store, path)
    assert _lines(path) == store.ledger()


WHOLE_WRITE = ("import sys\nfrom pathlib import Path\nfrom register_service.app import write_ledger\n"
               "from register_service.store import Store\nstore = Store(Path(sys.argv[1]))\nprint('ready', flush=True)\n"
               "write_ledger(store, Path(sys.argv[2]), whole=True)\n")


def test_a_whole_write_from_another_process_waits_for_an_append_under_way(tmp_path):
    """As register-cli log-export runs it: a process of its own, holding the lock file beside the ledger."""
    store, path = _ledger_one_row_behind(tmp_path)
    here = str(Path(A.__file__).parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (here, os.environ.get("PYTHONPATH")) if p)}
    other = []

    def start_whole():
        other.append(subprocess.Popen([sys.executable, "-c", WHOLE_WRITE, str(store.path), str(path)], env=env,
                                      stdout=subprocess.PIPE, text=True))
        assert other[0].stdout.readline().strip() == "ready"
        try:
            other[0].wait(1.5)
        except subprocess.TimeoutExpired:
            pass
    _append_with_a_whole_write_landing_after_its_read(store, path, start_whole)
    assert other[0].wait(60) == 0
    other[0].stdout.close()
    assert [r["seq"] for r in _lines(path)] == [r["seq"] for r in store.ledger()], "a row was filed twice"


def test_a_ledger_whose_tail_repeats_or_reorders_rows_is_written_whole(tmp_path):
    """Whatever left the file so, the next write notices that its tail is not the database's rows in order."""
    store, path = _ledger_one_row_behind(tmp_path)
    A.write_ledger(store, path)
    for n, spoil in enumerate((lambda rows: rows + rows[-1:], lambda rows: rows + rows[-2:],
                               lambda rows: rows[:-2] + [rows[-1], rows[-2]])):
        path.write_text("".join(spoil(path.read_text().splitlines(keepends=True))))
        store.create_account(f"account-after-{n}")
        A.write_ledger(store, path)
        assert _lines(path) == store.ledger(), n


# flock(1) where util-linux has none: the lock the named file or descriptor takes, held until the command exits.
FLOCK = """import fcntl, os, subprocess, sys, time
args = sys.argv[1:]
wait = float(args[1]) if args[0] == "-w" else 1e9
args = args[2:] if args[0] == "-w" else args
if args[0] == "-u":
    fcntl.flock(int(args[1]), fcntl.LOCK_UN)
    sys.exit(0)
fd = int(args[0]) if args[0].isdigit() else os.open(args[0], os.O_RDWR | os.O_CREAT)
deadline = time.monotonic() + wait
while True:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except BlockingIOError:
        if time.monotonic() > deadline:
            sys.exit(1)
        time.sleep(0.05)
sys.exit(subprocess.call(args[1:]) if len(args) > 1 else 0)
"""


def test_the_public_mirror_copies_the_ledger_only_while_no_writer_holds_the_mirror(tmp_path):
    """register-mirror.sh copies under the lock every writer of the mirror takes, so no copy holds half an append."""
    data = tmp_path / "srv" / "data"
    store = S.Store(data / "register.sqlite3")
    store.create_account("account-0")
    A.write_ledger(store, data / "mirror" / "ledger.jsonl", whole=True)
    repo = tmp_path / "ledger.git"
    subprocess.run(["git", "init", "-q", "--bare", str(repo)], check=True)
    tools = tmp_path / "bin"
    tools.mkdir()
    if shutil.which("flock") is None:
        (tools / "flock").write_text(f"#!{sys.executable}\n{FLOCK}")
        (tools / "flock").chmod(0o755)
    env = {**os.environ, "REGISTER_DATA_DIR": str(data), "LEDGER_REPO": str(repo), "HOME": str(tmp_path),
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}"}
    script = Path(__file__).resolve().parents[1] / "deploy" / "register-mirror.sh"
    copied = tmp_path / "srv" / "ledger-mirror" / "ledger.jsonl"
    with RT.export_lock(data / "mirror"):
        run = subprocess.Popen(["bash", str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not copied.exists() and run.poll() is None:
            time.sleep(0.05)
        early = copied.exists()
    out = run.communicate(timeout=60)[0]
    assert not early, "the ledger was copied while a writer held the mirror"
    assert run.returncode == 0, out
    assert copied.read_bytes() == (data / "mirror" / "ledger.jsonl").read_bytes()
