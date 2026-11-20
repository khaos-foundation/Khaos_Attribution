"""The export writes only what a verifier reads, and clears what an earlier layout left beside it (a failed removal is
retried); a corrupt local copy is unreadable, never a crash; one head per size; no entry past a missing leaf."""
import json
import sys

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
pytest.importorskip("rfc8785")

from register_service.app import create_app, export_all  # noqa: E402
from tests.test_register_service import _machine, _signed, register  # noqa: E402,F401
from tests import log_files as LF  # noqa: E402

pytestmark = pytest.mark.usefixtures("tile_widths")


def test_the_export_writes_no_file_no_reader_reads_and_clears_the_old_ones(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    client.withdraw(serial, "asked to")
    log = app.state.cfg["data"] / "mirror" / "log"
    assert not (log / "entries").exists() and not (log / "struck.json").exists() and not (log / "withheld.json").exists()
    for name in ("struck.json", "withheld.json", "records.json"):
        (log / name).write_text("{}")
    (log / "entries").mkdir()
    (log / "entries" / "stale.json").write_text("{}")
    assert export_all(app.state.cfg, store.log, store.log.signer, True, full=True)
    assert not (log / "entries").exists() and not any((log / n).exists() for n in ("struck.json", "withheld.json", "records.json"))
    assert LF.size(log) == store.log.size() and not (log / "hashes.json").exists() and not (log / "bodies.json").exists()


def test_a_corrupt_local_copy_reads_as_unreadable_and_the_register_still_starts(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    client.publish(_signed(key))
    latest = app.state.cfg["data"] / "mirror-public" / "log" / "sth" / "latest.json"
    latest.write_text("{not json")
    app2 = create_app(app.state.cfg)
    assert app2.state.store.log.blocked is None, "the full export still vouches for the database"
    from khaos_attribution.translog import verify as V
    latest.write_text("{not json")
    with pytest.raises(V.Unreadable):
        V.Mirror(str(latest.parents[2])).get("sth/latest.json")


def test_the_size_is_the_number_of_leaves(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    client.publish(_signed(key))
    n = store.db.execute("SELECT COUNT(*) AS n FROM log_leaves").fetchone()["n"]
    assert store.log.size() == n > 0 and store.log.head()["tree_size"] == n


def _as_an_older_export(app, store):
    """The mirror as a format-2 export left it: its rows say format 2, and entries/ holds a body file."""
    log = app.state.cfg["data"] / "mirror" / "log"
    with store._tx() as db:
        db.execute("UPDATE log_exports SET format = 2")
    (log / "entries").mkdir(exist_ok=True)
    (log / "entries" / "stale.json").write_text("{}")
    return log


def test_a_removal_that_fails_is_not_recorded_and_the_next_export_retries(register, monkeypatch):
    import register_service.translog as TL
    app, store, http = register
    key, client = _machine(register, "artist-one")
    client.publish(_signed(key))
    log = _as_an_older_export(app, store)

    def refuse(path, ignore_errors=False, **k):
        # As shutil does: a removal told to ignore errors fails silently.
        if not ignore_errors:
            raise PermissionError(f"cannot remove {path}")

    monkeypatch.setattr(TL.shutil, "rmtree", refuse)
    with pytest.raises(PermissionError):
        export_all(app.state.cfg, store.log, store.log.signer, True)
    assert store.db.execute("SELECT format FROM log_exports WHERE place = ?", (str(log.resolve()),)).fetchone()["format"] == 2
    monkeypatch.undo()
    assert export_all(app.state.cfg, store.log, store.log.signer, True)
    assert not (log / "entries").exists()
    assert store.db.execute("SELECT format FROM log_exports WHERE place = ?", (str(log.resolve()),)).fetchone()["format"] == TL.EXPORT_FORMAT


def test_an_older_export_is_cleared_by_the_next_export_and_a_later_withhold_serves_no_body(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    log = _as_an_older_export(app, store)
    assert export_all(app.state.cfg, store.log, store.log.signer, True)
    assert not (log / "entries").exists()
    leaf = store.log.admit_leaf_of(store.record(serial)["record_sha256"])
    store.log_withhold(leaf, "a private matter")
    assert export_all(app.state.cfg, store.log, store.log.signer, True)
    assert LF.bodies(log)[store.log.leaf(leaf)["idx"]] is None and not (log / "entries").exists()


def test_a_heads_table_from_before_one_head_per_size_gets_the_rule_and_keeps_both(tmp_path):
    import sqlite3

    from register_service.store import Store
    path = tmp_path / "register.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE log_heads (id INTEGER PRIMARY KEY, tree_size INTEGER NOT NULL, root_sha256 TEXT NOT NULL,"
                   " timestamp TEXT NOT NULL, key_id TEXT NOT NULL, signature TEXT NOT NULL)")
        db.executemany("INSERT INTO log_heads (tree_size, root_sha256, timestamp, key_id, signature) VALUES (?, ?, ?, 'k', 's')",
                       [(1, "aa", "t1"), (1, "aa", "t2"), (2, "bb", "t3")])
    store = Store(path)
    assert [h["timestamp"] for h in store.log.heads()] == ["t1", "t3"]
    assert [r["timestamp"] for r in store.db.execute("SELECT timestamp FROM log_heads_superseded")] == ["t2"]
    with pytest.raises(sqlite3.IntegrityError):
        store.db.execute("INSERT INTO log_heads (tree_size, root_sha256, timestamp, key_id, signature) VALUES (2, 'cc', 't4', 'k', 's')")
    Store(path)


def test_a_log_with_a_missing_leaf_takes_no_entry(register):
    from register_service.translog import LogBlocked
    app, store, http = register
    key, client = _machine(register, "artist-one")
    client.publish(_signed(key))
    before = store.log.size()
    with store._tx() as db:
        db.execute("DELETE FROM log_leaves WHERE idx = 0")
    with pytest.raises(LogBlocked):
        with store._tx():
            store.log.heartbeat_locked()
    assert store.db.execute("SELECT COUNT(*) AS n FROM log_heads WHERE tree_size > ?", (before,)).fetchone()["n"] == 0
