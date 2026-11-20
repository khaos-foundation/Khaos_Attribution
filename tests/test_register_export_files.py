"""The export writes only what a verifier reads, and a full export clears what an earlier layout left beside it; a
local copy whose files cannot be read is unreadable, never a crash at start; the log's size is its leaves'."""
import json
import sys

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
pytest.importorskip("rfc8785")

from register_service.app import create_app, export_all  # noqa: E402
from tests.test_register_service import _machine, _signed, register  # noqa: E402,F401


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
    assert json.loads((log / "hashes.json").read_text())["tree_size"] == store.log.size()


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
