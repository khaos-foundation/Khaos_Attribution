"""Both Register databases say which shape of file they are, an older file is brought up to this code's and stamped,
a newer one is refused with words before anything is written, and a backup is checked before it is put back."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")

from register_service.index import INDEX_VERSION, Index  # noqa: E402
from register_service.store import STORE_VERSION, NewerFile, Store, file_version  # noqa: E402

RESTORE = Path(__file__).resolve().parents[1] / "deploy" / "register-restore.sh"


def _stamp(path: Path, version: int) -> None:
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA user_version = {version}")


def test_a_new_file_is_stamped_with_this_codes_version(tmp_path):
    Store(tmp_path / "register.sqlite3")
    Index(tmp_path / "index.sqlite3")
    assert file_version(tmp_path / "register.sqlite3") == STORE_VERSION
    assert file_version(tmp_path / "index.sqlite3") == INDEX_VERSION


def test_an_older_unstamped_file_is_brought_up_and_stamped(tmp_path):
    path = tmp_path / "register.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE records (serial INTEGER PRIMARY KEY, record_sha256 TEXT UNIQUE NOT NULL, machine_key_id TEXT NOT NULL,"
                   " account_id INTEGER NOT NULL, signed_json TEXT NOT NULL, counter_json TEXT NOT NULL, state TEXT NOT NULL,"
                   " published_at TEXT NOT NULL, withdrawn_at TEXT, withdraw_reason TEXT)")
    assert file_version(path) == 0
    store = Store(path)
    assert "payload" in {r["name"] for r in store.db.execute("PRAGMA table_info(records)")}
    assert file_version(path) == STORE_VERSION


@pytest.mark.parametrize("cls,name,ours", [(Store, "register.sqlite3", STORE_VERSION), (Index, "index.sqlite3", INDEX_VERSION)])
def test_a_live_file_from_before_versions_opens_and_is_stamped(tmp_path, cls, name, ours):
    """A register's live store or index, unstamped (0), opens as it always did and is stamped; it is never refused."""
    path = tmp_path / name
    cls(path)
    _stamp(path, 0)
    cls(path)
    assert file_version(path) == ours


@pytest.mark.parametrize("cls,name,ours", [(Store, "register.sqlite3", STORE_VERSION), (Index, "index.sqlite3", INDEX_VERSION)])
def test_a_file_a_newer_register_wrote_is_refused_untouched(tmp_path, cls, name, ours):
    path = tmp_path / name
    _stamp(path, ours + 1)
    before = path.read_bytes()
    with pytest.raises(NewerFile, match="newer register"):
        cls(path)
    assert path.read_bytes() == before and file_version(path) == ours + 1


def test_the_service_will_not_start_on_a_newer_file(tmp_path):
    from register_service.app import create_app
    data = tmp_path / "data"
    data.mkdir()
    _stamp(data / "register.sqlite3", STORE_VERSION + 1)
    with pytest.raises(SystemExit, match="newer register"):
        create_app({"data": data, "key_file": data / "k.pem", "dev_key": True, "public_url": "x"})


def test_a_backup_is_checked_before_it_is_put_back(tmp_path, capsys):
    from register_service.__main__ import main
    Store(tmp_path / "register.sqlite3")
    Index(tmp_path / "index.sqlite3")
    assert main(["check-version", str(tmp_path)]) == 0
    _stamp(tmp_path / "index.sqlite3", INDEX_VERSION + 1)
    assert main(["check-version", str(tmp_path)]) == 1 and "newer register" in capsys.readouterr().out
    text = RESTORE.read_text()
    assert text.index("check-version") < text.index("systemctl stop register.service")
