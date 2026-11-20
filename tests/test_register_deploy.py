"""The Register's setup deploys the tag it names with that tag's own scripts, from the lock, with a pinned uv; a first
log start is the owner's explicit act, in the script and in the service; backups carry the verify keys."""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, "src")

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
SERVER = DEPLOY / "register-server.sh"


def _text() -> str:
    return SERVER.read_text(encoding="utf-8")


def test_the_same_tag_check_refuses_a_deploy_folder_that_is_not_the_tags(tmp_path):
    running, checkout = tmp_path / "running", tmp_path / "checkout"
    shutil.copytree(DEPLOY, running)
    shutil.copytree(DEPLOY, checkout)
    script = DEPLOY / "register-same-tag.sh"
    same = subprocess.run(["bash", str(script), str(running), str(checkout)], capture_output=True, text=True)
    assert same.returncode == 0, same.stdout + same.stderr
    (running / "register-mirror.sh").write_text("#!/usr/bin/env bash\n# an older copy\n")
    differs = subprocess.run(["bash", str(script), str(running), str(checkout)], capture_output=True, text=True)
    assert differs.returncode == 1
    assert "register-mirror.sh" in differs.stdout and str(checkout / "register-server.sh") in differs.stdout


def test_every_installed_file_comes_from_the_tags_checkout_after_the_check():
    text = _text()
    assert '"$(dirname "$0")"' not in text and "$(dirname \"$0\")/" not in text, "nothing is installed from the running clone"
    check = text.index("register-same-tag.sh")
    assert text.index('checkout -q --detach "\'"$TAG"\'"') < check
    installs = [m.start() for m in re.finditer(r"^\s*install -m", text, re.M)]
    assert installs and all(i > check for i in installs)
    for name in ("register-backup.sh", "register-mirror.sh", "register-key.sh", "register-log-key.sh", "journald-retention.conf",
                 "logrotate/"):
        assert f'"$CHECKOUT/deploy/{name}' in text or f'"$CHECKOUT"/deploy/{name}' in text, name


def test_the_tag_is_required_and_the_venv_comes_from_the_lock_with_a_pinned_uv():
    text = _text()
    assert 'TAG="${REGISTER_TAG:?' in text and "REGISTER_TAG:-" not in text
    assert "astral.sh/uv/install.sh" not in text and "UV_VERSION=0.11.32" in text
    assert re.search(r"UV_SHA256_X86_64=[0-9a-f]{64}", text) and re.search(r"UV_SHA256_AARCH64=[0-9a-f]{64}", text)
    assert "sha256sum -c" in text
    assert "uv sync --frozen --no-editable --extra registry --extra service" in text and "UV_PROJECT_ENVIRONMENT" in text
    assert "uv pip install" not in text


def test_a_first_log_start_needs_the_owners_explicit_enable_before_anything_is_installed():
    text = _text()
    guard = text.index('[ "${REGISTER_LOG_ENABLE:-}" != 1 ]')
    assert guard < text.index("uv sync --frozen")
    assert '[ ! -f "$DATA/log-started" ]' in text[guard - 200:guard]


def test_the_claim_paths_are_outside_the_ten_megabyte_rule():
    caddy = re.search(r"cat > /etc/caddy/Caddyfile <<CADDY\n(.*?)\nCADDY", _text(), re.S).group(1)
    assert "@claims path /api/claims /api/claims/check" in caddy
    assert "@rest not path /api/claims /api/claims/check" in caddy
    rules = re.findall(r"request_body (@\w+) \{\s*max_size (\d+)MB", caddy)
    assert sorted(rules) == [("@claims", "25"), ("@rest", "10")], "each request meets exactly one body limit"


def test_backup_and_restore_carry_the_verify_keys(tmp_path):
    data = tmp_path / "srv" / "data"
    data.mkdir(parents=True)
    for db in ("register", "index"):
        sqlite3.connect(data / f"{db}.sqlite3").execute("CREATE TABLE t (x)").connection.commit()
    (data / "verify-keys.json").write_text('{"keys": []}')
    env = {**os.environ, "REGISTER_DATA_DIR": str(data)}
    run = subprocess.run(["bash", str(DEPLOY / "register-backup.sh")], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    archive, = (tmp_path / "srv" / "backups").glob("register-*.tar.gz")
    with tarfile.open(archive) as tar:
        assert "./verify-keys.json" in tar.getnames()
    assert "verify-keys.json" in (DEPLOY / "register-restore.sh").read_text()


def test_the_key_switches_drop_the_public_development_setting_once_no_development_key_is_left():
    for name in ("register-key.sh", "register-log-key.sh"):
        text = (DEPLOY / name).read_text()
        switch = text[text.index("switch)"):]
        assert "REGISTER_DEV_KEY_PUBLIC" in switch, name


def test_the_journal_drop_in_is_a_file_the_space_holds_too():
    assert (DEPLOY / "journald-retention.conf").read_text().splitlines()[0] == "[Journal]"
    space = Path(__file__).resolve().parents[2] / "Khaos_Platform_Chris_Playground" / "deploy" / "journald-retention.conf"
    if space.is_file():
        assert space.read_bytes() == (DEPLOY / "journald-retention.conf").read_bytes()


def test_the_readme_starts_an_update_from_the_tags_own_script():
    readme = (DEPLOY / "README.md").read_text()
    update = readme[readme.index("## Updating"):]
    first = update.split("\n\n")[1]
    assert "checkout" in first and "register-server.sh" in first
    assert "v0.27.0" not in readme and "v0.27.0" not in _text()


# ---- the service itself ----

pytest.importorskip("cryptography")
pytest.importorskip("fastapi")


def test_the_service_refuses_to_start_a_log_the_owner_has_not_enabled(tmp_path):
    from register_service.app import create_app
    cfg = {"data": tmp_path / "data", "key_file": tmp_path / "data" / "register-key.pem", "dev_key": True,
           "public_url": "https://register.example", "log_enable": False}
    with pytest.raises(SystemExit, match="REGISTER_LOG_ENABLE"):
        create_app(cfg)
    assert not (tmp_path / "data" / "log-started").exists()
    assert create_app({**cfg, "log_enable": True}).state.store.log.size() > 0
    # Once started, the log opens on every later start with or without the setting.
    assert create_app(cfg).state.store.log.size() > 0


def test_the_setting_is_read_from_the_environment_and_the_daily_job_never_starts_a_log(tmp_path, monkeypatch):
    from register_service.__main__ import main
    from register_service.app import settings
    monkeypatch.setenv("REGISTER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REGISTER_DEV_KEY", "1")
    monkeypatch.delenv("REGISTER_LOG_ENABLE", raising=False)
    assert settings()["log_enable"] is False
    with pytest.raises(SystemExit, match="REGISTER_LOG_ENABLE"):
        main(["log-stamp"])
    monkeypatch.setenv("REGISTER_LOG_ENABLE", "1")
    assert settings()["log_enable"] is True


def test_the_bill_of_materials_names_what_the_register_runs(capsys):
    from register_service.__main__ import main
    import khaos_attribution
    assert main(["bill-of-materials"]) == 0
    rows = {r["name"]: r for r in json.loads(capsys.readouterr().out)["packages"]}
    assert rows["khaos-attribution"]["version"] == khaos_attribution.__version__
    assert rows["fastapi"]["version"] and all("licence" in r for r in rows.values())
