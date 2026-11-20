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
    assert "register-mirror.sh" in differs.stdout and "/root/Khaos_Attribution/deploy/register-server.sh" in differs.stdout


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


def test_the_script_keeps_an_unstarted_log_off_unless_the_owner_enables_it():
    text = _text()
    check = text.index('echo "== the signed log"')
    block = text[check:text.index('echo "==', check + 10)]
    assert "exit" not in block, "an unstarted log never stops the deploy: the service runs with it off"
    assert '[ -f "$DATA/log-started" ]' in block and "SELECT COUNT(*) FROM log_leaves" in block, "a log with entries has started"
    assert "REGISTER_LOG_ENABLE=1 bash" not in (DEPLOY / "README.md").read_text(), "no copy-paste line starts the log"


def test_the_setup_refuses_to_run_from_the_registers_own_checkout(tmp_path):
    text = _text()
    early = text.index('if [ "$HERE" = "$(readlink -f -- "$CHECKOUT/deploy"')
    assert early < text.index('echo "== packages"'), "refused before anything is changed"
    same = subprocess.run(["bash", str(DEPLOY / "register-same-tag.sh"), str(DEPLOY), str(DEPLOY)], capture_output=True, text=True)
    assert same.returncode == 1 and "/root/Khaos_Attribution/deploy/register-server.sh" in same.stdout


def test_caddy_checks_the_site_file_before_any_restart():
    text = _text()
    assert text.index("caddy validate --config /etc/caddy/Caddyfile") < text.index("systemctl restart register.service")


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


def _off(tmp_path, **over):
    return {"data": tmp_path / "data", "key_file": tmp_path / "data" / "register-key.pem", "dev_key": True,
            "log_dev_key": False, "public_url": "https://register.example", "log_enable": False, **over}


def test_a_register_whose_log_never_started_serves_everything_else_with_it_off(tmp_path):
    """No log key is needed and nothing of the log is written; the first start on the owner's word admits it all."""
    from fastapi.testclient import TestClient
    from khaos_attribution import registry as R
    from register_service.app import create_app
    from tests.test_register_service import _machine, _signed
    cfg = _off(tmp_path)
    app = create_app(cfg)
    store, http = app.state.store, TestClient(app)
    store.create_account("artist-one")
    key, client = _machine((app, store, http), "artist-one")
    s1, s2 = _signed(key), _signed(key, run_id="run_2")
    serial = client.publish(s1)["serial"]
    second = client.publish(s2)["serial"]
    assert client.withdraw(second, "the artist withdrew")["state"] == "withdrawn"
    assert http.get("/api/withdrawn").json()["withdrawn"] == [second] and http.get(f"/api/models/{serial}").status_code == 200
    assert http.get("/log/sth").status_code == 404 and http.get("/log/key").status_code == 404
    assert http.get("/health").json()["log"] == "off"
    _, warden = _machine((app, store, http), "artist-one", role="warden")
    with pytest.raises(R.RegisterError) as e:
        warden.log_append({"kind": "admit", "record_sha256": s1["record_sha256"], "sitting": "S1"})
    assert e.value.status == 503
    assert store.log.size() == 0 and not (cfg["data"] / "log-started").exists()
    assert not (cfg["data"] / "mirror" / "log").exists() and not (cfg["data"] / "mirror-public").exists()
    on = create_app(_off(tmp_path, log_enable=True, log_dev_key=True)).state.store
    kinds = [e["entry"]["kind"] for e in on.log.entries(0, on.log.size())]
    assert kinds == ["genesis", "admit", "admit", "revoke"], "the first start admits every model and strikes the withdrawn"
    assert (cfg["data"] / "log-started").exists()
    # Once started, the log runs on every later start, with or without the setting.
    assert create_app(_off(tmp_path, log_dev_key=True)).state.store.log.off is None


def test_the_setting_is_read_from_the_environment_and_the_daily_jobs_do_nothing_while_the_log_is_off(tmp_path, monkeypatch,
                                                                                                    capsys):
    from register_service.__main__ import main
    from register_service.app import settings
    monkeypatch.setenv("REGISTER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REGISTER_DEV_KEY", "1")
    monkeypatch.setenv("REGISTER_LOG_DEV_KEY", "1")
    monkeypatch.delenv("REGISTER_LOG_ENABLE", raising=False)
    assert settings()["log_enable"] is False
    assert main(["log-stamp"]) == 0 and "REGISTER_LOG_ENABLE" in capsys.readouterr().out
    assert not (tmp_path / "data" / "log-started").exists()
    with pytest.raises(SystemExit, match="REGISTER_LOG_ENABLE"):
        main(["log-withhold", "0" * 64, "a reason"])
    assert main(["check"]) == 0 and "log off" in capsys.readouterr().out
    monkeypatch.setenv("REGISTER_LOG_ENABLE", "1")
    assert settings()["log_enable"] is True


def test_the_bill_of_materials_names_what_the_register_runs(capsys):
    from register_service.__main__ import main
    import khaos_attribution
    assert main(["bill-of-materials"]) == 0
    rows = {r["name"]: r for r in json.loads(capsys.readouterr().out)["packages"]}
    assert rows["khaos-attribution"]["version"] == khaos_attribution.__version__
    assert rows["fastapi"]["version"] and all("licence" in r for r in rows.values())
