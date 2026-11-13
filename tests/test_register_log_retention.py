"""The Register's server keeps no log line past 30 days and copies none on to rsyslog."""

import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "register-server.sh"


def test_the_journal_drop_in_bounds_every_line_at_30_days():
    """The drop-in sorts after rsyslog's syslog.conf, bounds the journal by age, stops forwarding, and is applied."""
    text = SCRIPT.read_text()
    m = re.search(r"cat > /etc/systemd/journald\.conf\.d/(\S+) <<'CONF'\n(.*?)\nCONF", text, re.S)
    assert m, "the journald drop-in is written"
    name, body = m.group(1), m.group(2).splitlines()
    assert name > "syslog.conf", "rsyslog's own drop-in would win"
    days = int(re.fullmatch(r"MaxRetentionSec=(\d+)day", body[1]).group(1))
    file_days = int(re.fullmatch(r"MaxFileSec=(\d+)day", body[2]).group(1))
    assert body[0] == "[Journal]" and days + file_days <= 30 and "ForwardToSyslog=no" in body
    assert text.index(m.group(0)) < text.index("systemctl restart systemd-journald") < text.index("systemctl restart register.service")


def test_the_files_rsyslog_and_the_sign_in_records_keep_no_more_than_30_days():
    """Every logrotate file the script installs turns its logs over daily, empty or not, and keeps at most 28."""
    folder = SCRIPT.parent / "logrotate"
    names = {"rsyslog", "ufw", "fail2ban", "btmp", "wtmp"}
    assert {p.name for p in folder.iterdir()} == names
    for path in folder.iterdir():
        words = [line.split() for line in path.read_text().splitlines() if line.strip()]
        # An empty log is still turned over: only a turn-over drops the archive past the count.
        assert ["daily"] in words and not any(w[0] in ("weekly", "monthly", "yearly", "minsize", "notifempty") for w in words), path.name
        assert [int(w[1]) for w in words if w[0] == "rotate"] and all(int(w[1]) <= 28 for w in words if w[0] == "rotate"), path.name
    text = SCRIPT.read_text()
    assert 'install -m 644 "$(dirname "$0")"/logrotate/{rsyslog,ufw,fail2ban,btmp,wtmp} /etc/logrotate.d/' in text
