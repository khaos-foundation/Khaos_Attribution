#!/usr/bin/env bash
# The Guild Register on one small Ubuntu 24.04 server, run once as root and safe to run again:
# updates on a timer, a firewall open on 22, 80 and 443, logs kept 30 days, a service user with no shell, the register under
# systemd on loopback port 8912, Caddy in front of it with HTTPS, a nightly backup to a storage box.
#   REGISTER_DOMAIN=register.khaos.dev REGISTER_TAG=vX.Y.Z bash <a clone checked out at vX.Y.Z>/deploy/register-server.sh
# It refuses to go on when it is not that tag's own script, run from root's clone. A register whose signed log never
# started keeps it off, serving every other door, until REGISTER_LOG_ENABLE=1 is given: the owner's decision.
# Until the Guardians' key exists the register signs with a development key and says so on every page;
# the ceremony replaces /srv/register/data/register-key.pem and clears REGISTER_DEV_KEY in /etc/register.env.
# The signed append-only log has its own key and its own ceremony (register-log-key), and two timers anchor its root.
set -euo pipefail

DOMAIN="${REGISTER_DOMAIN:?set REGISTER_DOMAIN to the name people will open}"
TAG="${REGISTER_TAG:?set REGISTER_TAG to the release tag to deploy, e.g. v0.58.0}"
REPO="${REGISTER_REPO:-https://github.com/khaos-foundation/Khaos_Attribution.git}"
USER_NAME=register
HOME_DIR=/srv/register
CHECKOUT="$HOME_DIR/Khaos_Attribution"
VENV="$HOME_DIR/venv"
DATA="$HOME_DIR/data"
ENV_FILE=/etc/register.env
PORT=8912
HERE="$(dirname -- "$(readlink -f -- "$0")")"
ROOT_CLONE_STEPS="  git clone $REPO /root/Khaos_Attribution   (once)
  git -C /root/Khaos_Attribution fetch --tags && git -C /root/Khaos_Attribution checkout --detach $TAG
  REGISTER_DOMAIN=$DOMAIN REGISTER_TAG=$TAG bash /root/Khaos_Attribution/deploy/register-server.sh"
# The register's own checkout is checked out anew below, under this very script: it is never the one that runs.
if [ "$HERE" = "$(readlink -f -- "$CHECKOUT/deploy" 2>/dev/null || echo "$CHECKOUT/deploy")" ]; then
  echo "Run the setup from root's own clone of the tag, never from $CHECKOUT. Nothing was changed. As root:"
  echo "$ROOT_CLONE_STEPS"
  exit 1
fi
# uv is fetched at this version and checked against Astral's published SHA-256 before it runs: the values are the
# release's own uv-<arch>-unknown-linux-gnu.tar.gz.sha256 files at https://github.com/astral-sh/uv/releases/tag/0.11.32
UV_VERSION=0.11.32
UV_SHA256_X86_64=aab924fd522efd06f1c5f3b93a243864fc453132c94b2dc49f1371b528a4b967
UV_SHA256_AARCH64=4d4fa08d95b06642e5800df6a22bd71455f23f988269e18da2847971d8c0bf31
case "$(uname -m)" in
  x86_64) UV_ARCH=x86_64; UV_SHA256=$UV_SHA256_X86_64 ;;
  aarch64|arm64) UV_ARCH=aarch64; UV_SHA256=$UV_SHA256_AARCH64 ;;
  *) echo "no pinned uv for $(uname -m)"; exit 1 ;;
esac

[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
export DEBIAN_FRONTEND=noninteractive

echo "== packages"
apt-get update -q
apt-get install -y -q -o Dpkg::Options::=--force-confold unattended-upgrades ufw fail2ban git curl sqlite3 rsync debian-keyring debian-archive-keyring apt-transport-https ca-certificates
dpkg-reconfigure -f noninteractive unattended-upgrades

echo "== ssh: keys only"
cat > /etc/ssh/sshd_config.d/90-register.conf <<SSHD
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin prohibit-password
MaxAuthTries 3
SSHD
sshd -t && systemctl reload ssh

echo "== firewall"
ufw --force reset >/dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
systemctl enable --now fail2ban

echo "== caddy"
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -q && apt-get install -y -q caddy
fi

echo "== service user"
id -u "$USER_NAME" >/dev/null 2>&1 || useradd --system --home-dir "$HOME_DIR" --create-home --shell /usr/sbin/nologin "$USER_NAME"
mkdir -p "$DATA" "$HOME_DIR/backups"
chown -R "$USER_NAME:$USER_NAME" "$HOME_DIR"
chmod 750 "$HOME_DIR"

echo "== uv $UV_VERSION and the code at $TAG"
sudo -u "$USER_NAME" -H env UV_VERSION="$UV_VERSION" UV_ARCH="$UV_ARCH" UV_SHA256="$UV_SHA256" bash -c '
  set -euo pipefail
  cd "$HOME"
  if [ "$("$HOME/.local/bin/uv" --version 2>/dev/null | cut -d" " -f2)" != "$UV_VERSION" ]; then
    tgz="uv-$UV_ARCH-unknown-linux-gnu.tar.gz"; tmp="$(mktemp -d)"
    curl -fsSL -o "$tmp/$tgz" "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$tgz"
    echo "$UV_SHA256  $tmp/$tgz" | sha256sum -c --quiet
    tar -xzf "$tmp/$tgz" -C "$tmp"
    mkdir -p "$HOME/.local/bin"; cp "$tmp/uv-$UV_ARCH-unknown-linux-gnu/uv" "$HOME/.local/bin/uv"; chmod 755 "$HOME/.local/bin/uv"
    rm -rf "$tmp"
  fi
  if [ -d "'"$CHECKOUT"'/.git" ]; then git -C "'"$CHECKOUT"'" fetch -q --tags --force origin; else git clone -q "'"$REPO"'" "'"$CHECKOUT"'"; fi
  git -C "'"$CHECKOUT"'" checkout -q --detach "'"$TAG"'"
'

echo "== this script is the tag's own"
# Every unit, helper and setting below comes from this script and its folder: they must be the tag's, not an older clone's.
bash "$HERE/register-same-tag.sh" "$HERE" "$CHECKOUT/deploy"

echo "== the signed log"
# Started once its marker or a first entry exists, as the service reads it; until then off unless the owner says so.
LOG_ENTRIES="$(sqlite3 -readonly "$DATA/register.sqlite3" "SELECT COUNT(*) FROM log_leaves" 2>/dev/null || echo 0)"
if [ -f "$DATA/log-started" ] || [ "${LOG_ENTRIES:-0}" -gt 0 ]; then
  echo "the signed log has started here; it goes on"
elif [ "${REGISTER_LOG_ENABLE:-}" = 1 ] || grep -qs '^REGISTER_LOG_ENABLE=1' "$ENV_FILE"; then
  echo "REGISTER_LOG_ENABLE=1: the service starts the signed log when it next starts; /health says if it could not"
else
  echo "the signed log stays off; every other door is served. REGISTER_LOG_ENABLE=1 starts it, the owner's decision"
fi

echo "== the venv from the lock"
sudo -u "$USER_NAME" -H env UV_PROJECT_ENVIRONMENT="$VENV" bash -c '
  set -euo pipefail
  export PATH="$HOME/.local/bin:$PATH"
  cd "'"$CHECKOUT"'"
  uv sync --frozen --no-editable --extra registry --extra service --python 3.12 --quiet
'
# What the register runs, by name, version and licence, beside it.
(cd "$HOME_DIR" && sudo -u "$USER_NAME" "$VENV/bin/python" -m register_service bill-of-materials) > "$HOME_DIR/bill-of-materials.json"

echo "== logs: 30 days"
# No journal line outlives 30 days (a file goes once its first line is 29 days old; files turn over daily). Named to sort
# after rsyslog's syslog.conf, whose forwarding would keep a five-week copy in /var/log/syslog. The Space holds the same file.
mkdir -p /etc/systemd/journald.conf.d
install -m 644 "$CHECKOUT/deploy/journald-retention.conf" /etc/systemd/journald.conf.d/zz-retention.conf
systemctl restart systemd-journald
# The files rsyslog, the firewall and fail2ban still write, and the sign-in records, turn over daily and keep 28.
install -m 644 "$CHECKOUT"/deploy/logrotate/{rsyslog,ufw,fail2ban,btmp,wtmp} /etc/logrotate.d/

echo "== settings"
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<ENV
REGISTER_DATA_DIR=$DATA
REGISTER_KEY_FILE=$DATA/register-key.pem
REGISTER_PUBLIC_URL=https://$DOMAIN
# 1 until the Guardians' key is placed at REGISTER_KEY_FILE by the ceremony; then delete both lines and restart.
REGISTER_DEV_KEY=1
# The development key is reachable at REGISTER_PUBLIC_URL through Caddy, by decision; nothing it signs is a Guild record.
REGISTER_DEV_KEY_PUBLIC=1
# The log's own key, which signs tree heads; 1 until register-log-key switch places the Guild's.
REGISTER_LOG_KEY_FILE=$DATA/log-key.pem
REGISTER_LOG_DEV_KEY=1
# Days a new log entry's body waits before it reaches the public mirror's git history (0 to 7); a withhold inside it never lands there.
REGISTER_LOG_PUBLIC_DELAY_DAYS=7
# Published copies that witness the log beside this server's own, comma-separated folders or URLs. With none readable the
# register appends nothing unless its log is new and nothing anywhere was published (fail closed).
REGISTER_LOG_WITNESSES=https://raw.githubusercontent.com/guild-register/ledger/main
# The storage box for the nightly backup, as user@host:path over ssh (Hetzner: port 23); empty keeps backups local only.
BACKUP_TARGET=
BACKUP_PORT=23
# The public ledger mirror's repository (ssh address), pushed with ~register/.ssh/ledger_deploy; empty mirrors nothing.
LEDGER_REPO=
ENV
  chmod 640 "$ENV_FILE"; chown root:"$USER_NAME" "$ENV_FILE"
fi
# A settings file from before the exposure check keeps its development key serving: the decision it already made.
if grep -q '^REGISTER_DEV_KEY=1' "$ENV_FILE" && ! grep -q '^REGISTER_DEV_KEY_PUBLIC=' "$ENV_FILE"; then
  echo "REGISTER_DEV_KEY_PUBLIC=1" >> "$ENV_FILE"
fi
# The owner's word that the log may start is kept, so a later run and the service itself both read it.
if [ "${REGISTER_LOG_ENABLE:-}" = 1 ] && ! grep -q '^REGISTER_LOG_ENABLE=1' "$ENV_FILE"; then
  sed -i '/^REGISTER_LOG_ENABLE=/d' "$ENV_FILE"
  echo "REGISTER_LOG_ENABLE=1" >> "$ENV_FILE"
fi
# A settings file from before the log gets the log's lines: its key is a development one until its own ceremony.
if ! grep -q '^REGISTER_LOG_DEV_KEY=' "$ENV_FILE"; then
  grep -q '^REGISTER_LOG_KEY_FILE=' "$ENV_FILE" || echo "REGISTER_LOG_KEY_FILE=$DATA/log-key.pem" >> "$ENV_FILE"
  echo "REGISTER_LOG_DEV_KEY=1" >> "$ENV_FILE"
  grep -q '^REGISTER_DEV_KEY_PUBLIC=' "$ENV_FILE" || echo "REGISTER_DEV_KEY_PUBLIC=1" >> "$ENV_FILE"
fi

echo "== systemd"
cat > /etc/systemd/system/register.service <<UNIT
[Unit]
Description=The Guild Register
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_NAME
Group=$USER_NAME
EnvironmentFile=$ENV_FILE
WorkingDirectory=$HOME_DIR
ExecStart=$VENV/bin/python -m register_service serve --host 127.0.0.1 --port $PORT
Restart=on-failure
RestartSec=3
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DATA
PrivateTmp=true
PrivateDevices=true
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
LockPersonality=true
MemoryDenyWriteExecute=true

[Install]
WantedBy=multi-user.target
UNIT

install -m 755 "$CHECKOUT/deploy/register-backup.sh" /usr/local/bin/register-backup
cat > /etc/systemd/system/register-backup.service <<UNIT
[Unit]
Description=Back up the Guild Register's data

[Service]
Type=oneshot
User=$USER_NAME
EnvironmentFile=$ENV_FILE
ExecStart=/usr/local/bin/register-backup
UNIT
cat > /etc/systemd/system/register-backup.timer <<UNIT
[Unit]
Description=Nightly backup of the Guild Register

[Timer]
OnCalendar=*-*-* 03:30:00
RandomizedDelaySec=15m
Persistent=true

[Install]
WantedBy=timers.target
UNIT

install -m 755 "$CHECKOUT/deploy/register-mirror.sh" /usr/local/bin/register-mirror
install -m 755 "$CHECKOUT/deploy/register-key.sh" /usr/local/sbin/register-key
install -m 755 "$CHECKOUT/deploy/register-log-key.sh" /usr/local/sbin/register-log-key
cat > /etc/systemd/system/register-mirror.service <<UNIT
[Unit]
Description=Push the Guild Register's public mirror

[Service]
Type=oneshot
User=$USER_NAME
EnvironmentFile=$ENV_FILE
ExecStart=/usr/local/bin/register-mirror
UNIT
cat > /etc/systemd/system/register-mirror.timer <<UNIT
[Unit]
Description=The Guild Register's public mirror, every quarter hour

[Timer]
OnCalendar=*:0/15
RandomizedDelaySec=2m
Persistent=true

[Install]
WantedBy=timers.target
UNIT

for job in stamp upgrade; do
cat > /etc/systemd/system/register-log-$job.service <<UNIT
[Unit]
Description=The Guild Register's log: $job (stamp appends the day's heartbeat entry, then anchors the root with OpenTimestamps)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=$USER_NAME
EnvironmentFile=$ENV_FILE
WorkingDirectory=$HOME_DIR
ExecStart=$VENV/bin/python -m register_service log-$job
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DATA
PrivateTmp=true
UNIT
done
cat > /etc/systemd/system/register-log-stamp.timer <<UNIT
[Unit]
Description=Anchor the Guild Register's log root once a day

[Timer]
OnCalendar=*-*-* 02:10:00
RandomizedDelaySec=20m
Persistent=true

[Install]
WantedBy=timers.target
UNIT
cat > /etc/systemd/system/register-log-upgrade.timer <<UNIT
[Unit]
Description=Complete the Guild Register's pending anchor receipts, four times a day

[Timer]
OnCalendar=*-*-* 00/6:40:00
RandomizedDelaySec=10m
Persistent=true

[Install]
WantedBy=timers.target
UNIT

echo "== keys for the backup and the mirror (made once; the public halves are printed at the end)"
sudo -u "$USER_NAME" -H bash -c '
  set -euo pipefail
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  [ -f "$HOME/.ssh/id_ed25519" ] || ssh-keygen -q -t ed25519 -N "" -C "register-backup" -f "$HOME/.ssh/id_ed25519"
  [ -f "$HOME/.ssh/ledger_deploy" ] || ssh-keygen -q -t ed25519 -N "" -C "guild-register-ledger-deploy" -f "$HOME/.ssh/ledger_deploy"
'

echo "== caddy site"
cat > /etc/caddy/Caddyfile <<CADDY
$DOMAIN {
	encode zstd gzip
	# A twelve-minute track's four-phase claim runs to about 13 MB; the code's own cap is 24 MB. Each request meets one
	# limit: two request_body handlers on one request both apply, and the smaller wins.
	@claims path /api/claims /api/claims/check
	@rest not path /api/claims /api/claims/check
	request_body @claims {
		max_size 25MB
	}
	request_body @rest {
		max_size 10MB
	}
	header {
		Strict-Transport-Security "max-age=31536000; includeSubDomains"
		X-Content-Type-Options nosniff
		X-Frame-Options DENY
		Referrer-Policy strict-origin-when-cross-origin
		Content-Security-Policy "default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
		-Server
	}
	reverse_proxy 127.0.0.1:$PORT {
		header_down -Server
	}
}
CADDY
# A site file Caddy would not load stops the run here, before any service is restarted on it.
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile

systemctl daemon-reload
systemctl enable register.service register-backup.timer register-mirror.timer register-log-stamp.timer register-log-upgrade.timer
systemctl start register-backup.timer register-mirror.timer register-log-stamp.timer register-log-upgrade.timer
# The restart moves an index from before version 2 to the new layout; the copy kept here as it was is what a rollback
# to an earlier tag restores (an earlier register would read the moved file as empty).
INDEX_DB="$DATA/index.sqlite3"
if [ -f "$INDEX_DB" ] && [ "$(sqlite3 -readonly "$INDEX_DB" 'PRAGMA user_version' 2>/dev/null || echo 0)" -lt 2 ]; then
  KEPT="$INDEX_DB.before-v2-$(date -u +%Y%m%dT%H%M%SZ)"
  sqlite3 "$INDEX_DB" ".backup '$KEPT'"
  chown "$USER_NAME:$USER_NAME" "$KEPT"
  echo "index: its copy before the move to version 2 is $KEPT; a rollback to an earlier tag restores it"
fi
# A re-run serves the tag it just put in place: a running service is restarted, not left on the old code.
systemctl restart register.service
systemctl restart caddy
sleep 2
systemctl --no-pager --quiet is-active register.service && echo "register: running" || { journalctl -u register.service --no-pager -n 30; exit 1; }
curl -fsS "http://127.0.0.1:$PORT/health" && echo
echo
echo "Backup key (install on the storage box):   $(cat $HOME_DIR/.ssh/id_ed25519.pub)"
echo "Ledger deploy key (add to the repository, write access):   $(cat $HOME_DIR/.ssh/ledger_deploy.pub)"
cat > /usr/local/bin/register-cli <<CLI
#!/usr/bin/env bash
# The register's commands as the service user with its settings: register-cli account <name> | invite <name> [--role space|warden] | check | key ...
exec sudo -u $USER_NAME bash -c 'set -a; . $ENV_FILE; set +a; cd $HOME_DIR && exec $VENV/bin/python -m register_service "\$@"' -- "\$@"
CLI
chmod 755 /usr/local/bin/register-cli
echo "Done. Next: the first account and invites:"
echo "  register-cli account <name>"
echo "  register-cli invite <name> --role workshop"
