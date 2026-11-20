#!/usr/bin/env bash
# The short key ceremony, run as root on the register server with both directors present.
#   register-key.sh make                     a new key beside the development one; the service keeps running
#   register-key.sh switch "Name, Name"      retire the development key, sign with the new one, record who was there
# Between the two, register-key-backup.sh locks the key to the directors' YubiKeys; nothing else holds it unencrypted.
set -euo pipefail
ENV_FILE="${REGISTER_ENV_FILE:-/etc/register.env}"
SYSTEMCTL="${SYSTEMCTL:-systemctl}"
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
set -a; . "$ENV_FILE"; set +a
DATA="${REGISTER_DATA_DIR:-/srv/register/data}"
KEY="${REGISTER_KEY_FILE:-$DATA/register-key.pem}"
STAGED="$DATA/staged/register-key.pem"
PY="${REGISTER_PY:-/srv/register/venv/bin/python}"
PORT="${REGISTER_PORT:-8912}"

# A development key must never be promoted by deleting the setting alone: both steps need it still set.
grep -q '^REGISTER_DEV_KEY=1' "$ENV_FILE" || { echo "REGISTER_DEV_KEY=1 is not set: the production key is already in place"; exit 1; }

case "${1:-}" in
make)
  [ -e "$STAGED" ] && { echo "$STAGED exists: a key is made once; switch to it, or remove it to start again"; exit 1; }
  install -d -o register -g register -m 700 "$(dirname "$STAGED")"
  sudo -u register "$PY" -m register_service make-key "$STAGED"
  chmod 600 "$STAGED"
  echo
  echo "sha256 of the private key file, for the backup stick and the sheet:"
  sha256sum "$STAGED" | cut -d' ' -f1
  echo
  echo "On the laptop, lock it to both YubiKeys and check each one opens it (deploy/key-ceremony.md, step 3):"
  echo "  deploy/register-key-backup.sh lock register-key.age <recipient-1> <recipient-2>"
  echo "  deploy/register-key-backup.sh check register-key.age yubikey-<name>.txt"
  echo "Then: register-key switch \"Name, Name\""
  ;;
switch)
  PRESENT="${2:?who was there, comma-separated}"
  [ -f "$STAGED" ] || { echo "no staged key: run register-key.sh make first"; exit 1; }
  [ -f "$KEY" ] || { echo "no development key at $KEY to retire; this server is not in the state the ceremony expects"; exit 1; }
  NEW_ID="$(sudo -u register "$PY" -c 'import sys; from khaos_attribution.registry import SigningKey; print(SigningKey.load(sys.argv[1]).key_id)' "$STAGED")"
  RETIRED="$DATA/retired/dev-$(date -u +%Y%m%dT%H%M%SZ)"
  rollback() {
    echo "switch failed: back on the development key"
    $SYSTEMCTL stop register.service || true
    [ -f "$RETIRED.pem" ] && mv -f "$RETIRED.pem" "$KEY" && mv -f "$RETIRED.pub" "${KEY%.pem}.pub" 2>/dev/null || true
    grep -q '^REGISTER_DEV_KEY=1' "$ENV_FILE" || echo 'REGISTER_DEV_KEY=1' >> "$ENV_FILE"
    grep -q '^REGISTER_DEV_KEY_PUBLIC=' "$ENV_FILE" || echo 'REGISTER_DEV_KEY_PUBLIC=1' >> "$ENV_FILE"
    $SYSTEMCTL start register.service
  }
  trap rollback ERR
  $SYSTEMCTL stop register.service
  install -d -o register -g register -m 700 "$DATA/retired"
  mv "$KEY" "$RETIRED.pem"; mv "${KEY%.pem}.pub" "$RETIRED.pub" 2>/dev/null || true
  install -o register -g register -m 600 "$STAGED" "$KEY"
  install -o register -g register -m 644 "${STAGED%.pem}.pub" "${KEY%.pem}.pub"
  sed -i '/^REGISTER_DEV_KEY=/d' "$ENV_FILE"
  # Serving a development key behind the proxy was said for that key; with none left, the setting goes too.
  grep -q '^REGISTER_LOG_DEV_KEY=1' "$ENV_FILE" || sed -i '/^REGISTER_DEV_KEY_PUBLIC=/d' "$ENV_FILE"
  $SYSTEMCTL start register.service
  for _ in $(seq 1 20); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
  SERVED="$(curl -fsS "http://127.0.0.1:$PORT/api/public-key")"
  echo "$SERVED" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d["register_key_id"]==sys.argv[1] and d["development"] is False else 1)' "$NEW_ID"
  trap - ERR
  sudo -u register bash -c 'set -a; . "$1"; set +a; cd "$(dirname "$2")" && exec "$2" -m register_service key --present "$3" --note "first ceremony"' -- "$ENV_FILE" "$PY" "$PRESENT" \
    || LEDGER_FAILED=1
  rm -rf "$(dirname "$STAGED")"
  [ -z "${LEDGER_FAILED:-}" ] || { echo "the register signs with $NEW_ID but its ledger entry failed; run: register-cli key --present \"$PRESENT\" --note \"first ceremony\""; exit 1; }
  echo "The register now signs with key $NEW_ID; the development key is kept, unused, at $RETIRED.pem"
  ;;
*)
  sed -n '2,5p' "$0"; exit 1 ;;
esac
