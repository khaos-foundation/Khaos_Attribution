#!/usr/bin/env bash
# One consistent copy of the register's two databases and its mirror files, kept locally for 30 days and sent to
# BACKUP_TARGET over ssh when one is set. Run as the register user; the environment comes from /etc/register.env.
set -euo pipefail
DATA="${REGISTER_DATA_DIR:?}"
cd "$(dirname "$DATA")"
OUT="$(dirname "$DATA")/backups"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"
for db in register index; do
  [ -f "$DATA/$db.sqlite3" ] && sqlite3 "$DATA/$db.sqlite3" ".backup '$WORK/$db.sqlite3'"
done
[ -d "$DATA/mirror" ] && cp -a "$DATA/mirror" "$WORK/mirror"
# The marker that this register started its log: an empty log is never opened on "nothing published" once it exists.
[ -f "$DATA/log-started" ] && cp -a "$DATA/log-started" "$WORK/log-started"
# The signing key is never in a backup: the Guardians hold its backup.
tar -C "$WORK" -czf "$OUT/register-$STAMP.tar.gz" .
sha256sum "$OUT/register-$STAMP.tar.gz" > "$OUT/register-$STAMP.tar.gz.sha256"
find "$OUT" -name 'register-*.tar.gz*' -mtime +30 -delete
if [ -n "${BACKUP_TARGET:-}" ]; then
  rsync -a -e "ssh -p ${BACKUP_PORT:-22} -o StrictHostKeyChecking=accept-new" "$OUT/" "$BACKUP_TARGET/"
fi
echo "backed up: $OUT/register-$STAMP.tar.gz"
