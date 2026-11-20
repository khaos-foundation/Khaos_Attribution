#!/usr/bin/env bash
# Refuses a setup run whose deploy folder is not the deployed tag's own: the units, helpers and settings it writes come
# from that folder, so an older clone's would put an older register's scripts around the new code.
#   register-same-tag.sh <the running script's folder> <the tag's checkout>/deploy
set -euo pipefail
RUNNING="${1:?the folder the running setup script is in}"
TAGGED="${2:?the deploy folder of the checkout at the tag}"
if ! diff -rq "$RUNNING" "$TAGGED" > /dev/null 2>&1; then
  echo "This setup script is not the one at the tag it deploys. These differ:"
  diff -rq "$RUNNING" "$TAGGED" | sed 's/^/  /' || true
  echo "Nothing was changed. Run the tag's own script, as root, with the same settings:"
  echo "  bash $TAGGED/register-server.sh"
  exit 1
fi
