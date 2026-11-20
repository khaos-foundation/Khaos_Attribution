#!/usr/bin/env bash
# Refuses a setup run whose deploy folder is not the deployed tag's own: the units, helpers and settings it writes come
# from that folder, so an older clone's would put an older register's scripts around the new code.
#   register-same-tag.sh <the running script's folder> <the tag's checkout>/deploy
set -euo pipefail
RUNNING="${1:?the folder the running setup script is in}"
TAGGED="${2:?the deploy folder of the checkout at the tag}"
STEPS="Check the tag out in root's own clone and run that copy, as root, with the same settings:
  git -C /root/Khaos_Attribution fetch --tags && git -C /root/Khaos_Attribution checkout --detach <tag>
  REGISTER_DOMAIN=<domain> REGISTER_TAG=<tag> bash /root/Khaos_Attribution/deploy/register-server.sh"
# Compared with itself the folders always agree: the register's own checkout is never the script that runs.
if [ "$(cd "$RUNNING" && pwd -P)" = "$(cd "$TAGGED" && pwd -P)" ]; then
  echo "This setup script runs from the checkout it deploys, so it cannot tell it is the tag's own."
  echo "Stopped before the new code was installed. $STEPS"
  exit 1
fi
if ! diff -rq "$RUNNING" "$TAGGED" > /dev/null 2>&1; then
  echo "This setup script is not the one at the tag it deploys. These differ:"
  diff -rq "$RUNNING" "$TAGGED" | sed 's/^/  /' || true
  echo "Stopped before the new code was installed. $STEPS"
  exit 1
fi
