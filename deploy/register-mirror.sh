#!/usr/bin/env bash
# The public ledger mirror: the register's mirror files copied into a clone of the public repository and pushed
# when anything changed. Run as the register user; LEDGER_REPO (ssh address) and the deploy key come from
# /etc/register.env and ~/.ssh/ledger_deploy. Nothing but mirror files ever goes in.
set -euo pipefail
DATA="${REGISTER_DATA_DIR:?}"
cd "$(dirname "$DATA")"
REPO="${LEDGER_REPO:-}"
CLONE="$(dirname "$DATA")/ledger-mirror"
[ -n "$REPO" ] || { echo "LEDGER_REPO is not set; nothing mirrored"; exit 0; }
export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/ledger_deploy -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
if [ ! -d "$CLONE/.git" ]; then
  git clone -q "$REPO" "$CLONE"
  git -C "$CLONE" config user.name "Guild Register"
  git -C "$CLONE" config user.email "register@khaos.dev"
fi
git -C "$CLONE" fetch -q origin
default="$(git -C "$CLONE" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#origin/##' || true)"
[ -n "$default" ] || default=main
git -C "$CLONE" checkout -q -B "$default" "origin/$default" 2>/dev/null || git -C "$CLONE" checkout -q -B "$default"
mkdir -p "$DATA/mirror/models"
rsync -a --delete --exclude .git --exclude README.md "$DATA/mirror/" "$CLONE/"
if [ ! -f "$CLONE/README.md" ]; then
  printf '# The Guild Register, mirrored\n\nEvery registered model as a JSON record under `models/`, the hash-chained ledger as `ledger.jsonl`, and the signed append-only log under `log/` (signed tree heads, entries, inclusion and consistency proofs, the log key, anchor receipts), enough for `khaos-log-verify --mirror <this folder>`. Written by the register at https://register.khaos.dev and pushed here; nothing is edited by hand.\n' > "$CLONE/README.md"
fi
git -C "$CLONE" add -A
if git -C "$CLONE" diff --cached --quiet; then
  echo "mirror unchanged"
  exit 0
fi
entries=0
[ -f "$DATA/mirror/ledger.jsonl" ] && entries="$(wc -l < "$DATA/mirror/ledger.jsonl")"
git -C "$CLONE" commit -q -m "Ledger at $entries entries, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git -C "$CLONE" push -q origin "$default"
echo "mirror pushed: $entries ledger entries"
