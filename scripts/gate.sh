#!/usr/bin/env bash
# Runs this repository's gates: those a change touches (--changed), named ones or all, or a Stop hook's (--hook).

ALL_GATES="pytest"

gate_cmd() {
  case "$1" in
    pytest) echo "$VENV/python -m pytest -q -p no:cacheprovider" ;;
    *) return 1 ;;
  esac
}

# Changed paths on stdin, the gates they need on stdout.
gates_for() {
  grep -qE '^(src/|tests/|pyproject\.toml$|uv\.lock$)' && echo pytest
}

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.." || exit 2
VENV=.venv/bin
[ -d "$VENV" ] || VENV=.venv/Scripts
GITDIR=$(git rev-parse --git-common-dir) || exit 2
LOCK="$GITDIR/khaos-gate.lock"
GREEN="$GITDIR/khaos-gate.green"

usage() {
  echo "usage: scripts/gate.sh [all | <gate>... | --changed | --hook] [--budget SECONDS]"
  echo "gates: $ALL_GATES"
}

changed_paths() {
  local base
  base=$(git merge-base HEAD origin/main 2>/dev/null) || base=HEAD
  { git diff --name-only "$base" --; git ls-files --others --exclude-standard; } | sort -u
}

state_key() {
  {
    echo "$*"
    git rev-parse HEAD
    git diff HEAD --
    git ls-files --others --exclude-standard | while IFS= read -r f; do git hash-object -- "$f"; done
  } 2> /dev/null | git hash-object --stdin
}

take_lock() {
  if ! mkdir "$LOCK" 2> /dev/null; then
    local pid
    pid=$(cat "$LOCK/pid" 2> /dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2> /dev/null; then return 1; fi
    rm -rf "$LOCK"
    mkdir "$LOCK" 2> /dev/null || return 1
  fi
  echo $$ > "$LOCK/pid"
  trap 'rm -rf "$LOCK"' EXIT
}

# macOS has no timeout(1); perl runs the gate in its own process group so the whole group can be stopped.
with_timeout() {
  local secs=$1
  shift
  if [ "$secs" -le 0 ]; then "$@"; return; fi
  if command -v timeout > /dev/null; then timeout "$secs" "$@"; return; fi
  if command -v gtimeout > /dev/null; then gtimeout "$secs" "$@"; return; fi
  perl -e '$t = shift; $p = fork; if (!$p) { setpgrp(0, 0); exec @ARGV; exit 127 }
    $SIG{ALRM} = sub { kill "TERM", -$p; sleep 2; kill "KILL", -$p; exit 124 }; alarm $t;
    waitpid($p, 0); exit($? & 127 ? 128 + ($? & 127) : $? >> 8)' "$secs" "$@"
}

run_gates() {
  local failed=0 skipped=0 start left cmd log rc g
  local deadline=$(( $(date +%s) + BUDGET ))
  for g in "$@"; do
    left=0
    if [ "$BUDGET" -gt 0 ]; then
      left=$(( deadline - $(date +%s) ))
      if [ "$left" -le 0 ]; then echo "SKIP $g: out of time"; skipped=1; continue; fi
    fi
    cmd=$(gate_cmd "$g")
    echo "== gate $g: $cmd"
    log=$(mktemp)
    start=$(date +%s)
    with_timeout "$left" bash -c "$cmd" > "$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
      summary=$(grep -E '([0-9]+ passed|^Results:)' "$log" | tail -1 | sed -E 's/=+//g; s/^ +//; s/ +$//')
      echo "PASS $g ($(( $(date +%s) - start ))s)${summary:+: $summary}"
      grep -E '^[[:space:]]*SKIP:' "$log" | head -8 | sed -E 's/^[[:space:]]*SKIP: */  skipped inside: /'
    else
      echo "FAIL $g ($(( $(date +%s) - start ))s, exit $rc)"
      tail -60 "$log"
      failed=1
    fi
    rm -f "$log"
  done
  [ "$failed" -eq 1 ] && return 1
  [ "$skipped" -eq 1 ] && return 3
  return 0
}

hook_answer() {
  local py=$VENV/python
  [ -x "$py" ] || py=$(command -v python3 || command -v python) || return 0
  "$py" -I -S -c '
import json, sys
red, path = sys.argv[1] == "1", sys.argv[2]
try:
    active = bool(json.loads(sys.argv[3] or "{}").get("stop_hook_active"))
except ValueError:
    active = False
text = open(path, encoding="utf-8", errors="replace").read().strip()
if not text:
    sys.exit(0)
lines = text.splitlines()
summary = "; ".join(l for l in lines if l.startswith(("== gate ", "PASS ", "FAIL ", "SKIP ")))
if red and not active:
    print(json.dumps({"decision": "block", "systemMessage": summary,
                      "reason": "A gate is red. Fix it before stopping:\n" + "\n".join(lines[-150:])}))
elif red:
    print(json.dumps({"systemMessage": "Still red after a fix attempt; not blocking again. " + summary}))
else:
    print(json.dumps({"systemMessage": summary}))
' "$1" "$2" "$3"
}

MODE=names
BUDGET=0
NAMES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --changed) MODE=changed ;;
    --hook) MODE=hook ;;
    --budget) BUDGET=${2:-0}; shift ;;
    -h|--help) usage; exit 0 ;;
    *) NAMES+=("$1") ;;
  esac
  shift
done

case "$MODE" in
  hook)
    input=$(cat)
    out=$(mktemp)
    bash "$SELF" --changed --budget 570 > "$out" 2>&1
    rc=$?
    red=0
    [ "$rc" -eq 1 ] && red=1
    hook_answer "$red" "$out" "$input"
    rm -f "$out"
    exit 0 ;;
  changed)
    gates=$(changed_paths | gates_for | tr '\n' ' ')
    gates=${gates% }
    [ -z "$gates" ] && exit 0
    key=$(state_key "$gates")
    if [ -f "$GREEN" ] && grep -qxF "$key" "$GREEN"; then echo "PASS $gates: unchanged since its last green run"; exit 0; fi
    take_lock || { echo "SKIP $gates: another gate run holds this repository's lock"; exit 3; }
    # shellcheck disable=SC2086
    run_gates $gates
    rc=$?
    if [ "$rc" -eq 0 ]; then
      { tail -49 "$GREEN" 2> /dev/null; echo "$key"; } > "$GREEN.tmp" && mv "$GREEN.tmp" "$GREEN"
    fi
    exit "$rc" ;;
  names)
    [ ${#NAMES[@]} -eq 0 ] && { usage; exit 2; }
    expanded=()
    for g in "${NAMES[@]}"; do
      if [ "$g" = all ]; then read -r -a every <<< "$ALL_GATES"; expanded+=("${every[@]}"); else expanded+=("$g"); fi
    done
    NAMES=("${expanded[@]}")
    for g in "${NAMES[@]}"; do gate_cmd "$g" > /dev/null || { echo "no gate named $g"; usage; exit 2; }; done
    take_lock || { echo "another gate run holds this repository's lock"; exit 3; }
    run_gates "${NAMES[@]}"
    exit $? ;;
esac
