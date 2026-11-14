#!/usr/bin/env bash
# Runs this repository's gates: those a change touches (--changed), named ones or all, or the quick check a Stop runs
# (--quick, which --hook answers with): only the test files for what changed since its last run that was not red.

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
# Every gate reads this checkout's code: a worktree shares a venv whose editable install is the main checkout's.
export PYTHONPATH="$PWD/src"
VENV=.venv/bin
[ -d "$VENV" ] || VENV=.venv/Scripts
GITDIR=$(git rev-parse --git-common-dir) || exit 2
LOCK="$GITDIR/khaos-gate.lock"
GREEN="$GITDIR/khaos-gate.green"
# Per worktree: each checkout remembers its own last quick run, and the test files a quick run had to skip.
QUICK="$(git rev-parse --absolute-git-dir)/khaos-gate.quick"
OWED="$QUICK.owed"
# The gate that runs every test file: only a green run of it, or of --changed, settles what is owed.
SUITE_GATE=pytest
# A Stop waits at most this long; a change that names more test files waits for --changed before the PR.
QUICK_SECONDS=90
QUICK_MAX_FILES=25
QUICK_PYTEST="$VENV/python -m pytest -q -p no:cacheprovider"

usage() {
  echo "usage: scripts/gate.sh [all | <gate>... | --changed | --quick | --hook] [--budget SECONDS]"
  echo "gates: $ALL_GATES"
}

changed_paths() {
  local base
  base=$(git merge-base HEAD origin/main 2>/dev/null) || base=HEAD
  { git diff --name-only "$base" --; git ls-files --others --exclude-standard; } | sort -u
}

# Each changed file once, as "<blob> <path>" ("gone <path>" when deleted).
branch_files() {
  local p
  changed_paths | while IFS= read -r p; do
    if [ -f "$p" ]; then printf '%s %s\n' "$(git hash-object -- "$p")" "$p"; else printf 'gone %s\n' "$p"; fi
  done
}

# Changed paths on stdin; on stdout, each once, a changed test file itself or the test files that name a changed file.
quick_tests() {
  local p base
  while IFS= read -r p; do
    case "$p" in
      archive/*|docs/*|reports/*|*.md) continue ;;
      tests/*test_*.py) [ -f "$p" ] && echo "$p"; continue ;;
    esac
    base=${p##*/}
    case "$base" in
      *.py) git grep --untracked -l -w -F -e "$base" -e "import ${base%.py}" -e "from ${base%.py}" -- 'tests/*.py' ;;
      *) git grep --untracked -l -w -F -e "$base" -- 'tests/*.py' ;;
    esac 2> /dev/null
  done | grep -E '(^|/)test_[^/]*\.py$' | awk '!seen[$0]++'
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

# The test files a quick run had to skip, those that still exist.
owed_tests() {
  local t
  [ -s "$OWED" ] && while IFS= read -r t; do [ -f "$t" ] && printf '%s\n' "$t"; done < "$OWED"
}

# Runs them in one pytest call: only their own green run takes them off the list.
run_owed() {
  local tests log rc start
  tests=$(owed_tests | tr '\n' ' ')
  echo "== owed: $QUICK_PYTEST $tests"
  log=$(mktemp)
  start=$(date +%s)
  bash -c "$QUICK_PYTEST $tests" > "$log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 5 ]; then
    echo "PASS owed ($(( $(date +%s) - start ))s): $(grep -E '[0-9]+ passed' "$log" | tail -1 | sed -E 's/=+//g; s/^ +//; s/ +$//')"
    rm -f "$OWED"
    rc=0
  else
    echo "FAIL owed ($(( $(date +%s) - start ))s, exit $rc)"
    tail -60 "$log"
    rc=1
  fi
  rm -f "$log"
  return "$rc"
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
    --quick) MODE=quick ;;
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
    bash "$SELF" --quick > "$out" 2>&1
    rc=$?
    red=0
    [ "$rc" -eq 1 ] && red=1
    hook_answer "$red" "$out" "$input"
    rm -f "$out"
    exit 0 ;;
  quick)
    # The test files a quick run had to skip are owed: every later run counts them until they have run green.
    owe() { { cat "$OWED" 2> /dev/null; printf '%s\n' "$tests"; } | sort -u > "$OWED.tmp" && mv "$OWED.tmp" "$OWED"; }
    finish() {
      if [ -s "$OWED" ]; then
        owed_tests > "$OWED.tmp"
        mv "$OWED.tmp" "$OWED"
        owed=$(wc -l < "$OWED" | tr -d ' ')
        [ "$owed" -gt 0 ] && echo "SKIP quick: $owed test files were never run; scripts/gate.sh --changed runs them"
      fi
      exit "$1"
    }
    now=$(branch_files)
    fresh=$(comm -23 <(printf '%s\n' "$now" | sort -u) <(sort -u "$QUICK" 2> /dev/null) | cut -d' ' -f2-)
    [ -z "$fresh" ] && finish 0
    remember() { printf '%s\n' "$now" > "$QUICK.tmp" && mv "$QUICK.tmp" "$QUICK"; }
    tests=$(printf '%s\n' "$fresh" | quick_tests)
    [ -z "$tests" ] && { remember; finish 0; }
    count=$(printf '%s\n' "$tests" | wc -l | tr -d ' ')
    if [ "$count" -gt "$QUICK_MAX_FILES" ]; then
      echo "SKIP quick: $count test files name the change; gate.sh --changed checks it before the PR"
      remember
      owe
      finish 0
    fi
    take_lock || { echo "SKIP quick: another gate run holds this repository's lock"; finish 0; }
    secs=$QUICK_SECONDS
    [ "$BUDGET" -gt 0 ] && [ "$BUDGET" -lt "$secs" ] && secs=$BUDGET
    log=$(mktemp)
    start=$(date +%s)
    with_timeout "$secs" bash -c "$QUICK_PYTEST $(printf '%s' "$tests" | tr '\n' ' ')" > "$log" 2>&1
    rc=$?
    took=$(( $(date +%s) - start ))
    summary=$(grep -E '[0-9]+ (passed|failed)' "$log" | tail -1 | sed -E 's/=+//g; s/^ +//; s/ +$//')
    case "$rc" in
      0|5) echo "PASS quick (${took}s, $count test files)${summary:+: $summary}"; remember; rc=0 ;;
      124)
        echo "SKIP quick: over ${secs}s ($count test files); gate.sh --changed checks it before the PR"
        remember
        owe
        rc=0 ;;
      *) echo "FAIL quick (${took}s, exit $rc): $(printf '%s' "$tests" | tr '\n' ' ')"; tail -60 "$log"; rc=1 ;;
    esac
    rm -f "$log"
    finish "$rc" ;;
  changed)
    gates=$(changed_paths | gates_for | tr '\n' ' ')
    gates=${gates% }
    owed=$(owed_tests)
    [ -z "$gates" ] && [ -z "$owed" ] && exit 0
    key=$(state_key "$gates")
    green=0
    if [ -n "$gates" ] && [ -f "$GREEN" ] && grep -qxF "$key" "$GREEN"; then
      echo "PASS $gates: unchanged since its last green run"
      green=1
      [ -z "$owed" ] && exit 0
    fi
    take_lock || { echo "SKIP ${gates:-owed}: another gate run holds this repository's lock"; exit 3; }
    rc=0
    if [ -n "$gates" ] && [ "$green" -eq 0 ]; then
      # shellcheck disable=SC2086
      run_gates $gates
      rc=$?
      if [ "$rc" -eq 0 ]; then
        { tail -49 "$GREEN" 2> /dev/null; echo "$key"; } > "$GREEN.tmp" && mv "$GREEN.tmp" "$GREEN"
      fi
    fi
    # What the quick check skipped is run itself, unless the suite gate covered it: another gate may not hold it.
    if [ "$rc" -eq 0 ] && [ -n "$owed" ]; then
      case " $gates " in *" $SUITE_GATE "*) rm -f "$OWED" ;; *) run_owed; rc=$? ;; esac
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
    rc=$?
    [ "$rc" -eq 0 ] && case " ${NAMES[*]} " in *" $SUITE_GATE "*) rm -f "$OWED" ;; esac
    exit "$rc" ;;
esac
