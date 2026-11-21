#!/usr/bin/env bash
# Runs this repository's gates: those a change touches (--changed), named ones or all, or the quick check a Stop runs
# (--quick, which --hook answers with): only the test files for what changed since its last run that was not red.

ALL_GATES="pytest"
# Named only, never in all, --changed or --quick: the release needs it, and a machine without torch must not go red.
NAMED_GATES="torch"
# Python files under src/ and scripts/ that no test names yet, known: exact paths only, and the list only shrinks
# (gate.sh --stale fails on a path that got a test or a new glob).
NO_TEST_OK="scripts/watermark_survival.py src/khaos_attribution/engines/mimetic/convert_worker.py
  src/khaos_attribution/engines/sa3/convert_worker.py src/khaos_attribution/seal/__main__.py src/register_service/bom.py"

gate_cmd() {
  local py
  case "$1" in
    pytest) echo "$VENV/python -m pytest -q -p no:cacheprovider" ;;
    torch)
      # Without torch the suite skips the torch tests, so a missing interpreter is a FAIL, never a SKIP.
      if py=$(torch_python); then
        printf 'KHAOS_TORCH_RUN=1 %q -m pytest -q -p no:cacheprovider\n' "$py"
      elif [ -n "$KHAOS_TORCH_PYTHON" ]; then
        echo "echo 'KHAOS_TORCH_PYTHON names no interpreter with all of: $TORCH_RUN_NEEDS pytest'; exit 1"
      else
        echo "echo 'no interpreter with all of $TORCH_RUN_NEEDS pytest in the checkout venv or on PATH; set KHAOS_TORCH_PYTHON to one'; exit 1"
      fi ;;
    *) return 1 ;;
  esac
}

# Every package the torch run needs; tests/conftest.py's TORCH_RUN_NEEDS names the same, and a test holds them equal.
TORCH_RUN_NEEDS="torch torchaudio librosa huggingface_hub soundfile"

has_torch() {
  [ -x "$1" ] && "$1" -c 'import importlib.util as u, sys; sys.exit(not all(u.find_spec(m) for m in sys.argv[1:]))' \
    $TORCH_RUN_NEEDS pytest 2> /dev/null
}

# KHAOS_TORCH_PYTHON when set, else the first of this checkout's venv, python3 and python on PATH that has them.
torch_python() {
  local py
  if [ -n "$KHAOS_TORCH_PYTHON" ]; then
    has_torch "$KHAOS_TORCH_PYTHON" && echo "$KHAOS_TORCH_PYTHON"
    return
  fi
  for py in "$VENV/python" "$(command -v python3)" "$(command -v python)"; do
    [ -n "$py" ] && has_torch "$py" && { echo "$py"; return 0; }
  done
  return 1
}

# Changed paths on stdin, the gates they need on stdout.
gates_for() {
  local paths named
  paths=$(cat)
  # A path some test file names is that test's to check, as --quick checks it.
  named=$(printf '%s\n' "$paths" | quick_tests)
  if [ -n "$named" ] || printf '%s\n' "$paths" | grep -qE '^(src/|tests/|pyproject\.toml$|uv\.lock$)'; then
    echo pytest
  fi
  return 0
}

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.." || exit 2
# Every gate reads this checkout's code: a worktree shares a venv whose editable install is the main checkout's.
export PYTHONPATH="$PWD/src"
# Only the torch gate sets it, on its own command: left in a shell it would change every other gate's skip rule.
unset KHAOS_TORCH_RUN
VENV=.venv/bin
[ -d "$VENV" ] || VENV=.venv/Scripts
GITDIR=$(git rev-parse --git-common-dir) || exit 2
LOCK="$GITDIR/khaos-gate.lock"
GREEN="$GITDIR/khaos-gate.green"
# Per worktree: each checkout remembers its own last quick run, and the test files a quick run had to skip.
WORKTREE_GITDIR=$(git rev-parse --absolute-git-dir) || exit 2
QUICK="$WORKTREE_GITDIR/khaos-gate.quick"
OWED="$QUICK.owed"
# The pins this branch turned red ("<holder> <test id>"), and those a run had to skip ("<holder> <why>"), one pair of
# files per branch: each later stop on it names them, and only a run of them settles them.
# A branch is named by its readable name and a hash of it, so a/b and a_b keep apart; a detached HEAD by its commit.
PINS_BRANCH=$(git symbolic-ref --short -q HEAD) || PINS_BRANCH="HEAD $(git rev-parse --short HEAD 2> /dev/null)"
PINS_RED="$QUICK.pins.$(printf '%s' "$PINS_BRANCH" | LC_ALL=C tr -c 'A-Za-z0-9._-' '_').$(printf '%s' "$PINS_BRANCH" | git hash-object --stdin | cut -c1-8)"
PINS_OWED="$PINS_RED.owed"
# The gate that runs every test file: only a green run of it, or of --changed, settles what is owed.
SUITE_GATE=pytest
# A Stop waits at most this long; a change that names more test files waits for --changed before the PR.
QUICK_SECONDS=90
QUICK_MAX_FILES=25
QUICK_PYTEST="$VENV/python -m pytest -q -p no:cacheprovider"

usage() {
  echo "usage: scripts/gate.sh [all | <gate>... | --changed | --plan | --quick | --hook | --unnamed | --stale] [--budget SECONDS]"
  echo "gates: $ALL_GATES; by name only: $NAMED_GATES"
  echo "pins: the tests a repository beside this one holds on it, which --changed and --quick also run"
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

# Changed paths on stdin; on stdout, each once, the test files for them. A changed test file itself, and the files
# that import from it; under a changed conftest.py, every test file it reaches; for a doc, the test files that name it
# or its folder, and the repository's test_docs_current files; for any other file, the test files that name it by its
# file name, its module name or its dotted path under src/, also as `from <package> import (... <name> ...)`.
quick_tests() {
  local p base mod item parent name f
  local -a pats=() parents=()
  {
    while IFS= read -r p; do
      base=${p##*/}
      case "$p" in
        ''|*/|archive/*|reports/*|docs/research/*) continue ;;
        conftest.py|*/conftest.py)
          git ls-files -co --exclude-standard -- "${p%conftest.py}" 2> /dev/null
          continue ;;
        tests/*test_*.py)
          [ -f "$p" ] || continue
          echo "$p"
          pats+=(-e "from ${base%.py} import" -e "import ${base%.py}")
          continue ;;
        docs/*|*.md)
          pats+=(-e "$base" -e "$p")
          case "$p" in docs/*/*) pats+=(-e "${p%/*}") ;; esac
          git ls-files -co --exclude-standard -- 'tests/*test_docs_current*.py' 2> /dev/null
          continue ;;
      esac
      mod=""
      # A package's __init__.py is named by its file: its package name would select every importer of the package.
      case "$p" in */__init__.py) ;; src/*.py) mod=$(printf '%s' "${p#src/}" | sed -e 's#/#.#g' -e 's#\.py$##') ;; esac
      case "$base" in
        *.py) pats+=(-e "$base" -e "import ${base%.py}" -e "from ${base%.py}" ${mod:+-e "$mod"}) ;;
        *) pats+=(-e "$base") ;;
      esac
      case "$mod" in *.*) parents+=("${mod%.*} ${mod##*.}") ;; esac
    done
    # One search for every pattern: a quick run's time goes to its tests.
    [ "${#pats[@]}" -eq 0 ] || git grep --untracked -l -w -F "${pats[@]}" -- 'tests/*.py' 2> /dev/null
    for item in ${parents[@]+"${parents[@]}"}; do
      parent=${item% *}
      name=${item#* }
      git grep --untracked -l -F "from $parent import" -- 'tests/*.py' 2> /dev/null | while IFS= read -r f; do
        grep -q -w -F -- "$name" "$f" && echo "$f"
      done
    done
  } | grep -E '(^|/)test_[^/]*\.py$' | awk '!seen[$0]++'
}

# Changed paths on stdin; on stdout the Python files under src/ and scripts/ that no test file names, which a quick run
# cannot check, less those this repository's NO_TEST_OK globs say are known to have none.
unnamed_sources() {
  local p glob skip
  local -a globs=()
  read -r -d "" -a globs <<< "${NO_TEST_OK:-} ${NO_TEST_OK_GLOBS:-}" || true
  while IFS= read -r p; do
    case "$p" in src/*.py|scripts/*.py) [ -f "$p" ] || continue ;; *) continue ;; esac
    skip=0
    for glob in ${globs[@]+"${globs[@]}"}; do
      # shellcheck disable=SC2254
      case "$p" in $glob) skip=1 ;; esac
    done
    [ "$skip" -eq 0 ] && [ -z "$(printf '%s\n' "$p" | quick_tests)" ] && echo "$p"
  done
  return 0
}

# Paths on stdin; on stdout those the glob $1 matches that no test file names.
unnamed_under() {
  local p
  while IFS= read -r p; do
    # shellcheck disable=SC2254
    case "$p" in $1) [ -z "$(printf '%s\n' "$p" | quick_tests)" ] && echo "$p" ;; esac
  done
  return 0
}

# The venv's packages and the contract's file list: a venv another checkout moved is not the one that ran green.
venv_packages() {
  local site
  for site in .venv/lib/python*/site-packages .venv/Lib/site-packages; do
    [ -d "$site" ] || continue
    (cd "$site" && ls -d -- *.dist-info; cat -- khaos_attribution-*.dist-info/RECORD) 2> /dev/null
  done
}

# One line when the venv's contract is not this checkout's pin: its tests then fail for the venv, not the change.
contract_note() {
  local file pin="" have
  for file in src/common/contract.py src/listening_space/contract.py; do
    [ -f "$file" ] && pin=$(sed -n 's/^CONTRACT_VERSION = "\(.*\)"$/\1/p' "$file") && break
  done
  have=$(venv_packages | sed -n 's/^khaos_attribution-\(.*\)\.dist-info$/\1/p' | head -1)
  if [ -n "$pin" ] && [ -n "$have" ] && [ "$pin" != "$have" ]; then
    echo "NOTE contract: the venv holds khaos_attribution $have, this checkout pins $pin; re-run the installer (never uv sync)"
  fi
}

state_key() {
  {
    echo "$*"
    git rev-parse HEAD
    git diff HEAD --
    git ls-files --others --exclude-standard | while IFS= read -r f; do git hash-object -- "$f"; done
    venv_packages
  } 2> /dev/null | git hash-object --stdin
}

take_lock() {
  if ! mkdir "$LOCK" 2> /dev/null; then
    local pid
    pid=$(cat "$LOCK/pid" 2> /dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2> /dev/null; then return 1; fi
    rm -rf "${LOCK:?}"
    mkdir "$LOCK" 2> /dev/null || return 1
  fi
  echo $$ > "$LOCK/pid"
  trap 'rm -rf "${LOCK:?}"' EXIT
}

# The test files a quick run had to skip, those that still exist.
owed_tests() {
  local t
  [ -s "$OWED" ] && while IFS= read -r t; do [ -f "$t" ] && printf '%s\n' "$t"; done < "$OWED"
}

# Runs test files with $QUICK_PYTEST, one pytest process per top-level test folder: in one process a module name is
# imported once, so two folders' modules of the same name (a conftest.py, a helper imported by name) collide. Each
# red folder's FAILED and ERROR lines come again at the end, then the folders' counts summed; the exit is the highest
# red folder's (5, nothing collected, only when every folder has none). Output is held in variables: no temp file to leak.
pytest_by_folder() {
  local folder files out one worst=0 red=0 ran=0 counts="" failures=""
  for folder in $(printf '%s\n' "$@" | awk -F/ '{ print (NF > 2 ? $1 "/" $2 : $1) }' | awk '!seen[$0]++'); do
    files=$(printf '%s\n' "$@" | awk -F/ -v f="$folder" '(NF > 2 ? $1 "/" $2 : $1) == f' | tr '\n' ' ')
    out=$(bash -c "$QUICK_PYTEST $files" 2>&1)
    one=$?
    printf '%s\n' "$out"
    counts="$counts$(printf '%s\n' "$out" | grep -E '[0-9]+ (passed|failed|errors?|skipped)' | tail -1)
"
    case "$one" in
      0) ran=1 ;;
      5) ;;
      *)
        [ "$one" -eq 1 ] && red=1
        [ "$one" -gt "$worst" ] && worst=$one
        failures="$failures$folder (exit $one)
$(printf '%s\n' "$out" | grep -E '^(FAILED|ERROR) ' | head -20)
"
        ;;
    esac
  done
  [ -n "$failures" ] && printf 'red folders:\n%s' "$failures"
  printf '%s passed, %s failed, %s skipped, %s errors (one pytest per folder)\n' \
    "$(printf '%s' "$counts" | grep -oE '[0-9]+ passed' | awk '{ n += $1 } END { print n + 0 }')" \
    "$(printf '%s' "$counts" | grep -oE '[0-9]+ failed' | awk '{ n += $1 } END { print n + 0 }')" \
    "$(printf '%s' "$counts" | grep -oE '[0-9]+ skipped' | awk '{ n += $1 } END { print n + 0 }')" \
    "$(printf '%s' "$counts" | grep -oE '[0-9]+ errors?' | awk '{ n += $1 } END { print n + 0 }')"
  [ "$red" -eq 1 ] && return 1
  [ "$worst" -ne 0 ] && return "$worst"
  [ "$ran" -eq 1 ] && return 0
  return 5
}
export -f pytest_by_folder
export QUICK_PYTEST

# Runs them, a pytest per folder: only their own green run takes them off the list.
run_owed() {
  local tests log rc start
  tests=$(owed_tests | tr '\n' ' ')
  echo "== owed: $QUICK_PYTEST $tests"
  log=$(mktemp)
  start=$(date +%s)
  # shellcheck disable=SC2086
  pytest_by_folder $tests > "$log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 5 ]; then
    echo "PASS owed ($(( $(date +%s) - start ))s): $(grep -E '[0-9]+ passed' "$log" | tail -1 | sed -E 's/=+//g; s/^ +//; s/ +$//')"
    rm -f "${OWED:?}"
    rc=0
  else
    echo "FAIL owed ($(( $(date +%s) - start ))s, exit $rc)"
    tail -60 "$log"
    rc=1
  fi
  rm -f "${log:?}"
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

# Changed paths on stdin; on stdout those a pin could read: not archive/, docs/, reports/ or a .md file.
code_paths() {
  grep -vE '^(archive|docs|reports)/|\.md$'
}

# The SIBLING_PINS entries the gate.sh at $1 declares, one per line: `<repo>[,<repo>]:<test file or id>`, and
# `!<repo>[,<repo>]:<test id>` for a test that repository's own gate keeps.
pins_declared() {
  [ -f "$1" ] && awk '/^SIBLING_PINS="/ { on = 1; sub(/^SIBLING_PINS="/, "") }
    on { last = sub(/"[[:space:]]*$/, ""); n = split($0, words, /[[:space:]]+/)
         for (i = 1; i <= n; i++) if (words[i] != "") print words[i]
         if (last) exit }' "$1"
}

# Each repository beside this checkout whose gate.sh holds a pin on it, as its folder.
pin_holders() {
  local ws me folder entry
  ws=$(cd .. && pwd)
  me=$(basename "$PWD")
  for folder in "$ws"/*/; do
    folder=${folder%/}
    [ "$(basename "$folder")" = "$me" ] && continue
    while IFS= read -r entry; do
      case "$entry" in '!'*) continue ;; esac
      case ",${entry%%:*}," in *",$me,"*) echo "$folder"; break ;; esac
    done < <(pins_declared "$folder/scripts/gate.sh")
  done
}

# Sets the lines of the record $1 for the holder $2 to the rest of the arguments, one line each; none clears them.
record_set() {
  local file=$1 holder=$2 item
  shift 2
  { grep -v "^$holder " "$file" 2> /dev/null; for item in "$@"; do printf '%s %s\n' "$holder" "$item"; done; } > "$file.tmp"
  mv "$file.tmp" "$file"
  [ -s "$file" ] || rm -f "${file:?}"
  # The branch a record belongs to, so the records of a branch that is gone can be found and cleared.
  if [ -s "$PINS_RED" ] || [ -s "$PINS_OWED" ]; then
    printf '%s\n' "$PINS_BRANCH" > "$PINS_RED.branch"
  else
    rm -f "${PINS_RED:?}.branch"
  fi
}

# The record $1's entries for the holder $2, one per line.
record_get() {
  grep "^$2 " "$1" 2> /dev/null | cut -d' ' -f2-
}

# Clears the records of a holder no longer beside this checkout, or holding no pin on it, saying so once.
pins_prune() {
  local holders holder named branch
  # Records of a branch that is gone, or of a detached HEAD left four whole days ago (-mtime +3), go with it.
  for named in "$QUICK".pins.*.branch; do
    [ -f "$named" ] && [ "$named" != "$PINS_RED.branch" ] || continue
    case "$named" in "${QUICK:?}".pins.*.branch) ;; *) continue ;; esac
    branch=$(cat "$named")
    case "$branch" in
      "HEAD "*) [ -n "$(find "$named" -mtime +3 2> /dev/null)" ] || continue ;;
      *) git show-ref --verify -q "refs/heads/$branch" && continue ;;
    esac
    rm -f "${named:?}" "${named%.branch}" "${named%.branch}.owed"
  done
  [ -s "$PINS_RED" ] || [ -s "$PINS_OWED" ] || return 0
  holders=$(pin_holders | while IFS= read -r holder; do basename "$holder"; done)
  for holder in $(cut -d' ' -f1 "$PINS_RED" "$PINS_OWED" 2> /dev/null | sort -u); do
    printf '%s\n' "$holders" | grep -qxF "$holder" && continue
    record_set "$PINS_RED" "$holder"
    record_set "$PINS_OWED" "$holder"
    echo "SKIP pins $holder: no longer beside this checkout holding pins on it; its record is cleared"
  done
}

# Each line of what the pins records still hold, as a stop names it.
pins_reminder() {
  local holder ids why
  pins_prune
  if [ -s "$PINS_RED" ]; then
    cut -d' ' -f1 "$PINS_RED" | sort -u | while IFS= read -r holder; do
      ids=$(record_get "$PINS_RED" "$holder" | tr '\n' ' ')
      echo "SKIP pins $holder: ${ids}red since an earlier run on this branch; scripts/gate.sh --changed shows it"
    done
  fi
  if [ -s "$PINS_OWED" ]; then
    while IFS=' ' read -r holder why; do
      echo "SKIP pins $holder: not run ($why); scripts/gate.sh --changed runs them"
    done < "$PINS_OWED"
  fi
}

# The test ids a pytest_by_folder run reported red, each once.
red_ids() {
  grep -E '^(FAILED|ERROR) ' | awk '{ print $2 }' | sort -u
}

# Of the pins $2 (test ids) the repository at $1 holds on this one, those red on this checkout's committed base too,
# run beside the same siblings within $3 seconds (0: no limit). Fails when that run could not be made.
red_before() {
  local holder=$1 ids=$2 secs=$3 ws me base tmp sibling out rc began
  ws=$(cd .. && pwd)
  me=$(basename "$PWD")
  base=$(git merge-base HEAD origin/main 2> /dev/null) || base=HEAD
  began=$(date +%s)
  tmp=$(mktemp -d) || return 1
  mkdir "$tmp/$me"
  # Where a link is made as a copy (Git Bash on Windows), the copy would hold a venv and every sibling: not made.
  if ! ln -s "$tmp/$me" "$tmp/link-probe" 2> /dev/null || [ ! -L "$tmp/link-probe" ]; then
    rm -rf "${tmp:?}"
    return 1
  fi
  rm -f "${tmp:?}/link-probe"
  if with_timeout "$secs" bash -c 'git archive --format=tar "$1" | tar -x -C "$2"' _ "$base" "$tmp/$me"; then
    # The copy holds what git tracks; a pin may also read the venv beside it, which no change to the files moves.
    [ -e .venv ] && ln -s "$PWD/.venv" "$tmp/$me/.venv"
    if [ "$secs" -gt 0 ]; then
      secs=$(( secs - ($(date +%s) - began) ))
      [ "$secs" -lt 1 ] && secs=1
    fi
    for sibling in "$ws"/*/; do
      sibling=${sibling%/}
      if [ "$(basename "$sibling")" != "$me" ] && [ -e "$sibling/.git" ]; then ln -s "$sibling" "$tmp/$(basename "$sibling")"; fi
    done
    out=$(bash "$holder/scripts/gate.sh" --pins-for "$me" --workspace "$tmp" --only "$ids" --budget "$secs" < /dev/null 2>&1)
    rc=$?
  else
    rc=2
  fi
  rm -rf "${tmp:?}"
  case "$rc" in
    0|5) return 0 ;;
    1) printf '%s\n' "$out" | red_ids ;;
    *) return 1 ;;
  esac
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
    if [ "$g" = pins ]; then cmd="bash $(printf %q "$SELF") --pins"; else cmd=$(gate_cmd "$g"); fi
    echo "== gate $g: $cmd"
    log=$(mktemp)
    start=$(date +%s)
    with_timeout "$left" bash -c "$cmd" > "$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
      summary=$(grep -E '([0-9]+ passed|^Results:)' "$log" | tail -1 | sed -E 's/=+//g; s/^ +//; s/ +$//')
      echo "PASS $g ($(( $(date +%s) - start ))s)${summary:+: $summary}"
      grep -E '^[[:space:]]*SKIP:' "$log" | head -8 | sed -E 's/^[[:space:]]*SKIP: */  skipped inside: /'
      [ "$g" = pins ] && grep -E '^SKIP pins ' "$log" | head -8 | sed 's/^/  /'
    elif [ "$g" = pins ] && [ "$rc" -eq 3 ]; then
      echo "SKIP pins ($(( $(date +%s) - start ))s): a pin held on this repository did not run, or was red before this change"
      grep -E '^SKIP pins ' "$log" | head -8 | sed 's/^/  /'
      skipped=1
    else
      echo "FAIL $g ($(( $(date +%s) - start ))s, exit $rc)"
      tail -60 "$log"
      failed=1
    fi
    rm -f "${log:?}"
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
PLAN_ONLY=""
STALE_ONLY=""
TARGET=""
WORKSPACE=""
ONLY=""
NAMES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --changed) MODE=changed ;;
    --quick) MODE=quick ;;
    --hook) MODE=hook ;;
    --budget) BUDGET=${2:-0}; shift ;;
    --pins) MODE=pins ;;
    --plan) MODE=changed; PLAN_ONLY=1 ;;
    --unnamed) MODE=unnamed ;;
    --stale) MODE=unnamed; STALE_ONLY=1 ;;
    --pins-for) MODE=pins_for; TARGET=${2:-}; shift ;;
    --workspace) WORKSPACE=${2:-}; shift ;;
    --only) ONLY=${2:-}; shift ;;
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
    rm -f "${out:?}"
    exit 0 ;;
  quick)
    # The test files a quick run had to skip are owed: every later run counts them until they have run green.
    owe() { { cat "$OWED" 2> /dev/null; printf '%s\n' "$1"; } | grep -v '^$' | sort -u > "$OWED.tmp" && mv "$OWED.tmp" "$OWED"; }
    finish() {
      if [ -s "$OWED" ]; then
        owed_tests > "$OWED.tmp"
        mv "$OWED.tmp" "$OWED"
        owed=$(wc -l < "$OWED" | tr -d ' ')
        [ "$owed" -gt 0 ] && echo "SKIP quick: $owed test files were never run; scripts/gate.sh --changed runs them"
      fi
      [ "$pins_ran" -eq 0 ] && pins_reminder
      exit "$1"
    }
    # The pins siblings hold on this repository run after its own tests, inside the same seconds. Only a pin newly red
    # on this branch blocks, once; the pins run keeps the records later stops name.
    pins_then_finish() {
      local left out prc holder
      if [ -z "$(printf '%s\n' "$fresh" | code_paths)" ] || [ -z "$(pin_holders)" ]; then finish "$1"; fi
      left=$(( secs - ($(date +%s) - began) ))
      if [ "$left" -lt 3 ]; then
        echo "SKIP pins: the stop's seconds are spent; scripts/gate.sh --changed runs them"
        while IFS= read -r holder; do
          record_set "$PINS_OWED" "$(basename "$holder")" "no time left in a stop"
        done < <(pin_holders)
        finish "$1"
      fi
      out=$(bash "$SELF" --pins --budget "$left" < /dev/null 2>&1)
      prc=$?
      pins_ran=1
      [ -n "$out" ] && printf '%s\n' "$out"
      [ "$prc" -eq 1 ] && finish 1
      finish "$1"
    }
    pins_ran=0
    began=$(date +%s)
    secs=$QUICK_SECONDS
    [ "$BUDGET" -gt 0 ] && [ "$BUDGET" -lt "$secs" ] && secs=$BUDGET
    now=$(branch_files)
    fresh=$(comm -23 <(printf '%s\n' "$now" | sort -u) <(sort -u "$QUICK" 2> /dev/null) | cut -d' ' -f2-)
    remember() { printf '%s\n' "$now" > "$QUICK.tmp" && mv "$QUICK.tmp" "$QUICK"; }
    if [ -z "$fresh" ]; then
      # A change undone counts for a pin it had turned red: the pins run again to see.
      undone=$(comm -13 <(printf '%s\n' "$now" | sort -u) <(sort -u "$QUICK" 2> /dev/null) | cut -d' ' -f2-)
      if { [ -s "$PINS_RED" ] || [ -s "$PINS_OWED" ]; } && [ -n "$undone" ]; then
        fresh=$undone
        remember
        pins_then_finish 0
      fi
      finish 0
    fi
    tests=$(printf '%s\n' "$fresh" | quick_tests)
    unnamed=$(printf '%s\n' "$fresh" | unnamed_sources)
    if [ -n "$unnamed" ]; then
      n=$(printf '%s\n' "$unnamed" | wc -l | tr -d ' ')
      shown=$(printf '%s\n' "$unnamed" | head -5 | tr '\n' ' ')
      [ "$n" -gt 5 ] && shown="${shown}and $((n - 5)) more "
      echo "SKIP quick: no test file names ${shown% }; gate.sh --changed runs the suite and fails until a test names it or NO_TEST_OK lists it"
    fi
    [ -z "$tests" ] && { remember; pins_then_finish 0; }
    count=$(printf '%s\n' "$tests" | wc -l | tr -d ' ')
    if [ "$count" -gt "$QUICK_MAX_FILES" ]; then
      # Over the cap the changed test files still run; the files that only import from them are owed.
      own=$(printf '%s\n' "$fresh" | grep -E '^tests/(.*/)?test_[^/]*\.py$' | while IFS= read -r t; do [ -f "$t" ] && echo "$t"; done)
      n_own=$(printf '%s' "$own" | grep -c . || true)
      if [ "$n_own" -eq 0 ] || [ "$n_own" -gt "$QUICK_MAX_FILES" ]; then
        echo "SKIP quick: $count test files name the change; gate.sh --changed checks it before the PR"
        remember
        owe "$tests"
        pins_then_finish 0
      fi
      echo "SKIP quick: $((count - n_own)) more test files name the change; gate.sh --changed checks them before the PR"
      owe "$(comm -23 <(printf '%s\n' "$tests" | sort -u) <(printf '%s\n' "$own" | sort -u))"
      tests=$own
      count=$n_own
    fi
    take_lock || { echo "SKIP quick: another gate run holds this repository's lock"; finish 0; }
    log=$(mktemp)
    start=$(date +%s)
    left=$(( secs - ($(date +%s) - began) ))
    [ "$left" -lt 1 ] && left=1
    with_timeout "$left" bash -c "pytest_by_folder $(printf '%s' "$tests" | tr '\n' ' ')" > "$log" 2>&1
    rc=$?
    took=$(( $(date +%s) - start ))
    summary=$(grep -E '[0-9]+ (passed|failed)' "$log" | tail -1 | sed -E 's/=+//g; s/^ +//; s/ +$//')
    case "$rc" in
      0|5) echo "PASS quick (${took}s, $count test files)${summary:+: $summary}"; remember; rc=0 ;;
      124)
        echo "SKIP quick: over ${secs}s ($count test files); gate.sh --changed checks it before the PR"
        remember
        owe "$tests"
        rc=0 ;;
      *) echo "FAIL quick (${took}s, exit $rc): $(printf '%s' "$tests" | tr '\n' ' ')"; tail -60 "$log"; rc=1 ;;
    esac
    rm -f "${log:?}"
    [ "$rc" -eq 0 ] && pins_then_finish 0
    finish "$rc" ;;
  changed)
    gates=$(changed_paths | gates_for | tr '\n' ' ')
    gates=${gates% }
    owed=$(owed_tests)
    unnamed=$(changed_paths | unnamed_sources)
    if [ -n "$unnamed" ]; then
      # The suite holds every test; the named tests it would run again are dropped.
      case " $gates " in
        *" $SUITE_GATE "*) ;;
        *) gates=$(printf '%s' " $gates " | sed 's/ named / /g; s/^ *//; s/ *$//'); gates="${gates:+$gates }$SUITE_GATE"; NAMED_TESTS="" ;;
      esac
    fi
    pins=""
    if [ -n "$(changed_paths | code_paths)" ] && [ -n "$(pin_holders)" ]; then pins=pins; fi
    if [ -n "$PLAN_ONLY" ]; then
      echo "gates: ${gates:-none}"
      echo "named: ${NAMED_TESTS:-none}"
      echo "owed: $(printf '%s' "$owed" | tr '\n' ' ')"
      echo "pins: ${pins:-none}"
      echo "unnamed: $(printf '%s' "$unnamed" | tr '\n' ' ')"
      exit 0
    fi
    contract_note
    if [ -z "$gates" ] && [ -z "$owed" ] && [ -z "$pins" ]; then
      echo "SKIP changed: no changed path maps to a gate or is named by a test; nothing to run"
      exit 0
    fi
    key=$(state_key "$gates")
    green=0
    if [ -n "$gates" ] && [ -f "$GREEN" ] && grep -qxF "$key" "$GREEN"; then
      echo "PASS $gates: unchanged since its last green run"
      green=1
      [ -z "$owed" ] && [ -z "$pins" ] && [ -z "$unnamed" ] && exit 0
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
      case " $gates " in *" $SUITE_GATE "*) rm -f "${OWED:?}" ;; *) run_owed; rc=$? ;; esac
    fi
    if [ -n "$unnamed" ]; then
      printf '%s\n' "$unnamed" | sed 's/^/FAIL changed: no test file names /; s/$/; write one, or add it to NO_TEST_OK/'
      rc=1
    fi
    # The pins siblings hold on this repository read them too, so a green run of them is never remembered.
    if [ -n "$pins" ]; then
      run_gates pins
      prc=$?
      if [ "$prc" -eq 1 ]; then rc=1; elif [ "$prc" -ne 0 ] && [ "$rc" -eq 0 ]; then rc=3; fi
    fi
    exit "$rc" ;;
  pins)
    me=$(basename "$PWD")
    ws=$(cd .. && pwd)
    began=$(date +%s)
    rc=0
    worse() { [ "$rc" -eq 0 ] && rc=$1; }
    pins_prune
    while IFS= read -r holder; do
      name=$(basename "$holder")
      left=0
      if [ "$BUDGET" -gt 0 ]; then
        left=$(( BUDGET - ($(date +%s) - began) ))
        if [ "$left" -lt 2 ]; then
          echo "SKIP pins $name: no time left; scripts/gate.sh --changed runs them"
          record_set "$PINS_OWED" "$name" "no time left"
          worse 3
          continue
        fi
      fi
      start=$(date +%s)
      out=$(bash "$holder/scripts/gate.sh" --pins-for "$me" --workspace "$ws" --budget "$left" < /dev/null 2>&1)
      hrc=$?
      took=$(( $(date +%s) - start ))
      case "$hrc" in
        0|5)
          echo "PASS pins $name (${took}s): $(printf '%s\n' "$out" | tail -1)"
          record_set "$PINS_RED" "$name"
          record_set "$PINS_OWED" "$name" ;;
        1)
          ids=$(printf '%s\n' "$out" | red_ids)
          [ "$BUDGET" -gt 0 ] && left=$(( BUDGET - ($(date +%s) - began) ))
          if [ -z "$ids" ]; then
            echo "SKIP pins $name (${took}s): red without naming a test; scripts/gate.sh --changed shows it"
            printf '%s\n' "$out" | tail -30
            record_set "$PINS_OWED" "$name" "red without naming a test"
            worse 3
          elif [ "$BUDGET" -gt 0 ] && [ "$left" -lt 2 ]; then
            echo "SKIP pins $name (${took}s): red, unconfirmed: no time left to run them on this branch's base: $(printf '%s ' $ids)"
            record_set "$PINS_OWED" "$name" "red, unconfirmed against this branch's base"
            worse 3
          elif ! before=$(red_before "$holder" "$(printf '%s ' $ids)" "$left"); then
            echo "SKIP pins $name (${took}s): red, unconfirmed: they could not be run on this branch's base: $(printf '%s ' $ids)"
            record_set "$PINS_OWED" "$name" "red, unconfirmed against this branch's base"
            worse 3
          else
            new=$(comm -23 <(printf '%s\n' "$ids") <(printf '%s\n' "$before" | sort -u))
            added=$(comm -23 <(printf '%s\n' "$new") <(record_get "$PINS_RED" "$name" | sort -u))
            record_set "$PINS_OWED" "$name"
            # shellcheck disable=SC2086
            record_set "$PINS_RED" "$name" $new
            if [ -n "$added" ]; then
              echo "FAIL pins $name (${took}s): red since this change: $(printf '%s ' $added)"
              printf '%s\n' "$out" | tail -60
              rc=1
            elif [ -n "$new" ]; then
              echo "SKIP pins $name (${took}s): $(printf '%s ' $new)red since an earlier run on this branch; settled by a change here or in $name"
              worse 3
            else
              echo "SKIP pins $name (${took}s): red before this change too, so not this change's to fix: $(printf '%s ' $ids)"
              worse 3
            fi
          fi ;;
        3)
          printf '%s\n' "$out" | grep -E '^SKIP '
          record_set "$PINS_OWED" "$name" "$(printf '%s\n' "$out" | grep -E '^SKIP ' | head -1 | sed 's/^SKIP pins [^:]*: //')"
          worse 3 ;;
        124)
          echo "SKIP pins $name: ran past the time left; scripts/gate.sh --changed runs them"
          record_set "$PINS_OWED" "$name" "ran past the time left"
          worse 3 ;;
        *)
          echo "SKIP pins $name: its gate.sh could not run them (exit $hrc)"
          printf '%s\n' "$out" | tail -10
          record_set "$PINS_OWED" "$name" "its gate.sh could not run them (exit $hrc)"
          worse 3 ;;
      esac
    done < <(pin_holders)
    exit "$rc" ;;
  pins_for)
    [ -n "$TARGET" ] && [ -n "$WORKSPACE" ] || { usage; exit 2; }
    files=""
    drop=""
    while IFS= read -r entry; do
      case "$entry" in
        '!'*) entry=${entry#!}; case ",${entry%%:*}," in *",$TARGET,"*) drop="$drop --deselect ${entry#*:}" ;; esac ;;
        *) case ",${entry%%:*}," in *",$TARGET,"*) files="$files ${entry#*:}" ;; esac ;;
      esac
    done < <(pins_declared "$SELF")
    [ -n "$ONLY" ] && files=" $ONLY"
    [ -n "$files" ] || { echo "SKIP pins $(basename "$PWD"): it holds none on $TARGET"; exit 3; }
    take_lock || { echo "SKIP pins $(basename "$PWD"): another gate run holds its lock"; exit 3; }
    # Every pin must be able to look, and looks at the workspace of the checkout it is run for.
    QUICK_PYTEST="KHAOS_WORKSPACE=$(printf %q "$WORKSPACE") KHAOS_REQUIRE_SIBLINGS=1 $QUICK_PYTEST$drop"
    export QUICK_PYTEST
    with_timeout "$BUDGET" bash -c "pytest_by_folder$files"
    exit $? ;;
  unnamed)
    # Every tracked Python file under src/ and scripts/ no test names (not with --stale), and each NO_TEST_OK glob that
    # no longer matches an unnamed file: the list only shrinks, so a stale glob fails and a new unnamed file is shown.
    [ -n "$STALE_ONLY" ] || git ls-files -- 'src/*.py' 'scripts/*.py' | unnamed_sources | sed 's/^/UNNAMED /'
    rc=0
    read -r -d "" -a globs <<< "${NO_TEST_OK:-}" || true
    for glob in ${globs[@]+"${globs[@]}"}; do
      case "$glob" in *[*?[]*) echo "STALE $glob: NO_TEST_OK takes exact paths; a glob would hide every new file it matches"; rc=1 ;; esac
    done
    read -r -d "" -a globs <<< "${NO_TEST_OK:-} ${NO_TEST_OK_GLOBS:-}" || true
    for glob in ${globs[@]+"${globs[@]}"}; do
      if [ -z "$(git ls-files -- 'src/*.py' 'scripts/*.py' | unnamed_under "$glob")" ]; then
        echo "STALE $glob: every file it matches has a test, or none is left"
        rc=1
      fi
    done
    exit "$rc" ;;
  names)
    [ ${#NAMES[@]} -eq 0 ] && { usage; exit 2; }
    expanded=()
    for g in "${NAMES[@]}"; do
      if [ "$g" = all ]; then read -r -a every <<< "$ALL_GATES"; expanded+=("${every[@]}"); else expanded+=("$g"); fi
    done
    NAMES=("${expanded[@]}")
    for g in "${NAMES[@]}"; do
      [ "$g" = pins ] || gate_cmd "$g" > /dev/null || { echo "no gate named $g"; usage; exit 2; }
    done
    contract_note
    take_lock || { echo "another gate run holds this repository's lock"; exit 3; }
    run_gates "${NAMES[@]}"
    rc=$?
    [ "$rc" -eq 0 ] && case " ${NAMES[*]} " in *" $SUITE_GATE "*) rm -f "${OWED:?}" ;; esac
    exit "$rc" ;;
esac
