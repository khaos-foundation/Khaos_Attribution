# shellcheck shell=bash disable=SC2034
# This repository's own part of scripts/gate.sh, which sources it: its gates, the commands they run and what a change
# maps to. Every name here has a default in gate.sh; GATE_CORE is gate.sh's blob, which the suite holds it to.
GATE_CORE=0d702b01651536ff1a8d384e8ef17ace9d8a33d7

ALL_GATES="pytest"
# Named only, never in all, --changed or --quick: the release needs it, and a machine without torch must not go red.
OTHER_GATES="torch"
# Python files under src/ and scripts/ that no test can name yet: exact paths only, and the list only shrinks
# (gate.sh --stale fails on a path that got a test or a new glob).
# engines/sa3/forward_mlx.py: its load_bundle and conditioning are reached only through measure_worker_mlx, and no
# stand-in test calls them yet.
NO_TEST_OK="scripts/watermark_survival.py src/khaos_attribution/engines/mimetic/convert_worker.py
  src/khaos_attribution/engines/sa3/convert_worker.py src/khaos_attribution/engines/sa3/forward_mlx.py
  src/khaos_attribution/seal/__main__.py src/register_service/bom.py"

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

# Only the torch gate sets it, on its own command: left in a shell it would change every other gate's skip rule.
unset KHAOS_TORCH_RUN
