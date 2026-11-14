#!/usr/bin/env bash
# PreToolUse guard: nothing under this repository's data/, artists/ or outputs/ folder is edited, written or removed
# from a session. The rule is guard.py beside this file; a missing python or guard.py refuses every call.
ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY=$(command -v python3 || command -v python) || { echo "refused: no python to read the tool call" >&2; exit 2; }
exec "$PY" -I -S "$(dirname "$0")/guard.py" "$ROOT"
