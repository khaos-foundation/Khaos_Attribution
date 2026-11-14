#!/usr/bin/env bash
# PreToolUse guard: nothing under this repository's data/, artists/ or outputs/ folder is edited, written or removed
# from a session. The rule is guard.py beside this file; a missing python or guard.py refuses every call.
ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
# The first python that runs: Windows' Store stub answers to the name and runs nothing.
PY=""
for name in python3 python py; do
  found=$(command -v "$name") && "$found" -I -S -c "" </dev/null >/dev/null 2>&1 && { PY=$found; break; }
done
[ -n "$PY" ] || { echo "refused: no python to read the tool call" >&2; exit 2; }
"$PY" -I -S "$(dirname "$0")/guard.py" "$ROOT"
code=$?
# Only guard.py's allow lets a call through: any other ending refuses it.
[ "$code" -eq 0 ] && exit 0
[ "$code" -eq 2 ] || echo "refused: the guard stopped with code $code" >&2
exit 2
