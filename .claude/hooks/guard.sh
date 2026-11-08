#!/usr/bin/env bash
# PreToolUse guard: nothing under this repository's data/ folder is edited, written or removed from a session.
ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY=$(command -v python3 || command -v python) || { echo "refused: no python to read the tool call" >&2; exit 2; }
exec "$PY" -I -S -c '
import json, os, re, sys

def deny(why):
    print("refused: " + why, file=sys.stderr)
    sys.exit(2)

root = os.path.abspath(sys.argv[1])
call = json.load(sys.stdin)
tool, args = call.get("tool_name", ""), call.get("tool_input") or {}
if tool == "Bash":
    cmd = args.get("command", "")
    # Deliberately crude: an rm-like word together with any path segment named data/.
    if re.search(r"(?<![A-Za-z0-9_])rm", cmd) and re.search(r"(?<![A-Za-z0-9_])data/", cmd):
        deny("the command names rm and a data/ path; artist data is never removed from a session")
    sys.exit(0)
path = args.get("file_path") or args.get("notebook_path") or ""
if not path:
    sys.exit(0)
given = os.path.abspath(os.path.join(call.get("cwd") or root, path))
# A data/ folder may be a link to another disk, so the path is judged as written and as resolved.
for full in {given, os.path.realpath(given)}:
    for base in {root, os.path.realpath(root)}:
        if full.startswith(base + os.sep) and "data" in os.path.relpath(full, base).replace(os.sep, "/").split("/")[:-1]:
            deny(path + " is under this repository'"'"'s data/ folder; artist data is never edited from a session")
' "$ROOT"
