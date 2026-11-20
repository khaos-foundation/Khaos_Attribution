"""The test goldens a deliberate change moves, written by one command each, from the repository's root.
    python scripts/goldens.py seal                 re-render tests/golden/seal.svg and marks_row.svg: look, then commit
    python scripts/goldens.py schema-pin <name>    append <name>'s newest version, fields and shape to schema_fields.json
A schema pin is appended only for a version above the last one pinned: a pinned version is never pinned again."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def seal() -> None:
    from khaos_attribution.seal import render_marks_row, render_seal
    from tests.test_seal import ARGS, GOLDEN_MARKS, GOLDEN_SEAL
    GOLDEN_SEAL.write_text(render_seal(*ARGS), encoding="utf-8")
    GOLDEN_MARKS.write_text(render_marks_row(*ARGS[1:]), encoding="utf-8")
    print(f"wrote {GOLDEN_SEAL.name} and {GOLDEN_MARKS.name}; open both and look before committing")


def schema_pin(name: str) -> int:
    from khaos_attribution import load_schema
    from khaos_attribution.validation import known_versions
    from tests.test_schema_versions import PINS, _version, fields, shape
    schema = load_schema(f"{name}.schema.json")
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    newest = max(known_versions(schema), key=_version)
    history = pins.setdefault(name, [])
    if history and _version(newest) <= _version(history[-1]["version"]):
        print(f"{name} {newest} is not above the last pin ({history[-1]['version']}): a change needs a new schema_version")
        return 1
    history.append({"version": newest, "fields": fields(schema), "shape": shape(schema)})
    PINS.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"pinned {name} {newest}")
    return 0


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(prog="goldens.py", description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seal")
    sub.add_parser("schema-pin").add_argument("name")
    a = p.parse_args(argv)
    if a.cmd == "seal":
        seal()
        return 0
    return schema_pin(a.name)


if __name__ == "__main__":
    sys.exit(main())
