"""Provision an engine on this machine, from a room's installer: its licence read once and accepted once (recorded
beside the room), then the engine's own provision slot when it has one. --check reports and changes nothing.
Usage: python -m khaos_attribution.engines.provision <engine> [--check] [--accept-licence] [--marker-dir DIR]
[--repo-dir DIR] [--weights-dir DIR] [--device DEV]"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

from khaos_attribution.engines import KNOWN_ENGINES, EngineUnavailable, descriptor

MARKER = ".khaos-engine-licence-{engine}.json"


def statement(engine_id: str) -> str:
    """The engine's licence as its descriptor states it: name, notice, obligations; never a summary of something
    nobody looked at."""
    d = descriptor(engine_id)
    lines = [f"{d.name} is under the {d.licence['name']}."]
    if d.status == "experimental":
        lines.append(f"{d.name} is experimental: {d.status_note or 'not yet proven in Khaos'}.")
    if d.licence.get("notice"):
        lines.append(d.licence["notice"])
    for o in d.licence.get("obligations") or []:
        lines.append(f"  - {o}")
    gated = [w for w in d.weights if w.get("gated")]
    for w in gated:
        lines.append(f"  - {w['repo']} is gated ({w.get('gated_by', 'by its owner')}): request access on its page and sign in to "
                     "Hugging Face yourself; this installer never handles a token.")
    return "\n".join(lines)


def marker_path(marker_dir: Path | str, engine_id: str) -> Path:
    return Path(marker_dir) / MARKER.format(engine=engine_id)


def accepted(marker_dir: Path | str, engine_id: str) -> bool:
    p = marker_path(marker_dir, engine_id)
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return rec.get("engine") == engine_id and rec.get("licence") == descriptor(engine_id).licence["name"]


def record_acceptance(marker_dir: Path | str, engine_id: str) -> Path:
    p = marker_path(marker_dir, engine_id)
    p.write_text(json.dumps({"engine": engine_id, "licence": descriptor(engine_id).licence["name"],
                             "accepted_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    return p


def provision_slot(engine_id: str):
    """The engine's provision module, or None when the engine needs no provisioning here."""
    package = KNOWN_ENGINES[engine_id]
    try:
        return import_module(package + ".provision")
    except ImportError:
        return None


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser(description="Provision an engine for Khaos on this machine.")
    ap.add_argument("engine", choices=sorted(KNOWN_ENGINES))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--accept-licence", action="store_true", help="accept without the prompt (the statement still prints)")
    ap.add_argument("--marker-dir", type=Path, default=Path.cwd())
    ap.add_argument("--repo-dir", type=Path, default=None)
    ap.add_argument("--weights-dir", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    try:
        d = descriptor(a.engine)
    except EngineUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"==> {d.name}: the licence")
    print(statement(a.engine))
    if not Path(a.marker_dir).is_dir():
        print(f"[FAIL] the marker directory {a.marker_dir} does not exist", file=sys.stderr)
        return 1
    if accepted(a.marker_dir, a.engine):
        print(f"[ok] accepted earlier ({marker_path(a.marker_dir, a.engine)})")
    elif a.check:
        print(f"[FAIL] the {d.licence['name']} has not been accepted here; run without --check to read and accept it")
        return 1
    elif a.accept_licence:
        record_acceptance(a.marker_dir, a.engine)
        print("[ok] accepted by --accept-licence")
    elif sys.stdin.isatty():
        answer = input("  Type yes to accept: ").strip().lower()
        if answer != "yes":
            print(f"The {d.licence['name']} was not accepted; stopping here.", file=sys.stderr)
            return 1
        record_acceptance(a.marker_dir, a.engine)
        print("[ok] accepted")
    else:
        print(f"[FAIL] no terminal to accept the {d.licence['name']} on; pass --accept-licence once you have read it", file=sys.stderr)
        return 1
    slot = provision_slot(a.engine)
    if slot is None:
        print(f"[ok] {d.name} needs no provisioning of its own here")
        return 0
    try:
        report = slot.provision(a.repo_dir, a.weights_dir, device=a.device, check_only=a.check)
    except (OSError, RuntimeError) as exc:
        print(f"[FAIL] {d.name} could not be provisioned: {exc}", file=sys.stderr)
        return 1
    print(report.text())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
