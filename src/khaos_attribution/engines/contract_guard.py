"""The engine environment's contract, checked where an engine process is started: an interpreter that imports
another contract than this room's is refused in a sentence. A refusal is read again at every start, so a repaired
environment needs no restart; a pass is kept until the environment's contract is installed again. The running
interpreter is never checked, so a test's fake worker spawns nothing."""

from __future__ import annotations

import sys
from pathlib import Path

_PASSED: dict = {}
# What _fingerprint answers for an interpreter that cannot be started at all.
MISSING = "missing"


def _fingerprint(python) -> str | None:
    from khaos_attribution.engines.sa3.provision import contract_fingerprint  # noqa: PLC0415
    try:
        return contract_fingerprint(python)
    except OSError:
        return MISSING


def _is_this_interpreter(python) -> bool:
    try:
        return Path(python).resolve() == Path(sys.executable).resolve()
    except OSError:
        return False


def _installed(python) -> tuple | None:
    """When the contract in the interpreter's venv was last installed (its dist-info's name and mtime), or None
    where it cannot be read; the interpreter's own path is kept, since a venv's python links outside it."""
    venv = Path(python).parent.parent
    try:
        sites = [venv / "Lib" / "site-packages", *sorted((venv / "lib").glob("python*/site-packages"))]
        found = sorted(d for site in sites for d in site.glob("khaos_attribution-*.dist-info"))
        return tuple((d.name, d.stat().st_mtime_ns) for d in found) or None
    except OSError:
        return None


def refusal(python, *, engine: str, engine_id: str) -> str | None:
    """Why ``python`` must not run this room's work for ``engine``, or None."""
    path = str(python)
    if _is_this_interpreter(path):
        return None
    installed = _installed(path)
    if path in _PASSED and _PASSED[path] == installed:
        return None
    from khaos_attribution.engines.sa3.provision import mismatch_words  # noqa: PLC0415
    ours, theirs = _fingerprint(None), _fingerprint(Path(path))
    if theirs is MISSING:
        _PASSED.pop(path, None)
        return f"the {engine} environment has no interpreter at {path}: run the installer with --engines {engine_id}"
    if theirs == ours:
        _PASSED[path] = installed
        return None
    _PASSED.pop(path, None)
    return f"the {engine} environment at {path} {mismatch_words(theirs, ours)}: run the installer with --engines {engine_id}"


def require(python, *, engine: str, engine_id: str, error: type) -> None:
    """``refusal`` raised as the calling slot's own error type."""
    why = refusal(python, engine=engine, engine_id=engine_id)
    if why:
        raise error(why)
