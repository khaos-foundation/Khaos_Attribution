"""The engine environment's contract, checked where an engine process is started: an interpreter that imports
another contract than this room's is refused in a sentence, once per interpreter for the life of the process. The
running interpreter is never checked, so a test's fake worker spawns nothing."""

from __future__ import annotations

import sys
from pathlib import Path

_CHECKED: dict = {}
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


def refusal(python, *, engine: str, engine_id: str) -> str | None:
    """Why ``python`` must not run this room's work for ``engine``, or None; the answer is remembered per path."""
    path = str(python)
    if _is_this_interpreter(path):
        return None
    if path not in _CHECKED:
        from khaos_attribution.engines.sa3.provision import mismatch_words  # noqa: PLC0415
        ours, theirs = _fingerprint(None), _fingerprint(Path(path))
        if theirs is MISSING:
            _CHECKED[path] = f"the {engine} environment has no interpreter at {path}: run the installer with --engines {engine_id}"
        else:
            _CHECKED[path] = None if theirs == ours else (f"the {engine} environment at {path} {mismatch_words(theirs, ours)}: "
                                                          f"run the installer with --engines {engine_id}")
    return _CHECKED[path]


def require(python, *, engine: str, engine_id: str, error: type) -> None:
    """``refusal`` raised as the calling slot's own error type."""
    why = refusal(python, engine=engine, engine_id=engine_id)
    if why:
        raise error(why)
