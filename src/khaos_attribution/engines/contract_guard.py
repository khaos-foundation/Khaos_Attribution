"""The engine environment's contract, checked where an engine process is started: an interpreter that imports
another contract than this room's is refused in a sentence. A refusal is read again at every start, so a repaired
environment needs no restart; a pass is kept until the environment's contract is installed again. The running
interpreter is never checked, so a test's fake worker spawns nothing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from khaos_attribution.engines import environment

_PASSED: dict = {}
# What _fingerprint answers for an interpreter that cannot be started at all.
MISSING = "missing"


def _fingerprint(python) -> str | None:
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


# ── the contract an engine's environment holds ────────────────────────────────

CONTRACT_SRC_ENV = "KHAOS_CONTRACT_SRC"
CONTRACT_WHEEL_ENV = "KHAOS_CONTRACT_WHEEL"
CONTRACT_REPO = "https://github.com/khaos-foundation/Khaos_Attribution.git"
CONTRACT_FINGERPRINT = "from khaos_attribution.engines.pins import contract_code_hash; print(contract_code_hash())"


def contract_spec(root: Path) -> str:
    """Where the contract comes from for the engine's environment: a checkout named in KHAOS_CONTRACT_SRC, a wheel
    named in KHAOS_CONTRACT_WHEEL (the app ships one), the room's sibling checkout while it holds the code this room
    imports, else the tagged release of this room's version."""
    from khaos_attribution import __version__  # noqa: PLC0415
    from khaos_attribution.engines.pins import contract_code_hash  # noqa: PLC0415
    src = os.environ.get(CONTRACT_SRC_ENV, "").strip()
    if src:
        return src
    wheel = os.environ.get(CONTRACT_WHEEL_ENV, "").strip()
    if wheel:
        return wheel
    sibling = Path(root).parent / "Khaos_Attribution"
    package = sibling / "src" / "khaos_attribution"
    if (sibling / "pyproject.toml").is_file() and package.is_dir() and \
            contract_code_hash(package).split()[1] == contract_code_hash().split()[1]:
        return str(sibling)
    return f"khaos-attribution @ git+{CONTRACT_REPO}@v{__version__}"


def contract_fingerprint(python: Path | None = None) -> str | None:
    """"<version> <hash>" of the contract an interpreter imports (this process's own when None), or None when it
    imports none or one too old to say."""
    if python is None:
        from khaos_attribution.engines.pins import contract_code_hash  # noqa: PLC0415
        return contract_code_hash()
    done = environment.run([python, "-c", CONTRACT_FINGERPRINT])
    return done.stdout.strip() if done.returncode == 0 else None


def contract_mismatch(python: Path) -> str:
    """Why the engine's environment does not hold this contract, in words."""
    return mismatch_words(contract_fingerprint(python), contract_fingerprint())


def mismatch_words(theirs: str | None, ours: str) -> str:
    """Why the fingerprint ``theirs`` is not this room's ``ours``, in words."""
    if not theirs:
        return "holds no contract, or one older than 0.41.1"
    if theirs.split()[0] == ours.split()[0]:
        return f"holds contract {theirs.split()[0]} built from other code than this room's"
    return f"holds contract {theirs.split()[0]}, this room runs {ours.split()[0]}"


def contract_present(python: Path) -> bool:
    theirs = contract_fingerprint(python)
    return theirs is not None and theirs == contract_fingerprint()


def contract_into(python: Path, root: Path) -> tuple:
    """The contract's gradient maths into the engine's environment, the version this room runs: the measure worker
    projects its features with it."""
    from khaos_attribution import __version__  # noqa: PLC0415
    # With its base dependencies (jsonschema, numpy): the engine's own pins are its own, the contract's are light.
    done = environment.run([environment.uv_path(), "pip", "install", "-q", "--python", python, "--reinstall-package",
                            "khaos-attribution", contract_spec(root)])
    if done.returncode != 0:
        return False, f"the contract could not be installed into the engine's environment: {done.stderr.strip()[-200:]}"
    if not contract_present(python):
        return False, f"khaos-attribution was installed into the engine's environment, but it {contract_mismatch(python)}"
    return True, f"khaos-attribution {__version__} in the engine's environment"
