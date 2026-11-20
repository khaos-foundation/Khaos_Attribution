"""Start one of the contract's scripts in an engine's own interpreter, on the contract of the room that started it.

khaos_attribution is loaded from beside this file, never from the engine environment, which may hold an older copy,
and nothing is written into the room's install. Usage: <engine python> launch.py <script> [args...]. Each such script
runs this file first under CHECK_NAME, so one started any other way, or a child the script spawns, cannot drift onto
another copy. Standard library only, and nothing from the contract until it is loaded.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

PACKAGE_ENV = "KHAOS_CONTRACT_PACKAGE"
CHECK_NAME = "khaos_contract_check"
# The first statement after the imports of every contract script that runs in an engine interpreter.
STUB = ('if __name__ in ("__main__", "__mp_main__"):  # the room\'s contract, only through engines/launch.py\n'
        '    import pathlib\n'
        '    import runpy\n'
        '    runpy.run_path(str(pathlib.Path(__file__).resolve().parents[1] / "launch.py"), '
        'run_name="khaos_contract_check")\n')
REFUSED_EXIT = 3
NAME = "khaos_attribution"
PACKAGE = Path(__file__).resolve().parent.parent


def _same(a, b) -> bool:
    return os.path.normcase(os.path.realpath(str(a))) == os.path.normcase(os.path.realpath(str(b)))


def refuse(why: str):
    """One sentence on stderr, the same as a worker's error reply on stdout, and a non-zero exit."""
    print(json.dumps({"error": why}), flush=True)
    print(why, file=sys.stderr, flush=True)
    raise SystemExit(REFUSED_EXIT)


class _ReadOnly(importlib.machinery.SourceFileLoader):
    """The room's files read, nothing written beside them: the install may be a signed app or an image."""

    def set_data(self, path, data, *, _mode=0o666):
        return None


class RoomContract:
    """Finds khaos_attribution and its modules in the room's package only, ahead of anything on sys.path."""

    def __init__(self, package: Path):
        self.package = Path(package)

    def find_spec(self, name, path=None, target=None):
        if name == NAME:
            origin = str(self.package / "__init__.py")
            return importlib.util.spec_from_file_location(name, origin, loader=_ReadOnly(name, origin),
                                                          submodule_search_locations=[str(self.package)])
        if not name.startswith(NAME + "."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            spec.loader = _ReadOnly(name, spec.origin)
        return spec


def ensure(package) -> None:
    """khaos_attribution from ``package``, loaded once; refused when another copy is already loaded."""
    package = Path(package)
    loaded = sys.modules.get(NAME)
    if loaded is not None:
        if not _same(Path(getattr(loaded, "__file__", "") or "").parent, package):
            refuse(f"this process already holds the contract at {Path(loaded.__file__).parent}, not the room's at "
                   f"{package}: start it through {PACKAGE / 'engines' / 'launch.py'}")
        return
    if not any(isinstance(f, RoomContract) and _same(f.package, package) for f in sys.meta_path):
        sys.meta_path.insert(0, RoomContract(package))
    try:
        importlib.import_module(NAME)
    except ImportError as exc:
        sys.modules.pop(NAME, None)
        refuse(f"this engine's environment cannot load the room's contract ({exc}): run the room's installer for this "
               "engine again")


def check() -> None:
    """What every script of the contract's runs first: the room's contract, and only through the launcher."""
    package = os.environ.get(PACKAGE_ENV, "").strip()
    if not package:
        refuse("this engine script was started outside the contract's launcher, so it could run an engine "
               "environment's own older contract: start it through worker_process.command()")
    if not _same(package, PACKAGE):
        refuse(f"this engine script belongs to the contract at {PACKAGE}, but its launcher named {package}")
    ensure(PACKAGE)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        print("usage: <engine python> launch.py <script> [args...]", file=sys.stderr)
        raise SystemExit(2)
    script = Path(argv[0]).resolve()
    os.environ[PACKAGE_ENV] = str(PACKAGE)
    ensure(PACKAGE)
    # As `python <script>` would leave them: the script's own folder first, its own name as argv[0].
    sys.argv = [str(script), *argv[1:]]
    sys.path[0] = str(script.parent)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == CHECK_NAME:
    check()
elif __name__ == "__main__":
    main()
