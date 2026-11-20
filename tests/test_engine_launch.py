"""The launcher runs a contract script on the room's own contract, whatever the engine environment holds, writes nothing
into the room's install, and a script started any other way refuses in one sentence instead of drifting."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import khaos_attribution
from khaos_attribution.engines import launch as L

PACKAGE = Path(khaos_attribution.__file__).resolve().parent
NO_PY311 = "no Python 3.11 on this machine to load the contract under"


@pytest.fixture()
def room(tmp_path) -> Path:
    """A room's installed contract, copied without bytecode, with a toy engine folder for the scripts below."""
    package = tmp_path / "room" / "khaos_attribution"
    shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (package / "engines" / "toy").mkdir()
    return package


@pytest.fixture()
def decoy(tmp_path) -> Path:
    """An older contract an engine environment still holds, first on the child's path."""
    folder = tmp_path / "decoy"
    (folder / "khaos_attribution").mkdir(parents=True)
    (folder / "khaos_attribution" / "__init__.py").write_text('__version__ = "0.0.0-decoy"\n')
    return folder


def _script(room: Path, name: str, body: str) -> Path:
    path = room / "engines" / "toy" / name
    path.write_text("from __future__ import annotations\n\n" + L.STUB + textwrap.dedent(body))
    return path


def _env(decoy: Path | None = None, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in (L.PACKAGE_ENV, "PYTHONPATH")}
    if decoy is not None:
        env["PYTHONPATH"] = str(decoy)
    return {**env, **extra}


def _launch(room: Path, script: Path, *args, env=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(room / "engines" / "launch.py"), str(script), *args],
                          capture_output=True, text=True, env=env, timeout=120)


def _reply(done: subprocess.CompletedProcess) -> dict:
    return json.loads(done.stdout.strip().splitlines()[-1])


REPORT = """
    import json, sys
    import khaos_attribution
    from khaos_attribution.engines import descriptor
    from khaos_attribution import gradient
    print(json.dumps({"file": khaos_attribution.__file__, "version": khaos_attribution.__version__,
                      "gradient": gradient.__file__, "name": __name__, "argv": sys.argv, "path0": sys.path[0],
                      "mock": descriptor("mock").id}))
"""


def test_a_launched_script_runs_on_the_rooms_contract_not_the_environments(room, decoy):
    """With an older copy first on the engine's path, the script still imports the room's package and its modules."""
    script = _script(room, "report.py", REPORT)
    done = _launch(room, script, "--flag", "x", env=_env(decoy))
    assert done.returncode == 0, done.stderr
    got = _reply(done)
    assert Path(got["file"]).resolve() == (room / "__init__.py").resolve()
    assert Path(got["gradient"]).resolve().is_relative_to(room.resolve())
    assert got["version"] == khaos_attribution.__version__
    assert got["mock"] == "mock"


def test_a_launched_script_sees_what_a_direct_run_would(room):
    """__main__, its own argv and its own folder first on sys.path, as `python <script>` gives."""
    script = _script(room, "report.py", REPORT)
    got = _reply(_launch(room, script, "--flag", "x", env=_env()))
    assert got["name"] == "__main__"
    assert got["argv"] == [str(script.resolve()), "--flag", "x"]
    assert Path(got["path0"]).resolve() == script.parent.resolve()


def test_nothing_is_written_into_the_rooms_install(room, decoy):
    """No bytecode lands beside the room's files: the install may be a signed app or a read-only image."""
    _launch(room, _script(room, "report.py", REPORT), env=_env(decoy))
    assert not [p for p in room.rglob("__pycache__")], "the launcher wrote bytecode into the room's package"


def test_a_script_started_outside_the_launcher_refuses(room, decoy):
    """Run directly, as a missed start site would, the script exits non-zero in one sentence and never imports."""
    script = _script(room, "report.py", REPORT)
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=_env(decoy), timeout=60)
    assert done.returncode == L.REFUSED_EXIT
    assert "outside the contract's launcher" in _reply(done)["error"]
    assert "0.0.0-decoy" not in done.stdout


def test_a_launcher_naming_another_contract_is_refused(room, decoy):
    script = _script(room, "report.py", REPORT)
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60,
                          env=_env(decoy, **{L.PACKAGE_ENV: str(decoy / "khaos_attribution")}))
    assert done.returncode == L.REFUSED_EXIT
    assert "its launcher named" in _reply(done)["error"]


def test_another_copy_already_loaded_is_refused(room, decoy):
    """A process that imported the environment's copy first cannot go on to run the room's script on it."""
    code = ("import khaos_attribution, runpy, sys; "
            f"runpy.run_path({str(room / 'engines' / 'launch.py')!r}, run_name='khaos_contract_check')")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                          env=_env(decoy, **{L.PACKAGE_ENV: str(room)}))
    assert done.returncode == L.REFUSED_EXIT
    assert "already holds the contract" in _reply(done)["error"]


def test_an_environment_that_cannot_load_the_contract_is_refused_in_one_sentence(room):
    """A missing base requirement of the contract names the installer, not a traceback."""
    shadow = room.parent / "shadow"
    shadow.mkdir()
    (shadow / "jsonschema.py").write_text("raise ImportError('no jsonschema here')\n")
    done = _launch(room, _script(room, "report.py", REPORT), env=_env(shadow))
    assert done.returncode == L.REFUSED_EXIT
    assert "cannot load the room's contract" in _reply(done)["error"]
    assert "installer" in done.stderr


SPAWN = """
    import importlib, json, multiprocessing


    def where(queue):
        queue.put(importlib.import_module("khaos_attribution").__file__)


    if __name__ == "__main__":
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        child = ctx.Process(target=where, args=(queue,))
        child.start()
        found = queue.get(timeout=60)
        child.join(60)
        print(json.dumps({"child": found, "exit": child.exitcode}))
"""


def test_a_spawned_child_of_a_launched_script_loads_the_rooms_contract(room, decoy):
    """A data loader's or a pool's spawn child re-runs the script, and its stub keeps it off the environment's copy."""
    done = _launch(room, _script(room, "spawn.py", SPAWN), env=_env(decoy))
    assert done.returncode == 0, done.stderr
    got = _reply(done)
    assert got["exit"] == 0
    assert Path(got["child"]).resolve() == (room / "__init__.py").resolve()


def _python311() -> str | None:
    found = shutil.which("python3.11")
    if found:
        return found
    uv = shutil.which("uv")
    if uv is None:
        return None
    done = subprocess.run([uv, "python", "find", "3.11"], capture_output=True, text=True, timeout=30,
                          env={**os.environ, "UV_PYTHON_DOWNLOADS": "never"})
    return done.stdout.strip() or None if done.returncode == 0 else None


def test_the_contract_and_its_launcher_load_under_python_311(tmp_path):
    """Engine environments run Python 3.11: every module of the contract compiles there and the launcher starts."""
    python = _python311()
    if python is None:
        pytest.skip(NO_PY311)
    sources = [str(p) for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts]
    code = ("import sys\n"
            "bad = []\n"
            "for f in sys.argv[1:]:\n"
            "    try:\n"
            "        compile(open(f, encoding='utf-8').read(), f, 'exec')\n"
            "    except SyntaxError as exc:\n"
            "        bad.append(f'{f}: {exc}')\n"
            "print('\\n'.join(bad))\n"
            "sys.exit(1 if bad else 0)\n")
    done = subprocess.run([python, "-c", code, *sources], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, f"modules that Python 3.11 cannot compile:\n{done.stdout}{done.stderr}"
    usage = subprocess.run([python, str(PACKAGE / "engines" / "launch.py")], capture_output=True, text=True, timeout=60)
    assert usage.returncode == 2 and "usage" in usage.stderr


def test_the_contracts_own_scripts_start_through_the_launcher_and_anyone_elses_directly(tmp_path):
    from khaos_attribution.engines import worker_process as W
    ours = PACKAGE / "engines" / "sa3" / "worker.py"
    assert W.command("py", ours, "--x", 1) == ["py", str(PACKAGE / "engines" / "launch.py"), str(ours), "--x", "1"]
    theirs = tmp_path / "optimized" / "mlx" / "scripts" / "pre_encode_mlx.py"
    assert W.command("py", theirs, "-i", "c") == ["py", str(theirs), "-i", "c"]


def test_a_contract_copy_refuses_to_load_in_a_process_started_on_another(room, decoy):
    """A stray child of a launched script that finds another copy of this contract on its path stops there."""
    done = subprocess.run([sys.executable, "-c", "import khaos_attribution"], capture_output=True, text=True, timeout=60,
                          env=_env(PYTHONPATH=str(room.parent), **{L.PACKAGE_ENV: str(decoy / "khaos_attribution")}))
    assert done.returncode != 0 and "ImportError: this process runs the room's contract at" in done.stderr
