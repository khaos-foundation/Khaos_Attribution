"""Every process the engines start is either one of the contract's scripts through the launcher or a named outside
program, and every contract script an engine interpreter runs checks its launcher first: a start that would run an
engine environment's own older contract fails to build, never drifts."""
from __future__ import annotations

import ast
from pathlib import Path

import khaos_attribution
from khaos_attribution.engines import launch as L

ENGINES = Path(khaos_attribution.__file__).resolve().parent / "engines"
SUBPROCESS = {"Popen", "run", "call", "check_call", "check_output"}
MULTIPROCESSING = {"Process", "Pool", "get_context", "set_start_method", "ProcessPoolExecutor"}
# (file, enclosing function): why that start is not a contract script, or needs no launcher.
ALLOWED = {
    ("environment.py", "run"): "uv, git and `python -c` probes for provisioning; never a contract script",
    ("pins.py", "checkout_commit"): "git, reading an engine checkout's HEAD",
    ("worker_process.py", "_ps"): "ps, reading the process table",
    ("worker_process.py", "_stop_windows"): "taskkill, stopping a tree on Windows",
    ("sa3/worker.py", "_run_forwarding_stop"): "its argv comes from command() in the MLX render",
    ("sa3/train_driver.py", "__init__"): "Stability's loader; children fork, or re-run this driver under spawn",
    ("mimetic/train_driver.py", "main"): "the trainer's loader; children fork, or re-run this driver under spawn",
    ("acestep/batching.py", "train_dataloader"): "ACE-Step trains in the room's own interpreter, not an engine's",
}
# Contract modules with a main block that a room runs in its own interpreter as `python -m`.
ROOM_MODULES = {"provision.py", "checklist.py", "launch.py", "acestep/seams.py", "sa3/provision.py",
                "mimetic/provision.py"}


def _files():
    return sorted(p for p in ENGINES.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: Path) -> str:
    return path.relative_to(ENGINES).as_posix()


def _starts(tree: ast.AST):
    """(enclosing function, call) for every call that starts a process or a loader's workers."""
    found = []

    def visit(node, where):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else where
            if isinstance(child, ast.Call):
                f = child.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                owner = getattr(f.value, "id", "") if isinstance(f, ast.Attribute) else ""
                if ((owner == "subprocess" and name in SUBPROCESS) or name in MULTIPROCESSING or "DataLoader" in name
                        or (owner == "os" and name.startswith(("spawn", "exec", "posix_spawn")))
                        or any(k.arg == "num_workers" for k in child.keywords)):
                    found.append((where, child))
            visit(child, inner)

    visit(tree, "<module>")
    return found


def _through_command(call: ast.Call) -> bool:
    first = call.args[0] if call.args else None
    if not isinstance(first, ast.Call):
        return False
    f = first.func
    return (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")) == "command"


def test_every_start_is_a_contract_script_through_the_launcher_or_a_named_outside_program():
    unexplained, stale = [], set(ALLOWED)
    for path in _files():
        for where, call in _starts(ast.parse(path.read_text(encoding="utf-8"))):
            key = (_rel(path), where)
            if _through_command(call):
                continue
            if key in ALLOWED:
                stale.discard(key)
                continue
            unexplained.append(f"{key[0]}:{call.lineno} in {where}()")
    assert not unexplained, ("starts that neither go through worker_process.command() nor are named here: "
                             + ", ".join(unexplained))
    assert not stale, f"named starts that no longer exist: {sorted(stale)}"


def _scripts():
    """The contract's scripts an engine interpreter runs: every module under engines/ with a main block."""
    for path in _files():
        rel = _rel(path)
        if rel not in ROOM_MODULES and '__name__ == "__main__"' in path.read_text(encoding="utf-8"):
            yield path, rel


def test_every_engine_script_checks_its_launcher_before_it_imports_the_contract():
    scripts = dict((rel, path) for path, rel in _scripts())
    assert {"sa3/worker.py", "sa3/measure_worker.py", "sa3/heldout_worker.py", "sa3/train_driver.py"} <= set(scripts)
    for rel, path in scripts.items():
        text = path.read_text(encoding="utf-8")
        assert L.STUB in text, f"{rel} does not start with the launcher's check"
        body = ast.parse(text).body
        stub = next(i for i, n in enumerate(body) if isinstance(n, ast.If) and "__mp_main__" in ast.unparse(n.test))
        first = next((i for i, n in enumerate(body) if isinstance(n, (ast.Import, ast.ImportFrom))
                      and "khaos_attribution" in ast.unparse(n)), len(body))
        assert stub < first, f"{rel} imports the contract before it checks which copy it runs on"


def test_every_script_the_engines_launch_is_one_that_checks():
    """A script path an engine starts by name is a checked engine script."""
    checked = {rel for _, rel in _scripts()}
    for path in _files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "with_name"
                    and node.args and isinstance(node.args[0], ast.Constant) and str(node.args[0].value).endswith(".py")):
                target = (path.parent / node.args[0].value).relative_to(ENGINES).as_posix()
                if target == "launch.py":
                    continue
                assert target in checked, f"{_rel(path)}:{node.lineno} starts {target}, which does not check its launcher"


def test_a_trainer_driver_starts_through_the_launcher_and_its_record_keeps_its_own_command(tmp_path, monkeypatch):
    """The record names the driver as the run's readers expect; the process itself is started through the launcher."""
    import json

    from khaos_attribution.engines import process_engine as PE
    from khaos_attribution.engines import worker_process as W
    from khaos_attribution.engines.sa3 import train as T
    started = []

    def fake_popen(argv, **_kw):
        started.append(argv)
        raise RuntimeError("stop here")

    monkeypatch.setattr(PE.subprocess, "Popen", fake_popen)
    session = PE.ProcessSession(PE.TrainingApi(root=tmp_path, runtime="torch", python=tmp_path / "py"), {},
                                checkpoint_root=tmp_path, tensor_dir=tmp_path, output_dir=tmp_path / "run", seed=1,
                                steps={"per_epoch": 1, "total": 1})
    session.save_dir.mkdir(parents=True)
    cmd = [str(tmp_path / "py"), str(T.TRAIN_DRIVER), "--steps", "1"]
    gen = session._run_trainer(cmd, {"command": cmd}, cwd=tmp_path, env={}, follow=lambda: iter(()), stop_tree=lambda p: None)
    try:
        next(gen)
    except (RuntimeError, StopIteration):
        pass
    assert started == [[cmd[0], str(W.LAUNCHER), *cmd[1:]]]
    assert json.loads((session.save_dir / "train_command.json").read_text())["command"] == cmd
