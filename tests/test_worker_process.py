"""An engine child is bounded and owned: it has a deadline, a stop reaches whatever it started, nothing that ends
the wait leaves it running, and the host decides what its environment carries."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from khaos_attribution.engines import worker_process as W

POSIX = os.name != "nt"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _gone(pid: int, within: float = 8.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def host_env_reset():
    yield
    W.set_host_env(None)


@pytest.mark.skipif(not POSIX, reason=W.POSIX_ONLY)
def test_a_stop_reaches_what_the_worker_started(tmp_path):
    """A worker that started a process of its own and forwards nothing: both are gone after the stop."""
    script = tmp_path / "parent.py"
    script.write_text("import subprocess, sys, time\n"
                      "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                      "print(child.pid, flush=True)\ntime.sleep(120)\n")
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    grandchild = int(proc.stdout.readline())
    try:
        W.stop_tree(proc)
        assert proc.poll() is not None
        assert _gone(grandchild), "the process the worker started outlived the stop"
    finally:
        proc.stdout.close()
        if _alive(grandchild):
            os.kill(grandchild, 9)


def _sleeper(tmp_path, seconds: int = 30) -> tuple:
    script, pidfile = tmp_path / "sleeper.py", tmp_path / "pid"
    script.write_text(f"import os, sys, time\nopen({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
                      f"sys.stdin.read()\ntime.sleep({seconds})\nprint('{{\"seed\": 1}}')\n")
    return script, pidfile


def test_a_worker_past_the_default_deadline_is_stopped(tmp_path, monkeypatch):
    """With no limit named, the module's own deadline ends a worker that never answers."""
    script, pidfile = _sleeper(tmp_path)
    monkeypatch.setattr(W, "DEFAULT_DEADLINE_S", 0.2)
    start = time.monotonic()
    with pytest.raises(W.WorkerFailed, match="time limit"):
        W.run(Path(sys.executable), script, {}, cwd=tmp_path, env=dict(os.environ), label="test")
    assert time.monotonic() - start < 15
    assert _gone(int(pidfile.read_text()))


def test_a_caller_may_wait_longer_than_the_default_or_for_ever(tmp_path, monkeypatch):
    script, _ = _sleeper(tmp_path, seconds=3)
    monkeypatch.setattr(W, "DEFAULT_DEADLINE_S", 0.2)
    assert W.run(Path(sys.executable), script, {}, cwd=tmp_path, env=dict(os.environ), label="test", timeout=60)["seed"] == 1
    assert W.run(Path(sys.executable), script, {}, cwd=tmp_path, env=dict(os.environ), label="test", timeout=None)["seed"] == 1


def test_whatever_ends_the_wait_stops_the_worker(tmp_path):
    """An interrupt in the caller is not one of the worker's own errors; the worker is stopped all the same."""
    script, pidfile = _sleeper(tmp_path)

    def interrupted():
        if pidfile.exists():
            raise KeyboardInterrupt
        return False
    with pytest.raises(KeyboardInterrupt):
        W.run(Path(sys.executable), script, {}, cwd=tmp_path, env=dict(os.environ), label="test", should_stop=interrupted)
    assert _gone(int(pidfile.read_text()))


def test_rows_stream_to_the_caller_and_the_reply_is_the_last_line(tmp_path):
    script = tmp_path / "rows.py"
    script.write_text("import json, sys\nreq = json.loads(sys.stdin.read())\n"
                      "sys.stderr.write('x' * 200000)\n"
                      "for i in range(3):\n    print('khaos-row ' + json.dumps({'row': i}), flush=True)\n"
                      "print('a log line')\nprint(json.dumps({'rows': req['n']}))\n")
    rows = []
    reply = W.stream(Path(sys.executable), script, {"n": 3}, cwd=tmp_path, env=dict(os.environ), label="test",
                     on_row=rows.append, required=("rows",))
    assert reply == {"rows": 3} and rows == [{"row": 0}, {"row": 1}, {"row": 2}]
    with pytest.raises(W.WorkerFailed, match="the test worker's reply lacks matrix"):
        W.stream(Path(sys.executable), script, {"n": 3}, cwd=tmp_path, env=dict(os.environ), label="test", required=("matrix",))
    bad = tmp_path / "bad.py"
    bad.write_text("import sys\nsys.stdin.read()\nsys.stderr.write('the reason\\n')\nsys.exit(3)\n")
    with pytest.raises(W.WorkerFailed, match="the test worker failed: the reason"):
        W.stream(Path(sys.executable), bad, {}, cwd=tmp_path, env=dict(os.environ), label="test")


ECHO = ("import json, os, sys\nsys.stdin.read()\n"
        "print(json.dumps({'seed': 1, 'n_params': 1, 'n_tensors': 1, 'signature': 's', 'env': dict(os.environ)}))\n")


def _children_envs(tmp_path, monkeypatch) -> dict:
    """The environment each sa3 and mimetic child is started with: render and measure by running a worker that
    answers with its own, the trainer by the environment its processes are given."""
    from khaos_attribution.engines.mimetic import measure as MM, render as MR, train as MT
    from khaos_attribution.engines.sa3 import measure as SM, render as SR, train as ST
    echo = tmp_path / "echo.py"
    echo.write_text(ECHO)
    python = Path(sys.executable)
    monkeypatch.setattr(SR, "python_for", lambda root, runtime: python)
    monkeypatch.setattr(MR, "python_for", lambda root: python)
    for module, name in ((SR, "WORKER"), (MR, "WORKER"), (SM, "MEASURE_WORKER"), (MM, "MEASURE_WORKER")):
        monkeypatch.setattr(module, name, echo)
    adapter = tmp_path / "a.safetensors"
    return {"sa3 render": SR.run_worker(tmp_path, "torch", {})["env"],
            "mimetic render": MR.run_worker(tmp_path, {})["env"],
            "sa3 measure": SM._ask(SM.Handle(root=tmp_path, python=python, adapter=adapter), {"command": "parameters"})["env"],
            "mimetic measure": MM._ask(MM.Handle(root=tmp_path, python=python, adapter=adapter), {"command": "parameters"})["env"],
            "sa3 train": ST._env(ST.TrainingApi(root=tmp_path, runtime="torch", python=python)),
            "mimetic train": MT._env(MT.TrainingApi(root=tmp_path, runtime="torch", python=python))}


def test_the_host_decides_what_no_engine_child_inherits(tmp_path, monkeypatch, host_env_reset):
    """One registration keeps a server's own secrets out of every sa3 and mimetic render, measure and train process."""
    monkeypatch.setenv("HOST_SECRET_TOKEN", "s3cret")
    monkeypatch.setenv("HOST_PLAIN", "kept")
    W.set_host_env(lambda: {k: v for k, v in os.environ.items() if not k.startswith("HOST_SECRET_")})
    for who, env in _children_envs(tmp_path, monkeypatch).items():
        assert "HOST_SECRET_TOKEN" not in env, who
        assert env["HOST_PLAIN"] == "kept" and env["PYTHONPATH"] == str(tmp_path) and env["WANDB_MODE"] == "disabled", who
        assert env["HF_HUB_OFFLINE"] == "1", who


def test_with_no_host_registered_a_child_inherits_the_process_environment(tmp_path, monkeypatch, host_env_reset):
    monkeypatch.setenv("HOST_SECRET_TOKEN", "s3cret")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    envs = _children_envs(tmp_path, monkeypatch)
    assert all(env["HOST_SECRET_TOKEN"] == "s3cret" and env["HF_HUB_OFFLINE"] == "0" for env in envs.values())
    assert envs["sa3 measure"]["SA3_ROOT"] == envs["sa3 train"]["SA3_ROOT"] == str(tmp_path)
    assert envs["mimetic render"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_the_contract_probe_of_an_engine_interpreter_takes_the_host_environment_too(monkeypatch, host_env_reset):
    from khaos_attribution.engines.sa3 import provision as P
    monkeypatch.setenv("HOST_SECRET_TOKEN", "s3cret")
    code = "import os; print(os.environ.get('HOST_SECRET_TOKEN', 'absent'))"
    assert P._run([sys.executable, "-c", code]).stdout.strip() == "s3cret"
    W.set_host_env(lambda: {k: v for k, v in os.environ.items() if not k.startswith("HOST_SECRET_")})
    assert P._run([sys.executable, "-c", code]).stdout.strip() == "absent"


LEAVES_A_CHILD = ("import json, subprocess, sys\nsys.stdin.read()\n"
                  "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                  "print('khaos-row ' + json.dumps({'row': 0}), flush=True)\n"
                  "print(json.dumps({'seed': 1, 'child': child.pid}), flush=True)\nsys.exit(%d)\n")


def _kill(pid) -> None:
    if pid and _alive(pid):
        os.kill(pid, 9)


@pytest.mark.skipif(not POSIX, reason=W.POSIX_ONLY)
def test_a_worker_that_exits_leaving_a_child_on_its_output_is_answered_at_once_and_the_child_stopped(tmp_path):
    """The child holds the worker's pipes open; the worker's own exit decides, and nothing it started is left."""
    script = tmp_path / "leaves.py"
    script.write_text(LEAVES_A_CHILD % 0)
    start = time.monotonic()
    reply = W.run(Path(sys.executable), script, {}, cwd=tmp_path, env=dict(os.environ), label="test", timeout=8, required="seed")
    try:
        assert reply["seed"] == 1 and time.monotonic() - start < 7.5, "answered by the worker's exit, not its deadline"
        assert _gone(reply["child"]), "the process the worker left behind is still running"
    finally:
        _kill(reply["child"])


@pytest.mark.skipif(not POSIX, reason=W.POSIX_ONLY)
def test_a_worker_that_fails_leaving_a_child_fails_by_its_own_exit_status(tmp_path):
    script, seen = tmp_path / "leaves.py", {}
    script.write_text(LEAVES_A_CHILD.replace("print(json.dumps({'seed': 1, 'child': child.pid}), flush=True)",
                                             f"open({str(tmp_path / 'child')!r}, 'w').write(str(child.pid))") % 3)
    try:
        with pytest.raises(W.WorkerFailed, match=r"the test worker failed \(3\)"):
            W.run(Path(sys.executable), script, {}, cwd=tmp_path, env=dict(os.environ), label="test", timeout=8)
        seen["child"] = int((tmp_path / "child").read_text())
        assert _gone(seen["child"])
    finally:
        _kill(seen.get("child") or (int((tmp_path / "child").read_text()) if (tmp_path / "child").exists() else None))


@pytest.mark.skipif(not POSIX, reason=W.POSIX_ONLY)
def test_a_streaming_worker_that_exits_leaving_a_child_on_its_output_does_not_hang(tmp_path):
    import threading
    script, got = tmp_path / "leaves.py", {}
    script.write_text(LEAVES_A_CHILD % 0)

    def ask():
        got["reply"] = W.stream(Path(sys.executable), script, {}, cwd=tmp_path, env=dict(os.environ), label="test",
                                on_row=lambda row: got.setdefault("rows", []).append(row))
    asking = threading.Thread(target=ask, daemon=True)
    asking.start()
    asking.join(15)
    child = got.get("reply", {}).get("child")
    try:
        assert not asking.is_alive(), "the caller is still waiting on a pipe the worker's child holds"
        assert got["reply"]["seed"] == 1 and got["rows"] == [{"row": 0}] and _gone(child)
    finally:
        _kill(child)


def test_parents_are_read_from_proc_whatever_a_process_is_called(tmp_path):
    """A command name may hold spaces and brackets; the parent is the second field after the last bracket."""
    for pid, text in ((12, "12 (python3) S 7 12 7 0 -1"), (13, "13 (a b) c) R 12 13 7 0"), (14, "14 (tail) S 1 14 14")):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_text(text)
    (tmp_path / "self").mkdir()
    (tmp_path / "15").mkdir()
    assert W._parents_from_proc(tmp_path) == {12: 7, 13: 12, 14: 1}
    assert W._below(7, {12: 7, 13: 12, 14: 1}) == {12, 13}


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason=W.LINUX_ONLY)
def test_on_linux_a_stop_reaches_the_tree_with_no_ps_installed(tmp_path, monkeypatch):
    script = tmp_path / "parent.py"
    script.write_text("import subprocess, sys, time\n"
                      "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                      "print(child.pid, flush=True)\ntime.sleep(120)\n")
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    grandchild = int(proc.stdout.readline())
    monkeypatch.setenv("PATH", str(tmp_path))
    try:
        W.stop_tree(proc)
        assert _gone(grandchild)
    finally:
        proc.stdout.close()
        _kill(grandchild)


@pytest.mark.skipif(not POSIX, reason=W.POSIX_ONLY)
def test_a_machine_that_cannot_name_a_workers_children_says_so_once_and_still_stops_the_worker(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(W, "_parents", lambda: None)
    monkeypatch.setattr(W, "_marked", lambda token: None)
    monkeypatch.setattr(W, "_warned", set())
    with caplog.at_level("WARNING", logger=W.log.name):
        for _ in range(2):
            proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
            W.stop_tree(proc)
            assert proc.poll() is not None
    said = [r.getMessage() for r in caplog.records if "cannot be read" in r.getMessage()]
    assert len(said) == 1 and "what a worker started may be left running" in said[0]


def test_on_windows_the_tree_is_taken_by_taskkill_and_a_failure_is_said(monkeypatch, caplog):
    """Driven with a stand-in for taskkill: the command it is given, and the fall back to killing the worker alone."""
    calls = []

    class Proc:
        pid, dead, killed = 4321, False, False

        def poll(self):
            return 0 if self.dead else None

        def kill(self):
            self.killed = self.dead = True

        def wait(self, timeout=None):
            return 0
    took = Proc()
    monkeypatch.setattr(W.subprocess, "run", lambda cmd, **k: calls.append(cmd) or setattr(took, "dead", True))
    W._stop_windows(took)
    assert calls == [["taskkill", "/T", "/F", "/PID", "4321"]] and not took.killed
    left = Proc()
    monkeypatch.setattr(W, "_warned", set())
    monkeypatch.setattr(W.subprocess, "run", lambda cmd, **k: (_ for _ in ()).throw(FileNotFoundError("taskkill")))
    with caplog.at_level("WARNING", logger=W.log.name):
        W._stop_windows(left)
    assert left.killed and any("taskkill" in r.getMessage() for r in caplog.records)
