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


@pytest.mark.skipif(not POSIX, reason="the process table is read with ps")
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
