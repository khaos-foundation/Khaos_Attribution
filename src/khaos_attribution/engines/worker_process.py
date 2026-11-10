"""One engine worker as a child process: the request as JSON on stdin, the reply as the last JSON line of stdout,
a stop request that kills it and keeps nothing. Engines whose code lives in their own environment run through this.
Every such child takes its environment from engine_env, which starts from what the host registered."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

# Far past any render inside the engines' bounds (two minutes of audio, 200 steps); longer work names its own limit.
DEFAULT_DEADLINE_S = 3600.0
# As a timeout: DEFAULT_DEADLINE_S, read when the worker starts. None waits for ever.
DEFAULT = object()
ROW_PREFIX = "khaos-row "
STOP_GRACE_S = 3.0


class WorkerFailed(RuntimeError):
    """The worker could not start, exited badly or wrote no reply; the sentence says which."""


class WorkerCancelled(RuntimeError):
    """A stop was asked while the worker ran; it was killed and nothing was kept."""


# ── the environment a child is given ──────────────────────────────────────────

_host_env = None


def set_host_env(source) -> None:
    """The host's one say over what engine children inherit: a callable giving the environment they start from,
    called for each child. None puts this process's own environment back."""
    global _host_env
    _host_env = source


def host_env() -> dict:
    return dict(os.environ if _host_env is None else _host_env())


def engine_env(root, **extra: str) -> dict:
    """What an engine child is started with: the host's environment, the engine's checkout importable, the hub
    offline unless the host says otherwise, no experiment tracker, and the engine's own settings."""
    base = host_env()
    return {**base, "PYTHONPATH": str(root), "HF_HUB_OFFLINE": base.get("HF_HUB_OFFLINE", "1"),
            "WANDB_MODE": "disabled", **extra}


# ── one request, one reply ────────────────────────────────────────────────────

def run(python: Path, script: Path, request: dict, *, cwd: Path, env: dict, label: str, should_stop=None,
        timeout=DEFAULT, required: str | None = None) -> dict:
    """The worker's reply; WorkerFailed names the worker by ``label`` and carries the tail of its stderr."""
    if timeout is DEFAULT:
        timeout = DEFAULT_DEADLINE_S
    try:
        proc = subprocess.Popen([str(python), str(script)], cwd=str(cwd), env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    except OSError as exc:
        raise WorkerFailed(f"the {label} worker could not start: {exc}") from exc
    try:
        stdout, stderr = communicate(proc, json.dumps(request), should_stop, timeout, label)
    except BaseException:
        # Not only the worker's own errors: an interrupt in the caller must not leave it holding the device.
        stop_tree(proc)
        raise
    reply = _last_json(stdout)
    if isinstance(reply, dict) and reply.get("error"):
        raise WorkerFailed(str(reply["error"]))
    if proc.returncode != 0 or not isinstance(reply, dict) or (required and required not in reply):
        tail = "\n".join((stderr or "").strip().splitlines()[-8:]) or (stdout or "")[-300:]
        raise WorkerFailed(f"the {label} worker failed ({proc.returncode}): {tail}")
    return reply


def _last_json(stdout: str):
    last = (stdout or "").strip().splitlines()[-1:] or [""]
    try:
        return json.loads(last[0])
    except ValueError:
        return None


def stream(python: Path, script: Path, request: dict, *, cwd: Path, env: dict, label: str, on_row=None,
           required=()) -> dict:
    """The worker's reply, its ``khaos-row`` lines handed to ``on_row`` as they come, so long work shows its
    progress; WorkerFailed says what went wrong and names the reply keys in ``required`` that are missing."""
    lines = []
    # stderr to a file, never a pipe: a worker that writes much to it would block while stdout is being read.
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as err:
        try:
            proc = subprocess.Popen([str(python), str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=err, cwd=str(cwd), text=True, errors="replace", env=env)
        except OSError as exc:
            raise WorkerFailed(f"the {label} worker could not start: {exc}") from exc
        try:
            try:
                proc.stdin.write(json.dumps(request))
                proc.stdin.close()
            except BrokenPipeError:
                pass
            for line in proc.stdout:
                if not line.startswith(ROW_PREFIX):
                    lines.append(line)
                elif on_row is not None:
                    try:
                        on_row(json.loads(line[len(ROW_PREFIX):]))
                    except ValueError:
                        pass
            returncode = proc.wait()
        finally:
            # A worker left running would hold the device's memory after its caller has gone.
            stop_tree(proc)
            for pipe in (proc.stdin, proc.stdout):
                if pipe is not None and not pipe.closed:
                    pipe.close()
        err.seek(0)
        stderr = err.read()
    reply = _last_json("".join(lines))
    if not isinstance(reply, dict) or reply.get("error") or returncode != 0:
        why = (reply.get("error") if isinstance(reply, dict) else None) or "\n".join(stderr.splitlines()[-6:])
        raise WorkerFailed(f"the {label} worker failed: {why}")
    missing = [k for k in required if k not in reply]
    if missing:
        raise WorkerFailed(f"the {label} worker's reply lacks {', '.join(missing)}")
    return reply


# ── stopping ──────────────────────────────────────────────────────────────────

def _descendants(pid: int) -> list:
    """Every process under ``pid`` from one reading of the process table; none where there is no ps to ask."""
    if os.name == "nt":
        return []
    try:
        table = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    children: dict = {}
    for line in table.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found, queue = [], [pid]
    while queue:
        for child in children.get(queue.pop(0), []):
            if child not in found:
                found.append(child)
                queue.append(child)
    return found


def _signal(pid: int, sig) -> None:
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_tree(proc) -> None:
    """The process and whatever it spawned: asked first, killed after a short grace, always reaped. The tree is
    read from the process table, so the child stays in its host's process group and the host's own signals reach it."""
    if proc.poll() is not None:
        return
    # Read while the tree stands: a process whose parent has exited no longer names it.
    below = set(_descendants(proc.pid))
    proc.terminate()
    for pid in below:
        _signal(pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_GRACE_S
    while time.monotonic() < deadline and (proc.poll() is None or below):
        time.sleep(0.1)
        # Dropped the moment it is gone, so a number the system hands out again is never signalled.
        below = {pid for pid in below if _running(pid)}
    if proc.poll() is None:
        proc.kill()
    for pid in below:
        _signal(pid, signal.SIGKILL)
    proc.wait()


def communicate(proc, stdin: str, should_stop, timeout, label: str):
    """stdin written, then polled: a stop between polls cancels the work."""
    try:
        proc.stdin.write(stdin)
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass   # the worker died before reading; its exit code and stderr say why
    proc.stdin = None   # communicate() would flush it again
    deadline = time.monotonic() + timeout if timeout else None
    while True:
        try:
            out, err = proc.communicate(timeout=2.0)
            return out or "", err or ""
        except subprocess.TimeoutExpired:
            if should_stop is not None and should_stop():
                raise WorkerCancelled()
            if deadline is not None and time.monotonic() > deadline:
                raise WorkerFailed(f"the {label} worker ran past its time limit")
