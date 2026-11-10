"""One engine worker as a child process: the request as JSON on stdin, the reply as the last JSON line of stdout,
a stop request that kills it and keeps nothing. Engines whose code lives in their own environment run through this.
Every such child takes its environment from engine_env, which starts from what the host registered."""

from __future__ import annotations

import json
import logging
import os
import queue
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Far past any render inside the engines' bounds (two minutes of audio, 200 steps); longer work names its own limit.
DEFAULT_DEADLINE_S = 3600.0
# As a timeout: DEFAULT_DEADLINE_S, read when the worker starts. None waits for ever.
DEFAULT = object()
ROW_PREFIX = "khaos-row "
STOP_GRACE_S = 3.0
POLL_S = 2.0
# How long a worker that has exited is waited on for a pipe something it started still holds.
PIPE_GRACE_S = 2.0
# Carried by a worker and by everything it starts, so what it leaves behind is found after it has gone.
MARK_ENV = "KHAOS_ENGINE_WORKER"
POSIX_ONLY = "a process tree is told apart with POSIX signals"
LINUX_ONLY = "the process table is read from /proc on Linux only"


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

def _start(python: Path, script: Path, *, cwd: Path, env: dict, stderr) -> subprocess.Popen:
    """The worker, in its host's process group so the host's own signals reach it, and marked so its tree is found."""
    mark = secrets.token_hex(8)
    proc = subprocess.Popen([str(python), str(script)], cwd=str(cwd), env={**env, MARK_ENV: mark}, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=stderr, text=True, errors="replace")
    proc.khaos_mark = mark
    return proc


def run(python: Path, script: Path, request: dict, *, cwd: Path, env: dict, label: str, should_stop=None,
        timeout=DEFAULT, required: str | None = None) -> dict:
    """The worker's reply; WorkerFailed names the worker by ``label`` and carries the tail of its stderr."""
    if timeout is DEFAULT:
        timeout = DEFAULT_DEADLINE_S
    try:
        proc = _start(python, script, cwd=cwd, env=env, stderr=subprocess.PIPE)
    except OSError as exc:
        raise WorkerFailed(f"the {label} worker could not start: {exc}") from exc
    try:
        stdout, stderr = communicate(proc, json.dumps(request), should_stop, timeout, label)
    finally:
        # Whatever ended the wait, an interrupt in the caller included, nothing is left holding the device.
        stop_tree(proc)
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


def _reader(pipe, take) -> threading.Thread:
    """A pipe read to its end on a thread of its own, each line handed to ``take`` and None after the last."""
    def drain():
        try:
            for line in pipe:
                take(line)
        except (OSError, ValueError):
            pass   # closed under the reader: what was read stands
        finally:
            take(None)
            try:
                pipe.close()
            except OSError:
                pass
    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return thread


def _released(proc, readers) -> None:
    """After the worker has exited: its pipes are waited on briefly, then whatever it left holding them is stopped."""
    def wait(seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        for r in readers:
            r.join(max(0.0, deadline - time.monotonic()))
        return not any(r.is_alive() for r in readers)
    if wait(PIPE_GRACE_S):
        return
    stop_tree(proc)
    if not wait(STOP_GRACE_S):
        log.warning("a worker's output is still held open after it exited (pid %s); its answer is read as it stands", proc.pid)


def stream(python: Path, script: Path, request: dict, *, cwd: Path, env: dict, label: str, on_row=None,
           required=()) -> dict:
    """The worker's reply, its ``khaos-row`` lines handed to ``on_row`` as they come, so long work shows its
    progress; WorkerFailed says what went wrong and names the reply keys in ``required`` that are missing."""
    lines, arriving = [], queue.Queue()
    # stderr to a file, never a pipe: a worker that writes much to it would block while stdout is being read.
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as err:
        try:
            proc = _start(python, script, cwd=cwd, env=env, stderr=err)
        except OSError as exc:
            raise WorkerFailed(f"the {label} worker could not start: {exc}") from exc
        try:
            try:
                proc.stdin.write(json.dumps(request))
                proc.stdin.close()
            except BrokenPipeError:
                pass
            reader = _reader(proc.stdout, arriving.put)
            while True:
                try:
                    line = arriving.get(timeout=POLL_S)
                except queue.Empty:
                    if proc.poll() is not None:
                        # The worker has gone and its output has not ended: something it started holds the pipe.
                        _released(proc, [reader])
                        if reader.is_alive():
                            arriving.put(None)
                    continue
                if line is None:
                    break
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
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
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

_warned: set = set()


def _warn_once(what: str, message: str) -> None:
    if what not in _warned:
        _warned.add(what)
        log.warning(message)


def _parents_from_proc(root: Path) -> dict:
    """pid -> parent pid from ``<root>/<pid>/stat``; the name between brackets may itself hold brackets and spaces."""
    out = {}
    for entry in os.listdir(root):
        if not entry.isdigit():
            continue
        try:
            text = (Path(root) / entry / "stat").read_text(errors="replace")
            out[int(entry)] = int(text[text.rindex(")") + 1:].split()[1])
        except (OSError, ValueError, IndexError):
            continue   # gone between the listing and the read
    return out


def _ps(*columns: str) -> list | None:
    """ps's rows as (pid, rest), or None where there is no ps to ask."""
    try:
        done = subprocess.run(["ps", "-A", "-ww", *columns], capture_output=True, text=True, errors="replace", timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in done.stdout.splitlines():
        pid, _, rest = line.strip().partition(" ")
        if pid.isdigit():
            rows.append((int(pid), rest))
    return rows if done.returncode == 0 else None


def _parents() -> dict | None:
    """pid -> parent pid for every process: /proc on Linux, ps elsewhere; None where neither can be read."""
    if sys.platform.startswith("linux"):
        try:
            found = _parents_from_proc(Path("/proc"))
        except OSError:
            found = {}
        if found:
            return found
    rows = _ps("-o", "pid=,ppid=")
    if rows is None:
        return None
    return {pid: int(rest) for pid, rest in rows if rest.strip().isdigit()}


def _marked(mark: str) -> set | None:
    """The processes whose environment carries a worker's mark: its tree, whoever their parent now is."""
    needle = f"{MARK_ENV}={mark}"
    if sys.platform.startswith("linux"):
        found = set()
        try:
            entries = [e for e in os.listdir("/proc") if e.isdigit()]
        except OSError:
            entries = None
        for entry in entries or ():
            try:
                if needle.encode() in (Path("/proc") / entry / "environ").read_bytes():
                    found.add(int(entry))
            except OSError:
                continue   # another user's, or gone
        if entries:
            return found
    rows = _ps("-E", "-o", "pid=,command=")
    return None if rows is None else {pid for pid, rest in rows if needle in rest}


def _below(pid: int, parents: dict) -> set:
    children: dict = {}
    for child, parent in parents.items():
        children.setdefault(parent, []).append(child)
    found, walk = set(), [pid]
    while walk:
        for child in children.get(walk.pop(), ()):
            if child not in found:
                found.add(child)
                walk.append(child)
    return found


def _tree(proc) -> set:
    """Everything the process started that still runs: by parent while it lives, by its mark when it carries one."""
    parents = _parents() if proc.poll() is None else {}
    mark = getattr(proc, "khaos_mark", None)
    marked = _marked(mark) if mark else set()
    if parents is None or marked is None:
        _warn_once("tree", "the process table cannot be read on this machine (no /proc and no ps): a worker is stopped, "
                           "but what a worker started may be left running")
    return (_below(proc.pid, parents or {}) | (marked or set())) - {proc.pid, os.getpid()}


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


def _stop_windows(proc) -> None:
    """taskkill takes the worker and its tree; where it cannot, the worker alone is killed and that is said."""
    if proc.poll() is None:
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as exc:
            _warn_once("taskkill", f"taskkill could not be run ({exc}): a worker is stopped, but what it started may be left running")
        if proc.poll() is None:
            proc.kill()
    proc.wait()


def stop_tree(proc) -> None:
    """The process and whatever it started: asked first, killed after a short grace, always reaped. The tree is read
    from the process table, so the worker stays in its host's process group and the host's own signals reach it."""
    if os.name == "nt":
        return _stop_windows(proc)
    if proc.poll() is not None and not getattr(proc, "khaos_mark", None):
        return
    # Read while the tree stands: a process whose parent has exited no longer names it.
    below = _tree(proc)
    if proc.poll() is None:
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
    """stdin written, the pipes drained on threads, the worker polled: a stop between polls cancels the work, and a
    worker that has exited is not waited on for a pipe something it started still holds."""
    try:
        proc.stdin.write(stdin)
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass   # the worker died before reading; its exit code and stderr say why
    out, err = [], []
    readers = [_reader(proc.stdout, out.append), _reader(proc.stderr, err.append)]
    deadline = time.monotonic() + timeout if timeout else None
    while True:
        try:
            proc.wait(timeout=POLL_S)
            break
        except subprocess.TimeoutExpired:
            if should_stop is not None and should_stop():
                raise WorkerCancelled()
            if deadline is not None and time.monotonic() > deadline:
                raise WorkerFailed(f"the {label} worker ran past its time limit")
    _released(proc, readers)
    return "".join(x for x in out if x is not None), "".join(x for x in err if x is not None)
