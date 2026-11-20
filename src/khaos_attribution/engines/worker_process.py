"""One engine worker as a child process: the request as JSON on stdin, the reply as the last JSON line of stdout,
a stop request that kills it and keeps nothing. Engines whose code lives in their own environment run through this.
Every such child takes its environment from engine_env, which starts from what the host registered."""

from __future__ import annotations

import contextlib
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
import traceback
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
    """The worker could not start, exited badly or wrote no reply; the sentence says which, and ``why`` the reason
    alone (the reply's error or the tail of stderr), for a caller that words the failure itself."""

    def __init__(self, message: str, why: str | None = None):
        super().__init__(message)
        self.why = message if why is None else why


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


# ── how a child is started ────────────────────────────────────────────────────

LAUNCHER = Path(__file__).with_name("launch.py")
_CONTRACT = LAUNCHER.parent.parent


def _ours(script) -> bool:
    try:
        return Path(script).resolve().is_relative_to(_CONTRACT.resolve())
    except OSError:
        return False


def command(python, script, *args) -> list:
    """The argv that runs ``script`` in ``python``: a script of the contract's own goes through the launcher, so it
    runs on this room's contract, never an engine environment's copy; anyone else's script runs directly."""
    head = [str(python), str(LAUNCHER)] if _ours(script) else [str(python)]
    return [*head, str(script), *(str(a) for a in args)]


# ── one request, one reply ────────────────────────────────────────────────────

def _start(python: Path, script: Path, *, cwd: Path, env: dict, stderr, args=()) -> subprocess.Popen:
    """The worker, in its host's process group so the host's own signals reach it, and marked so its tree is found."""
    mark = secrets.token_hex(8)
    proc = subprocess.Popen(command(python, script, *args), cwd=str(cwd), env={**env, MARK_ENV: mark},
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, errors="replace")
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
        _log_failure(label, stderr)
        raise WorkerFailed(str(reply["error"]))
    if proc.returncode != 0 or not isinstance(reply, dict) or (required and required not in reply):
        _log_failure(label, stderr)
        raise _ended(label, proc.returncode, stderr, stdout)
    return reply


def _exit_said(returncode) -> str:
    """An exit status as a person reads it: the code, or the signal that ended the worker."""
    if returncode is not None and returncode < 0:
        try:
            return f"killed by {signal.Signals(-returncode).name}"
        except ValueError:
            return f"killed by signal {-returncode}"
    return str(returncode)


def _ended(label: str, returncode, stderr: str | None, stdout: str | None) -> WorkerFailed:
    """A worker that ended without a reply: how it exited and what it last said, its stderr first, else its stdout,
    since a worker killed for memory often writes nothing to stderr."""
    tail = "\n".join((stderr or "").strip().splitlines()[-8:]) or (stdout or "").strip()[-300:]
    said = f": {tail}" if tail else ", having written nothing"
    return WorkerFailed(f"the {label} worker failed ({_exit_said(returncode)}){said}", why=tail)


def _log_failure(label: str, stderr: str | None) -> None:
    """A failed worker's last stderr line at ERROR and its traceback at DEBUG: a traceback can hold a request's
    words (a prompt, lyrics), which a server's kept logs must not."""
    lines = (stderr or "").strip().splitlines()
    if lines:
        log.error("the %s worker failed: %s", label, lines[-1])
        log.debug("the %s worker's stderr ends:\n%s", label, "\n".join(lines[-40:]))


def call(python: Path, script: Path, args, *, cwd: Path, env: dict, label: str, should_stop=None,
         timeout=None) -> tuple:
    """A plain command in an engine's environment, marked, bounded and stopped as a worker is: (exit code, stdout,
    stderr). None waits for ever."""
    try:
        proc = _start(python, script, cwd=cwd, env=env, stderr=subprocess.PIPE, args=args)
    except OSError as exc:
        raise WorkerFailed(f"the {label} could not start: {exc}") from exc
    try:
        stdout, stderr = communicate(proc, "", should_stop, timeout, label)
    finally:
        stop_tree(proc)
    return proc.returncode, stdout, stderr


def answer(work) -> int:
    """A worker's main: the request from stdin, ``work(request)``'s reply as the last line of stdout; a failure is
    replied as its type and message, and its traceback written to stderr for the caller's log."""
    req = json.loads(sys.stdin.read())
    try:
        reply = work(req)
    except Exception as exc:  # noqa: BLE001 — the reply carries it
        traceback.print_exc()
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("\n" + json.dumps(reply) + "\n")
    sys.stdout.flush()
    return 0 if not reply.get("error") else 1


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
           required=(), should_stop=None, timeout=None) -> dict:
    """The worker's reply, its ``khaos-row`` lines handed to ``on_row`` as they come; a stop or the deadline (None
    waits for ever) ends it as ``run`` does. WorkerFailed says what went wrong and names missing ``required`` keys."""
    if timeout is DEFAULT:
        timeout = DEFAULT_DEADLINE_S
    deadline = time.monotonic() + timeout if timeout else None
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
                    line = ""
                    if proc.poll() is not None:
                        # The worker has gone and its output has not ended: something it started holds the pipe.
                        _released(proc, [reader])
                        if reader.is_alive():
                            arriving.put(None)
                if line is None:
                    break
                # Read after every line as well as every quiet poll, so a worker that never falls silent still stops.
                if should_stop is not None and should_stop():
                    raise WorkerCancelled()
                if deadline is not None and time.monotonic() > deadline:
                    raise WorkerFailed(f"the {label} worker ran past its time limit")
                if not line:
                    continue
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
    stdout = "".join(lines)
    reply = _last_json(stdout)
    if not isinstance(reply, dict) or reply.get("error") or returncode != 0:
        _log_failure(label, stderr)
        error = reply.get("error") if isinstance(reply, dict) else None
        if error:
            raise WorkerFailed(f"the {label} worker failed: {error}", why=str(error))
        raise _ended(label, returncode, stderr, stdout)
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


def _marked(mark: str) -> tuple | None:
    """(the processes whose environment carries a worker's mark, those whose environment cannot be read), from the
    environment alone, never the command line; None where the process table cannot be read."""
    needle = f"{MARK_ENV}={mark}"
    found, unread = set(), set()
    if sys.platform.startswith("linux"):
        try:
            entries = [e for e in os.listdir("/proc") if e.isdigit()]
        except OSError:
            entries = None
        for entry in entries or ():
            try:
                env = (Path("/proc") / entry / "environ").read_bytes()
            except PermissionError:
                unread.add(int(entry))
                continue
            except OSError:
                continue   # gone
            if needle.encode() in env.split(b"\0"):
                found.add(int(entry))
        if entries:
            return found, unread
    # ps -E prints the environment after the command line; ps without it prints the command line alone.
    shown, commands = _ps("-E", "-o", "pid=,command="), _ps("-o", "pid=,command=")
    if shown is None or commands is None:
        return None
    commands = dict(commands)
    for pid, rest in shown:
        command = commands.get(pid)
        if command is None or not rest.startswith(command):
            continue   # started or changed between the two reads
        env = rest[len(command):]
        if not env.strip():
            unread.add(pid)   # macOS shows no environment for its own programs
        elif f" {needle} " in env + " ":
            found.add(pid)
    return found, unread


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
    """Everything the process started that still runs: what carries its mark, and what descends by parent from the
    process while it lives or from anything carrying its mark."""
    mark = getattr(proc, "khaos_mark", None)
    seen = _marked(mark) if mark else (set(), set())
    marked, unread = seen or (set(), set())
    roots = marked | ({proc.pid} if proc.poll() is None else set())
    parents = _parents() if roots else {}
    if parents is None or seen is None:
        _warn_once("tree", "the process table cannot be read on this machine (no /proc and no ps): a worker is stopped, "
                           "but what a worker started may be left running")
    below = set().union(*(_below(root, parents or {}) for root in roots))
    if below & unread:
        _warn_once("environment", f"the environment of a process a worker started cannot be read (pid {min(below & unread)}): "
                                  "it is found through its parent, and may be left running once that parent has gone")
    return (below | marked) - {proc.pid, os.getpid()}


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


# ── files a worker writes ─────────────────────────────────────────────────────

# A worker writes a file the caller keeps at this name beside it, renamed into place only when whole.
PART_SUFFIX = ".part"


def part_of(path) -> Path:
    return Path(str(path) + PART_SUFFIX)


@contextlib.contextmanager
def scratch_file(folder: Path, prefix: str, suffix: str):
    """A file of this call's own in ``folder`` for a worker to write, removed whatever ends the block."""
    folder.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(folder))
    os.close(fd)
    try:
        yield Path(name)
    finally:
        Path(name).unlink(missing_ok=True)
