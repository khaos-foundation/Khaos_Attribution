"""One engine worker as a child process: the request as JSON on stdin, the reply as the last JSON line of stdout,
a stop request that kills it and keeps nothing. Engines whose code lives in their own environment run through this."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path


class WorkerFailed(RuntimeError):
    """The worker could not start, exited badly or wrote no reply; the sentence says which."""


class WorkerCancelled(RuntimeError):
    """A stop was asked while the worker ran; it was killed and nothing was kept."""


def run(python: Path, script: Path, request: dict, *, cwd: Path, env: dict, label: str, should_stop=None,
        timeout: float | None = None, required: str | None = None) -> dict:
    """The worker's reply; WorkerFailed names the worker by ``label`` and carries the tail of its stderr."""
    try:
        proc = subprocess.Popen([str(python), str(script)], cwd=str(cwd), env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    except OSError as exc:
        raise WorkerFailed(f"the {label} worker could not start: {exc}") from exc
    try:
        stdout, stderr = communicate(proc, json.dumps(request), should_stop, timeout, label)
    except (WorkerCancelled, WorkerFailed):
        stop_tree(proc)
        raise
    last = (stdout or "").strip().splitlines()[-1:] or [""]
    try:
        reply = json.loads(last[0])
    except ValueError:
        reply = None
    if isinstance(reply, dict) and reply.get("error"):
        raise WorkerFailed(str(reply["error"]))
    if proc.returncode != 0 or not isinstance(reply, dict) or (required and required not in reply):
        tail = "\n".join((stderr or "").strip().splitlines()[-8:]) or (stdout or "")[-300:]
        raise WorkerFailed(f"the {label} worker failed ({proc.returncode}): {tail}")
    return reply


def stop_tree(proc) -> None:
    """The process and whatever it spawned: asked first, killed after a short grace, always reaped."""
    if proc.poll() is not None:
        return
    for ask in (proc.terminate, proc.kill):
        try:
            if hasattr(os, "getpgid") and os.getpgid(proc.pid) == proc.pid:
                os.killpg(proc.pid, signal.SIGTERM if ask is proc.terminate else signal.SIGKILL)
            else:
                ask()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        deadline = time.monotonic() + 3.0
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if proc.poll() is not None:
            break
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
