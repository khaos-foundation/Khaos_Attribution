"""What every engine's provision slot builds its environment with: the report a --check prints, uv, a quiet run of a
command without the room's PYTHONPATH, a venv on a pinned Python, a usable git, and a pinned checkout unpacked from
its archive when git is not."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from khaos_attribution import weights as mirror
from khaos_attribution.engines.worker_process import host_env

@dataclass
class Report:
    checks: list = field(default_factory=list)

    def ok(self, what: str) -> None:
        self.checks.append((True, what))

    def bad(self, what: str) -> None:
        self.checks.append((False, what))

    def note(self, what: str) -> None:
        """Said, never failed on."""
        self.checks.append((None, what))

    @property
    def passed(self) -> bool:
        return all(ok is not False for ok, _ in self.checks)

    def text(self) -> str:
        mark = {True: "ok", False: "FAIL", None: "note"}
        return "\n".join(f"[{mark[ok]}] {what}" for ok, what in self.checks) + \
            ("\nCHECK OK" if self.passed else "\nCHECK FAILED")


def uv_path() -> str:
    found = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    if not Path(found).exists():
        raise RuntimeError("uv is not installed; the engine's environment is made with it")
    return found


def run(cmd: list, cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    # The engine's interpreters never see the room's PYTHONPATH: its packages are not theirs.
    env = {k: v for k, v in (env if env is not None else host_env()).items() if k != "PYTHONPATH"}
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env, capture_output=True, text=True)


def make_venv(uv: str, venv: Path, python: Path, version: str, *, fallback: bool = False) -> tuple:
    """A fresh venv at ``venv`` on Python ``version``, or with ``fallback`` on the Python running this: (ok, note).
    The note is empty on ``version``, says why on the fallback, and gives uv's own words when nothing can be made."""
    # A folder left by an earlier failed try holds no interpreter; uv refuses it unless told to replace it.
    env = {**host_env(), "UV_VENV_CLEAR": "1"}
    done = run([uv, "venv", "-q", "--python", version, venv], env=env)
    note = ""
    if done.returncode != 0:
        why = done.stderr.strip()[:400]
        if not (fallback and (3, 11) <= sys.version_info[:2] <= (3, 12)):
            return False, f"the engine's Python {version} environment could not be made: {why}"
        here = f"{sys.version_info.major}.{sys.version_info.minor}"
        again = run([uv, "venv", "-q", "--python", sys.executable, venv], env=env)
        if again.returncode != 0:
            return False, (f"the engine's Python {version} environment could not be made: {why}; "
                           f"nor on this Python {here} ({sys.executable}): {again.stderr.strip()[:400]}")
        note = f"Python {version} could not be had ({why}); the environment was made on this Python {here}"
    if not python.is_file():
        return False, f"uv made the environment but left no interpreter at {python}"
    return True, note


def git_usable() -> bool:
    """Whether git here runs without a dialog: a Mac without the developer tools has a git stub that opens one."""
    if not shutil.which("git"):
        return False
    if sys.platform == "darwin":
        return run(["xcode-select", "-p"]).returncode == 0
    return True


def archive_url(repo: str, commit: str) -> str:
    base = repo[:-4] if repo.endswith(".git") else repo
    return f"{base}/archive/{commit}.tar.gz"


def remove_tree(path: Path) -> None:
    """``path`` removed, read-only files too (git marks its objects so on Windows); raises when any of it stays."""
    import stat  # noqa: PLC0415

    def writable_then_again(func, p, _exc):
        for q in (p, os.path.dirname(p)):
            try:
                os.chmod(q, os.stat(q).st_mode | stat.S_IWRITE | stat.S_IWUSR)
            except OSError:
                pass
        func(p)
    if path.exists():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=writable_then_again)
        else:
            shutil.rmtree(path, onerror=writable_then_again)
    if path.exists():
        raise OSError(f"{path} could not be removed")


def unpack_archive(url: str, root: Path, commit: str, kept: tuple) -> None:
    """The engine's code at the commit from its archive, when git is not usable here: the top folder stripped,
    the commit recorded in .khaos-commit so the pin can still be read; an earlier unpack's environments and
    weights carried across."""
    import io  # noqa: PLC0415
    import tarfile  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    from khaos_attribution.engines.pins import COMMIT_FILE  # noqa: PLC0415
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 — the descriptor's own code repository
        data = resp.read()
    part = root.with_name(root.name + ".part")
    old = root.with_name(root.name + ".old")
    # Both are cleared before anything of the checkout moves, so a failure leaves it whole.
    remove_tree(part)
    remove_tree(old)
    part.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if "/" in m.name]
        for m in members:
            m.name = m.name.split("/", 1)[1]
        tar.extractall(part, members=members, filter="data")
    (part / COMMIT_FILE).write_text(commit + "\n", encoding="utf-8")
    for rel in kept:
        if (root / rel).exists():
            (part / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(part / rel, ignore_errors=True)
            shutil.move(str(root / rel), str(part / rel))
    if root.exists():
        try:
            root.replace(old)
        except OSError:
            # The checkout could not be set aside (a file in it is open): its environments and weights go back.
            for rel in kept:
                if (part / rel).exists():
                    (root / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(part / rel), str(root / rel))
            raise
    part.replace(root)
    try:
        remove_tree(old)
    except OSError:
        pass  # the checkout is in place; the next move clears what is left


def from_mirror(fetch) -> bool:
    """Whether the set was served and checked; a source that cannot serve falls back to the hub library, but a file that
    fails its pin is refused (PinMismatch) and nothing falls back."""
    try:
        return fetch() is not None
    except mirror.PinMismatch:
        raise
    except mirror.Unserved as exc:
        if exc.pinned:
            raise
        return False
    except (mirror.MirrorError, OSError):
        return False


def refused(repo: str, exc: Exception) -> tuple:
    """The report line for a pinned set that no source served correctly: nothing was installed and no fallback was tried."""
    if isinstance(exc, mirror.PinMismatch):
        return (False, f"{repo} failed its pinned hash from every source; nothing was installed")
    return (False, f"{repo} could not be downloaded: {exc}")
