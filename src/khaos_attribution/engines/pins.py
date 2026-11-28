"""Which code an engine checkout is: the HEAD of a git checkout, read with every variable that would point git at
another repository stripped. Never an exception, never a guess: None when it cannot be read."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from khaos_attribution.files import real_rglob

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# Environment variables that make git answer about another repository no matter what `-C <dir>` says.
GIT_REPO_ENV_OVERRIDES = ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                          "GIT_CEILING_DIRECTORIES")


def git_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in GIT_REPO_ENV_OVERRIDES}


COMMIT_FILE = ".khaos-commit"


def head_from_files(git_dir: Path) -> str | None:
    """HEAD read from a repository's own files, for a machine where git cannot run: a detached SHA, or the ref it
    names in refs/ or packed-refs; None for anything else."""
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head[5:].strip()
            loose = git_dir / ref
            if loose.is_file():
                head = loose.read_text(encoding="utf-8").strip()
            else:
                packed = (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines()
                head = next((line.split()[0] for line in packed if line.endswith(" " + ref)), "")
    except (OSError, ValueError):
        return None
    return head if _SHA_RE.match(head) else None


def checkout_commit(root) -> str | None:
    """The 40-character HEAD of the checkout at *root*, or None (not a checkout, a slow git, no full SHA).
    A checkout unpacked from an archive carries its commit in .khaos-commit; a clone git cannot read is read from
    its own files."""
    if root is None:
        return None
    root = str(root)
    try:
        if not (Path(root) / ".git").exists():
            try:
                sha = (Path(root) / COMMIT_FILE).read_text(encoding="utf-8").strip()
            except OSError:
                return None
            return sha if _SHA_RE.match(sha) else None
    except OSError:
        return None
    try:
        proc = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              env=git_env())
    except FileNotFoundError:
        return head_from_files(Path(root) / ".git")
    except Exception:  # noqa: BLE001 — a timeout, an unreadable cwd: None
        return None
    if proc.returncode != 0:
        return head_from_files(Path(root) / ".git")
    sha = (proc.stdout or "").strip()
    return sha if _SHA_RE.match(sha) else None


def package_sha256(root=None) -> str:
    """sha256 of the khaos_attribution package this interpreter imports (or the package folder given): every .py and
    .json file by its path, line endings normalised so a Windows checkout and a wheel agree. A room pins it."""
    import hashlib  # noqa: PLC0415

    import khaos_attribution  # noqa: PLC0415
    root = Path(root) if root is not None else Path(khaos_attribution.__file__).parent
    h = hashlib.sha256()
    for p in real_rglob(root, "*"):
        if p.suffix in (".py", ".json") and "__pycache__" not in p.parts:
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def contract_code_hash(root=None) -> str:
    """"<version> <package_sha256>" of the package this interpreter imports, or of the package folder given."""
    import khaos_attribution  # noqa: PLC0415
    return f"{khaos_attribution.__version__} {package_sha256(root)}"
