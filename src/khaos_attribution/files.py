"""The one place that lists and removes folders, so what an operating system leaves beside a file is never read as one.

macOS writes a ``._`` twin beside every file and folder on an exFAT or FAT drive, and removes it with its file; a plain
listing returns the twins and a plain ``shutil.rmtree`` fails on them. Every listing here drops those leftovers but keeps
the app's own hidden names; ``is_real_file`` is the stricter rule for what may be a track.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import sys
from pathlib import Path

# What an operating system leaves beside a file, inside a folder it indexes, or at a drive's top (lower case).
OS_LEFTOVER_NAMES = frozenset({"thumbs.db", "ehthumbs.db", "desktop.ini", "icon\r", "__macosx", "$recycle.bin",
                               "system volume information"})
# Hidden names, so never a track already; a listing drops them too.
_HIDDEN_LEFTOVER_NAMES = frozenset({".ds_store", ".spotlight-v100", ".trashes", ".fseventsd", ".temporaryitems",
                                    ".documentrevisions-v100"})


def _base(name) -> str:
    return str(name).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def is_os_leftover(name) -> bool:
    """True for a ``._`` twin, ``.DS_Store`` and the other names an operating system writes on its own."""
    low = _base(name).lower()
    return low.startswith("._") or low in OS_LEFTOVER_NAMES or low in _HIDDEN_LEFTOVER_NAMES


def is_real_file(name) -> bool:
    """What may be a track: not an operating system's leftover and not hidden (a title opening with an ellipsis is not)."""
    base = _base(name)
    hidden = base.startswith(".") and not base.startswith("..")
    return bool(base) and not hidden and not is_os_leftover(base)


def real_glob(folder, pattern: str) -> list[Path]:
    """``Path.glob`` without the leftovers, sorted; a missing folder lists nothing, as glob does."""
    folder = Path(folder)
    return sorted(p for p in folder.glob(pattern) if not _leftover_below(p, folder))


def real_rglob(folder, pattern: str) -> list[Path]:
    """``Path.rglob`` without any path that passes through a leftover below the folder, sorted."""
    folder = Path(folder)
    return sorted(p for p in folder.rglob(pattern) if not _leftover_below(p, folder))


def real_iterdir(folder) -> list[Path]:
    """``Path.iterdir`` without the leftovers, sorted; a missing folder raises, as iterdir does."""
    return sorted(p for p in Path(folder).iterdir() if not is_os_leftover(p.name))


def real_listdir(folder) -> list[str]:
    """``os.listdir`` without the leftovers, sorted; a missing folder raises, as listdir does."""
    return sorted(n for n in os.listdir(folder) if not is_os_leftover(n))


def real_walk(top, **kwargs):
    """``os.walk`` that neither enters nor yields a leftover."""
    for root, dirs, files in os.walk(top, **kwargs):
        dirs[:] = [d for d in dirs if not is_os_leftover(d)]
        yield root, dirs, [f for f in files if not is_os_leftover(f)]


def real_match(names, pattern: str) -> list[str]:
    """The names matching a glob pattern, leftovers dropped, sorted."""
    return sorted(n for n in names if not is_os_leftover(n) and fnmatch.fnmatch(n, pattern))


def _leftover_below(path: Path, folder: Path) -> bool:
    try:
        parts = path.relative_to(folder).parts
    except ValueError:
        parts = (path.name,)
    return any(is_os_leftover(part) for part in parts)


def _gone_is_fine(func, path, exc) -> None:
    # A twin removed with its file, or anything already gone, is what removing wanted.
    if isinstance(exc, FileNotFoundError):
        return
    raise exc


def remove_tree(path, *, missing_ok: bool = False, ignore_errors: bool = False) -> None:
    """``shutil.rmtree`` on which a ``._`` twin that went with its file is not an error."""
    path = Path(path)
    if missing_ok and not os.path.lexists(path):
        return
    if ignore_errors:
        shutil.rmtree(path, ignore_errors=True)
    elif sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_gone_is_fine)
    else:
        shutil.rmtree(path, onerror=lambda f, p, info: _gone_is_fine(f, p, info[1]))


def _ignore_leftovers(_folder, names):
    return [n for n in names if is_os_leftover(n)]


def copy_tree(src, dst, **kwargs):
    """``shutil.copytree`` that leaves the leftovers behind."""
    kwargs.setdefault("ignore", _ignore_leftovers)
    return shutil.copytree(src, dst, **kwargs)


def move(src, dst) -> str:
    """``shutil.move``, whose copy to another drive leaves the leftovers behind and whose clean-up is ``remove_tree``."""
    src, dst = os.fspath(src), os.fspath(dst)
    real_dst = os.path.join(dst, os.path.basename(src.rstrip(os.sep))) if os.path.isdir(dst) else dst
    if os.path.isdir(dst) and os.path.lexists(real_dst):
        raise shutil.Error(f"Destination path '{real_dst}' already exists")
    try:
        os.rename(src, real_dst)
        return real_dst
    except OSError:
        pass
    if os.path.isdir(src) and not os.path.islink(src):
        copy_tree(src, real_dst, symlinks=True)
        remove_tree(src)
    else:
        shutil.copy2(src, real_dst)
        os.unlink(src)
    return real_dst
