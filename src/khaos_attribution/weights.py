"""The model weights the installers fetch: each file from Hugging Face at its pinned revision, else from Khaos's own
mirror, and from either only once it matches the sha256 in weights_pins.py. Anonymous: no token is read or sent.
Standard library only; a caller falls back to the hub library only where `may_fall_back` says so: a set with no pin
that no source served, never a refused file or one this computer could not write."""

from __future__ import annotations

import contextlib
import errno
import fnmatch
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Callable, NamedTuple

from khaos_attribution.weights_pins import PINS

MIRROR_ENV = "KHAOS_WEIGHTS_URL"
MIRROR_BASE = "https://weights.khaos.dev"
HUB_ENV = "HF_ENDPOINT"
HUB_BASE = "https://huggingface.co"
MANIFEST = "manifest.json"
CHUNK = 1 << 20
TIMEOUT = 30.0
HUB_TIMEOUT = 10.0
ATTEMPTS = 4
BACKOFF = (1.0, 2.0, 4.0)
SKIP_SECONDS = 60.0
# A shell caller falls back to the hub library on this code and nothing else.
EXIT_NO_MIRROR = 3
# A pinned set no source answered for: a script stops, because the hub library would install it unchecked.
EXIT_UNREACHABLE = 4
# A file could not be written or put in place on this computer (a full drive, a file another program holds): a script stops.
EXIT_LOCAL = 5
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")


class MirrorError(RuntimeError):
    """A file did not match its manifest, or the manifest names a place outside the set."""


class PinMismatch(MirrorError):
    """No source served a file that matches its pinned sha256: nothing was installed, and no fallback may be."""


class Unserved(urllib.error.URLError):
    """No source answered for a file; `pinned` says whether the hub library may still be tried (it may not for a pin)."""

    def __init__(self, reason, pinned: bool = False):
        super().__init__(reason)
        self.pinned = pinned


class NotPlaced(MirrorError):
    """A file could not be written or put in place on this computer: no other source and no hub library is tried."""


class NoSpace(NotPlaced):
    """The drive filled while a file was written."""


def may_fall_back(exc: BaseException) -> bool:
    """Whether a caller may still try the hub library after `exc`: only for a set with no pin that no source served."""
    if isinstance(exc, (PinMismatch, NotPlaced)):
        return False
    if isinstance(exc, Unserved):
        return not exc.pinned
    return isinstance(exc, MirrorError)


_FULL = {errno.ENOSPC} | ({errno.EDQUOT} if hasattr(errno, "EDQUOT") else set())
# A checked file another program holds (antivirus, on Windows) is asked again for about 15 s before it is refused.
PLACE_WAITS = (0.5, 1.0, 2.0, 4.0, 8.0)


def _free_gb(path: Path) -> str:
    for folder in (path, *path.parents):
        with contextlib.suppress(OSError):
            return f"{shutil.disk_usage(folder).free / 1e9:.1f} GB free"
    return "free space unknown"


def _unwritable(exc: OSError, path: Path) -> NotPlaced:
    """What a failed write or move on this computer is called: NoSpace for a full drive, else NotPlaced."""
    where = Path(exc.filename) if isinstance(exc.filename, str) and exc.filename else Path(path)
    if exc.errno in _FULL:
        return NoSpace(f"No space left on the drive for {where.name} in {where.parent} ({_free_gb(where.parent)}): "
                       "make room, then fetch again")
    return NotPlaced(f"{where.name} could not be written in {where.parent} on this computer "
                     f"({exc.strerror or exc}): check the drive is connected and writable, then fetch again")


def _here(step: Callable, path: Path):
    """A step on this computer's files; its OSError raised as NotPlaced or NoSpace."""
    try:
        return step()
    except OSError as exc:
        raise _unwritable(exc, path) from exc


def _put_in_place(part: Path, dest: Path) -> None:
    """The checked copy renamed to its name, asked again while another program holds it; it is kept when refused."""
    for wait in (*PLACE_WAITS, None):
        try:
            part.replace(dest)
            return
        except OSError as exc:
            if exc.errno in _FULL:
                raise _unwritable(exc, dest) from exc
            if wait is None:
                raise NotPlaced(f"{dest.name} was downloaded and matched its pin, but could not be put in place in "
                                f"{dest.parent} ({exc.strerror or exc}). Another program, often antivirus, may be holding "
                                "it: fetch again, and the checked copy is used without downloading it again") from exc
            _sleep(wait)


@contextlib.contextmanager
def _on_this_computer(where: Path):
    """Any OSError a fetch's own file steps raise becomes NotPlaced; Unserved, a URLError and so an OSError, passes."""
    try:
        yield
    except (Unserved, NotPlaced):
        raise
    except OSError as exc:
        raise _unwritable(exc, where) from exc


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _checked(repo: str, doc: dict) -> dict:
    """The manifest, refused before anything is written unless its revision is one plain folder name and every file a
    relative path inside the set with a sha256 for a name: the mirror's own word never chooses where a file lands."""
    revision = doc["revision"]
    if (not isinstance(revision, str) or revision in (".", "..") or any(c in revision for c in "/\\:")
            or revision != revision.strip()):
        raise MirrorError(f"{repo}: the manifest's revision {revision!r} is not a plain name")
    for f in doc["files"]:
        path = f.get("path") if isinstance(f, dict) else None
        parts = PurePosixPath(path).parts if isinstance(path, str) else ()
        if (not parts or PurePosixPath(path).is_absolute() or any(p in ("", ".", "..") for p in parts)
                or "\\" in path or ":" in path):
            raise MirrorError(f"{repo}: the manifest names a file outside the set: {path!r}")
        if not isinstance(f.get("sha256"), str) or not _SHA256.fullmatch(f["sha256"]):
            raise MirrorError(f"{repo}: the manifest's sha256 for {path!r} is not 64 hex characters")
        if not isinstance(f.get("size"), int) or isinstance(f.get("size"), bool) or f["size"] < 0:
            raise MirrorError(f"{repo}: the manifest's size for {path!r} is not a size")
    return doc


def _place_link(blob: Path, link: Path) -> None:
    """The snapshot's entry for a blob: a relative symlink, else a hard link, else a copy, as the hub's own downloader
    falls back where symlinks are refused (Windows without Developer Mode)."""
    try:
        link.symlink_to(os.path.relpath(blob, link.parent))
        return
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(blob, link)
    except OSError:
        shutil.copyfile(blob, link)


def base_url(base: str | None = None) -> str:
    return (base or os.environ.get(MIRROR_ENV) or MIRROR_BASE).rstrip("/")


def _get(url: str, timeout: float = TIMEOUT, headers: dict | None = None, agent: str | None = "khaos-weights"):
    """An anonymous GET: no credential is ever added, and the hub is sent no name for this project (agent=None)."""
    request = urllib.request.Request(url, headers={**({"User-Agent": agent} if agent else {}), **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout)


def manifest(repo: str, base: str | None = None, timeout: float = 10.0) -> dict | None:
    """The mirror's manifest for a repository, or None when the mirror is unreachable or holds no such set."""
    try:
        with _get(f"{base_url(base)}/{repo}/{MANIFEST}", timeout=timeout) as answer:
            doc = json.loads(answer.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("files"), list) or not doc.get("revision"):
        return None
    return doc


def _wanted(files: list, patterns: list | tuple | None) -> list:
    if not patterns:
        return list(files)
    return [f for f in files if any(fnmatch.fnmatch(f["path"], p) for p in patterns)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Source(NamedTuple):
    name: str
    url: str
    timeout: float
    agent: str | None
    hub: bool
    repo: str = ""


_skip: dict = {}
_now = time.monotonic
_sleep = time.sleep


def hub_url(base: str | None = None) -> str:
    return (base or os.environ.get(HUB_ENV) or HUB_BASE).rstrip("/")


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "on", "yes", "true")


def _hub_skipped(repo: str) -> bool:
    return any(_skip.get(key, 0.0) > _now() for key in ("*", repo))


def _skip_hub(key: str, why: str, say: Callable[[str], None] | None) -> None:
    _skip[key] = _now() + SKIP_SECONDS
    if say:
        say(f"Hugging Face {why}" + ("" if key == "*" else f" for {key}") + "; using the Khaos mirror")


def _pinned(repo: str) -> dict | None:
    pin = PINS.get(repo)
    if pin is None:
        return None
    return {"repo": repo, "revision": pin["revision"],
            "files": [{"path": p, "size": s, "sha256": h} for p, s, h in pin["files"]]}


def _plan(repo: str, base: str | None, revision: str | None) -> tuple | None:
    """(manifest, pinned): the pin when it is the revision asked, else the mirror's manifest and the mirror alone."""
    doc = _pinned(repo)
    if doc is not None and revision in (None, "", doc["revision"]):
        return doc, True
    doc = manifest(repo, base)
    if doc is None or (revision and doc["revision"] != revision):
        return None
    return doc, False


def _sources(repo: str, revision: str, path: str, base: str | None, pinned: bool) -> list:
    out = []
    if pinned and not _offline() and not _hub_skipped(repo):
        out.append(_Source("Hugging Face", f"{hub_url()}/{repo}/resolve/{revision}/{urllib.parse.quote(path)}",
                           HUB_TIMEOUT, None, True, repo))
    out.append(_Source("the Khaos mirror", f"{base_url(base)}/{repo}/{revision}/{path}", TIMEOUT, "khaos-weights", False))
    return out


def _retry_after(exc: urllib.error.HTTPError) -> float:
    try:
        return min(max(float(exc.headers.get("Retry-After", "")), 0.0), 30.0)
    except (TypeError, ValueError):
        return 0.0


def _receive(answer, part: Path, have: int, size: int, progress: Callable[[str], None] | None) -> str | None:
    """The answer's body appended to `part`: None once it ends, or why the connection broke. A failure to write here is
    this computer's, never the source's, and raises NotPlaced at once."""
    try:
        out = part.open("ab" if have else "wb")
    except OSError as exc:
        with contextlib.suppress(Exception):
            answer.close()
        raise _unwritable(exc, part) from exc
    try:
        if have and answer.status != 206:
            _here(lambda: (out.seek(0), out.truncate()), part)
        written = _here(out.tell, part)
        while written <= size:
            try:
                chunk = answer.read(CHUNK)
            except (OSError, http.client.HTTPException) as exc:
                return str(getattr(exc, "reason", None) or exc)
            if not chunk:
                return None
            _here(lambda: (out.write(chunk), out.flush()), part)    # a full drive is said at this chunk
            written += len(chunk)
            if progress:
                progress(f"{part.name.removesuffix('.part')}: {written * 100 // max(size, 1)}%")
        return None
    finally:
        with contextlib.suppress(Exception):
            answer.close()
        _here(out.close, part)


def _from_source(src: _Source, part: Path, size: int, sha256: str, progress: Callable[[str], None] | None) -> tuple:
    """One source's bytes into `part`, resumed: (ok | mismatch | gated | down | unavailable, why). A retry is made only
    once bytes have flowed, or against the mirror's own 5xx; a resumed copy that fails its hash is fetched once from zero."""
    broke = early = 0
    resumed = fresh = False
    why = ""
    while True:
        have = part.stat().st_size if part.is_file() else 0
        if have > size:
            part.unlink()
            have = 0
        resumed = resumed or have > 0
        if have < size:
            try:
                answer = _get(src.url, src.timeout, {"Range": f"bytes={have}-"} if have else None, src.agent)
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    return "gated", f"HTTP {exc.code}"
                if (exc.code == 429 or exc.code >= 500) and not src.hub and early < 2:
                    early += 1
                    _sleep(_retry_after(exc) or BACKOFF[early - 1])
                    continue
                return ("down" if exc.code == 429 or exc.code >= 500 else "unavailable"), f"HTTP {exc.code}"
            except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
                return "down", str(getattr(exc, "reason", None) or exc)
            declared = answer.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) != (size - have if answer.status == 206 and have else size):
                answer.close()
                part.unlink(missing_ok=True)
                return "mismatch", "its length is not the pinned file's"
            lost = _receive(answer, part, have, size, progress)
            if lost is not None:
                broke += 1
                why = lost
                if broke >= ATTEMPTS:
                    return "down", why
                _sleep(BACKOFF[min(broke, len(BACKOFF)) - 1])
                continue
        got = part.stat().st_size if part.is_file() else 0
        if got < size:
            broke += 1
            if broke >= ATTEMPTS:
                return "down", why or "the answer ended early"
            _sleep(BACKOFF[min(broke, len(BACKOFF)) - 1])
            continue
        if got == size and _sha256(part) == sha256:
            return "ok", ""
        part.unlink(missing_ok=True)
        if resumed and not fresh:
            fresh, resumed = True, False
            continue
        return "mismatch", "its hash is not the pinned one"


def _download(sources: list, dest: Path, size: int, sha256: str, progress: Callable[[str], None] | None = None,
              *, pinned: bool = False) -> None:
    """One file from the first source that serves bytes matching the manifest, resumed from a partial copy; a source
    whose bytes do not match is passed over, and a file no source could serve correctly is never installed."""
    with _on_this_computer(dest):
        _download_into(sources, dest, size, sha256, progress, pinned)


def _download_into(sources: list, dest: Path, size: int, sha256: str, progress: Callable[[str], None] | None,
                   pinned: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == size and _sha256(dest) == sha256:
        return
    if size == 0:
        if sha256 != hashlib.sha256(b"").hexdigest():
            raise MirrorError(f"{dest.name}: the manifest names an empty file with a hash that is not the empty hash")
        dest.write_bytes(b"")
        return
    part = dest.with_name(dest.name + ".part")
    mismatched, reasons = [], []
    for src in sources:
        status, why = _from_source(src, part, size, sha256, progress)
        if status == "ok":
            _put_in_place(part, dest)
            if progress:
                progress(f"{dest.name}: from {src.name}")
            return
        reasons.append(f"{src.name}: {why or status}")
        if status == "mismatch":
            mismatched.append(src.url)
            if progress:
                progress(f"{dest.name}: {src.name} did not match the pinned hash; discarded")
        elif src.hub and status == "gated":
            _skip_hub(src.repo, "needs an account", progress)
        elif src.hub and status == "down":
            _skip_hub("*", "did not answer", progress)
    if mismatched:
        raise (PinMismatch if pinned else MirrorError)(
            f"{mismatched[0]} did not match the manifest (size or sha256); the partial copy was discarded")
    raise Unserved(f"no source answered for {dest.name} ({'; '.join(reasons)})", pinned)


def fetch_into_dir(repo: str, dest: Path | str, patterns: list | tuple | None = None, *, base: str | None = None,
                   progress: Callable[[str], None] | None = None, revision: str | None = None,
                   place: Callable[[str], str] | None = None) -> str | None:
    """The repository's files, laid out under `dest` as the hub's local-dir download lays them out (or where `place`
    maps each path); the revision served, or None when no source holds this set or this revision."""
    plan = _plan(repo, base, revision)
    if plan is None:
        return None
    doc, pinned = plan
    _checked(repo, doc)
    dest = Path(dest)
    for f in _wanted(doc["files"], patterns):
        _download(_sources(repo, doc["revision"], f["path"], base, pinned), dest / (place(f["path"]) if place else f["path"]),
                  int(f["size"]), f["sha256"], progress, pinned=pinned)
    return doc["revision"]


def fetch_into_hub_cache(repo: str, cache_dir: Path | str, patterns: list | tuple | None = None, *,
                         base: str | None = None, progress: Callable[[str], None] | None = None,
                         revision: str | None = None) -> str | None:
    """The repository's files in the hub's own cache layout (blobs, a snapshot named by the revision, refs/main),
    so every library that reads the cache offline finds them; the revision served, or None."""
    plan = _plan(repo, base, revision)
    if plan is None:
        return None
    doc, pinned = plan
    _checked(repo, doc)
    repo_dir = Path(cache_dir) / ("models--" + repo.replace("/", "--"))
    blobs, snapshot = repo_dir / "blobs", repo_dir / "snapshots" / doc["revision"]
    for f in _wanted(doc["files"], patterns):
        blob = blobs / f["sha256"]
        _download(_sources(repo, doc["revision"], f["path"], base, pinned), blob, int(f["size"]), f["sha256"], progress,
                  pinned=pinned)
        link = snapshot / f["path"]
        with _on_this_computer(link):
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            _place_link(blob, link)
    refs = repo_dir / "refs"
    with _on_this_computer(refs / "main"):
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(doc["revision"], encoding="utf-8")
    return doc["revision"]


URLS_DIR = "urls"


def _hub_file(url: str) -> tuple | None:
    """(repo, revision, file) for a hub download address (https://huggingface.co/<repo>/resolve/<rev>/<file>)."""
    from urllib.parse import urlsplit  # noqa: PLC0415
    parts = urlsplit(url)
    if parts.hostname != "huggingface.co" or "/resolve/" not in parts.path:
        return None
    repo, _, rest = parts.path.strip("/").partition("/resolve/")
    revision, _, name = rest.partition("/")
    return (repo, revision, name) if repo and revision and name else None


def fetch_url(url: str, dest: Path | str, *, base: str | None = None,
              progress: Callable[[str], None] | None = None) -> bool:
    """One file the rooms would fetch by address, from its pinned set (hub first, then the mirror) or, for any other
    address, from the mirror's url manifest. True when it was served; False when it cannot be, so the caller fetches
    from the address itself."""
    hub = _hub_file(url)
    if hub:
        repo, revision, name = hub
        plan = _plan(repo, base, None if revision == "main" else revision)
        if plan is None:
            return False
        doc, pinned = plan
        entry = next((f for f in doc["files"] if f["path"] == name), None)
        if entry is None:
            return False
        _download(_sources(repo, doc["revision"], name, base, pinned), Path(dest), int(entry["size"]), entry["sha256"],
                  progress, pinned=pinned)
        return True
    doc = manifest(URLS_DIR, base)
    entry = next((f for f in (doc or {}).get("files", []) if f.get("url") == url), None)
    if entry is None:
        return False
    source = _Source("the Khaos mirror", f"{base_url(base)}/{URLS_DIR}/{entry['path']}", TIMEOUT, "khaos-weights", False)
    _download([source], Path(dest), int(entry["size"]), entry["sha256"], progress)
    return True


def anonymous_hub_env(env: dict | None = None) -> dict:
    """`env` (this process's by default) with no hub token, no saved login and no netrc for the hub library to find."""
    out = {k: v for k, v in (os.environ if env is None else env).items() if k not in TOKEN_VARS}
    out["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    out["NETRC"] = os.devnull
    return out


_anonymous = threading.RLock()


@contextlib.contextmanager
def anonymous_hub():
    """Hub library calls made inside send no token: not from the environment, the saved login or a netrc."""
    keys = (*TOKEN_VARS, "HF_HUB_DISABLE_IMPLICIT_TOKEN", "NETRC")
    with _anonymous:
        saved = {k: os.environ.get(k) for k in keys}
        constants = sys.modules.get("huggingface_hub.constants")
        implicit = getattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
        for k in TOKEN_VARS:
            os.environ.pop(k, None)
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        os.environ["NETRC"] = os.devnull
        if constants is not None:
            constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = True
        try:
            yield
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
            if constants is not None:
                constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = implicit


def default_hub_cache() -> Path:
    """Where the hub library keeps its cache, read the way it reads it, without importing it."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else Path(
        os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface"
    return home / "hub"


def main(argv: list | None = None) -> int:
    """fetch <repo> (--into DIR | --hub-cache [DIR]) [--pattern P]... [--revision R] [--base URL]; check <repo>.
    Exit 3 when no source can serve an unpinned set, so a script falls back to the hub library; 4 when none answered for a
    pinned one, which no fallback may replace; 1 when a file fails its pin; 5 when a file cannot be written here."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="khaos_attribution.weights")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("repo")
    where = fetch.add_mutually_exclusive_group(required=True)
    where.add_argument("--into")
    where.add_argument("--hub-cache", nargs="?", const="")
    fetch.add_argument("--pattern", action="append", default=[])
    fetch.add_argument("--revision")
    fetch.add_argument("--base")
    check = sub.add_parser("check")
    check.add_argument("repo")
    check.add_argument("--base")
    a = parser.parse_args(argv)
    say = lambda text: print(text, file=sys.stderr, flush=True)  # noqa: E731
    if a.command == "check":
        plan = _plan(a.repo, a.base, None)
        if plan is None:
            say(f"no pin for {a.repo}, and {base_url(a.base)} serves no {a.repo}")
            return EXIT_NO_MIRROR
        print(plan[0]["revision"])
        return 0
    try:
        if a.into is not None:
            served = fetch_into_dir(a.repo, a.into, a.pattern or None, base=a.base, progress=say, revision=a.revision)
        else:
            cache = Path(a.hub_cache) if a.hub_cache else default_hub_cache()
            served = fetch_into_hub_cache(a.repo, cache, a.pattern or None, base=a.base, progress=say, revision=a.revision)
    except NotPlaced as exc:
        say(str(exc))
        return EXIT_LOCAL
    except MirrorError as exc:
        say(str(exc))
        return 1
    except Unserved as exc:
        say(str(exc))
        return EXIT_UNREACHABLE if exc.pinned else EXIT_NO_MIRROR
    except OSError as exc:
        say(f"{a.repo} could not be fetched on this computer: {exc}")
        return EXIT_LOCAL
    if served is None:
        say(f"no source serves {a.repo}" + (f" at {a.revision}" if a.revision else ""))
        return EXIT_NO_MIRROR
    print(served)
    return 0


if __name__ == "__main__":
    sys.exit(main())
