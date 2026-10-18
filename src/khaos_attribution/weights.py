"""Khaos's own weights mirror: every model set an installer needs, served as plain files from a Khaos address with
a manifest (revision, path, size, sha256) per repository, so no room ever depends on a Hugging Face account or on
Hugging Face being reachable. Standard library only; a caller falls back to the hub when the mirror answers None."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

MIRROR_ENV = "KHAOS_WEIGHTS_URL"
MIRROR_BASE = "https://weights.khaos.dev"
MANIFEST = "manifest.json"
CHUNK = 1 << 20
TIMEOUT = 30.0
# A shell caller falls back to the hub on this code and nothing else.
EXIT_NO_MIRROR = 3


class MirrorError(RuntimeError):
    """A file the mirror served did not match its manifest."""


def base_url(base: str | None = None) -> str:
    return (base or os.environ.get(MIRROR_ENV) or MIRROR_BASE).rstrip("/")


def _get(url: str, timeout: float = TIMEOUT, headers: dict | None = None):
    request = urllib.request.Request(url, headers={"User-Agent": "khaos-weights", **(headers or {})})
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


def _download(url: str, dest: Path, size: int, sha256: str, progress: Callable[[str], None] | None = None) -> None:
    """One file, resumed from a partial copy, verified against the manifest before it takes its name."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == size and _sha256(dest) == sha256:
        return
    if size == 0:
        if sha256 != hashlib.sha256(b"").hexdigest():
            raise MirrorError(f"{url}: the manifest names an empty file with a hash that is not the empty hash")
        dest.write_bytes(b"")
        return
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with _get(url, headers=headers) as answer, part.open("ab" if have else "wb") as out:
            if have and answer.status != 206:
                out.seek(0)
                out.truncate()
            for chunk in iter(lambda: answer.read(CHUNK), b""):
                out.write(chunk)
                if progress:
                    progress(f"{dest.name}: {out.tell() * 100 // max(size, 1)}%")
    if part.stat().st_size != size or _sha256(part) != sha256:
        part.unlink()
        raise MirrorError(f"{url} did not match the manifest (size or sha256); the partial copy was discarded")
    part.replace(dest)


def fetch_into_dir(repo: str, dest: Path | str, patterns: list | tuple | None = None, *, base: str | None = None,
                   progress: Callable[[str], None] | None = None, revision: str | None = None,
                   place: Callable[[str], str] | None = None) -> str | None:
    """The repository's files, laid out under `dest` as the hub's local-dir download lays them out (or where `place`
    maps each path); the revision served, or None when the mirror cannot serve this set or this revision."""
    doc = manifest(repo, base)
    if doc is None or (revision and doc["revision"] != revision):
        return None
    dest = Path(dest)
    for f in _wanted(doc["files"], patterns):
        _download(f"{base_url(base)}/{repo}/{doc['revision']}/{f['path']}", dest / (place(f["path"]) if place else f["path"]),
                  int(f["size"]), f["sha256"], progress)
    return doc["revision"]


def fetch_into_hub_cache(repo: str, cache_dir: Path | str, patterns: list | tuple | None = None, *,
                         base: str | None = None, progress: Callable[[str], None] | None = None,
                         revision: str | None = None) -> str | None:
    """The repository's files in the hub's own cache layout (blobs, a snapshot named by the revision, refs/main),
    so every library that reads the cache offline finds them; the revision served, or None."""
    doc = manifest(repo, base)
    if doc is None or (revision and doc["revision"] != revision):
        return None
    repo_dir = Path(cache_dir) / ("models--" + repo.replace("/", "--"))
    blobs, snapshot = repo_dir / "blobs", repo_dir / "snapshots" / doc["revision"]
    for f in _wanted(doc["files"], patterns):
        blob = blobs / f["sha256"]
        _download(f"{base_url(base)}/{repo}/{doc['revision']}/{f['path']}", blob, int(f["size"]), f["sha256"], progress)
        link = snapshot / f["path"]
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(blob, link.parent))
    refs = repo_dir / "refs"
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
    """One file the rooms would fetch by address, from the mirror instead: a hub address from that repository's set
    (a pinned revision must be the one served), any other address from the mirror's url manifest. True when the
    mirror served it; False when it cannot, so the caller fetches from the address itself."""
    hub = _hub_file(url)
    if hub:
        repo, revision, name = hub
        doc = manifest(repo, base)
        if doc is None or (revision != "main" and revision != doc["revision"]):
            return False
        entry = next((f for f in doc["files"] if f["path"] == name), None)
        if entry is None:
            return False
        _download(f"{base_url(base)}/{repo}/{doc['revision']}/{name}", Path(dest), int(entry["size"]), entry["sha256"],
                  progress)
        return True
    doc = manifest(URLS_DIR, base)
    entry = next((f for f in (doc or {}).get("files", []) if f.get("url") == url), None)
    if entry is None:
        return False
    _download(f"{base_url(base)}/{URLS_DIR}/{entry['path']}", Path(dest), int(entry["size"]), entry["sha256"], progress)
    return True


def default_hub_cache() -> Path:
    """Where the hub library keeps its cache, read the way it reads it, without importing it."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else Path(
        os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface"
    return home / "hub"


def main(argv: list | None = None) -> int:
    """fetch <repo> (--into DIR | --hub-cache [DIR]) [--pattern P]... [--revision R] [--base URL]; check <repo>.
    Exit 3 when the mirror cannot serve the set, so a script falls back to the hub; 1 when a file fails its check."""
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
        doc = manifest(a.repo, a.base)
        if doc is None:
            say(f"{base_url(a.base)} serves no {a.repo}")
            return EXIT_NO_MIRROR
        print(doc["revision"])
        return 0
    try:
        if a.into is not None:
            served = fetch_into_dir(a.repo, a.into, a.pattern or None, base=a.base, progress=say, revision=a.revision)
        else:
            cache = Path(a.hub_cache) if a.hub_cache else default_hub_cache()
            served = fetch_into_hub_cache(a.repo, cache, a.pattern or None, base=a.base, progress=say, revision=a.revision)
    except MirrorError as exc:
        say(str(exc))
        return 1
    except (urllib.error.URLError, OSError) as exc:
        say(f"the mirror stopped answering: {exc}")
        return EXIT_NO_MIRROR
    if served is None:
        say(f"{base_url(a.base)} serves no {a.repo}" + (f" at {a.revision}" if a.revision else ""))
        return EXIT_NO_MIRROR
    print(served)
    return 0


if __name__ == "__main__":
    sys.exit(main())
