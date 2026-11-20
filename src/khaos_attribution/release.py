"""The release bundle: a model leaving a Workshop for a Space as one tar whose manifest names every file by
its hash. The Workshop writes it once per weights (content-addressed), a Space reads it only when every byte
matches the manifest and no member reaches outside the folder it is opened into.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

from khaos_attribution.canonical import compact_ascii

BUNDLE_VERSION = "khaos.release_bundle/1"
MANIFEST_NAME = "release.json"
MAX_MEMBERS = 4000


class BundleError(ValueError):
    """The bundle is not one this reader will open; the message says why."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


MAX_MANIFEST_BYTES = 4 << 20


def _clean(arcname: str) -> str:
    p = PurePosixPath(arcname)
    if (p.is_absolute() or not p.parts or any(part in ("", ".", "..") for part in p.parts) or arcname.startswith(".")
            or "\\" in arcname or ":" in arcname):
        raise BundleError(f"not a bundle path: {arcname!r}")
    return str(p)


def bundle_id(files: dict, meta: dict) -> str:
    """The content address: the hashed files and what the bundle says it is, nothing about when it was made."""
    key = {"files": {k: v["sha256"] for k, v in sorted(files.items())},
           "artist_id": meta.get("artist_id"), "run_id": meta.get("run_id"), "checkpoint": meta.get("checkpoint"),
           "serial": meta.get("serial"), "record_sha256": meta.get("record_sha256")}
    return hashlib.sha256(compact_ascii(key)).hexdigest()


def manifest_for(files: dict, meta: dict) -> dict:
    """The manifest the files and meta would make, hashed now, nothing written."""
    if not files:
        raise BundleError("a bundle holds at least one file")
    if MANIFEST_NAME in files:
        raise BundleError(f"{MANIFEST_NAME} is the bundle's own")
    entries = {}
    for arcname, src in files.items():
        src = Path(src)
        if not src.is_file():
            raise BundleError(f"not a file: {src}")
        entries[_clean(arcname)] = {"sha256": _sha256_file(src), "size": src.stat().st_size}
    manifest = {"bundle_version": BUNDLE_VERSION, **meta, "files": entries}
    manifest["bundle_id"] = bundle_id(entries, manifest)
    return manifest


def write_bundle(out_path: Path, files: dict, meta: dict, manifest: dict | None = None) -> dict:
    """Write the tar: the manifest first, then every file under its arcname. Returns the manifest."""
    manifest = manifest or manifest_for(files, meta)
    entries = manifest["files"]
    files = {_clean(k): v for k, v in files.items()}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".new")
    with tarfile.open(tmp, "w") as tar:
        raw = json.dumps(manifest, indent=1, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size, info.mode = len(raw), 0o644
        tar.addfile(info, io.BytesIO(raw))
        for arcname in sorted(entries):
            src = Path(files[arcname])
            info = tar.gettarinfo(str(src), arcname=arcname)
            info.type, info.linkname, info.size, info.mode = tarfile.REGTYPE, "", src.stat().st_size, 0o644
            with open(src, "rb") as fh:
                tar.addfile(info, fh)
    tmp.replace(out_path)
    return manifest


def read_manifest(tar_path: Path) -> dict:
    """The manifest alone, from the first member; nothing else is read."""
    with tarfile.open(tar_path, "r") as tar:
        first = tar.next()
        if first is None or first.name != MANIFEST_NAME or not first.isfile():
            raise BundleError(f"the bundle does not begin with {MANIFEST_NAME}")
        if first.size > MAX_MANIFEST_BYTES:
            raise BundleError("the manifest is too large to be one")
        fh = tar.extractfile(first)
        try:
            manifest = json.loads((fh.read() if fh else b"").decode("utf-8"))
        except ValueError:
            raise BundleError("the manifest is not JSON") from None
    return _checked_manifest(manifest)


def _checked_manifest(manifest) -> dict:
    if not isinstance(manifest, dict) or manifest.get("bundle_version") != BUNDLE_VERSION:
        raise BundleError("not a release bundle this reader knows")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > MAX_MEMBERS:
        raise BundleError("the manifest names no files, or too many")
    for arcname, entry in files.items():
        _clean(arcname)
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
            raise BundleError(f"the manifest's entry for {arcname!r} carries no hash")
        if isinstance(entry.get("size"), bool) or not isinstance(entry.get("size"), int) or entry["size"] < 0:
            raise BundleError(f"the manifest's entry for {arcname!r} carries no size")
    if manifest.get("bundle_id") != bundle_id(files, manifest):
        raise BundleError("the bundle id does not match the manifest")
    for key in ("artist_id", "run_id"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise BundleError(f"the manifest names no {key}")
    return manifest


def read_bundle(tar_path: Path, dest: Path) -> dict:
    """Open the bundle into `dest` (made here, must not exist): every member a plain file the manifest names, every
    manifest file present, every hash matching. Returns the manifest; on any refusal nothing is left in `dest`."""
    import shutil  # noqa: PLC0415
    dest = Path(dest)
    if dest.exists():
        raise BundleError(f"{dest} exists")
    manifest = read_manifest(tar_path)
    files = manifest["files"]
    dest.mkdir(parents=True)
    try:
        seen = set()
        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                if member.name == MANIFEST_NAME:
                    continue
                name = _clean(member.name)
                if not member.isfile() or member.issym() or member.islnk() or member.sparse:
                    raise BundleError(f"not a plain file: {member.name!r}")
                if name not in files:
                    raise BundleError(f"a member the manifest does not name: {name!r}")
                if member.size != files[name]["size"]:
                    raise BundleError(f"{name!r} is not the size the manifest says")
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                fh = tar.extractfile(member)
                h = hashlib.sha256()
                with open(target, "wb") as out:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                        out.write(chunk)
                if h.hexdigest() != files[name]["sha256"]:
                    raise BundleError(f"{name!r} does not match its hash")
                seen.add(name)
        missing = sorted(set(files) - seen)
        if missing:
            raise BundleError(f"the bundle lacks {len(missing)} file(s) the manifest names: {missing[0]}")
        (dest / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return manifest
