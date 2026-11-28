"""The one listing and removal rule: what an operating system leaves beside a file is never read, and never stops a removal."""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from khaos_attribution import files as F

SRC = Path(__file__).resolve().parents[1] / "src"
LEFTOVERS = ["._take_0.wav", "._lokr_weights.safetensors", ".DS_Store", "Thumbs.db", "desktop.ini", "__MACOSX",
             ".Spotlight-V100", ".fseventsd", "$RECYCLE.BIN"]


def _seed(folder: Path) -> None:
    """A folder as an exFAT drive leaves it: each real file and folder with a ``._`` twin, and the indexers' droppings."""
    (folder / "sub").mkdir(parents=True)
    for rel in ("take_0.wav", "lokr_weights.safetensors", "score.json", ".removing-x", "sub/inner.wav"):
        (folder / rel).write_bytes(b"real")
        twin = (folder / rel).with_name("._" + Path(rel).name)
        twin.write_bytes(b"\x00\x05\x16\x07" + b"\xff" * 60)
    (folder / "._sub").write_bytes(b"\x00\x05\x16\x07")
    for name in (".DS_Store", "Thumbs.db", "desktop.ini"):
        (folder / name).write_bytes(b"x")
    (folder / "__MACOSX").mkdir()
    (folder / "__MACOSX" / "take_0.wav").write_bytes(b"x")


@pytest.mark.parametrize("name", LEFTOVERS)
def test_a_leftover_is_known_as_one(name):
    """Every name an operating system writes on its own is a leftover, and never a real file."""
    assert F.is_os_leftover(name) and not F.is_real_file(name)


@pytest.mark.parametrize("name", [".removing-run_1", ".pending_removals.unfiled.jsonl", ".khaos-storage.json"])
def test_the_apps_own_hidden_names_are_listed_but_are_not_tracks(name):
    """A hidden name the app writes stays visible to a listing; only is_real_file keeps it out of the tracks."""
    assert not F.is_os_leftover(name) and not F.is_real_file(name)


@pytest.mark.parametrize("name", ["take_0.wav", "...And Then.mp3", "a/b/song.flac", "C:\\x\\song.wav"])
def test_a_real_name_is_real(name):
    assert F.is_real_file(name) and not F.is_os_leftover(name)


def test_listings_drop_the_leftovers_and_keep_the_apps_hidden_names(tmp_path):
    """glob, rglob, iterdir, listdir and walk each return the real entries only, sorted."""
    _seed(tmp_path)
    assert [p.name for p in F.real_glob(tmp_path, "*.wav")] == ["take_0.wav"]
    assert [p.name for p in F.real_glob(tmp_path, "*.safetensors")] == ["lokr_weights.safetensors"]
    assert sorted(p.relative_to(tmp_path).as_posix() for p in F.real_rglob(tmp_path, "*.wav")) == ["sub/inner.wav",
                                                                                                    "take_0.wav"]
    names = [p.name for p in F.real_iterdir(tmp_path)]
    assert names == sorted(names) == [".removing-x", "lokr_weights.safetensors", "score.json", "sub", "take_0.wav"]
    assert F.real_listdir(tmp_path) == names
    walked = {Path(r).relative_to(tmp_path).as_posix(): sorted(fs) for r, _, fs in F.real_walk(tmp_path)}
    assert walked == {".": [".removing-x", "lokr_weights.safetensors", "score.json", "take_0.wav"],
                      "sub": ["inner.wav"]}
    assert F.real_match(os.listdir(tmp_path), "*.json") == ["score.json"]


def test_a_missing_folder_lists_as_the_plain_call_does(tmp_path):
    """glob lists nothing for a missing folder and iterdir raises, so a call site keeps its behaviour."""
    gone = tmp_path / "gone"
    assert F.real_glob(gone, "*") == [] and F.real_rglob(gone, "*") == []
    with pytest.raises(FileNotFoundError):
        F.real_iterdir(gone)
    with pytest.raises(FileNotFoundError):
        F.real_listdir(gone)


def test_remove_tree_takes_a_folder_whose_twins_vanish_with_their_files(tmp_path, monkeypatch):
    """A twin that is gone by the time rmtree reaches it, as on exFAT, is not an error."""
    _seed(tmp_path / "run")
    real_unlink = os.unlink

    def unlink_with_twin(path, *args, **kwargs):
        real_unlink(path, *args, **kwargs)
        name = os.fsdecode(path)
        twin = os.path.join(os.path.dirname(name), "._" + os.path.basename(name))
        dir_fd = kwargs.get("dir_fd")
        try:
            real_unlink(twin, dir_fd=dir_fd) if dir_fd is not None else real_unlink(twin)
        except FileNotFoundError:
            pass

    monkeypatch.setattr(os, "unlink", unlink_with_twin)
    with pytest.raises(FileNotFoundError):
        shutil.rmtree(tmp_path / "run")
    F.remove_tree(tmp_path / "run")
    assert not (tmp_path / "run").exists()
    F.remove_tree(tmp_path / "run", missing_ok=True)


def test_copy_and_move_leave_the_leftovers_behind(tmp_path):
    _seed(tmp_path / "a")
    F.copy_tree(tmp_path / "a", tmp_path / "b")
    assert sorted(os.listdir(tmp_path / "b")) == F.real_listdir(tmp_path / "a")
    F.move(tmp_path / "b", tmp_path / "c")
    assert not (tmp_path / "b").exists() and (tmp_path / "c" / "sub" / "inner.wav").is_file()


@pytest.fixture(scope="module")
def exfat(tmp_path_factory):
    """A small real exFAT drive, where macOS itself writes and removes the twins."""
    if sys.platform != "darwin" or not shutil.which("hdiutil"):
        pytest.skip("needs macOS's hdiutil to make an exFAT drive")
    work = tmp_path_factory.mktemp("exfat")
    image, mnt = work / "drive.dmg", work / "mnt"
    mnt.mkdir()
    made = subprocess.run(["hdiutil", "create", "-size", "16m", "-fs", "ExFAT", "-volname", "KX", str(image)],
                          capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip(f"hdiutil could not make an exFAT image: {made.stderr.strip()[:200]}")
    att = subprocess.run(["hdiutil", "attach", "-nobrowse", "-mountpoint", str(mnt), str(image)],
                         capture_output=True, text=True)
    if att.returncode != 0:
        pytest.skip(f"hdiutil could not attach the image: {att.stderr.strip()[:200]}")
    try:
        yield mnt
    finally:
        subprocess.run(["hdiutil", "detach", "-force", str(mnt)], capture_output=True)


def test_on_a_real_exfat_drive(exfat):
    """On a real exFAT drive the plain calls read the twins and rmtree fails; the helpers do neither."""
    run = exfat / "run"
    (run / "final").mkdir(parents=True)
    for name in ("final/lokr_weights.safetensors", "take_0.wav", "score.json"):
        (run / name).write_bytes(b"real")
        subprocess.run(["xattr", "-w", "com.apple.provenance", "x", str(run / name)], capture_output=True)
    if not any(n.startswith("._") for n in os.listdir(run)):
        pytest.skip("macOS wrote no ._ twins on this drive")
    assert sorted((run / "final").glob("*.safetensors"))[0].name.startswith("._")
    assert [p.name for p in F.real_glob(run / "final", "*.safetensors")] == ["lokr_weights.safetensors"]
    assert [p.name for p in F.real_glob(run, "*.wav")] == ["take_0.wav"]
    copy = exfat / "copy"
    shutil.copytree(run, copy)
    with pytest.raises(FileNotFoundError):
        shutil.rmtree(copy)
    F.remove_tree(copy)
    F.remove_tree(run)
    assert not copy.exists() and not run.exists()


# What may still call the plain listing and removal functions: the helper itself, and a line marked with its reason.
_LISTING_ATTRS = {"glob", "rglob", "iterdir"}
_PLAIN = {("os", "listdir"), ("os", "scandir"), ("os", "walk"), ("glob", "glob"), ("glob", "iglob"),
          ("shutil", "rmtree"), ("shutil", "move"), ("shutil", "copytree")}
MARK = "# os-leftovers:"


def _plain_calls(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        f = node.func
        plain = (isinstance(f.value, ast.Name) and (f.value.id, f.attr) in _PLAIN) or f.attr in _LISTING_ATTRS
        if plain:
            line = lines[node.lineno - 1]
            reason = line.split(MARK, 1)[1].strip() if MARK in line else ""
            if not reason:
                yield f"{path.relative_to(SRC)}:{node.lineno}: {line.strip()}"


def test_every_listing_and_removal_goes_through_the_helper():
    """A plain glob, iterdir, listdir, walk, rmtree, move or copytree reads or trips on the twins; each line that keeps
    one names its reason with ``# os-leftovers:``."""
    found = [hit for p in sorted(SRC.rglob("*.py")) if p.name != "files.py" or p.parent.name != "khaos_attribution"
             for hit in _plain_calls(p)]
    assert found == [], "list or remove through khaos_attribution.files:\n" + "\n".join(found)
