"""The tests' own guard: a move, rename, replace or removal that would leave the test's folder fails and leaves the
file where it was; inside the folder each still works."""
import os
import shutil
import uuid

import pytest


@pytest.mark.parametrize("how", ["move", "rename", "replace"])
def test_nothing_is_moved_out_of_the_tests_folder(tmp_path, confined, how):
    kept = tmp_path / "kept.txt"
    kept.write_text("mine")
    away = tmp_path.parent / f"outside-{uuid.uuid4().hex}.txt"
    act = {"move": shutil.move, "rename": os.rename, "replace": os.replace}[how]
    with pytest.raises(AssertionError, match="outside the test's folder"):
        act(str(kept), str(away))
    assert kept.read_text() == "mine" and not away.exists()
    act(str(kept), str(tmp_path / "moved.txt"))
    assert (tmp_path / "moved.txt").read_text() == "mine" and not kept.exists()


@pytest.mark.parametrize("how", ["unlink", "remove", "rmtree", "rmdir"])
def test_nothing_outside_the_tests_folder_is_removed(tmp_path, confined, how):
    act = {"unlink": os.unlink, "remove": os.remove, "rmtree": shutil.rmtree, "rmdir": os.rmdir}[how]
    with pytest.raises(AssertionError, match="outside the test's folder"):
        act(str(tmp_path.parent / f"outside-{uuid.uuid4().hex}"))
