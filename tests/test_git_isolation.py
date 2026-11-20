"""Tests that run git read none of this machine's git settings: a global hook, identity or signing rule is not theirs."""
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("git") is None, reason="no git")
def test_git_reads_none_of_this_machines_settings():
    out = subprocess.run(["git", "config", "--global", "--list"], capture_output=True, text=True)
    assert out.stdout == ""
    out = subprocess.run(["git", "config", "--system", "--list"], capture_output=True, text=True)
    assert out.stdout == ""
