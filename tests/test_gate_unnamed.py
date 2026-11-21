"""scripts/gate.sh --unnamed: every Python file under src/ and scripts/ is named by a test or listed in NO_TEST_OK, and
no NO_TEST_OK entry is stale, so a branch never fails --changed for a file it did not touch."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_source_file_is_named_by_a_test_or_listed():
    """A new file no test names fails here; so does a NO_TEST_OK entry whose file now has a test, or a glob in it."""
    out = subprocess.run(["bash", str(ROOT / "scripts" / "gate.sh"), "--unnamed"], cwd=ROOT, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0 and "UNNAMED " not in out.stdout, out.stdout + out.stderr
