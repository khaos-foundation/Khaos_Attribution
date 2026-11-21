"""scripts/gate.sh --stale: every NO_TEST_OK glob still matches a Python file no test names, so the list only shrinks."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_known_untested_list_is_not_stale():
    """A file that got a test leaves NO_TEST_OK; a glob that matches no untested file fails here."""
    out = subprocess.run(["bash", str(ROOT / "scripts" / "gate.sh"), "--stale"], cwd=ROOT, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
