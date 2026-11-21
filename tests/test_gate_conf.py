"""scripts/gate.sh is one file in every Khaos repository, and scripts/gate.conf.sh, this repository's own part,
names its blob: an edit to gate.sh here alone goes red."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gate_sh_is_the_blob_its_conf_names():
    """Change the shared file where it is kept and write it into each repository, which sets GATE_CORE."""
    named = re.search(r"^GATE_CORE=([0-9a-f]{40})$", (ROOT / "scripts" / "gate.conf.sh").read_text(), re.M)
    assert named, "scripts/gate.conf.sh names no GATE_CORE"
    blob = subprocess.run(["git", "hash-object", "scripts/gate.sh"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    assert blob == named.group(1), "scripts/gate.sh is not the gate core scripts/gate.conf.sh names"
