"""Every file the package needs beside its code is shipped: each tracked non-Python file under src/ matches a
package-data pattern in pyproject.toml, so a wheel or a tag install never lacks one."""
import fnmatch
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _package_data() -> dict:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text[text.index("[tool.setuptools.package-data]"):]
    return {m.group(1): re.findall(r'"([^"]+)"', m.group(2)) for m in re.finditer(r"^(\w+) = \[(.*?)\]", block, re.M | re.S)}


def test_every_tracked_data_file_under_src_is_package_data():
    tracked = subprocess.run(["git", "ls-files", "src"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    patterns = _package_data()
    missing = []
    for name in tracked:
        path = Path(name)
        if path.suffix == ".py" or path.name in (".gitignore", "README.md"):
            continue
        package, rel = path.parts[1], "/".join(path.parts[2:])
        if not any(fnmatch.fnmatchcase(rel, p) for p in patterns.get(package, [])):
            missing.append(name)
    assert missing == [], f"tracked files no package-data pattern ships: {missing}"
