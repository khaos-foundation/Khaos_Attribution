"""__version__ is the literal in the package, and pyproject and uv.lock name the same version."""
from __future__ import annotations

import re
from pathlib import Path

import khaos_attribution

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "khaos_attribution"


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version = "([^"]+)"', text, re.M).group(1)


def test_the_version_is_the_literal_and_the_literal_is_pyprojects():
    """The code's own literal is the version a room reads, never an editable install's metadata, which goes
    stale on a fast-forward and would fail every fingerprint comparison on identical code."""
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    literal = re.search(r'^__version__ = "([^"]+)"', source, re.M).group(1)
    assert literal == _pyproject_version()
    assert khaos_attribution.__version__ == literal
    assert "importlib.metadata" not in source


def test_the_lock_records_the_version_pyproject_names():
    """uv.lock is relocked with every version; CI's `uv sync --locked` refuses any other disagreement with pyproject."""
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    entry = re.search(r'^\[\[package\]\]\nname = "khaos-attribution"\nversion = "([^"]+)"', lock, re.M)
    assert entry and entry.group(1) == _pyproject_version(), "run `uv lock` after setting the version"
