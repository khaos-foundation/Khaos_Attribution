"""__version__ is the literal in the package and matches pyproject, and every module with
`X | None` annotations defers them so requires-python >= 3.9 holds."""
from __future__ import annotations

import ast
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


def _uses_pep604_union(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        ann = None
        if isinstance(node, ast.arg):
            ann = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ann = node.returns
        elif isinstance(node, ast.AnnAssign):
            ann = node.annotation
        if ann is None:
            continue
        for sub in ast.walk(ann):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
                return True
    return False


def test_every_module_with_union_annotations_defers_them():
    """`int | None` is evaluated at definition time on 3.9 unless the module
    defers annotations."""
    offenders = []
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not _uses_pep604_union(tree):
            continue
        deferred = any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
                       and any(a.name == "annotations" for a in n.names)
                       for n in tree.body)
        if not deferred:
            offenders.append(path.name)
    assert offenders == [], f"union annotations without the future import: {offenders}"
