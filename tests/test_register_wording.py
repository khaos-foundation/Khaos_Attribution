"""The register is described as a tamper-evident public register or a signed append-only log; one word for a
different kind of system never appears anywhere in the repository, this test's own file excepted."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", "build", "dist"}
WORD = re.compile("block" + r"[\s_-]?" + "chain", re.IGNORECASE)


def test_the_word_never_appears_in_the_repository():
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_DIRS & set(path.relative_to(ROOT).parts) or path == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits += [f"{path.relative_to(ROOT)}:{text.count(chr(10), 0, m.start()) + 1}" for m in WORD.finditer(text)]
    assert hits == [], f"say 'tamper-evident public register' or 'signed append-only log': {hits}"
