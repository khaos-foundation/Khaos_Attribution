"""A skip the suite does not expect fails the run: a test that silently stops running is a guarantee nobody checks."""
import shutil
import subprocess
import sys
from pathlib import Path

CONFTEST = Path(__file__).with_name("conftest.py")


def _run(tmp_path, body: str) -> subprocess.CompletedProcess:
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_it.py").write_text("import pytest\n" + body, encoding="utf-8")
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
                          capture_output=True, text=True, cwd=tmp_path)


def test_a_skip_for_a_missing_register_package_fails_the_run(tmp_path):
    done = _run(tmp_path, "pytest.importorskip('a_package_the_register_needs')\ndef test_x():\n    pass\n")
    assert done.returncode == 1, done.stdout
    assert "unexpected skip" in done.stdout and "a_package_the_register_needs" in done.stdout


def test_a_skip_inside_a_test_for_another_reason_fails_the_run(tmp_path):
    done = _run(tmp_path, "def test_x():\n    pytest.skip('node is not installed')\n")
    assert done.returncode == 1 and "node is not installed" in done.stdout


def test_the_skips_the_suite_expects_pass(tmp_path):
    done = _run(tmp_path, "def test_a():\n    pytest.importorskip('torch_not_here', reason=\"could not import 'torch': x\")\n"
                          "def test_b():\n    pytest.skip('FFmpeg is not installed')\n"
                          "@pytest.mark.skipif(True, reason='torchaudio aligner not installed')\ndef test_c():\n    pass\n"
                          "@pytest.mark.xfail\ndef test_d():\n    assert False\n")
    assert done.returncode == 0, done.stdout
