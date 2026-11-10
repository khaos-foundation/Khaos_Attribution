"""A skip the suite does not expect fails the run: a test that silently stops running is a guarantee nobody checks."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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
    done = _run(tmp_path, "def test_a():\n    pytest.skip(\"could not import 'torch': No module named 'torch'\")\n"
                          "def test_b():\n    pytest.skip('FFmpeg is not installed')\n"
                          "@pytest.mark.skipif(True, reason='torchaudio aligner not installed')\ndef test_c():\n    pass\n"
                          "def test_d():\n    pytest.skip(\"could not import 'huggingface_hub.constants': No module named 'huggingface_hub'\")\n"
                          "@pytest.mark.xfail\ndef test_e():\n    pass\n")
    assert done.returncode == 0, done.stdout


@pytest.mark.parametrize("body, said", [
    ("def test_x():\n    pytest.xfail('fastapi is not installed')\n", "fastapi is not installed"),
    ("@pytest.mark.xfail(reason='cryptography is not installed')\ndef test_x():\n    assert False\n", "cryptography is not installed"),
    ("@pytest.mark.xfail\ndef test_x():\n    assert False\n", "test_x"),
    ("def test_x():\n    pytest.skip('ffmpeg would be nice, and so would fastapi')\n", "ffmpeg would be nice"),
    ("def test_x():\n    pytest.importorskip('torch_but_not_torch')\n", "torch_but_not_torch"),
    ("def test_x():\n    pytest.skip(\"could not import 'torchvision': No module named 'torchvision'\")\n", "torchvision"),
])
def test_a_reason_must_be_one_the_suite_names_exactly_and_an_expected_failure_is_held_to_the_same_rule(tmp_path, body, said):
    """A reason that only mentions an allowed word, a module whose name only starts like one, and every form of
    expected failure: none of them is a way round the rule."""
    done = _run(tmp_path, body)
    assert done.returncode == 1, done.stdout
    assert "unexpected skip" in done.stdout and said in done.stdout


def test_every_skip_reason_the_suite_uses_on_another_platform_is_named(tmp_path):
    from khaos_attribution.engines import worker_process as W
    done = _run(tmp_path, f"def test_a():\n    pytest.skip({W.POSIX_ONLY!r})\ndef test_b():\n    pytest.skip({W.LINUX_ONLY!r})\n")
    assert done.returncode == 0, done.stdout
