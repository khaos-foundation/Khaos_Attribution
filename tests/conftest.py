"""Every test runs with the weights mirror and the hub unreachable, no public log witness and none of this machine's git settings.
A test that does not run to a verdict, skipped or expected to fail, fails the run unless its reason is named here."""
import os
import re
import shutil
from pathlib import Path

import pytest

from khaos_attribution.engines.worker_process import LINUX_ONLY, POSIX_ONLY

# Packages only a room's or an engine's environment holds: a test that cannot import one of these, by this name, may skip.
ALLOWED_MISSING = ("torch", "torchaudio", "librosa", "huggingface_hub", "huggingface_hub.constants")
# Every other reason, word for word: a tool or a file a machine may lack, and the tests of another platform.
ALLOWED_REASONS = ("torchaudio aligner not installed", "ffmpeg unavailable", "FFmpeg is not installed", "no ffmpeg",
                   "the base model's config is not in this machine's cache", "no ACE-Step checkout beside this one",
                   "no Python 3.11 on this machine to load the contract under",
                   "no Python 3.11 with numpy and jsonschema on this machine to import the contract under",
                   POSIX_ONLY, LINUX_ONLY)
# gate.sh torch needs these all; only the register's own package may be missing, as the pytest gate runs its test.
TORCH_RUN_NEEDS = ("torch", "torchaudio", "librosa", "huggingface_hub", "huggingface_hub.constants", "soundfile")
if os.environ.get("KHAOS_TORCH_RUN") == "1":
    ALLOWED_MISSING = tuple(m for m in ALLOWED_MISSING if m not in TORCH_RUN_NEEDS) + ("opentimestamps",)
    ALLOWED_REASONS = tuple(r for r in ALLOWED_REASONS if r != "torchaudio aligner not installed")

_unexpected: list = []


@pytest.fixture(autouse=True)
def _no_real_weights_mirror(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("KHAOS_WEIGHTS_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    from khaos_attribution import weights
    weights._skip.clear()
    monkeypatch.setenv("REGISTER_LOG_WITNESSES", "")
    # Every git a test runs reads no configuration of this machine's.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    # khaos-log-verify's tile cache stays inside the test.
    monkeypatch.setenv("KHAOS_LOG_CACHE", str(tmp_path_factory.mktemp("log-cache")))


@pytest.fixture(params=[256, 4], ids=["tiles256", "tiles4"])
def tile_widths(request, monkeypatch):
    """The log's export at its own tile width and at 4, so a test's few leaves fill sealed tiles too."""
    from khaos_attribution import translog as T
    monkeypatch.setattr(T, "TILE", request.param)
    return request.param


def _reason(report) -> str | None:
    """Why a test did not run to a verdict, or None when it did: a skip's reason, or an expected failure's."""
    if not report.skipped:
        return None
    if hasattr(report, "wasxfail"):
        return f"expected to fail: {report.wasxfail or 'no reason given'}"
    text = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    return text.removeprefix("Skipped: ")


def _allowed(reason: str) -> bool:
    missing = re.fullmatch(r"could not import '([\w.]+)': .*", reason, re.S)
    return missing.group(1) in ALLOWED_MISSING if missing else reason in ALLOWED_REASONS


def _note(report) -> None:
    reason = _reason(report)
    if reason is not None and not _allowed(reason):
        _unexpected.append(f"{report.nodeid}: {reason}")


pytest_runtest_logreport = _note
pytest_collectreport = _note


def pytest_terminal_summary(terminalreporter):
    for line in _unexpected:
        terminalreporter.write_line(f"unexpected skip: {line}", red=True)


def pytest_sessionfinish(session, exitstatus):
    if _unexpected and exitstatus in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def confined(tmp_path, monkeypatch):
    """No removal, replacement, move or rename reaches outside this test's own folder, whatever the code under test
    asks: such a request fails the test and changes nothing. A hostile-path test must hold even against code that
    does not yet resist it."""
    import os
    root = tmp_path.resolve()
    real = {"rmtree": shutil.rmtree, "move": shutil.move, "unlink": os.unlink, "remove": os.remove, "rmdir": os.rmdir,
            "replace": os.replace, "rename": os.rename}

    def inside(path) -> bool:
        """The path's folder resolved, its last part kept as it is: a link inside the folder is the folder's to remove."""
        try:
            whole = Path(os.path.abspath(os.fsdecode(path)))
            return (whole.parent.resolve() / whole.name).is_relative_to(root)
        except (OSError, ValueError, TypeError):
            return False

    def guard(name):
        def guarded(*paths_and_args, **k):
            count = 2 if name in ("replace", "rename", "move") else 1
            # A call relative to an open folder came from a guarded call on that folder: it passes as before.
            if all(k.get(fd) is None for fd in ("dir_fd", "src_dir_fd", "dst_dir_fd")):
                for path in paths_and_args[:count]:
                    if not inside(path):
                        raise AssertionError(f"{name} outside the test's folder was asked for: {path}")
            return real[name](*paths_and_args, **k)
        return guarded
    for name in ("rmtree", "move"):
        monkeypatch.setattr(shutil, name, guard(name))
    for name in ("unlink", "remove", "rmdir", "replace", "rename"):
        monkeypatch.setattr(os, name, guard(name))
    return root
