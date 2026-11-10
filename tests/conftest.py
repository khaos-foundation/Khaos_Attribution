"""Every test runs with the weights mirror unreachable and no public log witness, so no test reaches the network.
A test that does not run to a verdict, skipped or expected to fail, fails the run unless its reason is named here."""
import re

import pytest

from khaos_attribution.engines.worker_process import LINUX_ONLY, POSIX_ONLY

# Packages only a room's or an engine's environment holds: a test that cannot import one of these, by this name, may skip.
ALLOWED_MISSING = ("torch", "torchaudio", "librosa", "huggingface_hub", "huggingface_hub.constants")
# Every other reason, word for word: a tool or a file a machine may lack, and the tests of another platform.
ALLOWED_REASONS = ("torchaudio aligner not installed", "ffmpeg unavailable", "FFmpeg is not installed", "no ffmpeg",
                   "the base model's config is not in this machine's cache", POSIX_ONLY, LINUX_ONLY)

_unexpected: list = []


@pytest.fixture(autouse=True)
def _no_real_weights_mirror(monkeypatch):
    monkeypatch.setenv("KHAOS_WEIGHTS_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("REGISTER_LOG_WITNESSES", "")


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
