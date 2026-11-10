"""Every test runs with the weights mirror unreachable and no public log witness, so no test reaches the network.
A skip outside ALLOWED_SKIPS fails the run: a test that stops running is a guarantee nobody checks."""
import pytest

# What only a room's or an engine's environment holds, and the tools a machine may lack. Anything else must run.
ALLOWED_SKIPS = ("could not import 'torch", "torchaudio aligner not installed", "ffmpeg", "the base model's config",
                 "could not import 'librosa'", "could not import 'huggingface_hub")

_unexpected: list = []


@pytest.fixture(autouse=True)
def _no_real_weights_mirror(monkeypatch):
    monkeypatch.setenv("KHAOS_WEIGHTS_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("REGISTER_LOG_WITNESSES", "")


def _note_skip(report) -> None:
    if not report.skipped or hasattr(report, "wasxfail"):
        return
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    if not any(allowed in reason.lower() for allowed in ALLOWED_SKIPS):
        _unexpected.append(f"{report.nodeid}: {reason}")


pytest_runtest_logreport = _note_skip
pytest_collectreport = _note_skip


def pytest_terminal_summary(terminalreporter):
    for line in _unexpected:
        terminalreporter.write_line(f"unexpected skip: {line}", red=True)


def pytest_sessionfinish(session, exitstatus):
    if _unexpected and exitstatus in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
