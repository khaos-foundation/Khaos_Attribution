"""Every test runs with the weights mirror unreachable and no public log witness, so no test reaches the network."""
import pytest


@pytest.fixture(autouse=True)
def _no_real_weights_mirror(monkeypatch):
    monkeypatch.setenv("KHAOS_WEIGHTS_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("REGISTER_LOG_WITNESSES", "")
