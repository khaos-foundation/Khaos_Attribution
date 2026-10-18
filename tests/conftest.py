"""Every test runs with the weights mirror unreachable, so no test ever fetches from Khaos's real mirror."""
import pytest


@pytest.fixture(autouse=True)
def _no_real_weights_mirror(monkeypatch):
    monkeypatch.setenv("KHAOS_WEIGHTS_URL", "http://127.0.0.1:9")
