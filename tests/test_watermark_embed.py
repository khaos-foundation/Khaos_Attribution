"""The embedded mark reads back: the id is strengthened in steps until it decodes clean, the first clean strength is
kept, and a mark that never reads back is still a mark (the strongest, with a warning). Needs torch; the rooms' venvs
have it, the contract's own may not."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from khaos_attribution import watermark as W  # noqa: E402

SR = W._SAMPLE_RATE
WATERMARK_ID = W.encode_payload(123)


class _Generator:
    """A constant delta: the marked audio's level says how strong the mark was."""

    def get_watermark(self, x, sample_rate=None, message=None):
        return torch.full_like(x, 1e-3)


class _Detector:
    """Reads the id correctly only when the mark is at least ``needs`` strong; below that two bits come back wrong."""

    def __init__(self, needs: float):
        self.needs = needs

    def detect_watermark(self, x, sample_rate=None):
        strength = float(x.abs().max()) / 1e-3
        bits = [(WATERMARK_ID >> (15 - i)) & 1 for i in range(16)]
        if strength + 1e-6 < self.needs:
            bits[3] ^= 1
            bits[9] ^= 1
        return torch.tensor(0.99), torch.tensor([bits], dtype=torch.float32)


def _install(monkeypatch, needs):
    monkeypatch.setattr(W, "_load_models", lambda: (_Generator(), _Detector(needs)))


def test_a_mark_that_reads_back_at_the_default_strength_stays_at_it(monkeypatch):
    _install(monkeypatch, needs=0.5)
    marked, strength = W.Watermarker.embed_verified(np.zeros((SR, 2), dtype=np.float32), SR, WATERMARK_ID)
    assert strength == 1.0 and marked.shape == (SR, 2)
    assert np.allclose(marked, 1e-3, atol=1e-6)


def test_a_mark_two_bits_short_is_strengthened_until_the_id_reads_back_clean(monkeypatch):
    _install(monkeypatch, needs=1.4)
    marked, strength = W.Watermarker.embed_verified(np.zeros((SR, 2), dtype=np.float32), SR, WATERMARK_ID)
    assert strength == 1.5
    assert np.allclose(marked, 1.5e-3, atol=1e-6)
    _, read, corrected = W.Watermarker.detect_array(marked, SR)
    assert read == WATERMARK_ID and not corrected
    assert W.Watermarker.embed_array(np.zeros((SR, 2), dtype=np.float32), SR, WATERMARK_ID).max() == pytest.approx(1.5e-3)


def test_a_mark_that_never_reads_back_is_kept_at_the_strongest_with_a_warning(monkeypatch, caplog):
    _install(monkeypatch, needs=10.0)
    with caplog.at_level("WARNING"):
        marked, strength = W.Watermarker.embed_verified(np.zeros((SR, 2), dtype=np.float32), SR, WATERMARK_ID)
    assert strength == W.Watermarker.STRENGTHS[-1] == 2.0
    assert np.allclose(marked, 2e-3, atol=1e-6)
    assert "does not read back clean" in caplog.text
