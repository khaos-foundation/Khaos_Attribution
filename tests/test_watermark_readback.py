"""A caller may refuse a mark whose id never reads back: with require_readback the embed raises instead of returning
the strongest mark, and without it nothing changes. Needs torch; the rooms' venvs have it, the contract's own may not."""
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


def _audio():
    return np.zeros((SR, 2), dtype=np.float32)


@pytest.fixture
def reads(monkeypatch):
    """Forces what every read-back returns: {"id": the codeword read or None, "corrected": bool}."""
    back = {"id": None, "corrected": False}
    monkeypatch.setattr(W, "_load_models", lambda: (_Generator(), None))
    monkeypatch.setattr(W.Watermarker, "detect_array", staticmethod(lambda audio, sr: (0.99, back["id"], back["corrected"])))
    return back


def test_a_mark_that_never_reads_back_is_refused_when_the_caller_requires_it(reads, caplog):
    with caplog.at_level("WARNING"), pytest.raises(W.WatermarkUnreadable, match=str(WATERMARK_ID)):
        W.Watermarker.embed_verified(_audio(), SR, WATERMARK_ID, require_readback=True)
    assert "the strongest mark is kept" not in caplog.text
    with pytest.raises(W.WatermarkUnreadable):
        W.Watermarker.embed_array(_audio(), SR, WATERMARK_ID, require_readback=True)


def test_another_id_read_back_is_not_this_one(reads):
    reads["id"] = W.encode_payload(124)
    with pytest.raises(W.WatermarkUnreadable):
        W.Watermarker.embed_verified(_audio(), SR, WATERMARK_ID, require_readback=True)


def test_without_the_keyword_the_strongest_mark_is_still_kept_with_its_warning(reads, caplog):
    for kwargs in ({}, {"require_readback": False}):
        caplog.clear()
        with caplog.at_level("WARNING"):
            marked, strength = W.Watermarker.embed_verified(_audio(), SR, WATERMARK_ID, **kwargs)
        assert strength == W.Watermarker.STRENGTHS[-1] and np.allclose(marked, 2e-3, atol=1e-6)
        assert "does not read back" in caplog.text
        assert W.Watermarker.embed_array(_audio(), SR, WATERMARK_ID, **kwargs).max() == pytest.approx(2e-3)


def test_a_mark_that_reads_back_clean_or_corrected_is_returned_as_before(reads):
    reads["id"] = WATERMARK_ID
    marked, strength = W.Watermarker.embed_verified(_audio(), SR, WATERMARK_ID, require_readback=True)
    assert strength == 1.0 and np.allclose(marked, 1e-3, atol=1e-6)
    reads["corrected"] = True
    marked, strength = W.Watermarker.embed_verified(_audio(), SR, WATERMARK_ID, require_readback=True)
    assert strength == W.Watermarker.STRENGTHS[-1] and np.allclose(marked, 2e-3, atol=1e-6)
    assert W.Watermarker.embed_array(_audio(), SR, WATERMARK_ID, require_readback=True).max() == pytest.approx(2e-3)


def test_the_refusal_is_a_named_error_a_room_can_tell_from_a_failed_embed():
    assert issubclass(W.WatermarkUnreadable, RuntimeError)
