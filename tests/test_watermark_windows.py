"""A recording is read in windows: memory stays flat for any length, a mark inside a longer recording is caught, and
the reading returned serves both the import's strong rule and Verify's likely rule."""
import numpy as np

from khaos_attribution import verify as V
from khaos_attribution import watermark as W

SR = 48000


def test_short_audio_is_one_window_read_whole():
    """Up to window plus tail, nothing changes from a whole-track reading."""
    assert W.detect_windows(20 * SR, SR) == [(0, 20 * SR)]
    assert W.detect_windows(35 * SR, SR) == [(0, 35 * SR)]


def test_longer_audio_is_cut_into_windows_and_a_short_tail_joins_the_last():
    """Thirty-second windows; a tail under five seconds is read with the window before it."""
    assert W.detect_windows(36 * SR, SR) == [(0, 30 * SR), (30 * SR, 36 * SR)]
    assert W.detect_windows(300 * SR, SR)[-1] == (270 * SR, 300 * SR) and len(W.detect_windows(300 * SR, SR)) == 10
    assert W.detect_windows(302 * SR, SR)[-1] == (270 * SR, 302 * SR)
    assert W.detect_windows(306 * SR, SR)[-1] == (300 * SR, 306 * SR)


def _readings(monkeypatch, readings):
    seen = []

    def one(audio, sample_rate):
        seen.append(len(audio))
        return readings[len(seen) - 1]

    monkeypatch.setattr(W, "_detect_one", one)
    return seen


def test_a_clean_likely_window_is_preferred_to_a_stronger_corrected_one(monkeypatch):
    """The import's strong rule needs a clean ID: one clean window at 0.95 wins over 0.998 corrected and 0.99 unreadable."""
    _readings(monkeypatch, [(0.99, None, False), (0.95, 0b1010011010110101, False), (0.998, 0b1, True)])
    assert W.Watermarker.detect_array(np.zeros((90 * SR, 2), dtype=np.float32), SR) == (0.95, 0b1010011010110101, False)


def test_with_no_clean_likely_window_the_most_likely_reading_is_returned(monkeypatch):
    """Verify still hears a corrected mark at 0.998 even when a weak window happens to decode cleanly."""
    _readings(monkeypatch, [(0.2, 0b1010011010110101, False), (0.9975, 0b1, True)])
    assert W.Watermarker.detect_array(np.zeros((60 * SR, 2), dtype=np.float32), SR) == (0.9975, 0b1, True)


def test_a_short_recording_is_read_once_whole(monkeypatch):
    """One reading of the whole array, as before windows."""
    seen = _readings(monkeypatch, [(0.97, 59368, False)])
    assert W.Watermarker.detect_array(np.zeros((30 * SR, 2), dtype=np.float32), SR) == (0.97, 59368, False)
    assert seen == [30 * SR]


def test_likely_is_the_probability_verify_hears():
    """One bar for 'likely marked' in the reading and in Verify."""
    assert W.LIKELY_MARKED == V.WATERMARK_THRESHOLD
