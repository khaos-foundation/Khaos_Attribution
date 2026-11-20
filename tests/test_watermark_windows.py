"""A recording is read in windows so memory stays flat for any length: the whole reading counts every window's frames
together, as AudioSeal reads a whole array, and the strongest window's reading says whether a mark sits inside it."""
import numpy as np

from khaos_attribution import verify as V
from khaos_attribution import watermark as W

SR = 48000
CODEWORD = W.encode_payload(123)


def _logits(codeword: int, frames: int, weight: float = 1.0, flips: tuple = ()):
    bits = [(codeword >> (15 - i)) & 1 for i in range(16)]
    for i in flips:
        bits[i] ^= 1
    return np.array([weight if b else -weight for b in bits]) * frames


def _windows(monkeypatch, windows):
    seen = []

    def frames(audio, sample_rate):
        seen.append(len(audio))
        return windows[len(seen) - 1]

    monkeypatch.setattr(W, "_window_frames", frames)
    return seen


def test_short_audio_is_one_window_read_whole():
    """Up to window plus tail, the recording is one reading."""
    assert W.detect_windows(20 * SR, SR) == [(0, 20 * SR)]
    assert W.detect_windows(35 * SR, SR) == [(0, 35 * SR)]


def test_longer_audio_is_cut_into_windows_and_a_short_tail_joins_the_last():
    """Thirty-second windows; a tail under five seconds is read with the window before it."""
    assert W.detect_windows(36 * SR, SR) == [(0, 30 * SR), (30 * SR, 36 * SR)]
    assert W.detect_windows(300 * SR, SR)[-1] == (270 * SR, 300 * SR) and len(W.detect_windows(300 * SR, SR)) == 10
    assert W.detect_windows(302 * SR, SR)[-1] == (270 * SR, 302 * SR)
    assert W.detect_windows(306 * SR, SR)[-1] == (300 * SR, 306 * SR)


def test_a_short_recording_reads_as_audioseal_reads_it_whole(monkeypatch):
    """One window: the share of frames read as marked, and each bit by the sign of its mean logit."""
    seen = _windows(monkeypatch, [(97, 100, _logits(CODEWORD, 100))])
    assert W.Watermarker.detect_array(np.zeros((30 * SR, 2), dtype=np.float32), SR) == (0.97, CODEWORD, False)
    assert seen == [30 * SR]


def test_the_whole_reading_counts_every_windows_frames_together(monkeypatch):
    """Frames and bit logits are summed across windows: a weakly marked window does not outvote a strong one."""
    _windows(monkeypatch, [(30, 100, _logits(CODEWORD, 100, 5.0)), (70, 100, _logits(CODEWORD, 100, 1.0, flips=(0, 1, 2)))])
    assert W.Watermarker.detect_array(np.zeros((60 * SR, 2), dtype=np.float32), SR) == (0.5, CODEWORD, False)


def test_the_strongest_window_prefers_a_clean_id_then_a_corrected_one(monkeypatch):
    """A clean likely window beats stronger corrected or unreadable ones; with none clean, a corrected id beats none."""
    clean = (95, 100, _logits(CODEWORD, 100))
    corrected = (998, 1000, _logits(CODEWORD, 1000, flips=(3,)))
    unreadable = (9997, 10000, _logits(CODEWORD, 10000, flips=(3, 9)))
    _windows(monkeypatch, [unreadable, clean, corrected])
    assert W.Watermarker.strongest_window(np.zeros((90 * SR, 2), dtype=np.float32), SR) == (0.95, CODEWORD, False)
    _windows(monkeypatch, [unreadable, corrected])
    assert W.Watermarker.strongest_window(np.zeros((60 * SR, 2), dtype=np.float32), SR) == (0.998, CODEWORD, True)


def test_with_no_likely_readable_window_the_most_likely_reading_is_returned(monkeypatch):
    """Below the likely bar a decodable id does not outrank a more likely reading."""
    _windows(monkeypatch, [(20, 100, _logits(CODEWORD, 100)), (40, 100, _logits(CODEWORD, 100, flips=(3, 9)))])
    reading = W.Watermarker.strongest_window(np.zeros((60 * SR, 2), dtype=np.float32), SR)
    assert reading[0] == 0.4


def test_likely_is_the_probability_verify_hears():
    """One bar for 'likely marked' in the reading and in Verify."""
    assert W.LIKELY_MARKED == V.WATERMARK_THRESHOLD
