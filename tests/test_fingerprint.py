"""Fingerprint v1: same audio matches (incl. excerpts + mp3), different
audio does not. Synthetic but structured audio — tones with onsets — so
the constellation has real peaks."""

import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from khaos_attribution.fingerprint import (  # noqa: E402
    fingerprint_array,
    is_confident,
    match_stats,
)


def _music_like(seed: int, seconds: float = 12.0, sr: int = 44100):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    audio = np.zeros_like(t)
    for _ in range(24):   # note events with attack/decay envelopes
        f = rng.uniform(80, 2000)
        start = rng.uniform(0, seconds - 1)
        dur = rng.uniform(0.2, 1.0)
        mask = (t >= start) & (t < start + dur)
        env = np.exp(-4 * (t[mask] - start) / dur)
        audio[mask] += env * np.sin(2 * np.pi * f * t[mask])
    return (0.5 * audio / np.max(np.abs(audio))).astype("float32")


def test_same_audio_matches_and_different_does_not():
    a = _music_like(1)
    b = _music_like(2)
    fa = fingerprint_array(a, 44100)
    fb = fingerprint_array(b, 44100)
    assert len(fa) > 50 and len(fb) > 50
    self_votes = match_stats(fa, fa)[0]
    cross_votes = match_stats(fa, fb)[0]
    assert is_confident(self_votes, len(fa))
    assert not is_confident(cross_votes, len(fa))
    assert self_votes > cross_votes * 3


def test_excerpt_matches_with_offset():
    a = _music_like(3, seconds=20.0)
    full = fingerprint_array(a, 44100)
    excerpt = fingerprint_array(a[int(7.3 * 44100):int(13.3 * 44100)], 44100)
    votes = match_stats(excerpt, full)[0]
    assert is_confident(votes, len(excerpt)), (votes, len(excerpt))


def test_survives_mp3(tmp_path):
    import soundfile as sf
    a = _music_like(4)
    wav = tmp_path / "a.wav"
    sf.write(str(wav), a, 44100)
    mp3 = tmp_path / "a.mp3"
    back = tmp_path / "b.wav"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                        "-b:a", "128k", str(mp3)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                        str(back)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg unavailable")
    b, sr = sf.read(str(back), dtype="float32", always_2d=True)
    votes = match_stats(fingerprint_array(b, sr), fingerprint_array(a, 44100))[0]
    assert is_confident(votes, len(fingerprint_array(b, sr)))


def test_repetitive_audio_limitation_is_known_and_dominance_separates():
    """Same-tempo click tracks are pairwise-confident against each other;
    dominance (best >= 2x runner-up) is what separates the true source."""
    sr = 44100
    t = np.arange(int(10 * sr)) / sr

    def click_loop(freq, period):
        audio = np.zeros_like(t)
        for k in range(int(10 / period)):
            mask = (t >= k * period) & (t < k * period + 0.05)
            audio[mask] += np.sin(2 * np.pi * freq * t[mask])
        return (0.5 * audio).astype("float32")

    from khaos_attribution.fingerprint import match_stats
    a = fingerprint_array(click_loop(440.0, 0.5), sr)
    true_votes, _ = match_stats(a, fingerprint_array(click_loop(440.0, 0.5), sr))
    cross_votes, cross_distinct = match_stats(
        a, fingerprint_array(click_loop(661.0, 0.5), sr))
    # the limitation: the cross-match IS pairwise-confident...
    assert is_confident(cross_votes, len(a), distinct=cross_distinct)
    # ...but the true source dominates it by well over 2x
    assert true_votes >= 2 * cross_votes, (true_votes, cross_votes)
    # different-tempo loops do not even get that far
    other_votes, other_distinct = match_stats(
        a, fingerprint_array(click_loop(440.0, 0.73), sr))
    assert not is_confident(other_votes, len(a), distinct=other_distinct)


def _drone(lfo_seed: int, seconds: float = 40.0, sr: int = 44100):
    """An F-minor chord held throughout; each partial's level drifts slowly, so one
    rendering of it differs from another only in that drift."""
    rng = np.random.default_rng(lfo_seed)
    t = np.arange(int(seconds * sr)) / sr
    audio = np.zeros_like(t)
    for base_hz in (174.6, 207.7, 261.6, 349.2, 415.3, 523.3):
        for k in (1, 2, 3):
            walk = np.cumsum(rng.normal(0, 1, int(seconds * 8)))
            level = np.interp(t, np.linspace(0, seconds, len(walk)), walk)
            level = (level - level.min()) / (np.ptp(level) + 1e-9)
            audio += level * np.sin(2 * np.pi * base_hz * k * t + rng.uniform(0, 6.28)) / k
    return (0.5 * audio / np.max(np.abs(audio))).astype("float32")


def test_a_held_chord_played_again_is_a_plateau_and_an_excerpt_of_it_a_peak():
    """A drone aligns at any offset: votes and distinct hashes pass on their own, and prominence is
    what refuses the chord played again while a true excerpt still passes."""
    from khaos_attribution.fingerprint import match_profile
    sr = 44100
    full = fingerprint_array(_drone(1), sr)
    excerpt = fingerprint_array(_drone(1)[int(10 * sr):int(25 * sr)], sr)
    again = fingerprint_array(_drone(2, seconds=15.0), sr)
    votes, distinct, runner = match_profile(excerpt, full)
    assert is_confident(votes, len(excerpt), distinct, runner), (votes, distinct, runner)
    votes, distinct, runner = match_profile(again, full)
    assert is_confident(votes, len(again), distinct), (votes, distinct)        # the limitation
    assert not is_confident(votes, len(again), distinct, runner), (votes, runner)
    assert match_stats(again, full) == (votes, distinct)
