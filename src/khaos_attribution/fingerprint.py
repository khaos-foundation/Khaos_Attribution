"""Landmark audio fingerprints, format v1 (watermarking v2 §5).

A classic constellation scheme, numpy-only, for closed-set identification
of Khaos's own stored outputs. Format v1 is stable: spectral peaks from a
~93 ms Hann STFT, mapped to Hz and seconds so fingerprints are sample-rate
independent; landmarks pair each peak with up to 5 later peaks, hashed as
(f1, f2, dt) in (10, 10, 5) bits; matching votes shared hashes onto a time
offset. Same-tempo percussive loops can be pairwise-confident against each
other, so closed-set callers must also require dominance (best candidate
>= 2x the runner-up). Sustained material (a drone, a held chord) votes almost
evenly onto many offsets, so ``match_profile`` also reports the best offset
outside the winner's neighbourhood and ``is_confident`` wants the winner at
``_MATCH_MIN_PROMINENCE`` times it. Heavy deps stay lazy.
"""

from __future__ import annotations

FINGERPRINT_VERSION = "v1"

_FREQ_MIN_HZ = 40.0
_FREQ_MAX_HZ = 5000.0
_WINDOW_S = 0.093
_HOP_S = 0.0465
_PEAKS_PER_SEC = 30
_FAN_OUT = 5
_PAIR_DT_S = (0.1, 1.6)
_PAIR_DF_HZ = 1500.0
_FREQ_BITS, _DT_BITS = 10, 5
_MATCH_MIN_VOTES = 12
_MATCH_MIN_FRACTION = 0.05
_MATCH_MIN_DISTINCT = 8
_MATCH_MIN_PROMINENCE = 1.5
_PROMINENCE_GUARD_BUCKETS = 20  # 2 s either side of the winner belongs to it: a copy's votes smear over its own beat grid


def _spectral_peaks(audio, sample_rate: int):
    """[(t_seconds, f_hz)] — the constellation."""
    import numpy as np

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    window = int(sample_rate * _WINDOW_S)
    hop = max(1, int(sample_rate * _HOP_S))
    if len(audio) < window:
        return []
    hann = np.hanning(window)
    n_frames = 1 + (len(audio) - window) // hop
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(n_frames, window),
        strides=(audio.strides[0] * hop, audio.strides[0])).copy()
    mags = np.abs(np.fft.rfft(frames * hann, axis=1))
    freqs = np.fft.rfftfreq(window, 1.0 / sample_rate)
    band = (freqs >= _FREQ_MIN_HZ) & (freqs <= _FREQ_MAX_HZ)
    mags = mags[:, band]
    freqs = freqs[band]

    # local maxima over a 3x3 (time x freq-neighbourhood) grid
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(mags, 1, mode="constant")
    neigh = sliding_window_view(padded, (3, 3)).max(axis=(2, 3))
    is_peak = (mags >= neigh) & (mags > 0)

    # adaptive floor + density cap: strongest peaks per one-second block
    duration = n_frames * hop / sample_rate
    budget = max(8, int(duration * _PEAKS_PER_SEC))
    t_idx, f_idx = np.nonzero(is_peak)
    if len(t_idx) == 0:
        return []
    order = np.argsort(mags[t_idx, f_idx])[::-1][:budget]
    peaks = [((t_idx[i] * hop) / sample_rate, float(freqs[f_idx[i]]))
             for i in order]
    peaks.sort()
    return peaks


def _quantise(f_hz: float, bits: int) -> int:
    span = _FREQ_MAX_HZ - _FREQ_MIN_HZ
    q = int((f_hz - _FREQ_MIN_HZ) / span * ((1 << bits) - 1))
    return max(0, min((1 << bits) - 1, q))


def fingerprint_array(audio, sample_rate: int) -> list:
    """[(hash, offset_ms)] landmarks for one piece of audio."""
    peaks = _spectral_peaks(audio, sample_rate)
    out = []
    for i, (t1, f1) in enumerate(peaks):
        fanned = 0
        for t2, f2 in peaks[i + 1:]:
            dt = t2 - t1
            if dt < _PAIR_DT_S[0]:
                continue
            if dt > _PAIR_DT_S[1] or fanned >= _FAN_OUT:
                break
            if abs(f2 - f1) > _PAIR_DF_HZ:
                continue
            dt_q = int(dt / _PAIR_DT_S[1] * ((1 << _DT_BITS) - 1))
            landmark = ((_quantise(f1, _FREQ_BITS) << (_FREQ_BITS + _DT_BITS))
                        | (_quantise(f2, _FREQ_BITS) << _DT_BITS) | dt_q)
            out.append((landmark, int(t1 * 1000)))
            fanned += 1
    return out


QUERY_PHASES = 4


def query_phases(audio, sample_rate: int, phases: int = QUERY_PHASES) -> list:
    """The query at `phases` sub-hop offsets: a clip starts anywhere on the analysis grid, and half a hop off it
    keeps a twentieth of its votes. The stored fingerprint stays as it is; each candidate takes its best phase."""
    hop = max(1, int(sample_rate * _HOP_S))
    return [fingerprint_array(audio[k * hop // phases:], sample_rate) for k in range(phases)]


def best_phase_profile(phases: list, candidate_fps: list) -> tuple:
    """(votes, distinct, runner_up) for a candidate against the query's best phase."""
    return max((match_profile(q, candidate_fps) for q in phases), key=lambda t: t[0]) if phases else (0, 0, 0)


def best_phase_stats(phases: list, candidate_fps: list) -> tuple:
    """(votes, distinct) for a candidate against the query's best phase."""
    return best_phase_profile(phases, candidate_fps)[:2]


def dominant(votes: int, runner_up: int, ratio: float = 2.0) -> bool:
    """The closed-set rule the module states: the best candidate at least `ratio` times the runner-up."""
    return votes >= ratio * max(runner_up, 1)


def fingerprint_file(path) -> list:
    from khaos_attribution.verify import load_audio  # noqa: PLC0415

    audio, sample_rate = load_audio(path)
    return fingerprint_array(audio, sample_rate)


def query_phases_file(path, phases: int = QUERY_PHASES) -> list:
    """The four-phase query of a file, for asking the register's index."""
    from khaos_attribution.verify import load_audio  # noqa: PLC0415

    audio, sample_rate = load_audio(path)
    return query_phases(audio, sample_rate, phases)


def match_profile(query_fps: list, candidate_fps: list) -> tuple:
    """(votes, distinct_hashes, runner_up) for the best-aligned offset between two
    fingerprint lists (both [(hash, offset_ms)]); ``runner_up`` is the strongest
    offset outside the winner's neighbourhood, the plateau test's other side."""
    by_hash: dict = {}
    for landmark, offset in candidate_fps:
        by_hash.setdefault(landmark, []).append(offset)
    votes: dict = {}
    hashes: dict = {}
    for landmark, q_offset in query_fps:
        for c_offset in by_hash.get(landmark, ()):
            # 100 ms offset buckets absorb codec/trim jitter
            bucket = (c_offset - q_offset) // 100
            votes[bucket] = votes.get(bucket, 0) + 1
            hashes.setdefault(bucket, set()).add(landmark)
    if not votes:
        return 0, 0, 0
    best = max(votes, key=votes.get)
    runner_up = max((v for b, v in votes.items() if abs(b - best) > _PROMINENCE_GUARD_BUCKETS), default=0)
    return votes[best], len(hashes[best]), runner_up


def match_stats(query_fps: list, candidate_fps: list) -> tuple:
    """(votes, distinct_hashes) for the best-aligned offset between two fingerprint lists."""
    return match_profile(query_fps, candidate_fps)[:2]


def is_confident(votes: int, query_landmarks: int,
                 distinct: int | None = None, runner_up: int | None = None) -> bool:
    """``distinct`` is the winning bucket's distinct-hash count, the defence against
    loops; ``runner_up`` is ``match_profile``'s third value, the defence against
    drones. Pass both whenever available."""
    if distinct is not None and distinct < _MATCH_MIN_DISTINCT:
        return False
    if runner_up is not None and votes < _MATCH_MIN_PROMINENCE * runner_up:
        return False
    return (votes >= _MATCH_MIN_VOTES
            and query_landmarks > 0
            and votes / query_landmarks >= _MATCH_MIN_FRACTION)
