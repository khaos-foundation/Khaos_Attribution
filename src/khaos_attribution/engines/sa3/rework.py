"""Stable Audio 3's reworks around its worker: the regions put on the latent grid, the noise a rework starts from,
an extend cut into chunks that fit the window, and the splice that keeps the source's own samples wherever the model
was not asked to change it. Audio is read and written in the source's own sample format, so a sample outside every
region and crossfade leaves exactly as it came in. Pure numpy; soundfile only where a file is read or written."""

from __future__ import annotations

import math

import numpy as np

from khaos_attribution.engines.render import RenderFailed

REPAINT_MODES = ("conservative", "balanced", "aggressive")


def _soundfile():
    try:
        import soundfile as sf  # noqa: PLC0415
    except ImportError:
        raise RenderFailed("a Stable Audio 3 rework reads and writes audio with soundfile: install the contract's "
                           "'rooms' extra") from None
    return sf


def noise_level(mode: str, req, table: dict) -> float:
    """Where sampling starts between the source (0) and pure noise (1), for a rework mode."""
    floor = float(table.get("noise_floor", 0.01))
    if mode == "repaint":
        level = table["repaint_noise"][req.repaint_mode or "balanced"]
    elif mode == "retake":
        level = req.retake_variance if req.retake_variance is not None else table["retake_noise"]
    elif mode == "audio2audio":
        # cover_strength is how closely to follow the source, so the noise is what is left of it.
        level = 1.0 - float(req.cover_strength) if req.cover_strength is not None else table["audio2audio_noise"]
    else:
        level = 1.0
    return min(1.0, max(floor, float(level)))


def plan_regions(regions, n: int, sr: int, frame: int) -> list:
    """Regions in seconds as ``[(start, end), ...]`` sample ranges, widened outward to whole latent frames, clipped
    to the piece and merged where they meet."""
    out = []
    for start, end in regions:
        start = max(0.0, float(start))
        end = n / sr if end is None or float(end) < 0 else float(end)
        if end <= start:
            raise RenderFailed(f"a part must end after it starts: {start:.2f} s to {end:.2f} s")
        s = int(math.floor(start * sr / frame)) * frame
        e = min(n, int(math.ceil(end * sr / frame)) * frame)
        if s < e:
            out.append((s, e))
    if not out:
        raise RenderFailed("no part of the piece was chosen to change")
    out.sort()
    merged = [out[0]]
    for s, e in out[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def repaint_window(regions: list, n: int, window: int, frame: int, margin: int, sr: int) -> tuple:
    """``(offset, length)`` of the stretch of the source the model is given: all of it when it fits the window,
    else a frame-aligned window holding every region with ``margin`` samples of context either side."""
    if n <= window:
        return 0, n
    a = max(0, regions[0][0] - margin)
    b = min(n, regions[-1][1] + margin)
    earliest = -(-max(0, b - window) // frame) * frame
    latest = (a // frame) * frame
    if earliest > latest:
        raise RenderFailed(f"the parts span {(regions[-1][1] - regions[0][0]) / sr:.0f} s, more than one "
                           f"{window / sr:.0f} s pass can hold: fix them in separate passes")
    centre = ((a + b) // 2 - window // 2) // frame * frame
    offset = min(max(centre, earliest), latest)
    return offset, min(window, n - offset)


def ramp(length: int) -> np.ndarray:
    """A raised-cosine rise from 0 to 1 over ``length`` samples, never touching either end."""
    return 0.5 - 0.5 * np.cos(np.pi * (np.arange(length) + 0.5) / max(1, length))


def weights(regions: list, n: int, crossfade: int) -> np.ndarray:
    """How much of the regenerated audio each sample takes: 1 inside a region, a crossfade centred on each edge
    that is not the piece's own start or end, 0 elsewhere."""
    w = np.zeros(n, dtype=np.float64)
    for s, e in regions:
        half = max(0, min(crossfade // 2, (e - s) // 2))
        r = np.zeros(n, dtype=np.float64)
        r[s:e] = 1.0
        if half and s > 0:
            lo = max(0, s - half)
            r[lo:s + half] = ramp(2 * half)[2 * half - (s + half - lo):]
        if half and e < n:
            hi = min(n, e + half)
            r[e - half:hi] = (1.0 - ramp(2 * half))[: hi - (e - half)]
        w = np.maximum(w, r)
    return w


def _fit(audio: np.ndarray, n: int) -> np.ndarray:
    if len(audio) >= n:
        return audio[:n]
    return np.concatenate([audio, np.zeros((n - len(audio), audio.shape[1]), dtype=audio.dtype)])


def blend(a: np.ndarray, b: np.ndarray, w: np.ndarray, like: np.ndarray, step: int = 1) -> np.ndarray:
    """``(1 - w) * a + w * b`` in ``like``'s dtype, rounded to ``step`` for integer audio."""
    mixed = (1.0 - w[:, None]) * a.astype(np.float64) + w[:, None] * b.astype(np.float64)
    if np.issubdtype(like.dtype, np.integer):
        info = np.iinfo(like.dtype)
        mixed = np.clip(np.round(mixed / step) * step, info.min, info.max)
    return mixed.astype(like.dtype)


def splice(source: np.ndarray, regenerated: np.ndarray, regions: list, crossfade: int, offset: int = 0,
           step: int = 1) -> np.ndarray:
    """The source with each region taken from the regenerated audio, which starts ``offset`` samples into it."""
    n = len(source)
    w = weights(regions, n, crossfade)
    full = np.zeros_like(source)
    part = _fit(regenerated.astype(source.dtype, copy=False), min(len(regenerated), n - offset))
    full[offset:offset + len(part)] = part
    out = source.copy()
    touched = w > 0
    out[touched] = blend(source[touched], full[touched], w[touched], source, step)
    return out


def extend_plan(add: int, current: int, window: int, context: int, chunk: int | None, frame: int,
                head: bool = False) -> list:
    """The chunks an extend renders, ``[(context_samples, new_samples), ...]``: each one's context is the end (or,
    for a head, the start) of what is there so far, whole frames of it, and each chunk fits the window."""
    plans, have, left = [], current, add
    while left > 0:
        ctx = (min(context, have) // frame) * frame
        room = window - ctx
        if head:
            room = (room // frame) * frame
        new = min(left, room, chunk or room)
        if new <= 0:
            raise RenderFailed("the extend context leaves no room in the window for new audio")
        plans.append((ctx, new))
        have += new
        left -= new
    return plans


def join_tail(current: np.ndarray, generated: np.ndarray, ctx: int, crossfade: int, step: int = 1) -> np.ndarray:
    """What is there, then the generated continuation; the last ``crossfade`` samples before the seam fade from the
    source into the model's own reading of it, so the join has no step."""
    xf = min(crossfade, ctx, len(current))
    seam = len(current)
    if not xf:
        return np.concatenate([current, generated[ctx:]])
    faded = blend(current[seam - xf:], generated[ctx - xf:ctx], ramp(xf), current, step)
    return np.concatenate([current[:seam - xf], faded, generated[ctx:]])


def join_head(current: np.ndarray, generated: np.ndarray, pad: int, new: int, crossfade: int,
              step: int = 1) -> np.ndarray:
    """The generated opening, then what is there; the first ``crossfade`` samples after the seam fade from the
    model's reading of the source back into the source itself."""
    xf = min(crossfade, len(current), max(0, len(generated) - pad))
    opening = generated[pad - new:pad]
    if not xf:
        return np.concatenate([opening, current])
    faded = blend(generated[pad:pad + xf], current[:xf], ramp(xf), current, step)
    return np.concatenate([opening, faded, current[xf:]])


def read_audio(path) -> tuple:
    """``(audio, sample_rate, subtype, step)``: PCM as int32 (lossless), anything else as float64; ``step`` is the
    int32 spacing of the file's own sample values."""
    sf = _soundfile()
    info = sf.info(str(path))
    bits = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32}.get(info.subtype)
    if bits:
        audio, sr = sf.read(str(path), dtype="int32", always_2d=True)
        return audio, sr, info.subtype, 1 << (32 - bits)
    audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return audio, sr, info.subtype, 1


def read_as(path, like: np.ndarray) -> np.ndarray:
    """Audio from ``path`` in the same units as ``like`` (int32 full scale, or float)."""
    sf = _soundfile()
    dtype = "int32" if np.issubdtype(like.dtype, np.integer) else "float64"
    audio, _ = sf.read(str(path), dtype=dtype, always_2d=True)
    if audio.shape[1] != like.shape[1]:
        audio = np.repeat(audio[:, :1], like.shape[1], axis=1) if audio.shape[1] == 1 else audio[:, : like.shape[1]]
    return audio


def write_audio(path, audio: np.ndarray, sr: int, subtype: str) -> None:
    sf = _soundfile()
    if subtype not in sf.available_subtypes("WAV"):
        subtype = "FLOAT"
    sf.write(str(path), audio, sr, subtype=subtype, format="WAV")
