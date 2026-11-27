"""ACE-Step's preparation reader on soundfile never reaches torchaudio's FFmpeg and returns exactly the tensor the
engine's own reader returns."""
import sys

import pytest


def _engine_reader(path, target_sr, max_duration):
    """The pinned engine's load_audio_stereo, word for word: torchaudio.load, then its resample, stereo and cut."""
    import torchaudio
    audio, sr = torchaudio.load(path)
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]
    max_samples = int(max_duration * target_sr)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]
    return audio, sr


@pytest.mark.parametrize("rate, channels, subtype, seconds", [
    (48000, 2, "PCM_16", 3.0), (44100, 1, "PCM_16", 3.0), (48000, 4, "PCM_16", 1.0), (48000, 2, "PCM_16", 6.0),
])
def test_the_soundfile_reader_returns_the_engine_s_tensor_without_ffmpeg(monkeypatch, tmp_path, rate, channels,
                                                                       subtype, seconds):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    sf = pytest.importorskip("soundfile")
    import numpy as np
    from khaos_attribution.engines.acestep import train as T

    rng = np.random.default_rng(7)
    path = tmp_path / "clip.wav"
    sf.write(path, (rng.uniform(-0.9, 0.9, (int(rate * seconds), channels))).astype("float32"), rate, subtype=subtype)
    try:
        want, want_sr = _engine_reader(str(path), 48000, 4.0)
    except (RuntimeError, ImportError, OSError) as exc:
        if "torchcodec" in str(exc) or "FFmpeg" in str(exc) or "ffmpeg" in str(exc):
            pytest.skip("no ffmpeg")
        raise
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    got, got_sr = T.load_audio_stereo(path, 48000, 4.0)
    assert got_sr == want_sr
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got, want)
