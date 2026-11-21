"""The trainer's forward on Stable Audio 3 has one owner, forward_torch.py: the held-out worker's per-sample losses and
the measure worker's clip, forward and loss on a toy model, from wav files, equal values frozen from the code before
it was shared."""
import numpy as np
import pytest

from khaos_attribution.engines.sa3 import heldout_worker as HW
from khaos_attribution.engines.sa3 import measure_worker as MW

RATE, RATIO, CHANNELS = 16, 4, 2
TIMESTEPS = [0.25, 0.75]


def _toy(torch):
    rng = np.random.default_rng(3)

    def draw(*shape):
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    class Pretransform:
        downsampling_ratio = RATIO

        def __init__(self):
            self.w = draw(CHANNELS, 2 * RATIO)

        def encode(self, x):
            b, c, n = x.shape
            frames = x.reshape(b, c, n // RATIO, RATIO).permute(0, 2, 1, 3).reshape(b, n // RATIO, c * RATIO)
            return torch.einsum("bfk,ck->bcf", frames, self.w)

    class Toy(torch.nn.Module):
        sample_rate = RATE
        # The held-out worker casts its input to the model's precision; the measure worker does not.
        dtype = torch.bfloat16

        def __init__(self):
            super().__init__()
            self.pretransform = Pretransform()
            self.w = torch.nn.Parameter(draw(CHANNELS, CHANNELS))
            self.e = draw(CHANNELS)

        def conditioner(self, items, device):
            seconds = torch.tensor([float(i["seconds_total"]) for i in items])
            words = torch.tensor([float(len(i["prompt"])) for i in items])
            return {"emb": (self.e.view(1, -1, 1) * (seconds + 0.1 * words).view(-1, 1, 1))}

        def forward(self, x, t, cond=None, cfg_dropout_prob=0.0, padding_mask=None, **kw):
            assert torch.count_nonzero(cond["inpaint_mask"][0]) == 0 and torch.count_nonzero(cond["inpaint_masked_input"][0]) == 0
            h = torch.einsum("bct,cd->bdt", x, self.w.to(x.dtype))
            return torch.tanh(h + cond["emb"].to(x.dtype) + t.view(1, 1, 1).to(x.dtype)) * padding_mask.unsqueeze(1)

    class Wrapped:
        def __init__(self):
            self.model, self.device = Toy(), "cpu"

        def set_lora_strength(self, s, lora_index=None):
            pass
    return Wrapped()


def _clips(tmp_path, torch):
    import soundfile as sf
    rng = np.random.default_rng(5)
    for name, n, words in (("a_seg_000", 29, "a bright riff"), ("b_seg_001", 16, "")):
        sf.write(str(tmp_path / f"{name}.wav"), (0.3 * rng.standard_normal((n, 2))).astype(np.float32), RATE, subtype="FLOAT")
        if words:
            (tmp_path / f"{name}.txt").write_text(words, encoding="utf-8")
    return sorted(tmp_path.glob("*.wav"))


# Produced by 3d9d28b's heldout_worker.py and measure_worker.py on this toy (CPU).
FROZEN = {
 "heldout": {
  "a_seg_000.wav": {
   "0.25": 3.546501398086548,
   "0.75": 4.232885360717773
  },
  "b_seg_001.wav": {
   "0.25": 2.6972880363464355,
   "0.75": 3.055788993835449
  }
 },
 "forward": [
  -0.465086966753006,
  0.4155265986919403,
  0.29261621832847595,
  0.6909005045890808,
  0.48216453194618225,
  0.738945722579956,
  0.4659501910209656,
  -0.0,
  0.9999748468399048,
  0.9999589323997498,
  0.9999502301216125,
  0.9999303221702576,
  0.9999332427978516,
  0.9998862147331238,
  0.9999608993530273,
  0.0
 ],
 "loss": 3.3510621786117554,
 "gradient": [
  0.13132111728191376,
  0.00012260081712156534,
  0.02185479737818241,
  -5.2118743042228743e-05
 ]
}


def _values(torch, tmp_path):
    wrapped = _toy(torch)
    wavs = _clips(tmp_path, torch)
    req = {"clips_dir": str(tmp_path), "timesteps": TIMESTEPS, "seed": 11}
    heldout = HW._score_torch(wrapped, req, torch)
    clip = MW._clip(wrapped.model, torch, wavs[0], "cpu", 1234)
    masks = [MW._clip(wrapped.model, torch, w, "cpu", 1)[2].reshape(-1).tolist() for w in wavs]
    forward = MW._forward(wrapped.model, torch, *clip, 0.5).reshape(-1).tolist()
    loss = MW._loss(wrapped.model, torch, clip, TIMESTEPS)
    flat, _residual = MW._gradient(wrapped.model, torch, [wrapped.model.w], clip, TIMESTEPS, "vnorm")
    return {"heldout": heldout, "forward": forward, "loss": loss, "gradient": [float(v) for v in flat], "_masks": masks}


def test_the_shared_forward_gives_the_frozen_values(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("soundfile")
    got = _values(torch, tmp_path)
    assert not all(all(m) for m in got.pop("_masks")), "a clip's padding mask leaves frames out"
    assert got["heldout"].keys() == FROZEN["heldout"].keys()
    for name, row in FROZEN["heldout"].items():
        np.testing.assert_allclose([got["heldout"][name][k] for k in sorted(row)], [row[k] for k in sorted(row)],
                                   rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(got["forward"], FROZEN["forward"], rtol=1e-5, atol=1e-6)
    assert got["loss"] == pytest.approx(FROZEN["loss"], rel=1e-5)
    np.testing.assert_allclose(got["gradient"], FROZEN["gradient"], rtol=1e-5, atol=1e-7)
