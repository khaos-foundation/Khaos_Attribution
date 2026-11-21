"""The trainer's forward on Stable Audio 3 has one owner: the held-out worker's per-sample losses and the measure
worker's clip, forward and loss on a toy model, from wav files, equal values frozen from the code before it was
shared."""
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
            h = torch.einsum("bct,cd->bdt", x, self.w)
            return torch.tanh(h + cond["emb"] + t.view(1, 1, 1)) * padding_mask.unsqueeze(1)

    class Wrapped:
        def __init__(self):
            self.model, self.device = Toy(), "cpu"

        def set_lora_strength(self, s, lora_index=None):
            pass
    return Wrapped()


def _clips(tmp_path, torch):
    import soundfile as sf
    rng = np.random.default_rng(5)
    for name, n, words in (("a_seg_000", 30, "a bright riff"), ("b_seg_001", 16, "")):
        sf.write(str(tmp_path / f"{name}.wav"), (0.3 * rng.standard_normal((n, 2))).astype(np.float32), RATE, subtype="FLOAT")
        if words:
            (tmp_path / f"{name}.txt").write_text(words, encoding="utf-8")
    return sorted(tmp_path.glob("*.wav"))


# Produced by d0f48fb's heldout_worker.py and measure_worker.py on this toy (CPU, float32).
FROZEN = {
 "heldout": {
  "a_seg_000.wav": {
   "0.25": 3.655775785446167,
   "0.75": 4.244961261749268
  },
  "b_seg_001.wav": {
   "0.25": 3.5399017333984375,
   "0.75": 3.6973724365234375
  }
 },
 "forward": [
  -0.4638978838920593,
  0.4167802333831787,
  0.2940020263195038,
  0.6916921138763428,
  0.4833274185657501,
  0.7396332621574402,
  0.4671363830566406,
  0.38252896070480347,
  0.9999792575836182,
  0.9999661445617676,
  0.9999589323997498,
  0.9999425411224365,
  0.9999449849128723,
  0.9999062418937683,
  0.9999677538871765,
  0.9999634027481079
 ],
 "loss": 4.709249973297119
}


def _values(torch, tmp_path):
    wrapped = _toy(torch)
    wavs = _clips(tmp_path, torch)
    req = {"clips_dir": str(tmp_path), "timesteps": TIMESTEPS, "seed": 11}
    heldout = HW._score_torch(wrapped, req, torch)
    clip = MW._clip(wrapped.model, torch, wavs[0], "cpu", 1234)
    forward = MW._forward(wrapped.model, torch, *clip, 0.5).reshape(-1).tolist()
    loss = MW._loss(wrapped.model, torch, clip, TIMESTEPS)
    return {"heldout": heldout, "forward": forward, "loss": loss}


def test_the_shared_forward_gives_the_frozen_values(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("soundfile")
    got = _values(torch, tmp_path)
    assert got["heldout"].keys() == FROZEN["heldout"].keys()
    for name, row in FROZEN["heldout"].items():
        np.testing.assert_allclose([got["heldout"][name][k] for k in sorted(row)], [row[k] for k in sorted(row)],
                                   rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(got["forward"], FROZEN["forward"], rtol=1e-5, atol=1e-6)
    assert got["loss"] == pytest.approx(FROZEN["loss"], rel=1e-5)
