"""Both MLX workers load the DiT, the conditioner and the adapter, and build a clip's conditioning, through one
module: run with numpy standing in for mx and a recording stand-in for the MLX trainer, the held-out worker and the
measure worker make the trainer's calls in its order and give the conditioning and losses they gave before."""
import json
import sys
import types

import numpy as np
import pytest

from khaos_attribution.engines.sa3 import heldout_worker as HW
from khaos_attribution.engines.sa3 import measure_worker_mlx as MLX

CALLS: list = []


def _fake_mx():
    mx = types.ModuleType("mlx.core")
    mx.float16, mx.float32 = np.float16, np.float32
    mx.array = lambda x, dtype=None: np.asarray(x, dtype=dtype)
    mx.stop_gradient = lambda x: x
    mx.concatenate = lambda xs, axis=0: np.concatenate(xs, axis=axis)
    mx.eval = lambda *a: None
    return mx


def _fake_trainer():
    T = types.ModuleType("lora_train_mlx")
    T.DIT_CHOICES = {"medium": {"loader": "fake_dit_loader"}}
    T.T5GEMMA_NPZ_REL = "t5gemma.npz"

    def log(name, *args):
        CALLS.append((name, *args))

    T.ensure_local = lambda rel: log("ensure_local", rel) or f"/cache/{rel}"
    T.T5Gemma = types.SimpleNamespace(from_npz=lambda path: log("t5", path) or "T5")

    class Secs:
        def __init__(self, W, b):
            self.W, self.b = W, b

        def __call__(self, seconds):
            s = np.asarray(seconds, np.float32).reshape(-1, 1, 1)
            return (s * self.W + self.b).astype(np.float32)

    T.load_conditioner_from_npz = lambda path, prefix: log("conditioner", path, prefix) or (
        "PAD", Secs(np.full((1, 1, 4), 0.5, np.float32), np.full((1, 1, 4), 0.25, np.float32)))
    T.load_lora_checkpoint = lambda path: log("lora_checkpoint", path) or (
        {"model.x.lora_A": 1, "conditioners.seconds_total.y.lora_A": 2}, {"rank": 4, "alpha": 8, "adapter_type": "lora"})
    T.underfit_lora_config = lambda: log("underfit") or {"rank": 16}
    T.inject_from_lora_config = lambda module, cfg, checkpoint_prefix: log("inject", type(module).__name__, dict(cfg),
                                                                           checkpoint_prefix)
    T.TrainableSecondsEmbedder = lambda W, b: log("trainable_secs") or Secs(W * 2, b)
    T.TrainBundle = lambda dit, secs: log("bundle", secs is not None) or types.SimpleNamespace(dit=dit, secs=secs)
    T.load_trainable_lora_state = lambda bundle, state: log("lora_state", sorted(state))

    def build_conditioning(t5, pad, prompts):
        return np.full((1, 2, 4), float(len(prompts[0])), np.float32)
    T.build_conditioning = build_conditioning
    T.grad_checkpoint = lambda layer: log("grad_checkpoint", layer)

    def rectified_flow_loss(model_fn, clean, t, noise=None, loss_mask=None):
        out = model_fn(clean.astype(np.float32), t)
        # The noise counts, so the worker's per-clip seed and draw are pinned with the loss.
        return float(np.mean(out * loss_mask[:, None, :]) + 0.1 * float(np.mean(noise.astype(np.float32) ** 2)))
    T.rectified_flow_loss = rectified_flow_loss
    return T


class _Dit:
    def __init__(self):
        self.transformer = types.SimpleNamespace(layers=["layer0"])

    def __call__(self, noised, t, cross, glob, local_add_cond=None):
        return noised * float(np.asarray(t).reshape(-1)[0]) + float(cross.mean()) + float(glob.mean())


@pytest.fixture
def mlx_stand_in(monkeypatch):
    CALLS.clear()
    mx = _fake_mx()
    pkg = types.ModuleType("mlx")
    pkg.core, pkg.nn = mx, types.ModuleType("mlx.nn")
    loader = types.ModuleType("fake_dit_loader")
    loader.load_dit = lambda path, T_lat, dtype, compile_: CALLS.append(("load_dit", path, T_lat, np.dtype(dtype).name,
                                                                          compile_)) or _Dit()
    for name, mod in (("mlx", pkg), ("mlx.core", mx), ("mlx.nn", pkg.nn), ("lora_train_mlx", _fake_trainer()),
                      ("fake_dit_loader", loader)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return mx


# Produced by main's heldout_worker.py on this stand-in, before the load was shared.
A_WITH, B_WITH, A_BASE, B_BASE = 4.819815158843994, 1.2706491947174072, 3.7541897296905518, 0.7373484969139099
LOAD_WITH_ADAPTER = [
    ("ensure_local", "t5gemma.npz"), ("t5", "/cache/t5gemma.npz"), ("conditioner", "/b/base.npz", "cond."),
    ("lora_checkpoint", "/a/adapter.safetensors"), ("underfit",),
    ("inject", "_Dit", {"rank": 4, "alpha": 8, "adapter_type": "lora"}, "model."), ("trainable_secs",),
    ("inject", "Secs", {"rank": 4, "alpha": 8, "adapter_type": "lora"}, "conditioners.seconds_total."),
    ("bundle", True), ("lora_state", ["conditioners.seconds_total.y.lora_A", "model.x.lora_A"]),
]


def test_the_held_out_worker_loads_and_conditions_as_the_trainer_does(mlx_stand_in, tmp_path):
    latents = tmp_path / "latents"
    latents.mkdir()
    for name, frames, prompt in (("a_seg_000", 5, "riff"), ("b_seg_001", 3, "")):
        np.save(latents / f"{name}.npy", np.arange(2 * frames, dtype=np.float32).reshape(2, frames) / 10)
        (latents / f"{name}.json").write_text(json.dumps({"padding_mask": [1] * (frames - 1), "seconds_total": 2.0,
                                                          "prompt": prompt}))
    req = {"latents_dir": str(latents), "base_npz": "/b/base.npz", "adapter": "/a/adapter.safetensors",
           "timesteps": [0.5], "seed": 3}
    out = HW.heldout_mlx(req)
    assert CALLS[0] == ("load_dit", "/b/base.npz", 5, "float16", False)
    assert CALLS[1:] == LOAD_WITH_ADAPTER
    assert out["per_sample"].keys() == {"a_seg_000.wav", "b_seg_001.wav"} and out["runtime"] == "mlx"
    assert out["per_sample"]["a_seg_000.wav"]["0.50"] == pytest.approx(A_WITH, rel=1e-5)
    assert out["per_sample"]["b_seg_001.wav"]["0.50"] == pytest.approx(B_WITH, rel=1e-5)
    CALLS.clear()
    base = HW.heldout_mlx({**req, "adapter": None})
    assert CALLS[1:] == LOAD_WITH_ADAPTER[:3] + [("bundle", False)], "the base alone, as the torch path scores it"
    assert base["per_sample"]["a_seg_000.wav"]["0.50"] == pytest.approx(A_BASE, rel=1e-5)
    assert base["per_sample"]["b_seg_001.wav"]["0.50"] == pytest.approx(B_BASE, rel=1e-5)


def test_the_measure_worker_loads_and_conditions_as_the_trainer_does(mlx_stand_in, monkeypatch):
    monkeypatch.setattr(MLX, "_HALF", [np.float32])
    bundle, t5, pad, secs, state = MLX._load({"base_npz": "/b/base.npz", "adapter": "/a/adapter.safetensors"}, 7)
    assert CALLS == [("load_dit", "/b/base.npz", 7, "float32", False)] + LOAD_WITH_ADAPTER
    assert sorted(state) == ["conditioners.seconds_total.y.lora_A", "model.x.lora_A"]
    item = {"latents": np.zeros((2, 6), np.float32), "prompt": "abc", "seconds": 2.0}
    cross, glob, lac = MLX._conditioning(mlx_stand_in, item, t5, pad, secs, bundle)
    # The seconds token is the trained embedder's (W doubled): 2 * 2.0 * 0.5 + 0.25.
    assert cross.shape == (1, 3, 4) and np.allclose(cross[0, :2], 3.0) and np.allclose(cross[0, 2], 2.25)
    assert np.allclose(glob, 2.25) and lac.shape == (1, 6, 3) and not lac.any()
    CALLS.clear()
    monkeypatch.setattr(MLX, "CHECKPOINT_ABOVE_LATENTS", 4)
    MLX._load({"base_npz": "/b/base.npz", "adapter": "/a/adapter.safetensors"}, 7)
    assert CALLS[-1] == ("grad_checkpoint", "layer0")
    with pytest.raises(RuntimeError, match="needs the adapter"):
        MLX._load({"base_npz": "/b/base.npz"}, 7)
