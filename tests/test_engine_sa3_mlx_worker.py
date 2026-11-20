"""The MLX measure worker's own arithmetic, run with numpy standing in for mx: the clip it prepares (the mask, the
torch convention's rounding, the contract's noise), the loss's normaliser beside the torch worker's, the adapter's
parameter order, and the strength that switches the adapter off."""
import sys
import types

import numpy as np
import pytest

from khaos_attribution import gradient as G
from khaos_attribution.engines.sa3 import DESCRIPTOR
from khaos_attribution.engines.sa3 import measure_worker_mlx as MLX

FAKE_MX = types.SimpleNamespace(array=lambda x, dtype=None: np.asarray(x, dtype=dtype), float16=np.float16,
                                float32=np.float32, sum=np.sum, maximum=np.maximum)


@pytest.fixture(autouse=True)
def full_precision(monkeypatch):
    """The half precision the worker runs at, set to float32 so hand values compare exactly."""
    monkeypatch.setattr(MLX, "_HALF", [np.float32])


def _item(frames=6, channels=2, valid=4, samples=None):
    lat = np.arange(channels * frames, dtype=np.float32).reshape(channels, frames) / 10.0
    return {"wav": None, "latents": lat, "padding_mask": [1] * valid + [0] * (frames - valid),
            "samples": samples or 0, "seconds": 1.0, "prompt": ""}


def test_a_clip_is_prepared_whole_with_its_own_mask_and_the_contracts_noise():
    item = _item()
    clean, noise, mask = MLX._prepared(FAKE_MX, item, pair_seed=123)
    assert clean.shape == (1, 2, 6) and np.array_equal(clean[0], item["latents"])
    assert np.array_equal(noise, G.pair_noise(123, (1, 2, 6))), "the same draw the torch worker takes"
    assert mask.tolist() == [[True] * 4 + [False] * 2]


def test_the_torch_convention_counts_valid_frames_by_the_torch_workers_rounding():
    per = MLX.SAMPLES_PER_LATENT
    assert per == DESCRIPTOR.rework["frame_samples"], "one frame size, the descriptor's"
    for samples, frames in ((3 * per + per // 2, 4), (3 * per + per // 2 - 1, 3), (per // 4, 1), (99 * per, 6)):
        _c, _n, mask = MLX._prepared(FAKE_MX, _item(samples=samples), 1, convention="torch")
        # The torch worker's rule: max(1, min(round(n / ratio), T)).
        assert int(mask.sum()) == max(1, min(int(round(samples / per)), 6)), samples
        assert int(mask.sum()) == frames


def test_the_torch_convention_is_refused_where_the_clip_names_no_sample_count():
    """Without the sample count the MLX mask would be used, while the reply still said "torch"."""
    with pytest.raises(RuntimeError, match="sample count"):
        MLX._prepared(FAKE_MX, _item(samples=0), 1, convention="torch")


def test_the_loss_is_the_mean_square_over_valid_frames_and_channels_as_the_torch_worker_takes_it():
    item = _item()
    clean, noise, mask = MLX._prepared(FAKE_MX, item, pair_seed=5)
    pred = np.linspace(-1, 1, 12, dtype=np.float32).reshape(1, 2, 6)
    bundle = types.SimpleNamespace(dit=lambda noised, t, cross, glob, local_add_cond=None: pred)
    t = 0.3
    vnorm = MLX._objective(FAKE_MX, bundle, clean, noise, mask, t, (None, None, None), "vnorm")
    assert float(vnorm) == pytest.approx(float((pred[:, :, :4] ** 2).mean()))
    loss = MLX._objective(FAKE_MX, bundle, clean, noise, mask, t, (None, None, None), "loss")
    target = noise - clean
    assert float(loss) == pytest.approx(float(((pred - target)[:, :, :4] ** 2).mean()))


class _Layer:
    def __init__(self, alpha, rank, scaling=None):
        self.alpha, self.rank = float(alpha), int(rank)
        self.scaling = self.alpha / self.rank if scaling is None else scaling


@pytest.fixture
def lora_module(monkeypatch):
    """The engine's models.defs.lora, standing in: the adapter's layers as a bundle lists them."""
    lora = types.ModuleType("models.defs.lora")
    lora.iter_trainable_lora_layers = lambda bundle: iter(bundle.layers)
    monkeypatch.setitem(sys.modules, "models", types.ModuleType("models"))
    monkeypatch.setitem(sys.modules, "models.defs", types.ModuleType("models.defs"))
    monkeypatch.setitem(sys.modules, "models.defs.lora", lora)
    return lora


def test_the_adapter_is_switched_off_and_on_by_each_layers_own_scaling(lora_module):
    bundle = types.SimpleNamespace(layers=[_Layer(16, 16), _Layer(32, 8)])
    units = MLX._unit_scalings(bundle)
    MLX._set_strength(bundle, 0.0, units)
    assert [layer.scaling for layer in bundle.layers] == [0.0, 0.0]
    MLX._set_strength(bundle, 1.0, units)
    assert [layer.scaling for layer in bundle.layers] == [1.0, 4.0]
    MLX._set_strength(bundle, 0.5, units)
    assert [layer.scaling for layer in bundle.layers] == [0.5, 2.0]


def test_a_layer_whose_scaling_is_not_alpha_over_rank_is_refused(lora_module):
    """The strength is set by the layer's scaling only while the engine scales a layer by alpha / rank."""
    with pytest.raises(RuntimeError, match="alpha / rank"):
        MLX._unit_scalings(types.SimpleNamespace(layers=[_Layer(16, 16), _Layer(16, 8, scaling=1.0)]))


def test_the_adapters_parameters_are_read_in_the_files_sorted_key_order_and_none_else(lora_module, monkeypatch):
    """The order the torch worker flattens the same adapter in (gradient.checkpoint_order), whatever the bundle's."""
    utils = types.ModuleType("mlx.utils")
    utils.tree_flatten = lambda tree: list(tree.items())
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)
    a, b = types.SimpleNamespace(checkpoint_name="model.blocks.0.attn"), types.SimpleNamespace(checkpoint_name="model.blocks.1.ff")
    bundle = types.SimpleNamespace(layers=[a, b], named_modules=lambda: [("dit.b0", a), ("dit.b1", b)],
                                   trainable_parameters=lambda: {"dit.b1.lora_A": 3, "dit.b0.lora_B": 2, "dit.b0.lora_A": 1})
    state = {"model.blocks.1.ff.parametrizations.weight.0.lora_A": 0, "model.blocks.0.attn.parametrizations.weight.0.lora_B": 0,
             "model.blocks.0.attn.parametrizations.weight.0.lora_A": 0}
    paths, keys = MLX._ordered_names(bundle, state)
    assert keys == sorted(state) and paths == ["dit.b0.lora_A", "dit.b0.lora_B", "dit.b1.lora_A"]
    bundle.trainable_parameters = lambda: {"dit.b0.lora_A": 1, "dit.norm.weight": 9}
    with pytest.raises(RuntimeError, match="not an adapter's"):
        MLX._ordered_names(bundle, state)


def test_the_signed_mean_targets_are_refused_once_before_mlx_is_loaded():
    for target in ("vmean", "dmean", "vdelta"):
        with pytest.raises(RuntimeError, match="not built on the MLX worker"):
            MLX.run({"command": "feature", "timesteps": [0.5], "noise_seed": 0, "files": [], "target": target})


class _LoraLinear(_Layer):
    """The pinned engine's plain LoRA layer, in numpy: its base output plus the adapter's, times its scaling."""

    def __init__(self, alpha, rank, rng):
        super().__init__(alpha, rank)
        self.W = rng.standard_normal((5, 4)).astype(np.float32)
        self.A = rng.standard_normal((rank, 4)).astype(np.float32)
        self.B = rng.standard_normal((5, rank)).astype(np.float32)

    def base(self, x):
        return x @ self.W.T

    def __call__(self, x):
        return self.base(x) + ((x.astype(np.float32) @ self.A.T) @ self.B.T * self.scaling).astype(np.float32)


def test_at_strength_0_every_adapter_layer_adds_nothing_to_its_base(lora_module):
    """What the loss without the adapter reads: each layer's output at strength 0 is its base output, exactly."""
    rng = np.random.default_rng(3)
    bundle = types.SimpleNamespace(layers=[_LoraLinear(16, 16, rng), _LoraLinear(32, 8, rng)])
    x = rng.standard_normal((3, 4)).astype(np.float32)
    units = MLX._unit_scalings(bundle)
    MLX._set_strength(bundle, 0.0, units)
    assert all(np.array_equal(layer(x), layer.base(x)) for layer in bundle.layers)
    MLX._set_strength(bundle, 1.0, units)
    assert all(not np.array_equal(layer(x), layer.base(x)) for layer in bundle.layers), "strength 1 adapts again"
