"""The pieces every runtime of an engine shares: one seeded noise draw and one parameter order taken from the
adapter file's keys, refused when a runtime's parameters and the file disagree."""
import numpy as np
import pytest

from khaos_attribution import gradient as G


def test_the_noise_is_one_draw_for_a_seed_on_any_runtime():
    a, b = G.pair_noise(7, (1, 4, 3)), G.pair_noise(7, (1, 4, 3))
    assert a.dtype == np.float32 and a.shape == (1, 4, 3) and np.array_equal(a, b)
    assert not np.array_equal(a, G.pair_noise(8, (1, 4, 3)))
    assert np.allclose(a.reshape(-1)[:3], np.random.default_rng(7).standard_normal(12)[:3].astype(np.float32))


def test_parameters_are_ordered_by_the_files_keys_whatever_prefix_the_runtime_gives():
    keys = ["model.b.parametrizations.weight.0.lora_A", "model.a.parametrizations.weight.0.lora_B",
            "model.a.parametrizations.weight.0.lora_A"]
    torch_names = ["model." + k for k in keys]
    order = G.checkpoint_order(torch_names, keys)
    assert [torch_names[i] for i in order] == ["model." + k for k in sorted(keys)]
    assert G.checkpoint_order(keys, keys) == [keys.index(k) for k in sorted(keys)]


def test_a_parameter_without_a_key_or_a_key_without_a_parameter_is_refused():
    keys = ["m.a.lora_A", "m.a.lora_B"]
    with pytest.raises(G.GradientUnavailable, match="no key"):
        G.checkpoint_order(["m.a.lora_A", "m.c.lora_B"], keys)
    with pytest.raises(G.GradientUnavailable, match="no parameter carries"):
        G.checkpoint_order(["m.a.lora_A"], keys)
    with pytest.raises(G.GradientUnavailable, match="two adapter parameters"):
        G.checkpoint_order(["x.m.a.lora_A", "y.m.a.lora_A"], ["m.a.lora_A"])
