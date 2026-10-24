"""The research settings of the measure: a parameter filter keeps named tensors or refuses, per-timestep
normalisation scales each timestep to unit length before the mean, and an output's feature follows its index."""
import numpy as np
import pytest

from khaos_attribution import gradient as G


def test_a_parameter_filter_keeps_the_named_tensors_and_refuses_when_nothing_matches():
    names = ["blocks.0.attn.to_q.lora_A", "blocks.0.attn.to_out.0.lora_B", "blocks.1.ff.net.0.lora_A"]
    assert G.param_filter_mask(names, None) == [True, True, True]
    assert G.param_filter_mask(names, "to_out") == [False, True, False]
    with pytest.raises(G.GradientUnavailable, match="matches none"):
        G.param_filter_mask(names, "nothing_like_this")


def test_timestep_normalisation_gives_every_timestep_the_same_weight():
    loud = np.array([3.0, 4.0], dtype=np.float32)      # norm 5
    quiet = np.array([0.0, 0.1], dtype=np.float32)     # norm 0.1
    plain = G.combine_timesteps([loud, quiet], timestep_norm=False)
    normed = G.combine_timesteps([loud, quiet], timestep_norm=True)
    assert np.allclose(plain, (loud + quiet) / 2)
    assert np.allclose(normed, (loud / 5.0 + quiet / 0.1) / 2, atol=1e-5), "each timestep's gradient at unit length first"
    with pytest.raises(ValueError):
        G.combine_timesteps([], timestep_norm=True)


def test_an_outputs_feature_refuses_a_filtered_index_on_an_engine_without_the_filter():
    with pytest.raises(G.GradientUnavailable, match="parameter filter"):
        G.output_feature(object(), {"projection": {"dim": 8, "seed": 0, "n_params": 4}, "timesteps": [0.5],
                                    "noise_seed": 0, "param_filter": "to_out"}, ["x.pt"])
