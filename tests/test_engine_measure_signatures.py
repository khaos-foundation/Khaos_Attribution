"""Every engine's measure takes the same index keywords, so the Workshop can call any of them alike."""
import importlib
import inspect

import pytest

from khaos_attribution.engines import KNOWN_ENGINES

PACKAGES = sorted(KNOWN_ENGINES.values())


@pytest.mark.parametrize("package", PACKAGES)
def test_index_rows_takes_every_keyword_the_index_builder_passes(package):
    params = inspect.signature(importlib.import_module(f"{package}.measure").index_rows).parameters
    for name in ("projection", "timesteps", "noise_seed", "target", "timestep_norm", "param_filter", "latent_transform",
                 "latent_pool", "noise_draws", "noise_shared", "param_groups", "groups_out", "on_row"):
        assert name in params, f"{package}.index_rows has no {name}"


@pytest.mark.parametrize("package", PACKAGES)
def test_output_feature_takes_the_block_sketch_keyword(package):
    params = inspect.signature(importlib.import_module(f"{package}.measure").output_feature).parameters
    assert "groups_out" in params, f"{package}.output_feature has no groups_out"
