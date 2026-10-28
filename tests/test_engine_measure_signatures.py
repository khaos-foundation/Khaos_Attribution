"""Every engine's measure takes the same index keywords, so the Workshop can call any of them alike."""
import importlib
import inspect

import pytest


@pytest.mark.parametrize("engine", ["acestep", "sa3", "mimetic"])
def test_index_rows_takes_every_keyword_the_index_builder_passes(engine):
    params = inspect.signature(importlib.import_module(f"khaos_attribution.engines.{engine}.measure").index_rows).parameters
    for name in ("projection", "timesteps", "noise_seed", "target", "timestep_norm", "param_filter", "on_row"):
        assert name in params, f"{engine}.index_rows has no {name}"
