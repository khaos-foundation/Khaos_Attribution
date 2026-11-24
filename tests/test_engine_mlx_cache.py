"""MLX's buffer cache limit, asked for by the room: a small Mac turns it off to fit, with the same adapter and audio."""
import inspect
import types

import pytest

from khaos_attribution.engines import mlx_cache


def _mx(seen):
    return types.SimpleNamespace(set_cache_limit=lambda n: seen.append(n))


def test_no_request_leaves_mlx_as_it_is(monkeypatch):
    monkeypatch.delenv(mlx_cache.ENV, raising=False)
    seen = []
    assert mlx_cache.apply(_mx(seen)) is None and seen == []


@pytest.mark.parametrize("raw,limit", [("0", 0), ("1.5", int(1.5 * 2**30))])
def test_the_rooms_limit_is_set_in_bytes(monkeypatch, raw, limit):
    monkeypatch.setenv(mlx_cache.ENV, raw)
    seen = []
    assert mlx_cache.apply(_mx(seen)) == float(raw) and seen == [limit]


def test_a_limit_that_is_not_a_number_is_refused(monkeypatch):
    monkeypatch.setenv(mlx_cache.ENV, "off")
    with pytest.raises(ValueError, match=mlx_cache.ENV):
        mlx_cache.apply(_mx([]))


@pytest.mark.parametrize("module,function", [("train_driver_mlx", "main"), ("render_driver_mlx", "main"),
                                             ("heldout_worker", "heldout_mlx")])
def test_every_mlx_process_that_trains_scores_or_renders_applies_it(module, function):
    import importlib
    mod = importlib.import_module(f"khaos_attribution.engines.sa3.{module}")
    assert "mlx_cache.apply(" in inspect.getsource(getattr(mod, function))
