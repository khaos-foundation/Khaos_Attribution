"""Group removal's held-out losses from one worker, behind the descriptor's switch: off, each adapter has its own
worker as before; on, one worker loads the base once and switches each adapter on in turn, or says it cannot."""
import types
from pathlib import Path

import pytest

from khaos_attribution.engines import contract_guard
from khaos_attribution.engines import worker_process as W
from khaos_attribution.engines.sa3 import DESCRIPTOR
from khaos_attribution.engines.sa3 import heldout_worker as HW
from khaos_attribution.engines.sa3 import measure as M
from khaos_attribution.engines.sa3 import train as T

API = types.SimpleNamespace(python=Path("/x/python"), root=Path("/x"), runtime="torch")
ADAPTERS = {"served": "/a/served.safetensors", "g1": "/a/g1.safetensors"}


@pytest.fixture
def switch(monkeypatch):
    monkeypatch.setattr(contract_guard, "refusal", lambda *a, **k: None)
    monkeypatch.setattr(T, "_env", lambda api: {})

    def turn(on: bool):
        monkeypatch.setitem(DESCRIPTOR.doc, "runtime", {"heldout_in_one_process": on})
    return turn


def _per_sample(value: float) -> dict:
    return {"t_seg_000.wav": {f"{t:.2f}": value for t in T.bucket_timesteps(T.DEFAULT_BUCKETS)}}


def test_the_switch_ships_off_and_off_gives_each_adapter_its_own_worker(switch, monkeypatch):
    assert DESCRIPTOR.runtime.get("heldout_in_one_process") is False
    switch(False)
    asked = []
    monkeypatch.setattr(T, "heldout_summary", lambda api, clips, adapter, **kw: asked.append(adapter) or {"mean": 1.0})
    monkeypatch.setattr(W, "run", lambda *a, **k: pytest.fail("one worker was asked for every adapter"))
    assert M.output_losses(API, "/c", ADAPTERS, seed=4) == {"served": 1.0, "g1": 1.0}
    assert asked == list(ADAPTERS.values())


def test_on_one_worker_scores_every_adapter_in_the_order_asked(switch, monkeypatch):
    switch(True)
    seen = []

    def run(python, script, request, **kw):
        seen.append((Path(script).name, request, kw["required"]))
        return {"per_adapter": [_per_sample(1.0), _per_sample(1.5)], "runtime": "torch"}
    monkeypatch.setattr(W, "run", run)
    assert M.output_losses(API, "/c", ADAPTERS, seed=4) == {"served": 1.0, "g1": 1.5}
    (script, request, required), = seen
    assert script == "heldout_worker.py" and required == "per_adapter"
    assert request["adapters"] == [str(Path(p).resolve()) for p in ADAPTERS.values()] and request["adapter"] is None
    assert request["seed"] == 4 and request["timesteps"] == T.bucket_timesteps(T.DEFAULT_BUCKETS)


def test_on_a_worker_that_cannot_vouch_for_switching_off_falls_back_to_one_worker_each(switch, monkeypatch):
    switch(True)
    monkeypatch.setattr(W, "run", lambda *a, **k: {"per_adapter": [], "fallback": "decomposed", "runtime": "torch"})
    asked = []
    monkeypatch.setattr(T, "heldout_summary", lambda api, clips, adapter, **kw: asked.append(adapter) or {"mean": 2.0})
    assert M.output_losses(API, "/c", ADAPTERS, seed=4) == {"served": 2.0, "g1": 2.0}
    assert asked == list(ADAPTERS.values())


def test_on_the_mac_or_for_one_adapter_each_keeps_its_own_worker(switch, monkeypatch):
    switch(True)
    monkeypatch.setattr(W, "run", lambda *a, **k: pytest.fail("the MLX runtime has no in-process path"))
    monkeypatch.setattr(T, "heldout_summary", lambda api, clips, adapter, **kw: {"mean": 3.0})
    assert M.output_losses(types.SimpleNamespace(**{**vars(API), "runtime": "mlx"}), "/c", ADAPTERS, seed=4)["g1"] == 3.0
    assert M.output_losses(API, "/c", {"served": "/a/s"}, seed=4) == {"served": 3.0}


# ── the switching itself, on real parametrizations ────────────────────────────

class _Adapter:
    """An adapter as Stability's parametrization holds one: an index, a strength buffer, a delta it adds."""

    def __init__(self, torch, index: int, delta, decomposed: bool = False):
        self.torch, self.index, self.delta, self.decomposed = torch, index, delta, decomposed

    def module(self):
        torch, adapter = self.torch, self

        class P(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_index = adapter.index
                self.register_buffer("lora_strength", torch.tensor(1.0))

            def forward(self, W):
                if adapter.decomposed:
                    # A decomposed adapter renormalises even at strength 0, so switching it off is not the base.
                    return (W + self.lora_strength * adapter.delta) / (1.0 + 1e-3)
                return W + self.lora_strength * adapter.delta
        return P()


class _Wrapped:
    def __init__(self, torch, layer):
        self.torch, self.model, self.adapters = torch, torch.nn.Sequential(layer), []

    def set_lora_strength(self, strength, lora_index=None):
        for mod in self.model.modules():
            for p in getattr(getattr(mod, "parametrizations", None), "weight", []) or []:
                if lora_index is None or p.lora_index == lora_index:
                    p.lora_strength.fill_(strength)


def _loaded(torch, deltas, decomposed=False):
    from torch.nn.utils import parametrize
    layer = torch.nn.Linear(3, 2, bias=False)
    wrapped = _Wrapped(torch, layer)
    for i, d in enumerate(deltas):
        parametrize.register_parametrization(layer, "weight", _Adapter(torch, i, d, decomposed).module())
    return wrapped, layer


def test_one_adapter_switched_on_among_several_is_that_adapter_alone():
    """Every other adapter at strength 0 leaves the base whole, so adapter i on is the weight it gives loaded alone."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    deltas = [torch.randn(2, 3) for _ in range(3)]
    wrapped, layer = _loaded(torch, deltas)
    base = layer.parametrizations.weight.original.detach().clone()
    assert HW._switched_off_is_base(wrapped, torch)
    for i, d in enumerate(deltas):
        wrapped.set_lora_strength(0.0)
        wrapped.set_lora_strength(1.0, lora_index=i)
        alone, solo = _loaded(torch, [d])
        with torch.no_grad():
            solo.parametrizations.weight.original.copy_(base)
            assert torch.equal(layer.weight, solo.weight)


def test_a_decomposed_adapter_is_not_vouched_for():
    torch = pytest.importorskip("torch")
    wrapped, _layer = _loaded(torch, [torch.ones(2, 3)], decomposed=True)
    assert not HW._switched_off_is_base(wrapped, torch)


def test_on_a_worker_that_fails_with_every_adapter_loaded_is_retried_one_adapter_at_a_time(switch, monkeypatch):
    """Several adapters take more memory than one: a failed one-process worker is a fallback, not a refusal."""
    switch(True)

    def fails(*a, **k):
        raise W.WorkerFailed("the Stable Audio 3 held-out worker failed (killed by SIGKILL)", why="")
    monkeypatch.setattr(W, "run", fails)
    monkeypatch.setattr(T, "heldout_summary", lambda api, clips, adapter, **kw: {"mean": 4.0})
    assert M.output_losses(API, "/c", ADAPTERS, seed=4) == {"served": 4.0, "g1": 4.0}


def test_an_adapter_at_another_position_than_its_index_is_not_vouched_for():
    """The loader keys an adapter's weights by its index; at another position its weights would load as zero."""
    torch = pytest.importorskip("torch")
    wrapped, layer = _loaded(torch, [torch.ones(2, 3), torch.ones(2, 3)])
    assert HW._switched_off_is_base(wrapped, torch)
    layer.parametrizations.weight[1].lora_index = 0
    assert not HW._switched_off_is_base(wrapped, torch)
