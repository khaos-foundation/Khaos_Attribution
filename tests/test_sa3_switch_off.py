"""The SA3 worker switches an adapter off only when, at strength 0, every adapted weight is its base: Stability's
plain LoRA and weight-decomposed forwards both pass, and one that keeps a learned magnitude at 0 is refused, by the
function and through the worker's ``run`` at the loss command and the adapter-delta target."""
from types import SimpleNamespace

import pytest

from khaos_attribution.engines.sa3 import measure_worker as W


class _Strength:
    device = "cpu"

    def __init__(self, layers):
        self.layers = layers

    def set_lora_strength(self, s):
        for layer in self.layers:
            layer.parametrizations.weight[0].lora_strength.fill_(float(s))


def _adapt(torch, layer, forward_kind: str, seed: int):
    """``layer``'s frozen weight parametrized as Stability's ``lora_forward`` or ``dora_forward`` adapts it, or with
    a magnitude kept at strength 0 (``scale_only``), with non-zero factors and a learned magnitude."""
    from torch.nn.utils import parametrize

    class Adapt(torch.nn.Module):
        def __init__(self, fan_in, fan_out):
            super().__init__()
            gen = torch.Generator().manual_seed(seed)
            self.scaling = 1.0
            self.register_buffer("lora_strength", torch.tensor(1.0))
            self.lora_A = torch.nn.Parameter(torch.randn(2, fan_in, generator=gen))
            self.lora_B = torch.nn.Parameter(torch.randn(fan_out, 2, generator=gen))
            self.magnitude = torch.nn.Parameter(torch.rand(fan_out, generator=gen) + 0.5)

        def forward(self, W0):
            delta = self.scaling * self.lora_strength * (self.lora_B @ self.lora_A)
            if forward_kind == "lora":
                return W0 + delta.to(W0.dtype)
            if forward_kind == "dora" and self.lora_strength == 0:
                return W0
            V = W0 + delta
            return V / V.norm(dim=1, keepdim=True) * self.magnitude.unsqueeze(1)

    for p in layer.parameters():
        p.requires_grad_(False)
    parametrize.register_parametrization(layer, "weight", Adapt(layer.in_features, layer.out_features), unsafe=True)
    return layer


def _model(torch, forward_kind: str):
    """Two Linears adapted alike."""
    model = torch.nn.Sequential(_adapt(torch, torch.nn.Linear(4, 3), forward_kind, 0),
                                _adapt(torch, torch.nn.Linear(3, 2), forward_kind, 1))
    return model, _Strength(list(model))


def _served(torch, forward_kind: str):
    """One adapted Linear behind the forward the worker calls, so ``run`` can take a loss or a gradient on it."""
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = _adapt(torch, torch.nn.Linear(3, 3), forward_kind, 0)

        def forward(self, x, t, cond=None, cfg_dropout_prob=0.0, padding_mask=None, **kw):
            return self.lin(x.transpose(1, 2)).transpose(1, 2) + cond["emb"] + t.view(1, 1, 1)

    model = Toy()
    return model, _Strength([model.lin])


def _run(torch, monkeypatch, forward_kind: str, request: dict) -> dict:
    """``W.run`` on the served toy, with the engine's loading and clip encoding replaced and nothing else."""
    model, wrapped = _served(torch, forward_kind)
    names = [n for n, _p in model.named_parameters() if "lora_" in n]
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    for p in params:
        p.requires_grad_(True)
    monkeypatch.setattr(W, "_load", lambda req: (wrapped, model, params, torch, sorted(names), names, False))
    gen = torch.Generator().manual_seed(1)

    def clip(model, torch, wav, device, pair_seed):
        latents, noise = torch.randn(1, 3, 5, generator=gen), torch.randn(1, 3, 5, generator=gen)
        return latents, {"emb": torch.zeros(1, 3, 1)}, torch.ones(1, 5, dtype=torch.bool), noise

    monkeypatch.setattr(W, "_clip", clip)
    try:
        return W.run({"adapter": "a.safetensors", "files": ["a_seg_000.wav"], "timesteps": [0.3, 0.7], "noise_seed": 1,
                      **request})
    finally:
        assert float(model.lin.parametrizations.weight[0].lora_strength) == 1.0, "the adapter is left at full strength"


def test_the_loss_command_reads_with_and_without_only_an_adapter_that_switches_off(monkeypatch):
    torch = pytest.importorskip("torch")
    reply = _run(torch, monkeypatch, "lora", {"command": "loss"})
    assert set(reply) == {"with", "without", "n"} and reply["n"] == 1 and reply["with"] != reply["without"]
    with pytest.raises(RuntimeError, match="1 of 1 adapted weights differ .* a measurement"):
        _run(torch, monkeypatch, "scale_only", {"command": "loss"})


def test_the_adapter_delta_target_is_refused_through_run_on_an_adapter_that_does_not_switch_off(monkeypatch):
    torch = pytest.importorskip("torch")
    meta = {"projection": {"n_params": 12, "dim": 4, "seed": 0}}
    reply = _run(torch, monkeypatch, "lora", {"command": "feature", "target": "vdelta", "index_meta": meta})
    assert reply["n"] == 1 and len(reply["feature"]) == 4
    with pytest.raises(RuntimeError, match="1 of 1 adapted weights differ .* the vdelta target"):
        _run(torch, monkeypatch, "scale_only", {"command": "feature", "target": "vdelta", "index_meta": meta})


@pytest.mark.parametrize("kind", ["lora", "dora"])
def test_an_adapter_whose_strength_zero_is_its_base_is_switched_off(kind):
    torch = pytest.importorskip("torch")
    model, wrapped = _model(torch, kind)
    W._require_switch_off(wrapped, model, torch, "a measurement")
    assert all(float(layer.parametrizations.weight[0].lora_strength) == 1.0 for layer in model)


def test_an_adapter_that_keeps_its_magnitude_at_zero_is_refused_and_left_at_full_strength():
    torch = pytest.importorskip("torch")
    model, wrapped = _model(torch, "scale_only")
    with pytest.raises(RuntimeError, match="2 of 2 adapted weights differ"):
        W._require_switch_off(wrapped, model, torch, "a measurement")
    assert all(float(layer.parametrizations.weight[0].lora_strength) == 1.0 for layer in model)


def test_a_model_with_no_adapted_weight_is_refused():
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="0 of 0"):
        W._require_switch_off(SimpleNamespace(set_lora_strength=lambda s: None), torch.nn.Linear(2, 2), torch, "x")
