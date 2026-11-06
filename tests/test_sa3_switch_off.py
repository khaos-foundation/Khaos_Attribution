"""The SA3 worker switches an adapter off only when, at strength 0, every adapted weight is its base: Stability's
plain LoRA and weight-decomposed forwards both pass, and one that keeps a learned magnitude at 0 is refused."""
from types import SimpleNamespace

import pytest

from khaos_attribution.engines.sa3 import measure_worker as W


class _Strength:
    def __init__(self, layers):
        self.layers = layers

    def set_lora_strength(self, s):
        for layer in self.layers:
            layer.parametrizations.weight[0].lora_strength.fill_(float(s))


def _model(torch, forward_kind: str):
    """Two Linears adapted as Stability's ``lora_forward`` or ``dora_forward`` adapts them, or with a magnitude kept
    at strength 0 (``scale_only``), each with non-zero factors and a learned magnitude."""
    from torch.nn.utils import parametrize

    class Adapt(torch.nn.Module):
        def __init__(self, fan_in, fan_out, seed):
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

    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    for i, layer in enumerate(model):
        for p in layer.parameters():
            p.requires_grad_(False)
        parametrize.register_parametrization(layer, "weight", Adapt(layer.in_features, layer.out_features, i),
                                             unsafe=True)
    return model, _Strength(list(model))


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
