"""ACE-Step's training seams behind fakes: the masked loss is claimed only where the training step computes the loss
it masks, a dataset's lengths and facts come from one read of its tensors, and the latent frame rate is the
descriptor's."""
import sys
import types

from khaos_attribution.engines.acestep import DESCRIPTOR
from khaos_attribution.engines.acestep import batching as B


def _module(name: str, step):
    mod = types.ModuleType(name)
    mod.F = types.SimpleNamespace(mse_loss=lambda a, b, **kw: 0.0)
    cls = type("FixedLoRAModule", (), {"training_step": step, "__module__": name})
    mod.FixedLoRAModule = cls
    return mod


def test_the_masked_loss_is_claimed_only_where_the_training_step_computes_the_loss_it_masks(monkeypatch):
    def computes(self, batch):
        return F.mse_loss(batch, batch)  # noqa: F821 — the module's own F, read when the step runs

    def does_not(self, batch):
        return batch
    good, bad = _module("fake_lora_good", computes), _module("fake_lora_bad", does_not)
    monkeypatch.setitem(sys.modules, good.__name__, good)
    monkeypatch.setitem(sys.modules, bad.__name__, bad)
    assert B.install_masked_loss(bad.FixedLoRAModule) is False, "a step that computes no mse_loss is never called masked"
    assert B.install_masked_loss(good.FixedLoRAModule) is True
    B.uninstall_masked_loss(good.FixedLoRAModule)


def test_the_latent_frame_rate_is_the_descriptors():
    assert B.LATENT_FPS == DESCRIPTOR.latent_fps == 25


def test_a_datasets_lengths_and_facts_come_from_one_read_of_its_tensors(tmp_path, monkeypatch):
    loads = []

    def load(path, map_location=None, weights_only=None):
        loads.append(path)
        frames = {"a.pt": 1200, "b.pt": 300}[path.rsplit("/", 1)[-1]]
        return {"target_latents": types.SimpleNamespace(shape=(frames, 64)), "metadata": {"lyrics": "[Instrumental]"}}
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(load=load))
    paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    for p in paths:
        p.write_bytes(b"x")
    dataset = types.SimpleNamespace(valid_paths=[str(p) for p in paths])
    assert B.dataset_lengths(dataset) == [1200, 300]
    assert B.dataset_facts(dataset) == [{"frames": 1200, "instrumental": True}, {"frames": 300, "instrumental": True}]
    assert len(loads) == 2, "each tensor is read once, whichever reader asks first"
