"""The ACE-Step engine's train and measure slots, behind fakes: the door refuses a missing checkout, configs are
built from only the fields a revision takes, the module is opened with the adapter's weights loaded, and the memory
plan and checkpointing are the rooms' rules, now in one place."""
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from khaos_attribution import engines as E
from khaos_attribution.engines.acestep import measure as M
from khaos_attribution.engines.acestep import train as T


def test_the_engine_exposes_its_three_slots_without_loading_them_first():
    engine = E.engine("acestep-1.5")
    assert isinstance(engine, E.Engine)
    assert engine.train is T and engine.measure is M and hasattr(engine.render, "AceStepRenderer")


def test_the_door_refuses_a_missing_checkout_with_the_path(tmp_path):
    with pytest.raises(T.TrainingUnavailable, match=str(tmp_path / "nowhere")):
        T.training_api(tmp_path / "nowhere")


def test_only_the_kwargs_a_revision_takes_reach_its_constructor(caplog):
    def cfg_cls(a=None, b=None):
        return dict(a=a, b=b)
    with caplog.at_level(logging.WARNING):
        assert T.accepted_config(cfg_cls, {"a": 1, "b": None, "c": 3}) == {"a": 1, "b": None}
        assert T.accepted_config(cfg_cls, {"a": 1, "b": None, "c": 3}, keep_none=False) == {"a": 1, "b": None}
    assert "does not accept ['c']" in caplog.text
    assert T.accepted_kwargs(cfg_cls, {"a": 1, "b": None}, keep_none=False) == {"a": 1}, "a None is dropped so the default applies"


def test_adapter_kwargs_follow_the_recipe_for_both_adapter_types():
    lokr = {"adapter_type": "lokr", "adapter_rank": 16, "adapter_alpha": 32, "factor": 8, "weight_decompose": True,
            "attention_type": "both", "target_modules": None}
    kind, kwargs = T.adapter_kwargs(lokr)
    assert kind == "lokr" and kwargs == {"linear_dim": 16, "linear_alpha": 32, "factor": 8, "weight_decompose": True,
                                         "attention_type": "both", "target_modules": None}
    kind, kwargs = T.adapter_kwargs({"adapter_type": "lora", "adapter_rank": 8, "adapter_alpha": 16, "attention_type": "self"})
    assert kind == "lora" and kwargs == {"r": 8, "alpha": 16, "dropout": 0.0, "attention_type": "self", "target_modules": None}
    kind, _ = T.adapter_kwargs({"adapter_rank": 4, "adapter_alpha": 4, "factor": 1, "weight_decompose": False, "attention_type": "x"})
    assert kind == "lokr", "the descriptor's adapter type when the recipe names none"


def test_precision_and_dtype_rules_are_the_trainers():
    assert T.measure_precision("fp16", "mps") == "bf16" and T.measure_precision("fp16", "cuda") == "fp16"
    assert T.preprocess_precision("mps") == "bf16" and T.preprocess_precision("cuda") == "auto"
    calls = []

    def modern(device_type, requested_precision=None):
        calls.append(requested_precision)
        return "dt"
    assert T.compute_dtype(modern, "cuda", "fp16") == "dt" and calls == ["fp16"]

    def legacy(device_type):
        return torch.float16
    assert T.compute_dtype(legacy, "mps", "fp16") is torch.bfloat16 and T.compute_dtype(legacy, "cpu", "bf16") is torch.float16
    assert T.dtype_tag(torch.bfloat16) == "bf16" and T.dtype_tag("torch.float16") == "fp16" and T.dtype_tag("odd") == "odd"
    assert T.resolve_device("cuda:0") == "cuda" and T.resolve_device("cpu") == "cpu"


def test_the_adapter_file_is_found_by_the_descriptors_names_then_any_safetensors(tmp_path):
    assert T.adapter_file_in(tmp_path) is None
    (tmp_path / "other.safetensors").write_bytes(b"x")
    assert T.adapter_file_in(tmp_path).name == "other.safetensors"
    (tmp_path / "lokr_weights.safetensors").write_bytes(b"x")
    assert T.adapter_file_in(tmp_path).name == "lokr_weights.safetensors"


def _fake_api(seen: dict, *, loadable: bool = True):
    class Net:
        def load_weights(self, path):
            seen["weights"] = path

    class Module:
        def __init__(self, *, model, adapter_config, training_config, device, dtype):
            seen["module"] = dict(model=model, adapter_config=adapter_config, training_config=training_config,
                                  device=device, dtype=dtype)
            self.dtype = dtype
            if loadable:
                self.lycoris_net = Net()

    def cfg_cls(adapter_type=None, model_variant=None, checkpoint_dir=None, dataset_dir=None, output_dir=None,
                device=None, precision=None, seed=None, batch_size=None, linear_dim=None, linear_alpha=None,
                factor=None, weight_decompose=None, attention_type=None, target_modules=None):
        return dict(locals())

    def preprocess(*, output_dir, checkpoint_dir, variant, dataset_json, device, precision):
        seen["preprocess"] = dict(locals())
        return {"processed": 2, "failed": 0}

    return T.TrainingApi(preprocess=preprocess, Trainer=None, TrainingConfig=cfg_cls, LoRAConfig=cfg_cls,
                         LoKRConfig=cfg_cls, load_decoder=lambda root, variant, device, precision: ("model", root, variant, device, precision),
                         save_final=None, Module=Module, select_dtype=lambda device_type: torch.float32)


CFG = {"adapter_type": "lokr", "adapter_rank": 16, "adapter_alpha": 32, "factor": 8, "weight_decompose": True,
       "attention_type": "both", "mixed_precision": "bf16", "model_variant": "turbo", "seed": 3}


def test_open_module_builds_the_trainers_module_and_loads_the_weights(tmp_path):
    seen: dict = {}
    module = T.open_module(_fake_api(seen), CFG, tmp_path / "w.safetensors", checkpoint_root=tmp_path / "ck",
                           device="cpu", dataset_dir=tmp_path / "t", output_dir=tmp_path / "o")
    assert seen["weights"] == str(tmp_path / "w.safetensors") and module.dtype is torch.float32
    m = seen["module"]
    assert m["model"] == ("model", str(tmp_path / "ck"), "turbo", "cpu", "bf16")
    assert m["adapter_config"]["linear_dim"] == 16 and m["adapter_config"]["target_modules"] is None
    assert m["training_config"]["batch_size"] == 1 and m["training_config"]["seed"] == 3
    assert m["training_config"]["dataset_dir"] == str(tmp_path / "t") and m["device"] == torch.device("cpu")


def test_open_module_refuses_a_revision_that_cannot_load_weights(tmp_path):
    with pytest.raises(T.TrainingUnavailable, match="untrained adapter"):
        T.open_module(_fake_api({}, loadable=False), CFG, tmp_path / "w", checkpoint_root=tmp_path, device="cpu",
                      dataset_dir=tmp_path, output_dir=tmp_path)


def test_preprocess_passes_only_what_the_revision_takes_and_the_mps_precision(tmp_path):
    seen: dict = {}
    out = T.preprocess_pairs(_fake_api(seen), out_dir=tmp_path, checkpoint_root=tmp_path / "ck", variant="turbo",
                             dataset_json=tmp_path / "d.json", device="mps")
    assert out == {"processed": 2, "failed": 0}
    assert seen["preprocess"]["precision"] == "bf16" and "audio_dir" not in seen["preprocess"]
    T.preprocess_pairs(_fake_api(seen), out_dir=tmp_path, checkpoint_root=tmp_path, variant="base", dataset_json=None,
                       device="cuda:0", precision="fp32")
    assert seen["preprocess"]["precision"] == "fp32" and seen["preprocess"]["dataset_json"] is None


def test_the_memory_numbers_come_from_the_descriptor_and_the_plan_takes_the_cheapest_way_that_fits():
    mem = E.descriptor("acestep-1.5").memory
    assert M.ENGINE_RESIDENT_GB == mem["resident_gb"] and M.SCORER_BASE_GB == mem["scorer"]["base_gb"]
    big = M.memory_plan("mps", 120, total_gb=64.0, env={})
    assert big["fits"] == "resident" and big["checkpointed"] is False and big["refusal"] is None
    assert M.memory_plan("mps", 300, total_gb=64.0, env={})["checkpointed"] is True
    for seconds in (30, 120, 300):
        tight = M.memory_plan("mps", seconds, total_gb=32.0, env={})
        assert tight["fits"] == "yielded" and tight["checkpointed"] is True
    small = M.memory_plan("mps", 120, total_gb=16.0, env={})
    assert small["fits"] is None and "16 GB" in small["refusal"]
    assert M.memory_plan("mps", 120, total_gb=16.0, env={M.LOW_MEMORY_OVERRIDE_ENV: "1"})["fits"] == "yielded"
    assert M.memory_plan("cuda", 120, total_gb=96.0, env={})["fits"] == "resident"
    assert M.memory_plan("cuda", 600, total_gb=96.0, env={})["checkpointed"] is True
    assert M.memory_plan("cuda", 120, total_gb=8.0, env={})["fits"] is None
    assert M.scorer_need_gb(100) == pytest.approx(9.4 + 12.2 + 1.6) and M.scorer_need_gb(-5) == pytest.approx(9.4 + 1.6)


def test_checkpointing_gives_the_same_gradient_with_the_layers_contents_in_eval():
    class GradientCheckpointingLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)
            self.drop = torch.nn.Dropout(0.5)
            self.gradient_checkpointing = False

        def forward(self, x):
            if self.gradient_checkpointing and self.training:
                return self._gradient_checkpointing_func(self._run, x)
            return self._run(x)

        def _run(self, x):
            return self.drop(torch.tanh(self.lin(x)))

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([GradientCheckpointingLayer() for _ in range(3)])
            self.config = SimpleNamespace(use_cache=True)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    torch.manual_seed(0)
    decoder = Decoder().eval()
    module = SimpleNamespace(model=SimpleNamespace(decoder=decoder), device_type="cpu", dtype=torch.float32)
    x = torch.randn(2, 4)
    plain = torch.autograd.grad(decoder(x).sum(), list(decoder.parameters()))
    assert M.enable_checkpointing(module) == 3 and decoder.config.use_cache is False
    saved = torch.autograd.grad(decoder(x).sum(), list(decoder.parameters()))
    for a, b in zip(plain, saved):
        assert torch.allclose(a, b)
    assert all(not layer.drop.training for layer in decoder.layers), "dropout stays in eval under the checkpoint"
    assert M.enable_checkpointing(module) == 3, "enabling twice is harmless"


def _session_api(tmp_path, seen: dict):
    class Trainer:
        def __init__(self, model, adapter_config, training_config):
            seen["trainer"] = dict(model=model, adapter_config=adapter_config, training_config=training_config)
            self.module = SimpleNamespace(lycoris_net=None)
            self.training_config = training_config

        def train(self):
            yield SimpleNamespace(step=0, loss=0.5, msg="[OK] Loaded", kind="info", epoch=0)
            yield SimpleNamespace(step=3, loss=0.25, msg="[OK] Epoch 1", kind="epoch", epoch=1, lr=1e-3, epoch_time=2.0)
            yield SimpleNamespace(step=3, loss=0.25, msg="[OK] Checkpoint", kind="checkpoint", epoch=1,
                                  checkpoint_path=str(tmp_path / "checkpoints" / "epoch_1_loss_0.2500"))

    def cfg_cls(adapter_type=None, model_variant=None, checkpoint_dir=None, dataset_dir=None, output_dir=None,
                learning_rate=None, batch_size=None, max_epochs=None, seed=None, precision=None, device=None,
                save_every_n_epochs=None, resume_from=None, linear_dim=None, linear_alpha=None, factor=None,
                weight_decompose=None, attention_type=None, target_modules=None, weight_decay=None,
                gradient_accumulation_steps=None, gradient_checkpointing=None, num_workers=None, pin_memory=None,
                persistent_workers=None, prefetch_factor=None, scheduler_type=None):
        return dict(locals())

    def save_final(trainer, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "lokr_weights.safetensors").write_bytes(b"w")

    seven = (lambda **k: {}, Trainer, cfg_cls, cfg_cls, cfg_cls,
             lambda root, variant, device, precision: ("model", variant, device, precision), save_final)
    return T.TrainingApi.from_tuple(seven)


RECIPE = {**CFG, "learning_rate": 1e-4, "batch_size": 1, "max_epochs": 2, "gradient_checkpointing": False,
          "num_workers": 0, "pin_memory": False, "persistent_workers": False, "prefetch_factor": 2,
          "checkpoint_every_n_epochs": 1, "lr_schedule": "cosine", "gradient_accumulation_steps": 1}


def test_a_session_opens_the_trainer_from_the_recipe_and_streams_its_updates(tmp_path):
    seen: dict = {}
    loaded = []
    s = T.open_session(_session_api(tmp_path, seen), RECIPE, checkpoint_root=tmp_path / "ck", tensor_dir=tmp_path / "t",
                       output_dir=tmp_path, device="cpu", precision="bf16", seed=5, on_model_loaded=loaded.append)
    assert loaded == [("model", "turbo", "cpu", "bf16")]
    tc = seen["trainer"]["training_config"]
    assert tc["max_epochs"] == 2 and tc["save_every_n_epochs"] == 1 and tc["seed"] == 5 and tc["resume_from"] is None
    assert seen["trainer"]["adapter_config"]["linear_dim"] == 16 and "target_modules" not in seen["trainer"]["adapter_config"] or \
        seen["trainer"]["adapter_config"].get("target_modules") is None
    updates = list(s.train())
    assert [u.kind for u in updates] == ["info", "epoch", "checkpoint"]
    assert updates[1].lr == 1e-3 and updates[1].epoch_time == 2.0 and updates[2].checkpoint_path.endswith("epoch_1_loss_0.2500")
    assert updates[0].checkpoint_path is None
    assert s.save_final(tmp_path / "final").name == "lokr_weights.safetensors"
    assert T.pair_files(tmp_path) == []
    (tmp_path / "b.pt").write_bytes(b"1"); (tmp_path / "a.pt").write_bytes(b"1")
    assert [p.name for p in T.pair_files(tmp_path)] == ["a.pt", "b.pt"]


def test_a_resume_on_a_revision_without_resume_from_is_refused(tmp_path):
    seen: dict = {}
    api = _session_api(tmp_path, seen)

    def old_cfg(adapter_type=None, model_variant=None, checkpoint_dir=None, dataset_dir=None, output_dir=None,
                max_epochs=None, seed=None, precision=None, device=None, batch_size=None, learning_rate=None):
        return dict(locals())
    api = T.TrainingApi(**{**api.__dict__, "TrainingConfig": old_cfg})
    with pytest.raises(RuntimeError, match="no resume_from support"):
        T.open_session(api, RECIPE, checkpoint_root=tmp_path, tensor_dir=tmp_path, output_dir=tmp_path, device="cpu",
                       precision="bf16", seed=1, resume_from=tmp_path / "ckpt")


def test_a_seam_that_cannot_install_is_recorded_off_never_claimed():
    """Fakes with no module-level F or collate cannot take the seams: every flag says off, nothing is left to undo,
    and the variety record is still written at exit."""
    record: dict = {}

    class Module:
        def training_step(self, batch):
            return 0.0

    class DM:
        def train_dataloader(self):
            return []
    hooks = T.install_loss_hooks({"masked_loss": True, "length_buckets": True, "latent_crop_sec": 0,
                                  "stem_loss_weight": 1.0}, seed=1, module_cls=Module, data_module_cls=DM,
                                 record_into=record)
    assert hooks.flags == {"masked_loss": False, "length_buckets": False} and hooks._undo == []
    with hooks:
        pass
    assert isinstance(record, dict) and record == hooks.manifest_fields()
