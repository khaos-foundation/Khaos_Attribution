"""ACE-Step's training side behind the socket: the one door to its training_v2 modules, the precision rule on
Apple silicon, the adapter and training configs built from a recipe, the preprocessing call, and a live module
with an adapter's weights loaded, which is the object the gradient feature is taken on. Both rooms call these;
neither imports ACE-Step's training code by itself."""

from __future__ import annotations

import inspect
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from khaos_attribution.engines.acestep import DESCRIPTOR

log = logging.getLogger(__name__)

PRECISION = "bf16"


from khaos_attribution.engines.train import TrainingUnavailable  # noqa: E402,F401


@dataclass(frozen=True)
class TrainingApi:
    """The engine's training classes, by Khaos's names, so a rename upstream is absorbed in one place."""
    preprocess: Any
    Trainer: Any
    TrainingConfig: Any
    LoRAConfig: Any
    LoKRConfig: Any
    load_decoder: Any
    save_final: Any
    Module: Any = None
    select_dtype: Any = None

    @classmethod
    def from_tuple(cls, seven: tuple) -> "TrainingApi":
        """The seven the Workshop's door hands out, back into an api (the module and dtype selector unset)."""
        preprocess, trainer, training_config, lora, lokr, load_decoder, save_final = seven
        return cls(preprocess=preprocess, Trainer=trainer, TrainingConfig=training_config, LoRAConfig=lora,
                   LoKRConfig=lokr, load_decoder=load_decoder, save_final=save_final)

    def as_tuple(self) -> tuple:
        """The seven the Workshop's door hands out, in its order."""
        return (self.preprocess, self.Trainer, self.TrainingConfig, self.LoRAConfig, self.LoKRConfig,
                self.load_decoder, self.save_final)


def preload_npp() -> None:
    """torch's CUDA wheels do not preload NVIDIA's NPP, which torchcodec's CUDA build links; load it when the venv
    carries it."""
    import ctypes  # noqa: PLC0415
    try:
        import nvidia.npp as npp  # noqa: PLC0415
    except ImportError:
        return
    lib = Path(next(iter(npp.__path__))) / "lib"
    for name in ("libnppc.so.12", "libnppicc.so.12"):
        try:
            ctypes.CDLL(str(lib / name), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            return


def training_api(repo_dir: Path | str) -> TrainingApi:
    """ACE-Step's training_v2 door: the checkout on the path, its classes imported, the MPS precision rule applied."""
    root = str(Path(repo_dir).resolve())
    if not Path(root).is_dir():
        raise TrainingUnavailable(f"ACE-Step not found at {root}")
    if root not in sys.path:
        sys.path.insert(0, root)
    preload_npp()
    try:
        from acestep.training_v2.configs import LoKRConfigV2, LoRAConfigV2, TrainingConfigV2  # noqa: PLC0415
        from acestep.training_v2.fixed_lora_module import FixedLoRAModule, _select_compute_dtype  # noqa: PLC0415
        from acestep.training_v2.model_loader import load_decoder_for_training  # noqa: PLC0415
        from acestep.training_v2.preprocess import preprocess_audio_files  # noqa: PLC0415
        from acestep.training_v2.trainer_fixed import FixedLoRATrainer  # noqa: PLC0415
        from acestep.training_v2.trainer_helpers import save_final  # noqa: PLC0415
    except ImportError as exc:
        raise TrainingUnavailable(f"ACE-Step's training modules are not importable from {root}: {exc}") from exc
    patch_mps_precision()
    patch_vae_precision()
    return TrainingApi(preprocess=preprocess_audio_files, Trainer=FixedLoRATrainer, TrainingConfig=TrainingConfigV2,
                       LoRAConfig=LoRAConfigV2, LoKRConfig=LoKRConfigV2, load_decoder=load_decoder_for_training,
                       save_final=save_final, Module=FixedLoRAModule, select_dtype=_select_compute_dtype)


def patch_vae_precision() -> None:
    """Preprocessing encodes the training latents with the codec (VAE) in fp32 whatever the pass runs in: in bf16
    the encoder drifts a quarter from the fp32 latents and adds a high-frequency hash. The preprocessor imports the
    loader by name at call time, so the loader is replaced; the text encoder keeps its precision. Idempotent."""
    from acestep.training_v2 import model_loader as _ml  # noqa: PLC0415

    if getattr(_ml, "_khaos_vae_fp32", False):
        return
    orig = _ml.load_vae

    def load_vae(checkpoint_dir, device="cpu", precision="bf16"):
        return orig(checkpoint_dir, device, "fp32")

    _ml.load_vae = load_vae
    _ml._khaos_vae_fp32 = True


def patch_mps_precision() -> None:
    """MPS trains in bf16 as CUDA does: the engine's fp16 trips the GradScaler there and fp32 gives non-finite
    gradients. Both modules import the selectors by value, so both are patched; a revision whose selectors take
    requested_precision honours the config already and is left alone."""
    import torch  # noqa: PLC0415
    from acestep.training_v2 import fixed_lora_module as _flm  # noqa: PLC0415
    from acestep.training_v2 import trainer_fixed as _tf  # noqa: PLC0415

    if getattr(_flm, "_khaos_mps_precision_patched", False):
        return
    orig_dtype, orig_precision = _flm._select_compute_dtype, _flm._select_fabric_precision
    if "requested_precision" in inspect.signature(orig_dtype).parameters:
        _flm._khaos_mps_precision_patched = True
        log.info("ACE-Step's dtype selector honours requested_precision; MPS precision patch not needed")
        return

    def _compute_dtype(device_type: str):
        return torch.bfloat16 if device_type == "mps" else orig_dtype(device_type)

    def _fabric_precision(device_type: str):
        return "bf16-mixed" if device_type == "mps" else orig_precision(device_type)

    for mod in (_flm, _tf):
        if hasattr(mod, "_select_compute_dtype"):
            mod._select_compute_dtype = _compute_dtype
        if hasattr(mod, "_select_fabric_precision"):
            mod._select_fabric_precision = _fabric_precision
    _flm._khaos_mps_precision_patched = True
    log.info("Patched ACE-Step MPS precision to bf16 / bf16-mixed")


def accepted_kwargs(cls: Any, kwargs: dict, *, keep_none: bool = True, what: str | None = None) -> dict:
    """Only the kwargs this revision's constructor takes, the rest logged and left out; a callable whose signature
    cannot be read takes them all. None values are dropped unless kept, so a default applies in their place."""
    try:
        accepted = set(inspect.signature(cls).parameters)
    except (TypeError, ValueError):
        accepted = set(kwargs)
    kept = {k: v for k, v in kwargs.items() if k in accepted and (keep_none or v is not None)}
    dropped = sorted(set(kwargs) - accepted)
    if dropped:
        log.warning("%s on this ACE-Step revision does not accept %s; omitting", what or getattr(cls, "__name__", cls),
                    dropped)
    return kept


def accepted_config(cls: Any, kwargs: dict, *, keep_none: bool = True) -> Any:
    """``cls`` built from the kwargs it takes."""
    return cls(**accepted_kwargs(cls, kwargs, keep_none=keep_none))


def adapter_kwargs(cfg: dict) -> tuple:
    """(adapter type, the adapter config's candidate kwargs) a recipe describes."""
    adapter_type = str(cfg.get("adapter_type") or DESCRIPTOR.adapter["type"])
    rank, alpha = int(cfg["adapter_rank"]), int(cfg["adapter_alpha"])
    if adapter_type == "lokr":
        return adapter_type, {"linear_dim": rank, "linear_alpha": alpha, "factor": int(cfg["factor"]),
                              "weight_decompose": bool(cfg["weight_decompose"]),
                              "attention_type": str(cfg["attention_type"]), "target_modules": cfg.get("target_modules")}
    return adapter_type, {"r": rank, "alpha": alpha, "dropout": float(cfg.get("dropout", 0.0)),
                          "attention_type": str(cfg["attention_type"]), "target_modules": cfg.get("target_modules")}


def adapter_config(api: TrainingApi, cfg: dict, *, keep_none: bool = True) -> tuple:
    """(adapter type, the engine's adapter config) for a recipe."""
    adapter_type, kwargs = adapter_kwargs(cfg)
    cls = api.LoKRConfig if adapter_type == "lokr" else api.LoRAConfig
    return adapter_type, accepted_config(cls, kwargs, keep_none=keep_none)


def resolve_device(requested: str = "auto") -> str:
    """The device TYPE ("cuda", "mps", "cpu"), never an index like cuda:0: what an index records."""
    import torch  # noqa: PLC0415
    if requested and requested != "auto":
        return torch.device(requested).type
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def measure_precision(requested: str, device_type: str) -> str:
    """The trainer's rule for a feature: fp16 becomes bf16 on MPS (no gradient scaling there), else as asked."""
    return PRECISION if device_type == "mps" and requested in {"fp16", "16-mixed"} else requested


def preprocess_precision(device_type: str) -> str:
    """What preprocessing runs in: bf16 on MPS, the engine's own choice elsewhere."""
    return PRECISION if device_type == "mps" else "auto"


def compute_dtype(selector: Any, device_type: str, precision: str) -> Any:
    """The compute dtype the trainer uses for ``precision``: the selector is asked for it when it takes
    requested_precision; older revisions get bf16 on MPS, as the patch gives the trainer."""
    import torch  # noqa: PLC0415
    try:
        takes = "requested_precision" in inspect.signature(selector).parameters
    except (TypeError, ValueError):
        takes = False
    if takes:
        return selector(device_type, requested_precision=precision)
    return torch.bfloat16 if device_type == "mps" else selector(device_type)


def dtype_tag(dtype: Any) -> str:
    """``torch.bfloat16`` -> ``"bf16"``: the short name results are filed under."""
    name = str(dtype).replace("torch.", "")
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(name, name)


def adapter_file_in(directory: Path | str) -> Path | None:
    """The adapter weights inside a checkpoint directory: the descriptor's names first, then any safetensors."""
    directory = Path(directory)
    for name in DESCRIPTOR.adapter["filenames"]:
        if (directory / name).is_file():
            return directory / name
    files = sorted(directory.glob("*.safetensors"))
    return files[0] if files else None


def preprocess_pairs(api: TrainingApi, *, out_dir: Path | str, checkpoint_root: Path | str, variant: str,
                     dataset_json: Path | str | None, device: str, precision: str | None = None,
                     audio_dir: Path | str | None = None) -> dict:
    """The engine's preprocessing of a dataset into pairs under ``out_dir``, with only the kwargs this revision
    takes; its own counts come back."""
    device_type = str(device).split(":")[0]
    kwargs = {"audio_dir": None if audio_dir is None else str(audio_dir), "output_dir": str(out_dir),
              "checkpoint_dir": str(checkpoint_root), "variant": str(variant),
              "dataset_json": None if dataset_json is None else str(dataset_json), "device": str(device),
              "precision": precision or preprocess_precision(device_type)}
    return api.preprocess(**accepted_kwargs(api.preprocess, kwargs, what="preprocess_audio_files")) or {}


def open_module(api: TrainingApi, cfg: dict, adapter_file: Path | str, *, checkpoint_root: Path | str, device: str,
                dataset_dir: Path | str, output_dir: Path | str, precision: str | None = None) -> Any:
    """A live training-side module for this recipe with the adapter's weights loaded: the object the gradient
    feature is taken on. Device and precision are this machine's, never the run's recorded ones."""
    import torch  # noqa: PLC0415
    torch_device = torch.device(device)
    precision = measure_precision(str(precision or cfg.get("mixed_precision") or PRECISION), torch_device.type)
    variant = str(cfg.get("model_variant") or DESCRIPTOR.default_variant)
    model = api.load_decoder(str(checkpoint_root), variant, device, precision)
    adapter_type, adapter_cfg = adapter_config(api, cfg)
    train_kwargs = {"adapter_type": adapter_type, "model_variant": variant, "checkpoint_dir": str(checkpoint_root),
                    "dataset_dir": str(dataset_dir), "output_dir": str(output_dir), "device": device,
                    "precision": precision, "seed": int(cfg.get("seed", 0)), "batch_size": 1}
    train_cfg = accepted_config(api.TrainingConfig, train_kwargs)
    dtype = compute_dtype(api.select_dtype, torch_device.type, precision)
    module = api.Module(model=model, adapter_config=adapter_cfg, training_config=train_cfg, device=torch_device,
                        dtype=dtype)
    net = getattr(module, "lycoris_net", None)
    if net is None or not hasattr(net, "load_weights"):
        raise TrainingUnavailable(f"adapter type {adapter_type!r} exposes no load_weights on this ACE-Step revision; "
                                  "the feature would be taken on an untrained adapter")
    net.load_weights(str(adapter_file))
    log.info("Opened %s with %s in %s", variant, Path(adapter_file).name, dtype_tag(module.dtype))
    return module


# ── the training session ─────────────────────────────────────────────────────

from khaos_attribution.engines.train import TrainingUpdate  # noqa: E402,F401

PAIR_SUFFIX = ".pt"


def pair_files(tensor_dir: Path | str) -> list:
    """The preprocessed pairs under a directory, sorted by name."""
    return sorted(Path(tensor_dir).glob(f"*{PAIR_SUFFIX}"))


class LossHooks:
    """What install_loss_hooks installed, plus the undo: a context manager, so a class patch never outlives the run
    that asked for it. ``flags`` is what was actually installed; the variety record is read at exit."""

    def __init__(self, record_into: dict | None = None) -> None:
        self.flags: dict = {"masked_loss": False, "length_buckets": False}
        self.record_fn = None
        self._record_into = record_into
        self._undo: list = []

    def add_undo(self, fn) -> None:
        self._undo.append(fn)

    def manifest_fields(self) -> dict:
        return dict(self.record_fn()) if self.record_fn is not None else {}

    def uninstall(self) -> None:
        if self._record_into is not None:
            self._record_into.update(self.manifest_fields())
        while self._undo:
            fn = self._undo.pop()
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                log.error("loss hook uninstall failed: %s", exc)

    def __enter__(self) -> "LossHooks":
        return self

    def __exit__(self, *exc) -> None:
        self.uninstall()


def install_loss_hooks(cfg: dict, seed: int, start_epoch: int = 0, module_cls=None, data_module_cls=None, stem_ids=(),
                       training_config=None, record_into: dict | None = None) -> LossHooks:
    """The masked loss, length-bucket and variety seams (batching.py) when the recipe asks for them, on ACE-Step's
    classes unless fakes are given; a seam that fails to install is logged and recorded as off, never silent."""
    from khaos_attribution.engines.acestep import batching as B  # noqa: PLC0415
    hooks = LossHooks(record_into=record_into)
    if cfg.get("masked_loss"):
        try:
            if module_cls is None:
                from acestep.training_v2.fixed_lora_module import FixedLoRAModule as module_cls  # noqa: PLC0415
            if B.install_masked_loss(module_cls):
                hooks.flags["masked_loss"] = True
                hooks.add_undo(lambda: B.uninstall_masked_loss(module_cls))
        except Exception as exc:  # noqa: BLE001
            log.error("masked_loss requested but not installed (%s) — training UNMASKED", exc)
    if cfg.get("length_buckets"):
        try:
            if data_module_cls is None:
                from acestep.training.data_module import PreprocessedDataModule as data_module_cls  # noqa: PLC0415
            if B.install_length_buckets(data_module_cls, seed, start_epoch=start_epoch):
                hooks.flags["length_buckets"] = True
                hooks.add_undo(lambda: B.uninstall_length_buckets(data_module_cls))
        except Exception as exc:  # noqa: BLE001
            log.error("length_buckets requested but not installed (%s) — upstream shuffle", exc)
    for k, v in hooks.flags.items():
        if cfg.get(k) and not v:
            log.warning("%s: requested but OFF for this run (see the error above)", k)
    # Crops go on after length_buckets: the crop loader wraps whichever train_dataloader the class has when installed.
    crop_state = step_state = timestep_rec = None
    window = float(cfg.get("latent_crop_sec") or 0.0)
    if window > 0:
        try:
            if data_module_cls is None:
                from acestep.training.data_module import PreprocessedDataModule as data_module_cls  # noqa: PLC0415
            crop_state = B.install_latent_crops(data_module_cls, window, seed)
            if crop_state is not None:
                hooks.add_undo(lambda: B.uninstall_latent_crops(data_module_cls))
        except Exception as exc:  # noqa: BLE001
            log.error("latent_crop_sec %.3g requested but not installed (%s) — whole latents", window, exc)
    weight = float(cfg.get("stem_loss_weight", 1.0))
    if weight != 1.0 or crop_state is not None:
        try:
            if module_cls is None:
                from acestep.training_v2.fixed_lora_module import FixedLoRAModule as module_cls  # noqa: PLC0415
            step_state = B.install_step_hooks(module_cls, stem_ids, weight, crop_state)
            if step_state is not None:
                hooks.add_undo(lambda: B.uninstall_step_hooks(module_cls))
        except Exception as exc:  # noqa: BLE001
            log.error("stem_loss_weight %.3g requested but not installed (%s) — every step at weight 1", weight, exc)
    if training_config is not None:
        timestep_rec = B.apply_timestep_bias(training_config, float(cfg.get("timestep_high_noise_bias") or 0.0))
        if timestep_rec is not None:
            hooks.add_undo(lambda: B.restore_timestep_bias(training_config))
    hooks.record_fn = lambda: B.variety_record(crop_state, step_state, timestep_rec, cfg)
    return hooks


def training_config_kwargs(cfg: dict, *, checkpoint_root, tensor_dir, output_dir, device: str, precision: str,
                           seed: int, resume_from=None) -> dict:
    """ACE-Step's TrainingConfigV2 fields from a Khaos recipe, every one explicit so the recipe records what runs."""
    kwargs = {
        "weight_decay": float(cfg.get("weight_decay", 0.01)),
        "adapter_type": cfg["adapter_type"],
        "model_variant": cfg["model_variant"],
        "checkpoint_dir": str(checkpoint_root),
        "dataset_dir": str(tensor_dir),
        "output_dir": str(output_dir),
        "learning_rate": float(cfg["learning_rate"]),
        "batch_size": int(cfg["batch_size"]),
        "max_epochs": int(cfg["max_epochs"]),
        "seed": int(seed),
        "precision": precision,
        "device": device,
        "gradient_checkpointing": bool(cfg["gradient_checkpointing"]),
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": bool(cfg["pin_memory"]),
        "persistent_workers": bool(cfg["persistent_workers"]),
        "prefetch_factor": int(cfg["prefetch_factor"]),
        "save_every_n_epochs": int(cfg["checkpoint_every_n_epochs"]),
        "scheduler_type": cfg["lr_schedule"],
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
    }
    if resume_from is not None:
        kwargs["resume_from"] = str(resume_from)
    return kwargs


class Session:
    """One training run on ACE-Step's trainer: the live module, the event stream, the held-out evaluator on the same
    module, and the final export. Checkpoints land under ``<output_dir>/checkpoints/epoch_<n>_loss_<x>/``."""

    def __init__(self, api: TrainingApi, trainer, *, cfg: dict, output_dir: Path, seed: int) -> None:
        self.api = api
        self.trainer = trainer
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.seed = int(seed)

    @property
    def module(self):
        return getattr(self.trainer, "module", None)

    @property
    def training_config(self):
        return getattr(self.trainer, "training_config", None)

    def install_hooks(self, *, start_epoch: int = 0, stem_ids=(), record_into: dict | None = None) -> LossHooks:
        return install_loss_hooks(self.cfg, self.seed, start_epoch=start_epoch, stem_ids=stem_ids,
                                  training_config=self.training_config, record_into=record_into)

    def train(self):
        """The trainer's own updates, each with step, loss, msg, kind, epoch and, on a checkpoint, its path."""
        for update in self.trainer.train():
            yield TrainingUpdate(step=int(getattr(update, "step", 0) or 0),
                                 loss=float(getattr(update, "loss", 0.0) or 0.0),
                                 msg=str(getattr(update, "msg", "") or ""),
                                 kind=str(getattr(update, "kind", "progress") or "progress"),
                                 epoch=int(getattr(update, "epoch", 0) or 0),
                                 lr=float(getattr(update, "lr", 0.0) or 0.0),
                                 epoch_time=float(getattr(update, "epoch_time", 0.0) or 0.0),
                                 samples_per_sec=float(getattr(update, "samples_per_sec", 0.0) or 0.0),
                                 checkpoint_path=(str(getattr(update, "checkpoint_path", "") or "") or None))

    def heldout_evaluator(self, tensor_val_dir: Path | str, *, n_buckets: int):
        from khaos_attribution.engines.acestep.heldout import HeldoutEvaluator  # noqa: PLC0415
        return HeldoutEvaluator(self.module, tensor_val_dir, seed=self.seed, n_buckets=int(n_buckets))

    def save_final(self, final_dir: Path | str) -> Path:
        """The adapter exported flat into ``final_dir``; the file it wrote, by the descriptor's names."""
        final_dir = Path(final_dir)
        self.api.save_final(self.trainer, str(final_dir))
        exported = adapter_file_in(final_dir) if final_dir.is_dir() else None
        if exported is None:
            raise RuntimeError(f"save_final wrote no adapter file into {final_dir}")
        return exported


def open_session(api: TrainingApi, cfg: dict, *, checkpoint_root, tensor_dir, output_dir, device: str,
                 precision: str, seed: int, resume_from=None, on_model_loaded=None) -> Session:
    """The decoder loaded, the adapter and training configs built from the recipe, the trainer made: a Session.
    A resume on a revision without resume_from is refused rather than silently started fresh."""
    model = api.load_decoder(str(checkpoint_root), cfg["model_variant"], device, precision)
    if on_model_loaded is not None:
        on_model_loaded(model)
    adapter_type, adapter_cfg = adapter_config(api, cfg, keep_none=False)
    kwargs = training_config_kwargs(cfg, checkpoint_root=checkpoint_root, tensor_dir=tensor_dir, output_dir=output_dir,
                                    device=device, precision=precision, seed=seed, resume_from=resume_from)
    try:
        accepted = set(inspect.signature(api.TrainingConfig).parameters)
    except (TypeError, ValueError):
        accepted = set(kwargs)
    if resume_from is not None and "resume_from" not in accepted:
        raise RuntimeError("This ACE-Step checkout has no resume_from support — cannot resume; update ACE-Step or "
                           "start a fresh run.")
    train_cfg = accepted_config(api.TrainingConfig, kwargs)
    log.info("Adapter config: type=%s rank=%d alpha=%d lr=%g epochs=%d", adapter_type, int(cfg["adapter_rank"]),
             int(cfg["adapter_alpha"]), float(cfg["learning_rate"]), int(cfg["max_epochs"]))
    trainer = api.Trainer(model, adapter_cfg, train_cfg)
    return Session(api, trainer, cfg=cfg, output_dir=output_dir, seed=seed)
