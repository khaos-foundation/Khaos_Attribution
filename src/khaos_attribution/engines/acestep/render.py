"""ACE-Step 1.5's render slot: one renderer for both rooms. Loads the serving weights once (base on first use, never
downloaded), attaches and scales an adapter, and renders one request in the engine's own dialect, with the rework
modes it offers. Everything Khaos adds to a render (the watermark, the signed record, loudness, keepers) stays outside.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import logging
import os
import random
from pathlib import Path

from khaos_attribution.engines.acestep import DESCRIPTOR

from khaos_attribution.engines.render import RenderCancelled, RenderFailed, RenderRequest, RenderResult  # noqa: E402,F401

log = logging.getLogger(__name__)

# ACE-Step's own weight-file names: config.json alone is a directory initialize_service would "complete" by downloading.
WEIGHT_FILENAMES = ("model.safetensors", "model.safetensors.index.json",
                    "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.safetensors.index.json")
# Travels to the UI, the job result and the sidecar, so never an absolute path; the path goes to the log, once.
BASE_MISSING_NOTE = ("Base checkpoint not installed — rendered on turbo instead (set generation.base_checkpoint, or place "
                     "acestep-v15-base under the checkpoints directory; the server log names the path)")
CHECKPOINTS_ENV = "ACESTEP_CHECKPOINTS_DIR"   # the engine's own name for where its weights are


def flush_accelerator() -> None:
    """Collect and hand the accelerator's cached memory back; never worth a failed render."""
    import gc  # noqa: PLC0415
    gc.collect()
    try:
        import torch  # noqa: PLC0415
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache") and torch.backends.mps.is_available():
            mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        log.debug("accelerator cache flush skipped", exc_info=True)


def force_mps_bf16(handler_cls) -> None:
    """Run the DiT in bf16 on Apple silicon: ACE-Step hardcodes float32 on MPS and exposes no dtype hook, and
    `self.dtype` is assigned inline mid-method, so the class gets a property that answers bf16 on MPS. Idempotent;
    CUDA and CPU untouched."""
    import torch  # noqa: PLC0415
    if getattr(handler_cls, "_khaos_mps_bf16_installed", False):
        return

    def _get_dtype(self):
        if str(getattr(self, "device", "")).startswith("mps"):
            return torch.bfloat16
        return getattr(self, "_khaos_dtype", torch.float32)

    def _set_dtype(self, value):
        self._khaos_dtype = value

    handler_cls.dtype = property(_get_dtype, _set_dtype)
    handler_cls._khaos_mps_bf16_installed = True


def force_codec_fp32(handler_cls) -> None:
    """The codec (VAE) runs in fp32 on every device: in bf16 or fp16 its decoder adds a high-frequency hash that
    quiet material exposes. The engine's own rule picks half precision on CUDA and MPS and every codec call site
    consults it, so the rule itself is replaced; the DiT keeps its precision. Idempotent."""
    import torch  # noqa: PLC0415
    if getattr(handler_cls, "_khaos_codec_fp32_installed", False):
        return

    def _get_vae_dtype(self, device=None):
        return torch.float32

    handler_cls._get_vae_dtype = _get_vae_dtype
    handler_cls._khaos_codec_fp32_installed = True


def engine_device(device: str) -> str:
    """ACE-Step wants an indexed CUDA device ("cuda:0") and a bare string for mps/cpu."""
    return "cuda:0" if device == "cuda" else device


def accepted_params(params_cls, kwargs: dict) -> tuple:
    """(the kwargs this engine's GenerationParams takes, the keys it does not): by its fields when it is a dataclass,
    else by trying each key, so an engine revision that renamed a field warns instead of crashing a render."""
    if dataclasses.is_dataclass(params_cls):
        known = {f.name for f in dataclasses.fields(params_cls)}
        return {k: v for k, v in kwargs.items() if k in known}, sorted(set(kwargs) - known)
    accepted, dropped = {}, []
    for k, v in kwargs.items():
        try:
            params_cls(**{**accepted, k: v})
            accepted[k] = v
        except TypeError:
            dropped.append(k)
    return accepted, sorted(dropped)


def lora_call_failed(result) -> bool:
    """ACE-Step's LoRA controls report failure in the RETURN STRING, not an exception."""
    s = str(result or "").strip()
    return s.startswith(("⚠️", "❌")) or "❌" in s or s.lower().startswith("error")


@contextlib.contextmanager
def _hub_offline(enabled: bool):
    """huggingface_hub reads HF_HUB_OFFLINE once at import, so the constant itself is flipped around a base init
    (config.json alone would start a multi-GB download) and restored after."""
    if not enabled:
        yield
        return
    from huggingface_hub import constants as hf_constants  # noqa: PLC0415
    previous, env_previous = hf_constants.HF_HUB_OFFLINE, os.environ.get("HF_HUB_OFFLINE")
    hf_constants.HF_HUB_OFFLINE = True
    os.environ["HF_HUB_OFFLINE"] = "1"   # the constant for this process, the variable for anything it starts
    try:
        yield
    finally:
        hf_constants.HF_HUB_OFFLINE = previous
        if env_previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = env_previous


class AceStepRenderer:
    """The weights once per process, an adapter per variant, one render at a time (the caller serialises)."""

    def __init__(self, repo_dir, checkpoint_dir=None, device: str = "auto", base_checkpoint=None,
                 base_resident: bool = False, default_adapter_scale: float = 0.85):
        self.repo_dir = Path(repo_dir).resolve()
        self.checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        self.device = device
        self.base_resident = bool(base_resident)
        self.default_adapter_scale = float(default_adapter_scale)
        self._base_checkpoint = Path(base_checkpoint).resolve() if base_checkpoint else None
        self.handler = None                  # the serving (turbo) handler
        self.llm = None                      # the 5 Hz LM, loaded only when a request thinks
        self._handlers: dict = {}
        self._loaded: dict = {v: set() for v in DESCRIPTOR.variants}
        self._active: dict = {v: None for v in DESCRIPTOR.variants}
        self._missing_logged = False
        self._yielded = False

    # ── weights ──────────────────────────────────────────────────────────────
    def checkpoints_root(self) -> Path:
        return self.checkpoint_dir or (self.repo_dir / "checkpoints")

    def base_checkpoint_path(self) -> Path:
        return self._base_checkpoint or (self.checkpoints_root() / DESCRIPTOR.variant_config("base"))

    def base_available(self) -> bool:
        path = self.base_checkpoint_path()
        return (path / "config.json").is_file() and any((path / n).is_file() for n in WEIGHT_FILENAMES)

    def resolve_variant(self, requested: str | None) -> tuple:
        """(variant that will render, note): base falls back to turbo and says so when its weights are absent."""
        wanted = requested or DESCRIPTOR.default_variant
        if wanted == DESCRIPTOR.default_variant or wanted in self._handlers:
            return wanted, None
        if wanted not in DESCRIPTOR.variants:
            return DESCRIPTOR.default_variant, f"Unknown engine variant {wanted!r} — rendered on turbo"
        if self.base_available():
            return wanted, None
        if not self._missing_logged:
            self._missing_logged = True
            log.warning("Base checkpoint not installed at %s — base renders go to turbo", self.base_checkpoint_path())
        return DESCRIPTOR.default_variant, BASE_MISSING_NOTE

    def _init_kwargs(self, config_path: str) -> dict:
        from acestep.handler import AceStepHandler  # noqa: PLC0415
        kwargs = dict(project_root=str(self.repo_dir), config_path=config_path, device=engine_device(self._resolved_device()),
                      use_flash_attention=False, compile_model=False, offload_to_cpu=False, offload_dit_to_cpu=False,
                      quantization=None)
        # MLX DiT must be OFF for adapter inference: it snapshots the torch weights before the adapter attaches.
        try:
            params = inspect.signature(AceStepHandler.initialize_service).parameters
        except (TypeError, ValueError):
            params = {}
        takes_any = not params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        known = {k: v for k, v in kwargs.items() if takes_any or k in params}
        if "use_mlx_dit" in params:
            known["use_mlx_dit"] = False
        return known

    def _resolved_device(self) -> str:
        if self.device and self.device != "auto":
            return str(self.device)
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self, variant: str | None = None) -> None:
        """The serving weights, once: the default variant, or the one named (a one-render process loads only the
        variant it renders on; base loads with the hub offline so config.json alone cannot start a download)."""
        if not self.repo_dir.exists():
            raise RuntimeError(f"ACE-Step not found at {self.repo_dir}")
        if self.checkpoint_dir:
            os.environ[CHECKPOINTS_ENV] = str(self.checkpoint_dir)
        variant = variant or DESCRIPTOR.default_variant
        if variant not in DESCRIPTOR.variants:
            raise RuntimeError(f"ACE-Step has no variant {variant!r}")
        from acestep.handler import AceStepHandler  # noqa: PLC0415
        if self._resolved_device() == "mps":
            force_mps_bf16(AceStepHandler)
        force_codec_fp32(AceStepHandler)
        handler = AceStepHandler()
        config_path = DESCRIPTOR.variant_config(variant)
        if variant == "base":
            path = self.base_checkpoint_path()
            config_path = path.name if path.parent == self.checkpoints_root().resolve() else str(path)
        with _hub_offline(variant == "base"):
            status, ok = handler.initialize_service(**self._init_kwargs(config_path))
        if not ok:
            raise RuntimeError(f"ACE-Step failed to initialise: {status}")
        self.handler = handler
        self._handlers[variant] = handler
        self._yielded = False
        log.info("ACE-Step %s model loaded: %s", variant, status)

    def _handler_for(self, variant: str):
        if variant in self._handlers:
            return self._handlers[variant]
        if variant != "base":
            raise RuntimeError(f"Engine variant {variant!r} is not loaded")
        if not self.base_available():
            log.error("Base checkpoint weights missing at %s", self.base_checkpoint_path())
            raise RuntimeError("Base checkpoint not installed (the server log names the path)")
        from acestep.handler import AceStepHandler  # noqa: PLC0415
        if self._resolved_device() == "mps":
            force_mps_bf16(AceStepHandler)
        force_codec_fp32(AceStepHandler)
        path = self.base_checkpoint_path()
        config_path = path.name if path.parent == self.checkpoints_root().resolve() else str(path)
        handler = AceStepHandler()
        with _hub_offline(True):
            status, ok = handler.initialize_service(**self._init_kwargs(config_path))
        if not ok:
            raise RuntimeError(f"ACE-Step base checkpoint failed to initialise: {status}")
        if handler.is_turbo_model():
            raise RuntimeError(f"{path.name} declares is_turbo — not a base checkpoint")
        self._handlers["base"] = handler
        log.info("ACE-Step BASE model loaded: %s", status)
        return handler

    def release_base(self) -> None:
        if self.base_resident or "base" not in self._handlers:
            return
        self._handlers.pop("base", None)
        self._loaded["base"] = set()
        self._active["base"] = None
        flush_accelerator()

    def yield_memory(self) -> None:
        """Every model dropped for another process; the next render loads the serving model again."""
        if self.handler is None and not self._handlers:
            return
        self._handlers.clear()
        self.handler = None
        self.llm = None
        self._loaded = {v: set() for v in DESCRIPTOR.variants}
        self._active = {v: None for v in DESCRIPTOR.variants}
        self._yielded = True
        flush_accelerator()

    def forget_adapter(self, name: str) -> None:
        for variant, active in self._active.items():
            if active == name:
                self._active[variant] = None

    # ── the adapter ──────────────────────────────────────────────────────────
    @staticmethod
    def descriptor_adapter_filenames() -> tuple:
        return tuple(DESCRIPTOR.adapter["filenames"])

    @staticmethod
    def _check(status: str, doing: str) -> None:
        stripped = str(status).lstrip()
        if stripped.startswith("❌"):
            raise RuntimeError(f"{doing}: {status}")
        if stripped.startswith("⚠️"):
            log.warning("%s: %s", doing, status)

    def apply_adapter(self, adapter_path, scale: float | None, variant: str) -> tuple:
        """Attach the adapter to the variant's weights (unload then load on a change: a LoKR patches the decoder in
        place) and set its strength; returns (applied, via). The LyCORIS multiplier takes the number whole; without it
        ACE-Step's control clamps to [0, 1]."""
        adapter_path = Path(adapter_path)
        handler = self._handler_for(variant)
        loaded, name = self._loaded[variant], adapter_path.parent.name
        if self._active[variant] != name:
            if loaded:
                self._check(handler.unload_lora(), "Unloading the previous adapter")
                loaded.clear()
                self._active[variant] = None
            self._check(handler.add_lora(str(adapter_path), adapter_name=name), f"Loading adapter {name!r}")
            loaded.add(name)
            self._active[variant] = name
        self._check(handler.set_use_lora(True), "Enabling adapter")
        scale = float(self.default_adapter_scale if scale is None else scale)
        self._check(handler.set_lora_scale(name, scale), f"Setting adapter strength {scale}")
        net = getattr(getattr(getattr(handler, "model", None), "decoder", None), "_lycoris_net", None)
        set_multiplier = getattr(net, "set_multiplier", None)
        if callable(set_multiplier):
            set_multiplier(scale)
            return scale, "lycoris_multiplier"
        applied = max(0.0, min(1.0, scale))
        if applied != scale:
            log.warning("No LyCORIS net on %s: adapter strength %s was clamped to %s", variant, scale, applied)
        return applied, "set_lora_scale"

    def text_embedder(self):
        """The loaded text encoder as texts -> (n, dim), or None before load."""
        handler = self.handler
        model = getattr(handler, "text_encoder", None)
        tokenizer = getattr(handler, "text_tokenizer", None)
        if model is None or tokenizer is None:
            return None
        from khaos_attribution.prompt_expansion import embed_texts  # noqa: PLC0415
        return lambda texts: embed_texts(model, tokenizer, list(texts))

    # ── the render ───────────────────────────────────────────────────────────
    def engine_kwargs(self, req: RenderRequest, variant: str, out_dir: Path) -> tuple:
        """The engine's GenerationParams kwargs for a request: (kwargs, temp files to remove after)."""
        row = DESCRIPTOR.render_settings(variant, steps=req.steps, shift=req.shift, guidance_scale=req.guidance_scale,
                                         cfg_interval_start=req.cfg_interval_start, cfg_interval_end=req.cfg_interval_end,
                                         dcw_enabled=req.dcw_enabled)
        kwargs = dict(caption=req.prompt, instrumental=bool(req.instrumental), duration=float(req.duration_seconds),
                      inference_steps=int(row["steps"]), shift=float(row["shift"]), thinking=bool(req.thinking),
                      dcw_enabled=bool(row["dcw_enabled"]), guidance_scale=float(row["guidance_scale"]),
                      cfg_interval_start=float(row["cfg_interval_start"]), cfg_interval_end=float(row["cfg_interval_end"]))
        if req.lyrics is not None:
            kwargs["lyrics"] = req.lyrics
        if req.bpm is not None:
            kwargs["bpm"] = req.bpm
        if req.keyscale:
            kwargs["keyscale"] = req.keyscale
        if req.timesignature:
            kwargs["timesignature"] = req.timesignature
        # Passed whenever given; whether an instrumental carries one is each room's rule, made before this call.
        if req.vocal_language:
            kwargs["vocal_language"] = str(req.vocal_language)
        if req.negative_prompt and req.thinking:
            kwargs["lm_negative_prompt"] = str(req.negative_prompt)
        if req.fade_in_seconds:
            kwargs["fade_in_duration"] = float(req.fade_in_seconds)
        if req.fade_out_seconds:
            kwargs["fade_out_duration"] = float(req.fade_out_seconds)
        if req.normalize:
            kwargs["enable_normalization"] = True
        temps = []
        mode = req.mode or "text2music"
        if mode == "retake":
            kwargs["task_type"] = "text2music"
            kwargs["retake_variance"] = req.retake_variance if req.retake_variance is not None else 0.3
            kwargs["retake_seed"] = req.retake_seed if req.retake_seed is not None else random.randint(0, 2**31 - 1)
        elif mode == "audio2audio":
            kwargs["task_type"] = "cover"
            kwargs["src_audio"] = str(req.source_audio)
            kwargs["audio_cover_strength"] = req.cover_strength if req.cover_strength is not None else 0.5
        elif mode in ("repaint", "extend"):
            kwargs["task_type"] = "repaint"
            src = Path(req.source_audio)
            if req.extend_head:
                # The engine only pads tails: prepend silence to a temp copy and repaint the silent window.
                import numpy as np  # noqa: PLC0415
                import soundfile as sf  # noqa: PLC0415
                audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
                head = int(round(float(req.extend_head) * sr))
                padded = np.concatenate([np.zeros((head, audio.shape[1]), dtype="float32"), audio])
                pad_dir = Path(out_dir) / ".head_src"
                pad_dir.mkdir(parents=True, exist_ok=True)
                src = pad_dir / "head_padded_source.wav"
                sf.write(str(src), padded, sr)
                temps.append(src)
            kwargs["src_audio"] = str(src)
            kwargs["repainting_start"] = float(req.repaint_start or 0.0)
            kwargs["repainting_end"] = float(req.repaint_end if req.repaint_end is not None else -1)
            kwargs["repaint_mode"] = req.repaint_mode or "balanced"
        elif mode != "text2music":
            kwargs["task_type"] = mode
            if req.source_audio is not None:
                kwargs["src_audio"] = str(req.source_audio)
        else:
            kwargs["task_type"] = "text2music"
        kwargs.update(req.extra or {})
        return kwargs, temps

    @staticmethod
    def _cancel_hooks(should_stop, handler) -> list:
        """Forward pre-hooks that abort the render between diffusion steps: the sampler has no stop flag."""
        model = getattr(handler, "model", None)
        decoder = getattr(model, "decoder", None)
        if decoder is not None and hasattr(decoder, "register_forward_pre_hook"):
            targets = [decoder]
        elif model is not None and hasattr(model, "children"):
            targets = [m for m in model.children() if hasattr(m, "register_forward_pre_hook")]
        else:
            targets = []

        def _abort(_module, _args):
            if should_stop():
                raise RenderCancelled("cancelled")
        return [t.register_forward_pre_hook(_abort) for t in targets]

    def generate(self, req: RenderRequest, adapter_path, out_path, on_progress=None, should_stop=None,
                 release_base: bool = False) -> RenderResult:
        """One render to *out_path* (its suffix follows the engine's file). The caller holds any lock."""
        if self.handler is None:
            self.load()
        from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: PLC0415
        variant = req.variant or DESCRIPTOR.default_variant
        handler = self._handler_for(variant)
        applied = None
        if adapter_path is not None:
            applied = self.apply_adapter(adapter_path, req.adapter_scale, variant)
        seed = int(req.seed) if req.seed is not None else random.randint(0, 2**31 - 1)
        out_path = Path(out_path)
        kwargs, temps = self.engine_kwargs(req, variant, out_path.parent)
        kwargs["seed"] = seed
        accepted, dropped = accepted_params(GenerationParams, kwargs)
        if dropped:
            log.warning("GenerationParams does not support %s — dropped", dropped)
        params = GenerationParams(**accepted)
        config_kwargs = dict(batch_size=1, use_random_seed=False, seeds=[seed])
        if req.audio_format:
            config_kwargs["audio_format"] = req.audio_format
        config = GenerationConfig(**config_kwargs)
        extra = {}
        if on_progress is not None and "progress" in inspect.signature(generate_music).parameters:
            def progress(ratio, desc=None, **_kw):
                try:
                    on_progress(float(ratio))
                except (TypeError, ValueError):
                    pass
            extra["progress"] = progress
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if should_stop and should_stop():
            raise RenderCancelled("cancelled before the render started")
        hooks = self._cancel_hooks(should_stop, handler) if should_stop else []
        try:
            result = generate_music(handler, self.llm, params, config, save_dir=str(out_path.parent), **extra)
        finally:
            for h in hooks:
                h.remove()
            for t in temps:
                Path(t).unlink(missing_ok=True)
            if variant == "base" and release_base:
                del handler
                self.release_base()
        if should_stop and should_stop():
            if getattr(result, "success", False) and getattr(result, "audios", None):
                produced = result.audios[0].get("path")
                if produced:
                    Path(produced).unlink(missing_ok=True)
            raise RenderCancelled("cancelled")
        if not getattr(result, "success", False) or not getattr(result, "audios", None):
            raise RenderFailed(f"Generation failed: {getattr(result, 'error', None) or getattr(result, 'status_message', None)}")
        audio = result.audios[0]
        if not audio.get("path"):
            raise RenderFailed("Generation produced audio the engine could not save — check the log and free disk space")
        produced = Path(audio["path"])
        target = out_path.with_suffix(produced.suffix or ".wav")
        produced.replace(target)
        if on_progress is not None:
            on_progress(1.0)
        record = dict(audio.get("params") or {})
        record["khaos_seed"] = seed
        record["khaos_variant"] = variant
        # Every output keeps its engine's own rate and its record says which.
        record.setdefault("sample_rate", DESCRIPTOR.sample_rate)
        if applied is not None:
            record["lora_scale"] = applied[0]
            record["khaos_adapter_scale_path"] = applied[1]
        return RenderResult(audio_path=target, seed=seed, variant=variant, params=record,
                            applied_scale=applied[0] if applied else None,
                            scale_applied_via=applied[1] if applied else None, dropped=dropped)


Renderer = AceStepRenderer
