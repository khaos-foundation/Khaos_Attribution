"""The mock engine's render slot: a seeded tone from nothing, in seconds, with the renderer surface every engine
offers. Same request and seed, same bytes; a different seed or adapter, different audio."""

from __future__ import annotations

import hashlib
import math
import random
import struct
import wave
from pathlib import Path

from khaos_attribution.engines.mock import DESCRIPTOR
from khaos_attribution.engines.render import RenderCancelled, RenderFailed, RenderRequest, RenderResult

STEPS = 4


class Renderer:
    """The surface the rooms drive: load, resolve_variant, apply_adapter, generate; nothing heavier than wave."""

    def __init__(self, repo_dir=None, checkpoint_dir=None, device: str = "auto", base_checkpoint=None,
                 base_resident: bool = False, default_adapter_scale: float | None = None):
        self.repo_dir = repo_dir
        self.device = "cpu" if device == "auto" else device
        self.default_adapter_scale = float(DESCRIPTOR.adapter_strength if default_adapter_scale is None else default_adapter_scale)
        self.loaded: str | None = None
        self.adapter: tuple | None = None

    @property
    def descriptor(self):
        return DESCRIPTOR

    def load(self, variant: str | None = None) -> None:
        variant = variant or DESCRIPTOR.default_variant
        if variant not in DESCRIPTOR.variants:
            raise RenderFailed(f"the mock engine has no variant {variant!r}")
        self.loaded = variant

    def resolve_variant(self, requested: str | None) -> tuple:
        wanted = requested or DESCRIPTOR.default_variant
        if wanted in DESCRIPTOR.variants:
            return wanted, None
        return DESCRIPTOR.default_variant, f"variant {wanted!r} is not available on this engine; rendered on {DESCRIPTOR.default_variant}"

    def apply_adapter(self, path, scale: float | None, variant: str | None = None) -> tuple:
        """(applied scale, how): the mock reads the adapter's bytes into its seed and takes any scale."""
        scale = self.default_adapter_scale if scale is None else float(scale)
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else ""
        self.adapter = (digest, scale)
        return scale, "mock"

    def forget_adapter(self) -> None:
        self.adapter = None

    def release_base(self) -> None:
        return None

    def yield_memory(self) -> None:
        return None

    def generate(self, req: RenderRequest, adapter_path=None, out_path=None, on_progress=None, should_stop=None,
                 release_base: bool = False) -> RenderResult:
        if self.loaded is None:
            self.load(req.variant)
        variant, _ = self.resolve_variant(req.variant)
        lo, hi = DESCRIPTOR.bound("duration_seconds")
        seconds = min(max(float(req.duration_seconds), lo), hi)
        seed = int(req.seed) if req.seed is not None else random.SystemRandom().randrange(1 << 31)
        if adapter_path is not None:
            self.apply_adapter(adapter_path, req.adapter_scale, variant)
        digest, scale = self.adapter or ("", 0.0)
        key = hashlib.sha256(f"{req.prompt}\x1f{seed}\x1f{digest}\x1f{scale}\x1f{variant}".encode()).digest()
        rng = random.Random(key)
        notes = [220.0 * 2 ** (rng.randrange(0, 24) / 12) for _ in range(4)]
        sr = DESCRIPTOR.sample_rate
        n = int(seconds * sr)
        out = Path(out_path or Path.cwd() / f"mock_{seed}.wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        frames = bytearray()
        for step in range(STEPS):
            if should_stop is not None and should_stop():
                raise RenderCancelled()
            if on_progress is not None:
                on_progress(step + 1, STEPS)
            start, end = n * step // STEPS, n * (step + 1) // STEPS
            for i in range(start, end):
                t = i / sr
                f = notes[int(t * 2) % len(notes)]
                v = 0.3 * math.sin(2 * math.pi * f * t) * (0.5 + 0.5 * scale)
                s = int(max(-1.0, min(1.0, v)) * 32767)
                frames += struct.pack("<hh", s, s)
        with wave.open(str(out), "wb") as fh:
            fh.setnchannels(2)
            fh.setsampwidth(2)
            fh.setframerate(sr)
            fh.writeframes(bytes(frames))
        row = DESCRIPTOR.render_settings(variant, steps=req.steps, shift=req.shift, guidance_scale=req.guidance_scale,
                                         cfg_interval_start=req.cfg_interval_start, cfg_interval_end=req.cfg_interval_end,
                                         dcw_enabled=req.dcw_enabled)
        params = {"caption": req.prompt, "duration": seconds, "seed": seed, "variant": variant, **row,
                  "instrumental": bool(req.instrumental), "lyrics": req.lyrics}
        return RenderResult(audio_path=out, seed=seed, variant=variant, params=params,
                            applied_scale=scale if self.adapter else None,
                            scale_applied_via="mock" if self.adapter else None, dropped=[])
