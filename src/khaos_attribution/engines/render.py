"""The render slot in Khaos's words: one request, one result, two refusals. Every engine's renderer takes a
RenderRequest and returns a RenderResult; the engine-dialect lives behind it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class RenderCancelled(Exception):
    """The caller's should_stop said stop between diffusion steps; nothing is kept."""


class RenderFailed(RuntimeError):
    """The engine made no audio; the sentence says what it reported."""


@dataclass
class RenderRequest:
    """One render in Khaos's words; the renderer speaks the engine's."""
    prompt: str
    duration_seconds: float
    lyrics: str | None = None
    instrumental: bool = True
    steps: int | None = None
    seed: int | None = None
    variant: str | None = None
    shift: float | None = None
    guidance_scale: float | None = None
    cfg_interval_start: float | None = None
    cfg_interval_end: float | None = None
    dcw_enabled: bool | None = None
    bpm: int | None = None
    keyscale: str | None = None
    timesignature: str | None = None
    vocal_language: str | None = None
    thinking: bool = False
    negative_prompt: str | None = None
    fade_in_seconds: float | None = None
    fade_out_seconds: float | None = None
    normalize: bool = False
    mode: str = "text2music"           # text2music | retake | repaint | extend | audio2audio
    source_audio: Path | None = None
    repaint_start: float | None = None
    repaint_end: float | None = None
    repaint_regions: list | None = None   # several [start, end] parts fixed in one pass, where the engine can
    repaint_mode: str | None = None
    extend_head: float | None = None
    retake_variance: float | None = None
    retake_seed: int | None = None
    cover_strength: float | None = None   # how closely audio2audio follows the source: higher follows more
    adapter_scale: float | None = None
    audio_format: str | None = None   # the engine's own default (its highest fidelity) unless a caller names one
    extra: dict = field(default_factory=dict)   # engine-dialect keys a caller already resolved


@dataclass
class RenderResult:
    audio_path: Path
    seed: int
    variant: str
    params: dict          # the engine's own record of the render, plus the facts Khaos pins on it
    applied_scale: float | None
    scale_applied_via: str | None
    dropped: list


__all__ = ["RenderCancelled", "RenderFailed", "RenderRequest", "RenderResult"]
