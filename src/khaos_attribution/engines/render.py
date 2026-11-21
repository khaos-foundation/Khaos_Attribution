"""The render slot in Khaos's words: one request, one result, two refusals. Every engine's renderer takes a
RenderRequest and returns a RenderResult; the engine-dialect lives behind it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


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


@runtime_checkable
class Renderer(Protocol):
    """What every engine's render slot offers the rooms as its ``Renderer``. Progress is ``on_progress(done,
    total=1.0)``: the fraction done is ``done / total``, never falling, ending at 1; a renderer may report a bare
    fraction. ``forget_adapter()`` with no name forgets every adapter the renderer holds; an engine's own options
    (ACE-Step's base checkpoint, a resident base) arrive as keywords another engine ignores."""

    def __init__(self, repo_dir=None, checkpoint_dir=None, device: str = "auto",
                 default_adapter_scale: float | None = None, **engine_options): ...

    def load(self, variant: str | None = None) -> None: ...

    def resolve_variant(self, requested: str | None) -> tuple: ...

    def apply_adapter(self, path, scale: float | None, variant: str | None = None) -> tuple: ...

    def forget_adapter(self, name: str | None = None) -> None: ...

    def release_base(self) -> None: ...

    def yield_memory(self) -> None: ...

    def generate(self, req: RenderRequest, adapter_path=None, out_path=None, on_progress=None, should_stop=None,
                 release_base: bool = False) -> RenderResult: ...


RENDERER_METHODS = ("load", "resolve_variant", "apply_adapter", "forget_adapter", "release_base", "yield_memory",
                    "generate")
# How the rooms call each method, by position and keyword: a renderer must take every one of these calls.
RENDERER_CALLS = {
    "__init__": [((), {"repo_dir": None, "checkpoint_dir": None, "device": "auto", "default_adapter_scale": None}),
                 ((), {"repo_dir": None, "base_checkpoint": None, "base_resident": False})],
    "load": [((), {}), (("v",), {})],
    "resolve_variant": [(("v",), {})],
    "apply_adapter": [(("path", 1.0, "v"), {}), (("path", None), {})],
    "forget_adapter": [((), {}), (("name",), {})],
    "release_base": [((), {})],
    "yield_memory": [((), {})],
    "generate": [(("req", None, "out"), {"on_progress": None, "should_stop": None}),
                 (("req", None, "out"), {"on_progress": None, "should_stop": None, "release_base": True})],
}
