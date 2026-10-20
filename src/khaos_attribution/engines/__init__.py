"""The socket every base-model engine plugs into: one registry, a descriptor per engine read as data, and the slots
(provision, train, render, measure) the rooms call. Engine packages live beside this module; their heavy imports are
lazy, so reading a descriptor never loads a model."""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

from khaos_attribution.engines.descriptor import Descriptor, DescriptorError, load_descriptor, packaged_descriptor

# Engine id -> the package that implements it. Adding an engine is one line here and one package.
KNOWN_ENGINES = {
    "acestep-1.5": "khaos_attribution.engines.acestep",
    "stable-audio-3": "khaos_attribution.engines.sa3",
    "mimetic-a0": "khaos_attribution.engines.mimetic",
    "mock": "khaos_attribution.engines.mock",
}


class EngineUnavailable(RuntimeError):
    """The engine is not installed here, or its package cannot import; the sentence says which."""


def descriptor(engine_id: str) -> Descriptor:
    """An engine's descriptor by id, without importing its code."""
    package = KNOWN_ENGINES.get(engine_id)
    if package is None:
        raise EngineUnavailable(f"no engine is known as {engine_id!r}; known: {', '.join(sorted(KNOWN_ENGINES))}")
    return packaged_descriptor(package)


def descriptors() -> dict:
    return {engine_id: descriptor(engine_id) for engine_id in KNOWN_ENGINES}


def engine(engine_id: str):
    """The engine object (its slots), importing its package; EngineUnavailable with the reason when it cannot."""
    package = KNOWN_ENGINES.get(engine_id)
    if package is None:
        raise EngineUnavailable(f"no engine is known as {engine_id!r}")
    try:
        module = importlib.import_module(package)
    except ImportError as exc:
        raise EngineUnavailable(f"engine {engine_id!r} cannot import: {exc}") from exc
    return module.Engine()


@runtime_checkable
class Engine(Protocol):
    """What every engine object carries: its descriptor, and the three slots as modules (render: a renderer class
    and request; train: the training door, configs and open_module; measure: memory plan and checkpointing)."""
    descriptor: Descriptor

    @property
    def render(self): ...

    @property
    def train(self): ...

    @property
    def measure(self): ...


# The conditioning a generation form offers as controls, in the order the forms show them.
MUSICAL_CONDITIONING = (("tempo", "tempo"), ("key", "key"), ("time_signature", "time signature"))
# Every rework mode a room offers and what the room calls doing it.
REWORK_VERBS = {"repaint": "fix a section", "extend": "make a piece longer", "retake": "take a piece again",
                "audio2audio": "take a piece somewhere new"}


def _either(words: list) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]


def capabilities(engine_id: str) -> dict:
    """What a generation form may offer on this engine, with one sentence per control the form must hide: both rooms
    read this so a missing control is explained in the same words."""
    d = descriptor(engine_id)
    c = d.conditioning
    absent = {}
    if not d.sings:
        absent["lyrics"] = f"{d.name} does not sing: every piece is instrumental, so there are no lyrics to write."
    elif not c.get("vocal_language"):
        absent["vocal_language"] = f"{d.name} picks the sung language itself."
    missing = [label for key, label in MUSICAL_CONDITIONING if not c.get(key)]
    if missing:
        absent["musical"] = f"{d.name} takes no {_either(missing)}; describe the feel in the prompt instead."
    for mode, verb in REWORK_VERBS.items():
        if mode not in d.rework_modes:
            absent[f"rework.{mode}"] = f"{d.name} cannot {verb}."
    return {
        "engine": d.id,
        "name": d.name,
        "status": d.status,
        "status_note": d.status_note,
        "sings": d.sings,
        "conditioning": c,
        # Any of tempo, key or time signature: the row the forms show them on.
        "musical": any(c.get(key) for key, _ in MUSICAL_CONDITIONING),
        "rework_modes": list(d.rework_modes),
        "bounds": {"duration": list(d.bound("duration_seconds")), "steps": list(d.bound("steps")),
                   "bpm": list(d.bound("bpm")), "takes": list(d.bound("takes"))},
        "variants": {name: {"step_clamp": d.step_clamp(name), "note": v.get("note")} for name, v in d.variants.items()},
        "default_variant": d.default_variant,
        "vocal_languages": d.vocal_languages,
        "default_vocal_language": d.default_vocal_language,
        "sample_rate": d.sample_rate,
        "absent": absent,
    }


__all__ = ["Descriptor", "DescriptorError", "Engine", "EngineUnavailable", "KNOWN_ENGINES", "MUSICAL_CONDITIONING",
           "REWORK_VERBS", "capabilities", "descriptor", "descriptors", "engine", "load_descriptor",
           "packaged_descriptor"]
