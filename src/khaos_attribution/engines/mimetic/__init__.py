"""mimetic-a0 as a Khaos engine: the descriptor is data (engine.json); the slots drive Mick Grierson's checkout and
stable-audio-tools in the engine's own environment only when called."""

from __future__ import annotations

from importlib import import_module

from khaos_attribution.engines.descriptor import packaged_descriptor

DESCRIPTOR = packaged_descriptor(__name__)


class Engine:
    descriptor = DESCRIPTOR

    @property
    def render(self):
        return import_module(__name__ + ".render")

    @property
    def train(self):
        return import_module(__name__ + ".train")

    @property
    def measure(self):
        return import_module(__name__ + ".measure")
