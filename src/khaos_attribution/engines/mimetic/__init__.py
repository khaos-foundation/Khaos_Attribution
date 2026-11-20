"""mimetic-a0 as a Khaos engine: the descriptor is data (engine.json); the slots drive the engine's checkout and
stable-audio-tools in the engine's own environment only when called."""

from __future__ import annotations

from khaos_attribution.engines import slots
from khaos_attribution.engines.descriptor import packaged_descriptor

DESCRIPTOR = packaged_descriptor(__name__)
Engine = slots(__name__, DESCRIPTOR)
