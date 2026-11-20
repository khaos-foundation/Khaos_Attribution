"""Stable Audio 3 Medium as a Khaos engine: the descriptor is data (engine.json); the slots import Stability's code,
or drive the MLX port on Apple silicon, only when called."""

from __future__ import annotations

from khaos_attribution.engines import slots
from khaos_attribution.engines.descriptor import packaged_descriptor

DESCRIPTOR = packaged_descriptor(__name__)
Engine = slots(__name__, DESCRIPTOR)
