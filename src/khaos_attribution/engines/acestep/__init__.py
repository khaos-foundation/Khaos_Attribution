"""ACE-Step 1.5 as a Khaos engine. The descriptor is data (engine.json); the slots import ACE-Step only when called."""

from __future__ import annotations

from khaos_attribution.engines import slots
from khaos_attribution.engines.descriptor import packaged_descriptor

DESCRIPTOR = packaged_descriptor(__name__)
Engine = slots(__name__, DESCRIPTOR)
