"""The stand-in engine every test and the checklist itself run against: it makes audio from nothing and takes
seconds, and it passes the same checklist a real engine must."""

from __future__ import annotations

from khaos_attribution.engines import slots
from khaos_attribution.engines.descriptor import packaged_descriptor

DESCRIPTOR = packaged_descriptor(__name__)
Engine = slots(__name__, DESCRIPTOR)
