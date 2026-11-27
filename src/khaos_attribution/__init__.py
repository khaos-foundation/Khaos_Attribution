"""khaos_attribution — the contract the Khaos rooms share: record schemas and validators, the watermark codec, the
attribution estimators, Content Credentials, the seal, the Guild Register's client and service, and the engine socket."""

import os as _os

# Set by engines/launch.py: a process started on one room's contract never goes on to import another copy.
_ROOM = _os.environ.get("KHAOS_CONTRACT_PACKAGE", "").strip()
_HERE = _os.path.dirname(__file__)
if _ROOM and _os.path.normcase(_os.path.realpath(_ROOM)) != _os.path.normcase(_os.path.realpath(_HERE)):
    raise ImportError(f"this process runs the room's contract at {_ROOM}, not the copy at {_HERE}: start it through "
                      "the contract's engines/launch.py")

from khaos_attribution.validation import (
    AttributionValidationError,
    WrittenByNewerContract,
    load_schema,
    validate_attribution_estimate,
    validate_catalogue_delivery,
    validate_evidence_review,
    evidence_review_key,
    validate_model_card,
    validate_provenance_record,
    validate_tombstone,
    validate_track_rights,
)

# The schema_version values writers stamp. A version is stamped only where a record uses what it added, so a room on
# an older contract still reads every other record: a track rights record from a DDEX delivery takes 1.1.0
# (track_rights_schema_version), a group-removal estimate 1.2.0 (group_removal.ESTIMATE_SCHEMA_VERSION), a dtrak
# estimate whose method has its declared shape 1.3.0; every other record the version named here.
PROVENANCE_SCHEMA_VERSION = "1.0.0"
MODEL_CARD_SCHEMA_VERSION = "1.1.0"  # writers stamp the newest; 1.0.0 still validates
TRACK_RIGHTS_SCHEMA_VERSION = "1.0.0"
CATALOGUE_DELIVERY_SCHEMA_VERSION = "1.0.0"
EVIDENCE_REVIEW_SCHEMA_VERSION = "1.0.0"
# 1.0.0 still validates; a group-removal estimate stamps its own, group_removal.ESTIMATE_SCHEMA_VERSION.
ATTRIBUTION_ESTIMATE_SCHEMA_VERSION = "1.1.0"
DTRAK_ESTIMATE_SCHEMA_VERSION = "1.3.0"


def track_rights_schema_version(record: dict) -> str:
    """The version a track rights record is stamped with: 1.1.0 when its source is ddex_ern, else 1.0.0."""
    return "1.1.0" if (record or {}).get("source") == "ddex_ern" else TRACK_RIGHTS_SCHEMA_VERSION

__all__ = [
    "ATTRIBUTION_ESTIMATE_SCHEMA_VERSION",
    "AttributionValidationError",
    "CATALOGUE_DELIVERY_SCHEMA_VERSION",
    "DTRAK_ESTIMATE_SCHEMA_VERSION",
    "EVIDENCE_REVIEW_SCHEMA_VERSION",
    "MODEL_CARD_SCHEMA_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "TRACK_RIGHTS_SCHEMA_VERSION",
    "WrittenByNewerContract",
    "load_schema",
    "track_rights_schema_version",
    "validate_attribution_estimate",
    "validate_catalogue_delivery",
    "validate_evidence_review",
    "evidence_review_key",
    "validate_model_card",
    "validate_provenance_record",
    "validate_tombstone",
    "validate_track_rights",
]

# The code's own number, never the installed metadata: an editable install's metadata goes stale on a fast-forward.
__version__ = "0.63.2"  # tests/test_version.py pins this to pyproject.toml
