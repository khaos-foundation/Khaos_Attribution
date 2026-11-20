"""The canonical JSON forms the contract hashes and signs, in one place. Each is byte-for-byte the form its callers
wrote before; a hash already published over one never moves.

- compact: sorted keys, no spaces, UTF-8 as written (a register record, a rights stamp, an evidence review).
- compact_ascii: the same with every non-ASCII character escaped (a release bundle's id, the seal's record digest).
- jcs: RFC 8785, the log's entries.
"""

from __future__ import annotations

import json


def compact(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compact_ascii(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def jcs(doc) -> bytes:
    """RFC 8785. It does not normalise unicode: callers send strings already in NFC."""
    import rfc8785  # noqa: PLC0415
    return rfc8785.dumps(doc)
