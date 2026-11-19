"""The industry's codes as a delivery carries them: each read into its stored form, or None when it is not one.

ISRC names a recording, ISWC a composition, UPC/EAN (the ICPN) a release, IPI a writer or publisher. Where a code
has a check digit it is checked; IPI is checked for form only.
"""

from __future__ import annotations

import re

_ISRC = re.compile(r"[A-Z]{2}[A-Z0-9]{3}[0-9]{7}")


def isrc(code) -> str | None:
    """An ISRC as stored (GBAYE2612345), from any of the forms it is printed in, or None."""
    text = re.sub(r"[\s\-]", "", str(code or "")).upper()
    return text if _ISRC.fullmatch(text) else None


def iswc(code) -> str | None:
    """An ISWC as stored (T-123456789-1) when its check digit holds, or None."""
    digits = re.sub(r"[\s.\-]", "", str(code or "")).upper()
    if not re.fullmatch(r"T[0-9]{10}", digits):
        return None
    body, check = digits[1:10], int(digits[10])
    total = 1 + sum(i * int(d) for i, d in enumerate(body, 1))
    return f"T-{body}-{check}" if (10 - total % 10) % 10 == check else None


def gtin(code) -> str | None:
    """A release's UPC (12 digits) or EAN (13) or GTIN-14 when its check digit holds, or None."""
    digits = re.sub(r"[\s\-]", "", str(code or ""))
    if not re.fullmatch(r"[0-9]{12,14}", digits):
        return None
    body, check = digits[:-1], int(digits[-1])
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return digits if (10 - total % 10) % 10 == check else None


def ipi(code) -> str | None:
    """An IPI name number in the form the rights record stores (9 to 11 digits), or None."""
    digits = re.sub(r"[\s.\-]", "", str(code or ""))
    return digits if re.fullmatch(r"[0-9]{9,11}", digits) else None
