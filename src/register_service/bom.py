"""What the register runs: every distribution in its environment with its version and the licence its metadata names."""

from __future__ import annotations

from importlib import metadata


def _licence(meta) -> str:
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        return "; ".join(classifiers)
    text = (meta.get("License") or "").strip()
    return text.splitlines()[0][:200] if text else "not stated"


def bill_of_materials() -> dict:
    """One row a distribution, by normalised name; this package's own version is the code's, never stale metadata."""
    from khaos_attribution import __version__  # noqa: PLC0415
    rows = {}
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip().lower().replace("_", "-")
        if name:
            rows[name] = {"name": name, "version": __version__ if name == "khaos-attribution" else dist.version,
                          "licence": _licence(dist.metadata)}
    return {"packages": sorted(rows.values(), key=lambda r: r["name"])}
