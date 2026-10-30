"""An engine's descriptor: what it is and can do, read as data from its package and validated against the contract's
schema. Both rooms read the same file; the Engine Room pins their copies equal."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

SCHEMA_NAME = "descriptor.schema.json"


class DescriptorError(ValueError):
    """A descriptor that does not say what an engine must say."""


@dataclass(frozen=True)
class Descriptor:
    doc: dict

    @property
    def id(self) -> str:
        return self.doc["id"]

    @property
    def name(self) -> str:
        return self.doc["name"]

    @property
    def status(self) -> str:
        """"stable", or "experimental": offered for trial and labelled so wherever the base model is named."""
        return self.doc.get("status", "stable")

    @property
    def status_note(self) -> str | None:
        return self.doc.get("status_note")

    @property
    def commit(self) -> str | None:
        return (self.doc.get("code") or {}).get("commit")

    @property
    def commit_note(self) -> str | None:
        return (self.doc.get("code") or {}).get("commit_note")

    @property
    def licence(self) -> dict:
        return self.doc["licence"]

    @property
    def variants(self) -> dict:
        return self.doc["variants"]

    @property
    def default_variant(self) -> str:
        return self.doc["default_variant"]

    @property
    def adapter_strength(self) -> float:
        """The default artist strength, one number for both rooms."""
        return float(self.doc.get("adapter_strength", 0.85))

    @property
    def render_setting_keys(self) -> tuple:
        return tuple(self.doc.get("render_setting_keys") or ())

    def render_defaults(self, variant: str | None = None) -> dict:
        """The render row for *variant*; an unknown variant resolves on the default's row."""
        v = self.variants.get(variant) or self.variants[self.default_variant]
        return dict(v["render"])

    def render_settings(self, variant: str, **overrides) -> dict:
        """Every render setting for *variant*: the caller's value where given (None means not given), the row's
        otherwise; steps 0 or None means the variant's own."""
        row = self.render_defaults(variant)
        unknown = set(overrides) - set(self.render_setting_keys)
        if unknown:
            raise TypeError(f"render_settings got unknown setting(s): {sorted(unknown)}")
        out = {}
        for k in self.render_setting_keys:
            given, default = overrides.get(k), row[k]
            if k == "steps":
                out[k] = int(given) if given else int(default)
            elif isinstance(default, bool):
                out[k] = bool(default if given is None else given)
            else:
                out[k] = float(default if given is None else given)
        return out

    def variant_config(self, variant: str) -> str | None:
        return (self.variants.get(variant) or {}).get("config")

    def step_clamp(self, variant: str) -> int | None:
        return (self.variants.get(variant) or {}).get("step_clamp")

    def bound(self, name: str) -> tuple:
        lo, hi = self.doc["bounds"][name]
        return lo, hi

    @property
    def vocal_languages(self) -> dict:
        return dict(self.doc.get("vocal_languages") or {})

    @property
    def default_vocal_language(self) -> str | None:
        return self.doc.get("default_vocal_language")

    @property
    def sings(self) -> bool:
        return bool(self.doc["sings"])

    @property
    def conditioning(self) -> dict:
        return dict(self.doc.get("conditioning") or {})

    @property
    def rework_modes(self) -> tuple:
        return tuple(self.doc.get("rework_modes") or ())

    @property
    def weights(self) -> list:
        return list(self.doc["weights"])

    @property
    def adapter(self) -> dict:
        return dict(self.doc["adapter"])

    @property
    def platforms(self) -> dict:
        return {k: tuple(v) for k, v in self.doc["platforms"].items()}

    def supports(self, platform: str, slot: str) -> bool:
        return slot in self.platforms.get(platform, ())

    @property
    def sample_rate(self) -> int:
        return int(self.doc["sample_rate"])

    @property
    def feature_version(self) -> str:
        return self.doc["measure"]["feature_version"]

    @property
    def train(self) -> dict:
        """The engine's own training defaults for a recipe; empty when the room's defaults apply."""
        return dict(self.doc.get("train") or {})

    @property
    def memory(self) -> dict:
        """The engine's footprint and its scorer's peak, in GB, as measured on it; empty when never measured."""
        return dict(self.doc.get("memory") or {})


def _schema() -> dict:
    return json.loads(resources.files("khaos_attribution.engines").joinpath(SCHEMA_NAME).read_text(encoding="utf-8"))


def validate_descriptor(doc: dict) -> dict:
    import jsonschema  # noqa: PLC0415
    try:
        jsonschema.validate(doc, _schema())
    except jsonschema.ValidationError as exc:
        raise DescriptorError(f"engine descriptor: {exc.message} (at {'/'.join(str(p) for p in exc.absolute_path) or 'top level'})") from None
    if doc["default_variant"] not in doc["variants"]:
        raise DescriptorError(f"engine descriptor: default_variant {doc['default_variant']!r} is not a variant")
    keys = doc.get("render_setting_keys") or []
    for name, v in doc["variants"].items():
        missing = [k for k in keys if k not in v["render"]]
        if missing:
            raise DescriptorError(f"engine descriptor: variant {name!r} lacks render settings {missing}")
    return doc


def load_descriptor(path: Path | str) -> Descriptor:
    return Descriptor(validate_descriptor(json.loads(Path(path).read_text(encoding="utf-8"))))


def packaged_descriptor(package: str) -> Descriptor:
    """The descriptor an engine package ships as engine.json, read without importing the package's code."""
    text = resources.files(package).joinpath("engine.json").read_text(encoding="utf-8")
    return Descriptor(validate_descriptor(json.loads(text)))
