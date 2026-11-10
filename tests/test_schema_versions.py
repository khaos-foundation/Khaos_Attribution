"""A new field moves schema_version: a closed schema read by an older room refuses a record with a field it does not
know, so the version is how that room says why. Each schema's fields are pinned here beside the version that holds
them; a change of fields needs a new, higher version and its own entry in tests/golden/schema_fields.json."""
import json
from pathlib import Path

import pytest

from khaos_attribution import load_schema
from khaos_attribution.validation import known_versions

PINS = Path(__file__).parent / "golden" / "schema_fields.json"
SCHEMAS = ("provenance_record", "model_card", "track_rights", "attribution_estimate", "tombstone")


def fields(schema, path="") -> list:
    """Every property a schema names, as dotted paths; a list's items carry []."""
    out = []
    if not isinstance(schema, dict):
        return out
    for name, sub in (schema.get("properties") or {}).items():
        out.append(f"{path}{name}")
        out += fields(sub, f"{path}{name}.")
    if "items" in schema:
        out += fields(schema["items"], f"{path.rstrip('.')}[]." if path else "[].")
    for key in ("oneOf", "anyOf", "allOf"):
        for sub in schema.get(key) or []:
            out += fields(sub, path)
    for key in ("$defs", "definitions"):
        for name, sub in (schema.get(key) or {}).items():
            out += fields(sub, f"{path}#{name}.")
    return sorted(set(out))


def _version(text: str) -> tuple:
    return tuple(int(part) for part in text.split("."))


@pytest.mark.parametrize("name", SCHEMAS)
def test_a_schemas_fields_are_the_fields_pinned_for_its_newest_version(name):
    schema = load_schema(f"{name}.schema.json")
    history = json.loads(PINS.read_text(encoding="utf-8"))[name]
    newest = max(known_versions(schema), key=_version)
    assert history[-1] == {"version": newest, "fields": fields(schema)}, (
        f"{name}'s fields or version changed: a new field needs a new schema_version, higher than {history[-1]['version']}, "
        "and its own entry at the end of tests/golden/schema_fields.json")


@pytest.mark.parametrize("name", SCHEMAS)
def test_every_pinned_change_of_fields_came_with_a_higher_version(name):
    history = json.loads(PINS.read_text(encoding="utf-8"))[name]
    for earlier, later in zip(history, history[1:]):
        assert _version(later["version"]) > _version(earlier["version"]), f"{name}: {later['version']} is not above {earlier['version']}"
        assert later["fields"] != earlier["fields"]
        assert set(earlier["fields"]) <= set(later["fields"]), f"{name}: a field left the schema, which an older record still carries"
    assert set(known_versions(load_schema(f"{name}.schema.json"))) >= {history[-1]["version"]}


def test_the_field_reader_sees_nested_objects_and_list_items():
    schema = {"properties": {"a": {"type": "object", "properties": {"b": {}}},
                             "c": {"type": "array", "items": {"properties": {"d": {}}}}}}
    assert fields(schema) == ["a", "a.b", "c", "c[].d"]
    assert "training_catalogue[].title" in fields(load_schema("model_card.schema.json"))
