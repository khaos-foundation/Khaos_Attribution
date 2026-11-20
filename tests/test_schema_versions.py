"""A change to what a schema accepts moves schema_version: a closed schema read by an older room refuses a record with a
field, a value or a shape it does not know, so the version is how that room says why. Each schema's fields and shape
(every rule but its words) are pinned here beside the version that holds them; a change of either needs a new, higher
version and its own entry in tests/golden/schema_fields.json (scripts/goldens.py schema-pin writes it)."""
import hashlib
import json
from pathlib import Path

import pytest

from khaos_attribution import load_schema
from khaos_attribution.validation import known_versions

PINS = Path(__file__).parent / "golden" / "schema_fields.json"
SCHEMAS = ("provenance_record", "model_card", "track_rights", "attribution_estimate", "tombstone", "catalogue_delivery", "evidence_review")


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


WORDS = ("description", "title", "$comment", "examples")


def shape(schema) -> str:
    """The schema's rules without its words or its own schema_version values: what a record must be to pass."""
    def strip(node, top=False):
        if isinstance(node, dict):
            out = {k: strip(v) for k, v in node.items() if k not in WORDS}
            if top and isinstance(out.get("properties"), dict) and isinstance(out["properties"].get("schema_version"), dict):
                out["properties"]["schema_version"] = {k: v for k, v in out["properties"]["schema_version"].items()
                                                       if k not in ("const", "enum")}
            return out
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node
    return hashlib.sha256(json.dumps(strip(schema, top=True), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _version(text: str) -> tuple:
    return tuple(int(part) for part in text.split("."))


@pytest.mark.parametrize("name", SCHEMAS)
def test_a_schemas_fields_are_the_fields_pinned_for_its_newest_version(name):
    schema = load_schema(f"{name}.schema.json")
    history = json.loads(PINS.read_text(encoding="utf-8"))[name]
    newest = max(known_versions(schema), key=_version)
    assert history[-1] == {"version": newest, "fields": fields(schema), "shape": shape(schema)}, (
        f"{name}'s fields, rules or version changed: a change needs a new schema_version, higher than {history[-1]['version']}, "
        "and its own entry at the end of tests/golden/schema_fields.json")


@pytest.mark.parametrize("name", SCHEMAS)
def test_every_pinned_change_of_fields_came_with_a_higher_version(name):
    history = json.loads(PINS.read_text(encoding="utf-8"))[name]
    for earlier, later in zip(history, history[1:]):
        assert _version(later["version"]) > _version(earlier["version"]), f"{name}: {later['version']} is not above {earlier['version']}"
        assert later["fields"] != earlier["fields"] or later.get("shape") != earlier.get("shape")
        assert set(earlier["fields"]) <= set(later["fields"]), f"{name}: a field left the schema, which an older record still carries"
    assert set(known_versions(load_schema(f"{name}.schema.json"))) >= {history[-1]["version"]}


def test_the_field_reader_sees_nested_objects_and_list_items():
    schema = {"properties": {"a": {"type": "object", "properties": {"b": {}}},
                             "c": {"type": "array", "items": {"properties": {"d": {}}}}}}
    assert fields(schema) == ["a", "a.b", "c", "c[].d"]
    assert "training_catalogue[].title" in fields(load_schema("model_card.schema.json"))


def test_a_rule_changed_under_the_same_version_is_caught():
    """A widened enum is what the field list alone never saw."""
    schema = load_schema("track_rights.schema.json")
    wider = json.loads(json.dumps(schema))
    wider["properties"]["source"]["enum"].append("a_new_source")
    assert fields(wider) == fields(schema) and shape(wider) != shape(schema)
    reworded = json.loads(json.dumps(schema))
    reworded["properties"]["source"]["description"] = "other words"
    assert shape(reworded) == shape(schema), "words are not rules"


def test_a_ddex_record_alone_takes_the_version_that_added_ddex():
    from khaos_attribution import TRACK_RIGHTS_SCHEMA_VERSION, track_rights_schema_version
    assert track_rights_schema_version({"source": "ddex_ern"}) == "1.1.0"
    assert track_rights_schema_version({"source": "operator-entered"}) == TRACK_RIGHTS_SCHEMA_VERSION == "1.0.0"
    assert set(known_versions(load_schema("track_rights.schema.json"))) == {"1.0.0", "1.1.0"}
