"""Validators for the provenance record and model card schemas."""

import json
import math
import re
from importlib import resources

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import best_match


class AttributionValidationError(ValueError):
    """Raised when a record does not match its schema."""


class WrittenByNewerContract(AttributionValidationError):
    """The record's schema_version is past every version this contract reads: update the contract, not the record."""


def load_schema(name):
    """Load a bundled schema by filename, e.g. 'provenance_record.schema.json'."""
    schema_file = resources.files("khaos_attribution").joinpath("schemas", name)
    return json.loads(schema_file.read_text(encoding="utf-8"))


def known_versions(schema):
    """The schema_version values a schema reads."""
    spec = schema.get("properties", {}).get("schema_version", {})
    return [spec["const"]] if "const" in spec else list(spec.get("enum") or [])


def _version(text):
    """A schema version as numbers to compare, or None when it is not three numbers."""
    if not isinstance(text, str) or not re.fullmatch(r"\d+\.\d+\.\d+", text):
        return None
    return tuple(int(part) for part in text.split("."))


def _words(error):
    """One problem as the validator found it; a field the schema does not name, in plain words."""
    if error.validator == "additionalProperties" and isinstance(error.instance, dict):
        unknown = sorted(k for k in error.instance if k not in (error.schema.get("properties") or {}))
        return f"fields this contract does not know: {', '.join(unknown)} (written by a newer contract, or misspelt)"
    return error.message


def _validate(record, schema_name, record_kind):
    schema = load_schema(schema_name)
    theirs = _version(record.get("schema_version")) if isinstance(record, dict) else None
    newest = max(known_versions(schema), key=_version, default=None)
    # Said before anything else: a closed schema would otherwise report the newer record's fields one by one.
    if theirs and newest and theirs > _version(newest):
        raise WrittenByNewerContract(
            f"Invalid {record_kind}: it was written by a newer contract (schema_version {record['schema_version']}; "
            f"this contract reads up to {newest}). Update the contract here to read it.")
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(record), key=lambda e: list(e.absolute_path))
    if not errors:
        return record

    lines = []
    for error in errors:
        location = "/".join(str(part) for part in error.absolute_path) or "(top level)"
        lines.append(f"  - at {location}: {_words(error)}")
    raise AttributionValidationError(
        f"Invalid {record_kind}: {_words(best_match(errors))}\n"
        f"All problems:\n" + "\n".join(lines)
    )


def validate_provenance_record(record):
    """Validate a provenance record dict; raise AttributionValidationError if invalid."""
    return _validate(record, "provenance_record.schema.json", "provenance record")


def validate_model_card(record):
    """Validate a model card dict; raise AttributionValidationError if invalid."""
    return _validate(record, "model_card.schema.json", "model card")


# JSON Schema can bound each share but cannot add them up; the sum checks live here.
_SHARE_TOLERANCE = 0.01


def _check_share_sum(parties, pool, record_kind, required=True):
    for p in parties:
        v = p.get("share_pct", 0)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise AttributionValidationError(
                f"Invalid {record_kind}: a {pool[:-1]} share is not a finite number")
    total = sum(p.get("share_pct", 0) for p in parties)
    if not parties:
        if required:
            raise AttributionValidationError(
                f"Invalid {record_kind}: {pool} is empty")
        return
    if abs(total - 100.0) > _SHARE_TOLERANCE:
        raise AttributionValidationError(
            f"Invalid {record_kind}: {pool} shares sum to {total:g}, not 100")


_CODE_RULES = {
    "isrc": (r"^[A-Z]{2}[A-Z0-9]{3}[0-9]{7}$",
             "an ISRC looks like GBAYE2612345 — two-letter country, "
             "three-character registrant, seven digits, no dashes"),
    "iswc": (r"^T-[0-9]{9}-[0-9]$",
             "an ISWC is the letter T, nine digits, and a final check "
             "digit, stored as T-123456789-1 (paperwork prints the same "
             "code dotted: T-123.456.789-1)"),
}


def _check_code_fields(record, record_kind):
    """Say what is wrong with an ISRC/ISWC in words, not in regex; the
    schema enforces identical rules, these run first."""
    import re

    for key, (pattern, hint) in _CODE_RULES.items():
        field = record.get(key)
        if not isinstance(field, dict):
            continue
        status, value = field.get("status"), field.get("value")
        if status == "assigned" and not value:
            raise AttributionValidationError(
                f"Invalid {record_kind}: {key.upper()} is marked 'assigned' "
                f"but carries no code — enter the code, or set the status to "
                f"'none assigned' / 'not yet looked up'.")
        if status != "assigned" and value:
            raise AttributionValidationError(
                f"Invalid {record_kind}: {key.upper()} has a code but its "
                f"status says '{status}' — set the status to 'assigned' or "
                f"clear the code.")
        if status != "assigned" and value == "":
            # An empty string is a pattern violation to the schema.
            raise AttributionValidationError(
                f"Invalid {record_kind}: {key.upper()} carries an empty code "
                f"string — omit the value entirely when the status is "
                f"'{status}'.")
        if value and isinstance(value, str) and not re.fullmatch(pattern, value):
            raise AttributionValidationError(
                f"Invalid {record_kind}: {value!r} is not a standard "
                f"{key.upper()} — {hint}.")


def validate_tombstone(record):
    """Validate an output tombstone (watermarking v2 §6), written at
    deletion so deleted material stays identifiable; returns the record."""
    _validate(record, "tombstone.schema.json", "output tombstone")
    return record


def validate_track_rights(record):
    """Validate a track rights record; raise AttributionValidationError if invalid.

    Beyond the schema: writer shares must sum to 100, publisher shares must
    sum to 100 whenever any publisher is listed (an empty publisher list is
    legitimate — unpublished or self-released work), and ISRC/ISWC problems
    are reported in words rather than schema regex.
    """
    _check_code_fields(record, "track rights record")
    _validate(record, "track_rights.schema.json", "track rights record")
    _check_share_sum(record["writers"], "writers", "track rights record")
    _check_share_sum(record["publishers"], "publishers", "track rights record",
                     required=False)
    _check_share_sum(record.get("masters") or [], "masters",
                     "track rights record", required=False)
    return record


def validate_catalogue_delivery(record):
    """Validate a catalogue delivery; beyond the schema, every UPC/EAN and ISWC check digit holds, each release names a
    recording once, and every recording of a release that is not a takedown names its file."""
    from khaos_attribution import industry_codes as C  # noqa: PLC0415
    _validate(record, "catalogue_delivery.schema.json", "catalogue delivery")
    for release in record["releases"]:
        where = f"release {release['release_reference']}"
        if release.get("icpn") and C.gtin(release["icpn"]) is None:
            raise AttributionValidationError(f"Invalid catalogue delivery: {where}'s UPC/EAN {release['icpn']} fails its check digit")
        refs = [t["resource_reference"] for t in release["tracks"]]
        if len(refs) != len(set(refs)):
            raise AttributionValidationError(f"Invalid catalogue delivery: {where} names a recording twice")
        for t in release["tracks"]:
            if not release["takedown"] and not (t.get("file") or {}).get("path"):
                raise AttributionValidationError(
                    f"Invalid catalogue delivery: {where}, {t['resource_reference']}: a delivered recording names its file")
            if t.get("iswc") and C.iswc(t["iswc"]) is None:
                raise AttributionValidationError(
                    f"Invalid catalogue delivery: {where}, {t['resource_reference']}: ISWC {t['iswc']} fails its check digit")
    return record


def validate_evidence_review(record):
    """Validate a run's evidence review; beyond the schema, each track is named once."""
    _validate(record, "evidence_review.schema.json", "evidence review")
    ids = [t["track_id"] for t in record["tracks"]]
    if len(ids) != len(set(ids)):
        raise AttributionValidationError("Invalid evidence review: a track is named twice")
    return record


def evidence_review_key(track):
    """What an operator's approval binds to: the matches and the evidence, so new evidence needs a new approval and a
    change of names or rights does not."""
    import hashlib  # noqa: PLC0415
    body = {"track_id": track["track_id"], "state": track["state"], "matches": track["matches"], "evidence": track["evidence"]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_attribution_estimate(record):
    """Validate an attribution estimate; raise AttributionValidationError if invalid.

    Beyond the schema: blended influence shares must sum to 100, and every
    range must contain its point estimate.
    """
    _validate(record, "attribution_estimate.schema.json", "attribution estimate")
    if (record.get("method") or {}).get("kind") == "group_removal":
        ids = [g["group_id"] for g in record["groups"]]
        if len(ids) != len(set(ids)):
            raise AttributionValidationError("Invalid attribution estimate: a group appears twice")
        group_total = sum(g["share_pct"] for g in record["groups"])
        if abs(group_total - 100.0) > _SHARE_TOLERANCE:
            raise AttributionValidationError(
                f"Invalid attribution estimate: group shares sum to {group_total:g}, not 100")
    else:
        influence_total = sum(t["blended_share_pct"] for t in record["influence"])
        if abs(influence_total - 100.0) > _SHARE_TOLERANCE:
            raise AttributionValidationError(
                f"Invalid attribution estimate: blended influence sums to "
                f"{influence_total:g}, not 100")
    for entry in record.get("influence") or []:
        lo, hi = entry["share_range_pct"]
        point = entry["blended_share_pct"]
        if not (lo - _SHARE_TOLERANCE <= point <= hi + _SHARE_TOLERANCE):
            raise AttributionValidationError(
                f"Invalid attribution estimate: track {entry['track_id']} "
                f"blended share {point:g} lies outside its own range "
                f"[{lo:g}, {hi:g}]")
    for pool in ("writers", "publishers", "masters"):
        for party in record["splits"].get(pool) or []:
            lo, hi = party["share_range_pct"]
            if not (lo - _SHARE_TOLERANCE <= party["share_pct"] <= hi + _SHARE_TOLERANCE):
                raise AttributionValidationError(
                    f"Invalid attribution estimate: {pool[:-1]} "
                    f"{party['name']!r} share lies outside its own range")
    # The money follows the influence: the writers' shares and the unattributed remainder are the whole.
    writers_total = sum(p["share_pct"] for p in record["splits"].get("writers") or [])
    unattributed = record["splits"].get("unattributed_pct", 0.0)
    if abs(writers_total + unattributed - 100.0) > _SHARE_TOLERANCE:
        raise AttributionValidationError(
            f"Invalid attribution estimate: writers ({writers_total:g}) plus unattributed "
            f"({unattributed:g}) is not 100")
    if (record.get("method") or {}).get("kind") == "dtrak":
        for entry in record["influence"]:
            if entry.get("influence_share_pct") is None or \
                    abs(entry["influence_share_pct"] - entry["blended_share_pct"]) > _SHARE_TOLERANCE:
                raise AttributionValidationError(
                    f"Invalid attribution estimate: track {entry['track_id']} measured influence is not "
                    "the share the money follows")
    return record
