"""A record's bytes and a log entry's bytes are two canonical forms. This pins, by example, where they part, so
nobody hashes a record in the log's form and expects the record's hash."""
import hashlib
import json

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("rfc8785")

from khaos_attribution import registry as R  # noqa: E402
from khaos_attribution import translog as T  # noqa: E402
from tests.test_registry import _record  # noqa: E402

SAME = [{"b": 1, "a": "x"}, {"n": -7, "list": [1, 2, {"z": None, "y": True}]}, {"share": 0.30000000000000004, "half": 0.5},
        {"name": "naïve ☃", "é": "ü"}, {}]
# (document, the record's form, the log's form)
DIFFERENT = [
    ({"share": 1.0}, b'{"share":1.0}', b'{"share":1}'),
    ({"duration_sec": 200.0}, b'{"duration_sec":200.0}', b'{"duration_sec":200}'),
    ({"x": 1e-07}, b'{"x":1e-07}', b'{"x":1e-7}'),
    ({"x": 1e16}, b'{"x":1e+16}', b'{"x":10000000000000000}'),
    # Keys beyond the basic plane: the record's form orders by code point, the log's by UTF-16 unit.
    ({"￿": 1, "\U0001f600": 2}, '{"￿":1,"\U0001f600":2}'.encode(), '{"\U0001f600":2,"￿":1}'.encode()),
]


@pytest.mark.parametrize("doc", SAME)
def test_whole_numbers_strings_and_plain_fractions_have_one_form(doc):
    assert R.canonical_bytes(doc) == T.jcs(doc)


@pytest.mark.parametrize("doc, as_record, as_entry", DIFFERENT)
def test_the_two_forms_part_on_whole_floats_exponents_and_keys_beyond_the_basic_plane(doc, as_record, as_entry):
    assert R.canonical_bytes(doc) == as_record
    assert T.jcs(doc) == as_entry


def test_the_logs_form_refuses_what_the_records_form_writes():
    """An integer past 2**53 is written by the record's form and refused by RFC 8785."""
    big = {"n": 2 ** 60}
    assert R.canonical_bytes(big) == b'{"n":1152921504606846976}'
    with pytest.raises(Exception, match="(?i)integer|domain|range"):
        T.jcs(big)


def test_a_records_hash_is_over_the_records_own_form():
    """A register record carries whole floats (a writer's share, a track's length): its RFC 8785 form is other
    bytes, and so another hash than record_sha256."""
    key = R.SigningKey.generate()
    record = _record()
    record["model"]["machine_key_id"] = key.key_id
    signed = R.sign_record(record, key)
    own = R.canonical_bytes(json.loads(json.dumps(signed["record"])))
    assert hashlib.sha256(own).hexdigest() == signed["record_sha256"]
    assert R.signed_problems(signed, key.public_pem()) == []
    assert hashlib.sha256(T.jcs(signed["record"])).hexdigest() != signed["record_sha256"]
