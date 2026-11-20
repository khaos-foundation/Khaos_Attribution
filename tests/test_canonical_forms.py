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


# ---- one home: each named form is byte-for-byte what its callers wrote before ----

GOLDEN_DOCS = [
    {"b": 1, "a": "x"},
    {"n": -7, "list": [1, 2, {"z": None, "y": True}]},
    {"share": 0.30000000000000004, "half": 0.5, "whole": 50.0, "tiny": 1e-05, "big": 1e16, "neg_zero": -0.0},
    {"name": "naïve ☃", "é": "ü", "ctl": "a\tb\nc\u0001\u007f "},
    {"￿": 1, "\U0001f600": 2},
    {"nested": {"k": [1.5, -2.25e-10, 123456789012345678]}},
]
# sha256 of each document in each form, taken from the code before the forms had one home; None where it refuses.
GOLDEN = {
    "compact": ["cdab067e9f3beb32d1252cfd63e492592fecbf591b0d08cadb24bb17f3864246",
                "dd4b0dcc82a272594711cf1325b338a412d70f3f5335796fc0711f035a28e27a",
                "b864007229d4d8aa9d740efc922e0d8023c1aa4dc8adaf489d6f568596e6f1d4",
                "9d8de3e5c428fc1f59f4cf0f11f9d52e370a0e6a1ae13ed1c805d8bb8708eaac",
                "c07399259f29babc3ab550e5dcfa10f3ac14b9e55a39a6b3e9100fc6100ef936",
                "52b2f2a00bf167b7874014818b4a4fbef35f4fd8862aa9adddd568f368ef1e38"],
    "compact_ascii": ["cdab067e9f3beb32d1252cfd63e492592fecbf591b0d08cadb24bb17f3864246",
                      "dd4b0dcc82a272594711cf1325b338a412d70f3f5335796fc0711f035a28e27a",
                      "b864007229d4d8aa9d740efc922e0d8023c1aa4dc8adaf489d6f568596e6f1d4",
                      "2ccbf461351b4b4c3bec9136661844fbb82a69dd1b2ceb38dd96827711dbe0ca",
                      "7399aff294a6b56f657db5fc869921bc94cd5f74936711c78e04981c8fb1b3e4",
                      "52b2f2a00bf167b7874014818b4a4fbef35f4fd8862aa9adddd568f368ef1e38"],
    "jcs": ["cdab067e9f3beb32d1252cfd63e492592fecbf591b0d08cadb24bb17f3864246",
            "dd4b0dcc82a272594711cf1325b338a412d70f3f5335796fc0711f035a28e27a",
            "74b03f9a57b9e9356b675ed2a6be839dcd289f6c96ddf4d23a3194e0c10c848c",
            "9d8de3e5c428fc1f59f4cf0f11f9d52e370a0e6a1ae13ed1c805d8bb8708eaac",
            "48c5d098713870a7b88be7fc0ffc0a79e912d20f80ea8a4a5604426f49add103",
            None],
    # A calibration block's identity (influence.calibration_sha256) keeps its own spaced form, outside the named three.
    "spaced": ["385820f0096fd558f4091319e7fa742cebf877dc3baca180981889f1c40eca84",
               "c0b29bdf310060f3d48a50501b4a66763c02721d9f68fc1c76b530ee4c773639",
               "8c9cd3dccb2cd0865e11a037aaaaa34a5ee2c4ce4fa9dceb91c5c2ccb2119161",
               "8d9d2088d5a6a2fb356e60fdb8a8339db3e9f968568ba77c2bbf577a3b4cdf2d",
               "bbf2181746efc73f47c5b6ee0379db8b0396876feeedb488fe4e9a01e2b05821",
               "736f849c751f8b0146c9b4c28f1f59b65e2698372695447b076cc44329e6dd57"],
}


def _sha(form, doc):
    try:
        return hashlib.sha256(form(doc)).hexdigest()
    except Exception:  # noqa: BLE001 — a form that refuses a document has no hash for it
        return None


@pytest.mark.parametrize("name", ["compact", "compact_ascii", "jcs"])
def test_each_named_form_writes_the_bytes_its_callers_wrote_before(name):
    from khaos_attribution import canonical as C
    assert [_sha(getattr(C, name), d) for d in GOLDEN_DOCS] == GOLDEN[name]


def test_every_caller_hashes_through_its_named_form(tmp_path):
    """The old names stay and each is its form: the rooms import them, and a hash already published must not move."""
    from khaos_attribution import canonical as C
    from khaos_attribution import release, rights_stamp
    from khaos_attribution.seal import _canonical, _record_digest
    for doc in GOLDEN_DOCS:
        assert R.canonical_bytes(doc) == C.compact(doc)
        assert _canonical(doc) == C.compact_ascii(doc) and _record_digest(doc) == hashlib.sha256(C.compact_ascii(doc)).digest()
        if _sha(C.jcs, doc) is not None:
            assert T.jcs(doc) == C.jcs(doc)
        (tmp_path / "rights.json").write_text(json.dumps({"tracks": doc}), encoding="utf-8")
        assert rights_stamp.rights_sha256(tmp_path / "rights.json") == hashlib.sha256(C.compact(doc)).hexdigest()
    files, meta = {"a.bin": {"sha256": "ab" * 32}}, {"artist_id": "artïst", "run_id": "r", "serial": 65536}
    key = {"files": {"a.bin": "ab" * 32}, "artist_id": "artïst", "run_id": "r", "checkpoint": None, "serial": 65536,
           "record_sha256": None}
    assert release.bundle_id(files, meta) == hashlib.sha256(C.compact_ascii(key)).hexdigest()


PAGE_TEXTS = [
    '{"share":50.0,"tiny":1e-05,"big":1e16,"neg":-0.0,"int":-0,"exact":123456789012345678901234567890}',
    '{"f":[0.1,0.30000000000000004,1e15,1e-4,1E5,2.5e-300,1.7976931348623157e308,1e400,5e-324,100.0,-2.25e-10]}',
    '{"b":1,"a":{"z":null,"y":[true,false]},"":"empty"}',
    '{"ctl":"a\\tb\\nc\\u0001\\u001f\\u007f\\u2028","q":"\\"\\\\/","nonascii":"naïve ☃ \\ud83d\\ude00"}',
    '{"\\uffff":1,"\\ud83d\\ude00":2,"é":3,"e":4}',
    '{"dup":1,"dup":2}',
    ' [ 1 , { "x" : 1.50 } ] ',
]
REFUSED = ['{"x":NaN}', '{"x":"\\ud800"}', '{"x":"\\udc00"}', '{"x":"\\ude00\\ud83d"}', '{"x":1} trailing', '{"x":01}',
           '{"x":"a\tb"}', '{"n":' + "9" * 4301 + "}"]


def _page_forms(tmp_path, texts):
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    spec = tmp_path / "texts.json"
    spec.write_text(json.dumps(texts))
    js = Path("src/register_service/static/log-verify.js").resolve()
    run = subprocess.run([node, "tests/js/canonical_harness.mjs", str(js), str(spec)], capture_output=True, text=True,
                         timeout=60, check=True)
    # One line per case: a form may hold U+2028, which splitlines() would also split on.
    return [json.loads(line) for line in run.stdout.strip().split("\n")]


def test_the_page_writes_a_records_own_form_byte_for_byte(tmp_path):
    """The page rehashes a pasted signed record as khaos-log-verify does: whole floats, exponents, exact integers, -0.0,
    controls, keys beyond the basic plane and duplicates all come out as Python writes them."""
    from khaos_attribution import canonical as C
    for text, page in zip(PAGE_TEXTS, _page_forms(tmp_path, PAGE_TEXTS)):
        assert page.encode("utf-8") == C.compact(json.loads(text)), text


def test_the_page_writes_any_double_as_python_does(tmp_path):
    """Thousands of doubles from every exponent, in Python's repr, %.17g and short exponent spellings."""
    import math
    import random
    import struct
    r, xs = random.Random(7), []
    while len(xs) < 3000:
        x = struct.unpack("<d", r.getrandbits(64).to_bytes(8, "little"))[0]
        if math.isfinite(x):
            xs.append(x)
    xs += [r.uniform(-1e6, 1e6) for _ in range(1000)] + [float(r.randint(-10 ** 17, 10 ** 17)) for _ in range(500)]
    spelled = [repr(x) for x in xs] + ["%.17g" % x for x in xs[:500]] + ["%.3e" % x for x in xs[:500]]
    text = "[" + ",".join(s if any(c in s for c in ".eE") else s + ".0" for s in spelled) + "]"
    from khaos_attribution import canonical as C
    assert _page_forms(tmp_path, [text])[0].encode("utf-8") == C.compact(json.loads(text))


def test_the_page_refuses_what_a_record_never_holds(tmp_path):
    """NaN, a lone surrogate, trailing text, a leading zero, a raw control character, an integer Python will not read."""
    for text, page in zip(REFUSED, _page_forms(tmp_path, REFUSED)):
        assert isinstance(page, dict) and "cannot be read" in page["error"], text


def test_an_evidence_reviews_key_is_over_the_compact_form():
    """Pins evidence_review_key's bytes to canonical.compact, so it can call it and no approval's key moves."""
    from khaos_attribution import canonical as C
    from khaos_attribution.validation import evidence_review_key
    track = {"track_id": "t-é", "state": "cleared", "matches": [{"serial": 65536, "score": 0.5, "whole": 50.0}],
             "evidence": {"note": "naïve ☃", "seconds": 1e-05}, "name": "not part of the key"}
    body = {k: track[k] for k in ("track_id", "state", "matches", "evidence")}
    assert evidence_review_key(track) == hashlib.sha256(C.compact(body)).hexdigest()


def test_the_calibration_identity_keeps_its_own_spaced_form():
    influence = pytest.importorskip("khaos_attribution.influence")
    assert [influence.calibration_sha256(d) for d in GOLDEN_DOCS] == GOLDEN["spaced"]
