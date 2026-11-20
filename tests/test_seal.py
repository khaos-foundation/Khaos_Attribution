"""The seal and marks of the Guild of Fine Tuners: byte-identical, font-free,
struck only for an output whose watermark agreed with its sidecar."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from khaos_attribution import seal
from khaos_attribution.seal import (SealRefused, marks_row_for_provenance, render_marks_row,
                                    render_seal, seal_for_provenance)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SEAL = Path(__file__).parent / "golden" / "seal.svg"
GOLDEN_MARKS = Path(__file__).parent / "golden" / "marks_row.svg"
ARGS = ("123456", "CG", "1.0", "A")
INTERLACED = seal._star_elements()[0]          # the first element of the interlaced artwork
REDUCED = seal.reduced_star_points()


def test_the_same_input_is_the_same_bytes():
    assert render_seal(*ARGS) == render_seal(*ARGS)
    assert render_marks_row("CGD", "0.20.0", "B", rough=True) == render_marks_row("CGD", "0.20.0", "B", rough=True)


def test_the_goldens_have_not_moved():
    """A geometry change is deliberate: re-render both, look, commit with the reason."""
    assert render_seal(*ARGS) == GOLDEN_SEAL.read_text(encoding="utf-8")
    assert render_marks_row("CG", "1.0", "A") == GOLDEN_MARKS.read_text(encoding="utf-8")


def test_no_text_element_font_or_fetch_in_either_output():
    for svg in (render_seal("GUILD/2026-000123", "CGD", "0.20.0", "Z"), render_marks_row("AB", "10", "C"),
                render_marks_row("AB", "10", "C", reverse=True, rough=True)):
        assert "<text" not in svg and "<textPath" not in svg
        assert "font" not in svg.lower() and "@import" not in svg
        assert "http" not in svg.replace('xmlns="http://www.w3.org/2000/svg"', "")
    # the seal carries a <title> (metadata, not type); the marks row carries none
    assert "<title>Guild of Fine Tuners seal" in render_seal(*ARGS)
    assert "<title>" not in render_marks_row("CG", "1.0", "A")


def test_rendering_needs_no_font_and_no_environment():
    """An isolated interpreter with an empty environment renders from package data alone."""
    code = ("import sys; sys.path.insert(0, %r); import khaos_attribution.seal as s; "
            "a = s.render_seal('123456', 'CG', '1.0', 'A'); b = s.render_marks_row('CG', '1.0', 'A'); "
            "assert 'fontTools' not in sys.modules; sys.stdout.write('%%d %%d' %% (len(a), len(b)))") % str(ROOT / "src")
    r = subprocess.run([sys.executable, "-I", "-c", code], env={}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout == f"{len(GOLDEN_SEAL.read_text())} {len(GOLDEN_MARKS.read_text())}"


def test_the_interlaced_star_is_only_in_the_seal_and_the_reduced_only_in_the_row():
    s, m = render_seal(*ARGS), render_marks_row("CG", "1.0", "A")
    assert INTERLACED in s and REDUCED not in s
    assert REDUCED in m and INTERLACED not in m
    assert 'transform="translate(269.00,167.00) scale(0.42429)"' in s


def test_the_filter_is_off_by_default_and_fixed_when_on():
    assert "<filter" not in render_seal(*ARGS) and "<filter" not in render_marks_row("CG", "1.0", "A")
    for svg in (render_seal(*ARGS, rough=True), render_marks_row("CG", "1.0", "A", rough=True)):
        assert 'id="struck"' in svg and 'seed="11"' in svg and 'filter="url(#struck)"' in svg


def test_values_outside_the_charset_or_the_lengths_are_refused():
    for args in (("abc", "CG", "1.0", "A"), ("", "CG", "1.0", "A"), ("1", "C", "1.0", "A"),
                 ("1", "CGDE", "1.0", "A"), ("1", "CG", "", "A"), ("1", "CG", "1.0", "AB"),
                 ("1", "CG", "1.0", ""), ("1", "C_G", "1.0", "A")):
        with pytest.raises(SealRefused):
            render_seal(*args)
    with pytest.raises(SealRefused):
        render_marks_row("C", "1.0", "A")


# ── punches: centred, and inside their dies ─────────────────────────────────

def _text_extent(text, size, baseline):
    used, glyphs = seal.punch_text_layout(text, size, baseline)
    xs = [g["x"] + g["bounds"][0] * g["scale"] for g in glyphs if g["bounds"]]
    xe = [g["x"] + g["bounds"][2] * g["scale"] for g in glyphs if g["bounds"]]
    ys = [baseline - g["bounds"][3] * g["scale"] for g in glyphs if g["bounds"]]
    ye = [baseline - g["bounds"][1] * g["scale"] for g in glyphs if g["bounds"]]
    return used, min(xs), max(xe), min(ys), max(ye)


@pytest.mark.parametrize("mark", ["CG", "CGD", "A1", "WWW"])
def test_artist_marks_of_two_and_three_characters_stay_centred_in_the_punch(mark):
    used, x0, x1, y0, y1 = _text_extent(mark, 44.0, 101.0)
    assert abs((x0 + x1) / 2 - seal.PUNCH_CX) < 2.5, "not centred on the punch"
    assert 10 + 9 < x0 and x1 < 130 - 9, "runs into the canted rect's stroke"
    assert 10 < y0 and y1 < 160
    # The widest three letters ("WWW") are cut smaller to the die.
    assert used == 44.0 if mark != "WWW" else used < 44.0


@pytest.mark.parametrize("version", ["10", "1.0", "0.20.0", "12.34.56"])
def test_short_and_long_standard_versions_fit_the_shield(version):
    used, x0, x1, y0, y1 = _text_extent(version, 44.0, 97.0)
    assert abs((x0 + x1) / 2 - seal.PUNCH_CX) < 2.5
    assert x1 - x0 <= seal.PUNCH_TEXT_MAX_W + 0.01
    assert 10 + 9 < x0 and x1 < 130 - 9, "escapes the shield"
    assert used <= 44.0 and (used == 44.0 if len(version) <= 3 else used < 44.0)
    assert render_marks_row("CG", version, "A")            # and it renders


def test_the_date_letter_sits_inside_the_cartouche():
    for letter in "AWZ1":
        used, x0, x1, y0, y1 = _text_extent(letter, 92.0, 118.0)
        assert 10 + 9 < x0 and x1 < 130 - 9 and 10 + 9 < y0 and y1 < 160 - 9


# ── the ring ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("number", ["7", "123456", "2026-000123", "A B C 1 2"])
def test_the_ring_run_is_centred_on_the_top_and_stays_inside_the_band(number):
    """An eleven-character number still sits within the top half; the
    number's length is not this module's to limit."""
    glyphs = seal.layout(seal.RING_PREFIX + number)
    r = seal.BASELINE_R
    first = glyphs[0]["theta"] - glyphs[0]["advance_px"] / 2 / r
    last = glyphs[-1]["theta"] + glyphs[-1]["advance_px"] / 2 / r
    assert abs(first + last) < 1e-9 and last < math.pi / 2
    s = seal.TYPE_SIZE / 1000.0
    for g in glyphs:
        box = seal._glyphs()[g["char"]]["bounds"]
        if box:
            assert seal.INNER_RULE_R + 3.5 < r + box[1] * s and r + box[3] * s < seal.OUTER_RULE_R - 3.5


def test_size_and_height_scale_the_canvas_not_the_geometry():
    small = render_seal(*ARGS, size=200)
    assert 'width="200" height="200" viewBox="0 0 800 800"' in small
    row = render_marks_row("CG", "1.0", "A", height=119)
    assert 'width="335" height="119" viewBox="0 0 670 238"' in row
    assert render_marks_row("CG", "1.0", "A", reverse=True).count('<rect width="670" height="238" fill="#0b0f0e"/>') == 1


# ── provenance binding ──────────────────────────────────────────────────────

RECORD = {"schema_version": "1.0.0", "artist_id": "chris_green", "artist_name": "Chris Green",
          "adapter_version": "run_1", "adapter_hash": "ab" * 32, "prompt": "x",
          "timestamp": "2026-09-13T00:00:00Z", "watermark_id": 36388}
DOC = {"record": RECORD, "embedded_agrees_with_sidecar": True}


def test_binding_refuses_unless_the_agreement_flag_is_literally_true():
    for doc in ({"record": RECORD}, {"record": RECORD, "embedded_agrees_with_sidecar": False},
                {"record": RECORD, "embedded_agrees_with_sidecar": None},
                {"record": RECORD, "embedded_agrees_with_sidecar": "true"},
                {"record": RECORD, "embedded_agrees_with_sidecar": 1}, RECORD, [], "nope"):
        with pytest.raises(SealRefused):
            seal_for_provenance(doc)
        with pytest.raises(SealRefused):
            marks_row_for_provenance(doc)


def test_binding_reads_every_value_from_the_record_and_invents_none():
    v = seal.provenance_values(DOC)
    assert (v["number"], v["artist_mark"], v["standard_version"], v["date_letter"]) == ("36388", "CG", "1.0", "A")
    assert seal_for_provenance(DOC) == render_seal("36388", "CG", "1.0", "A")
    assert marks_row_for_provenance(DOC) == render_marks_row("CG", "1.0", "A")
    assert seal_for_provenance(json.loads(json.dumps(DOC))) == seal_for_provenance(DOC)
    # explicit keys win over the derived readings
    explicit = {"record": {**RECORD, "artist_mark": "xyz", "standard_version": "2.1", "date_letter": "k"},
                "embedded_agrees_with_sidecar": True}
    assert seal_for_provenance(explicit) == render_seal("36388", "XYZ", "2.1", "K")
    # a later year is a later letter; three words give three initials
    v = seal.provenance_values({"record": {**RECORD, "timestamp": "2028-01-01T00:00:00Z",
                                           "artist_name": "North Sea Trio"}, "embedded_agrees_with_sidecar": True})
    assert (v["artist_mark"], v["date_letter"]) == ("NST", "C")
    for missing in ({"watermark_id": None}, {"artist_name": "", "artist_mark": None}, {"schema_version": "x"},
                    {"timestamp": "soon"}):
        with pytest.raises(SealRefused):
            seal.provenance_values({"record": {**RECORD, **missing}, "embedded_agrees_with_sidecar": True})


def test_jitter_from_record_is_deterministic_per_record_and_differs_across_records():
    a1 = seal_for_provenance(DOC, jitter_from_record=True)
    a2 = seal_for_provenance({"record": dict(reversed(list(RECORD.items()))), "embedded_agrees_with_sidecar": True},
                             jitter_from_record=True)
    assert a1 == a2, "key order is not part of the record"
    b = seal_for_provenance({"record": {**RECORD, "prompt": "y"}, "embedded_agrees_with_sidecar": True},
                            jitter_from_record=True)
    assert a1 != b
    assert a1 != seal_for_provenance(DOC), "jitter is off by default and never in a golden"
    m1 = marks_row_for_provenance(DOC, jitter_from_record=True)
    assert m1 == marks_row_for_provenance(DOC, jitter_from_record=True)
    assert m1 != marks_row_for_provenance(DOC)
    for rot, dx, dy in seal._jitter_from_digest(seal._record_digest(RECORD), 4):
        assert seal.JITTER_ROTATION[0] <= rot <= seal.JITTER_ROTATION[1]
        assert seal.JITTER_DX[0] <= dx <= seal.JITTER_DX[1] and seal.JITTER_DY[0] <= dy <= seal.JITTER_DY[1]


def test_the_cli_writes_the_same_bytes(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = tmp_path / "s.svg"
    r = subprocess.run([sys.executable, "-m", "khaos_attribution.seal", "seal", *ARGS, str(out)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.read_text(encoding="utf-8") == GOLDEN_SEAL.read_text(encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "khaos_attribution.seal", "marks", "CG", "1.0", "A", str(out)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0 and out.read_text(encoding="utf-8") == GOLDEN_MARKS.read_text(encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "khaos_attribution.seal", "seal", "bad_id", "CG", "1.0", "A", str(out)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "REFUSED" in r.stderr


def test_the_licence_travels_with_the_table_and_the_font_does_not():
    pkg = ROOT / "src" / "khaos_attribution" / "seal"
    text = (pkg / "OFL.txt").read_text()
    assert text.startswith("Copyright 2021 Red Hat, Inc.") and "Reserved Font Name Red Hat" in text
    assert not list((ROOT / "src").rglob("*.ttf")) and not list((ROOT / "tools").rglob("*.ttf"))


# ── the inscription, the opaque number, the naming ──────────────────────────

def test_the_ring_carries_the_fixed_label_and_the_number_exactly_as_given():
    """The number is opaque: never computed, padded, cased or reformatted."""
    assert seal.RING_PREFIX == "REGISTER No: "
    assert seal.TYPE_SIZE == 44.0 and seal.TRACKING == 9.0 and seal.BASELINE_R == 290.0   # 298.6 was rejected
    for number in ("123456", "000042", "7", "2026-000123", "A/9"):
        chars = [g["char"] for g in seal.layout(seal.RING_PREFIX + number)]
        assert "".join(chars) == "REGISTER No: " + number
        svg = render_seal(number, "CG", "1.0", "A")
        # the ring's glyph count is the inscription's non-space characters, no more, no fewer
        ring = svg.split('<g fill="#ffffff">')[1].split("</g>")[0]
        assert ring.count("<path ") == len(("REGISTER No: " + number).replace(" ", ""))
        assert f"<title>Guild of Fine Tuners seal — REGISTER No: {number}</title>" in svg
    # the provenance reading passes the number through untouched
    v = seal.provenance_values({"record": {**RECORD, "watermark_id": "000042"}, "embedded_agrees_with_sidecar": True})
    assert v["number"] == "000042"
    assert seal.provenance_values(DOC)["number"] == "36388"
    # lowercase is a character the table cannot set for an input — refused, never uppercased into something else
    with pytest.raises(SealRefused):
        render_seal("no", "CG", "1.0", "A")


def test_the_label_is_never_carried_by_the_marks_row():
    assert "REGISTER" not in render_marks_row("CG", "1.0", "A")
    assert "<title>" not in render_marks_row("CG", "1.0", "A")


def test_the_foundations_name_is_in_no_filename_module_or_public_identifier():
    """Human-readable text only: a rename must be a find-and-replace."""
    pkg = ROOT / "src" / "khaos_attribution" / "seal"
    for path in list(pkg.iterdir()) + [ROOT / "tools" / "build_glyphs.py", GOLDEN_SEAL, GOLDEN_MARKS]:
        assert "khaos" not in path.name.lower(), path.name
    for name in seal.__all__:
        assert "khaos" not in name.lower(), name
    for svg in (render_seal(*ARGS), render_marks_row("CG", "1.0", "A")):
        assert 'id="' not in svg or all("khaos" not in i.lower() for i in svg.split('id="')[1:])


def test_the_ring_reads_the_register_serial_whenever_the_record_carries_one():
    """A derived watermark ID names a bucket many models share; the Register's serial names the model."""
    v = seal.provenance_values({"record": {**RECORD, "register_serial": 65537}, "embedded_agrees_with_sidecar": True})
    assert v["number"] == "65537"


CARD = {"artist_name": "Example Artist", "schema_version": "1.1.0", "training_date": "2026-10-10", "watermark_id": 36388}


def test_a_models_seal_document_is_built_once_for_every_room():
    doc = seal.model_seal_document(CARD, serial=65540, verified=True, derived=True)
    assert seal.provenance_values(doc)["number"] == "65540"
    assert seal.provenance_values(seal.model_seal_document(CARD, serial=None, verified=True, derived=False))["number"] == "36388", \
        "a model from before the Register keeps its own ID"
    with pytest.raises(seal.SealRefused, match="register"):
        seal.provenance_values(seal.model_seal_document(CARD, serial=None, verified=True, derived=True))
    with pytest.raises(seal.SealRefused):
        seal.provenance_values(seal.model_seal_document(CARD, serial=65540, verified=False, derived=True))
