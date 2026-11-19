"""A catalogue delivery's normalised form and the industry codes it carries."""
import copy

import pytest

from khaos_attribution import AttributionValidationError, validate_catalogue_delivery, validate_track_rights
from khaos_attribution import industry_codes as C

GOOD = {
    "schema_version": "1.0.0", "delivery_id": "20261119120000000", "received_at_utc": "2026-11-19T12:00:00Z",
    "sender": {"name": "Example Records", "ddex_party_id": "PADPIDA2026111901X"}, "ern_version": "4.3",
    "releases": [{
        "message_id": "MSG-1", "release_reference": "R0", "icpn": "036000291452", "title": "Night Lines",
        "display_artist": "Example Artist", "territories": ["Worldwide"], "start_date": "2026-12-01", "takedown": False,
        "tracks": [{"resource_reference": "A1", "isrc": "GBAYE2600001", "iswc": "T-034524680-1", "title": "Harbour",
                    "display_artist": "Example Artist", "duration_s": 241.0, "p_line": "2026 Example Records",
                    "c_line": None, "contains_ai": None,
                    "contributors": [{"name": "A. Writer", "roles": ["Composer"], "instruments": [], "ipi": "00012345678"},
                                     {"name": "B. Player", "roles": ["Musician"], "instruments": ["Cello"]}],
                    "file": {"path": "resources/A1.flac", "hash_algorithm": "MD5", "hash": "0" * 32}}]}],
}


def test_published_codes_read_and_wrong_check_digits_do_not():
    """Each code reads from its printed forms into the stored one; a code whose check digit fails is no code."""
    assert C.isrc("gb-aye-26-00001") == "GBAYE2600001" and C.isrc("GBAYE260001") is None
    assert C.iswc("T-034.524.680-1") == "T-034524680-1" and C.iswc("T0345246801") == "T-034524680-1"
    assert C.iswc("T-034.524.680-2") is None and C.iswc("034524680-1") is None
    assert C.gtin("036000291452") == "036000291452" and C.gtin("036000291453") is None
    assert C.gtin("4006381333931") == "4006381333931" and C.gtin("4006381333932") is None
    assert C.gtin("12345") is None and C.ipi("00012345678") == "00012345678" and C.ipi("1234") is None


def test_a_good_delivery_validates():
    assert validate_catalogue_delivery(copy.deepcopy(GOOD)) is not None


@pytest.mark.parametrize("change, said", [
    (lambda d: d["releases"][0].__setitem__("icpn", "036000291453"), "check digit"),
    (lambda d: d["releases"][0]["tracks"][0].__setitem__("iswc", "T-034524680-2"), "check digit"),
    (lambda d: d["releases"][0]["tracks"].append(copy.deepcopy(d["releases"][0]["tracks"][0])), "twice"),
    (lambda d: d["releases"][0]["tracks"][0].__setitem__("isrc", "GB-AYE-26-00001"), "GB-AYE"),
    (lambda d: d["sender"].__setitem__("ddex_party_id", "Example"), "Example"),
    (lambda d: d.__setitem__("ern_version", "3.8"), "3.8"),
    (lambda d: d["releases"][0]["tracks"][0].__setitem__("shares", 50), "shares"),
])
def test_a_delivery_that_breaks_a_rule_is_refused_saying_which(change, said):
    bad = copy.deepcopy(GOOD)
    change(bad)
    with pytest.raises(AttributionValidationError, match=said):
        validate_catalogue_delivery(bad)


def test_a_rights_record_read_from_a_delivery_validates():
    """A record a person confirmed from a delivery's credits is written with source ddex_ern."""
    record = {"schema_version": "1.0.0", "track_id": "t1", "title": "Harbour",
              "isrc": {"status": "assigned", "value": "GBAYE2600001"}, "iswc": {"status": "not_yet_looked_up"},
              "writers": [{"name": "A. Writer", "role": "composer", "share_pct": 100}], "publishers": [],
              "masters": [{"name": "Example Records", "share_pct": 100}], "source": "ddex_ern"}
    assert validate_track_rights(record)["source"] == "ddex_ern"
