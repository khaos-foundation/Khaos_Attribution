"""The shared watermark contract: SECDED codec, issuer partition, legacy and reset records.

Pure-Python throughout — the AudioSeal embed/detect machinery is exercised
in the App and Platform suites, where the heavy dependencies live.
"""

import pytest

from khaos_attribution.watermark import (
    PAYLOAD_RANGES,
    PAYLOAD_SPACE,
    decode_codeword,
    encode_payload,
    legacy_watermark_ids,
    watermark_issuer,
)


def test_partition_covers_space_without_overlap():
    covered = sorted(p for r in PAYLOAD_RANGES.values() for p in r)
    assert covered == list(range(PAYLOAD_SPACE))


def test_codec_roundtrip_every_payload():
    seen = set()
    for payload in range(PAYLOAD_SPACE):
        codeword = encode_payload(payload)
        assert 0 <= codeword <= 0xFFFF  # provenance schema range
        assert codeword not in seen
        seen.add(codeword)
        assert decode_codeword(codeword) == (payload, codeword, False)


def test_codec_corrects_any_single_bit_error():
    for payload in range(0, PAYLOAD_SPACE, 89):
        codeword = encode_payload(payload)
        for bit in range(16):
            decoded = decode_codeword(codeword ^ (1 << bit))
            assert decoded is not None
            assert decoded[0] == payload
            assert decoded[2] is True


def test_codec_refuses_double_bit_errors():
    for payload in (0, 1, 682, 1365, PAYLOAD_SPACE - 1):
        codeword = encode_payload(payload)
        for i in range(16):
            for j in range(i + 1, 16):
                assert decode_codeword(codeword ^ (1 << i) ^ (1 << j)) is None


def test_issuer_recovered_from_codeword():
    assert watermark_issuer(encode_payload(0)) == "platform"
    assert watermark_issuer(encode_payload(1023)) == "platform"
    assert watermark_issuer(encode_payload(1024)) == "app"
    assert watermark_issuer(encode_payload(2047)) == "app"
    # a corrupted word is not an issuer claim
    assert watermark_issuer(encode_payload(7) ^ 0b1) is None


def test_resets_are_on_record_and_freed_ids_say_so():
    from khaos_attribution.watermark import reset_before, reset_history
    history = reset_history()
    two = next(e for e in history if e["epoch"] == 2)
    freed = two["freed"]
    assert freed and all(decode_codeword(c) is not None for c in freed)
    assert len(freed) == len(set(freed))
    assert set(two["sources"]["legacy_frozen"]) <= set(freed)
    assert set(two["sources"]["loto_calibration_renders"]) <= set(freed)
    assert reset_before(freed[0])["reset_at_utc"].startswith("2026-08-22")
    for kept in two["kept_active"]:
        assert reset_before(int(kept)) is None          # the active runs kept theirs
        assert int(kept) not in freed
    assert reset_before(5) is None                      # never in play
    # the history is a copy: a caller cannot corrupt the record
    history[0]["freed"].clear()
    assert reset_history()[0]["freed"]
    # the legacy list is history: the reset record holds what it held
    assert legacy_watermark_ids() == frozenset()


def test_legacy_ids_are_frozen_valid_codewords():
    """Whatever the frozen list holds must be clean codewords: a verify reads them as Khaos's."""
    legacy = legacy_watermark_ids()
    assert isinstance(legacy, frozenset)
    for codeword in legacy:
        decoded = decode_codeword(codeword)
        assert decoded is not None and decoded[1] == codeword, \
            f"legacy {codeword} is not a clean codeword"


def test_retired_ids_are_spent_forever(tmp_path):
    from khaos_attribution.watermark import (RETIRED_IDS_FILENAME, retire_watermark_ids,
                                             retired_watermark_ids)
    home = tmp_path / "artists"
    assert retired_watermark_ids(home) == set()
    n = retire_watermark_ids(home, [{"watermark_id": 1300, "artist_id": "a", "run_id": "run_1"},
                                    {"watermark_id": "1301", "artist_id": "a"},
                                    {"watermark_id": 1300, "artist_id": "dup"}, {"no": "id"}])
    assert n == 2 and retired_watermark_ids(home) == {1300, 1301}
    import json
    doc = json.loads((home / RETIRED_IDS_FILENAME).read_text())
    assert doc["retired"][0]["run_id"] == "run_1" and doc["retired"][0]["retired_at_utc"]
    # append-only; a second retirement keeps the first
    assert retire_watermark_ids(home, [{"watermark_id": 1302}]) == 1
    assert retired_watermark_ids(home) == {1300, 1301, 1302}
    # a corrupt record is set aside, never overwritten in place
    (home / RETIRED_IDS_FILENAME).write_text("{not json")
    assert retire_watermark_ids(home, [{"watermark_id": 1303}]) == 1
    assert retired_watermark_ids(home) == {1303}
    assert list(home.glob(f"{RETIRED_IDS_FILENAME}.corrupt-*"))


def test_audio_that_is_not_frames_by_channels_is_refused_in_words_before_any_model_loads():
    """One-dimensional mono, an empty array and a batch are a caller's mistake: named, never marked or read on a guess."""
    import numpy as np

    from khaos_attribution import watermark as wm
    for bad in (np.zeros(1600, dtype=np.float32), np.zeros((0, 2), dtype=np.float32), np.zeros((2, 3, 4), dtype=np.float32)):
        with pytest.raises(ValueError, match=r"\(frames, channels\)"):
            wm.Watermarker.embed_verified(bad, 16000, wm.encode_payload(40))
        with pytest.raises(ValueError, match=r"\(frames, channels\)"):
            wm.Watermarker.embed_array(bad, 16000, wm.encode_payload(40))
        with pytest.raises(ValueError, match=r"\(frames, channels\)"):
            wm.Watermarker.detect_array(bad, 16000)


def test_no_allocator_is_left_to_issue_an_id_outside_a_derived_bucket():
    """Every new mark is a derived bucket: the band and per-output allocators are gone, the readers stay."""
    from khaos_attribution import watermark as wm
    for gone in ("allocate_model_watermark_id", "allocate_watermark_id", "MACHINE_BANDS", "artist_band", "ARTIST_BANDS"):
        assert not hasattr(wm, gone), gone
    for kept in ("PAYLOAD_RANGES", "watermark_issuer", "legacy_watermark_ids", "reset_history", "retire_watermark_ids"):
        assert hasattr(wm, kept), kept
