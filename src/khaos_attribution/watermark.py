"""Watermarking contract shared by every Khaos issuer.

The watermark_id (0-65535) is AudioSeal's 16-bit message payload, carrying an
extended-Hamming (16,11) SECDED codeword: single-bit detector errors are
corrected, double errors refused. The 2048-payload space is partitioned by
issuer (platform 0-1023, app 1024-2047) and, for run IDs, by machine band.
The watermark delta is computed at 16 kHz and resampled onto the untouched
original. AudioSeal is imported lazily so the codec and ranges work without
the audio extras.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading

os.environ.setdefault("NO_TORCH_COMPILE", "1")

log = logging.getLogger("khaos_attribution.watermark")

# ---- extended Hamming (16,11) SECDED codec ----
# Positions 1..15 hold parity at powers of two and data elsewhere; position 0
# is overall parity.

_DATA_POSITIONS = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]  # 11 data bits
_PARITY_POSITIONS = [1, 2, 4, 8]

PAYLOAD_SPACE = 1 << len(_DATA_POSITIONS)  # 2048 distinct IDs

# Issuer partition — the contract between the Platform and the App.
PAYLOAD_RANGES: dict[str, range] = {
    "platform": range(0, 1024),
    "app": range(1024, 2048),
}


def encode_payload(payload: int) -> int:
    """11-bit payload -> 16-bit SECDED codeword (the watermark_id)."""
    if not 0 <= payload < PAYLOAD_SPACE:
        raise ValueError(f"Payload out of range: {payload}")
    bits = [0] * 16
    for i, pos in enumerate(_DATA_POSITIONS):
        bits[pos] = (payload >> (len(_DATA_POSITIONS) - 1 - i)) & 1
    for p in _PARITY_POSITIONS:
        bits[p] = sum(bits[pos] for pos in range(1, 16) if pos & p and pos != p) % 2
    bits[0] = sum(bits[1:]) % 2
    return int("".join(map(str, bits)), 2)


def decode_codeword(codeword: int) -> tuple[int, int, bool] | None:
    """16-bit word (possibly corrupted) -> (payload, clean codeword,
    corrected?) or None if uncorrectable (2+ bit errors)."""
    bits = [(codeword >> (15 - i)) & 1 for i in range(16)]
    syndrome = 0
    for pos in range(1, 16):
        if bits[pos]:
            syndrome ^= pos
    parity_ok = sum(bits) % 2 == 0
    corrected = False
    if syndrome and parity_ok:
        return None  # double error: detectable, not correctable
    if syndrome:
        bits[syndrome] ^= 1
        corrected = True
    elif not parity_ok:
        bits[0] ^= 1
        corrected = True
    payload = 0
    for pos in _DATA_POSITIONS:
        payload = (payload << 1) | bits[pos]
    return payload, int("".join(map(str, bits)), 2), corrected


def watermark_issuer(codeword: int) -> str | None:
    """Which issuer a (clean) codeword belongs to, or None if invalid."""
    decoded = decode_codeword(codeword)
    if decoded is None or decoded[1] != codeword:
        return None
    payload = decoded[0]
    for issuer, payload_range in PAYLOAD_RANGES.items():
        if payload in payload_range:
            return issuer
    return None


# ---- watermarking v2: model-level allocation (docs/watermarking-v2.md) ----
# One ID per training run, allocated into the model card. The payload space is
# partitioned per training machine so machines allocate without coordination;
# payloads issued by the retired per-output scheme are frozen out
# (legacy_watermark_ids.json) so a decoded ID is owned by at most one thing.
# Each artist's Workshop draws from its own 24-payload band, carved from the
# old threadripper band; allocation reports exhaustion loudly.
ARTIST_BAND_START = 1376
ARTIST_BAND_SIZE = 24
ARTIST_BANDS = 20


def artist_band(n: int) -> str:
    """The band name for artist slot n (1 … ARTIST_BANDS): 'artist-07'."""
    if not 1 <= n <= ARTIST_BANDS:
        raise ValueError(f"artist slot must be 1..{ARTIST_BANDS}, got {n}")
    return f"artist-{n:02d}"


MACHINE_BANDS: dict[str, range] = {
    "reserved":     range(0, 32),      # calibration + migration tooling
    "laptop":       range(32, 704),
    "studio":       range(704, 1376),
    **{artist_band(n): range(ARTIST_BAND_START + (n - 1) * ARTIST_BAND_SIZE,
                             ARTIST_BAND_START + n * ARTIST_BAND_SIZE)
       for n in range(1, ARTIST_BANDS + 1)},                    # 1376 … 1855
    "threadripper": range(ARTIST_BAND_START + ARTIST_BANDS * ARTIST_BAND_SIZE, 2048),
}

# ---- the payload as a bucket: derived by the machine, never allocated ----
# A model's payload is a hash of its machine key and run id into the payloads
# outside the reserved 32; many models share one. The register's serial names
# the model; the payload narrows a verify to one bucket of fingerprints.
RESERVED_PAYLOADS = range(0, 32)
DERIVED_PAYLOADS = range(32, PAYLOAD_SPACE)


def derive_payload(machine_key_id: str, run_id: str) -> int:
    """The bucket for a run: deterministic from the machine key id and the run id.
    2**64 mod 2016 is not 0, so 1,024 buckets are heavier by about one part in 2**53: nothing."""
    import hashlib  # noqa: PLC0415
    digest = hashlib.sha256(f"{machine_key_id}:{run_id}".encode("utf-8")).digest()
    return DERIVED_PAYLOADS.start + int.from_bytes(digest[:8], "big") % len(DERIVED_PAYLOADS)


def derive_watermark_id(machine_key_id: str, run_id: str) -> int:
    """The codeword a run stamps: the derived bucket with its error-correcting bits."""
    return encode_payload(derive_payload(machine_key_id, run_id))


_LEGACY_IDS = None


def legacy_watermark_ids() -> frozenset:
    """The frozen set of pre-v2 per-output codewords (package data)."""
    global _LEGACY_IDS
    if _LEGACY_IDS is None:
        import json  # noqa: PLC0415
        from importlib.resources import files  # noqa: PLC0415
        data = json.loads(files("khaos_attribution")
                          .joinpath("legacy_watermark_ids.json").read_text())
        _LEGACY_IDS = frozenset(int(c) for c in data["codewords"])
    return _LEGACY_IDS


def allocate_model_watermark_id(band: str, used: set[int]) -> int | None:
    """A fresh run codeword from this machine's band, or None when the band
    is exhausted (the caller must fail loudly — a run may not complete
    unmarked). `used` is the codewords already in model cards; frozen legacy
    payloads are excluded automatically."""
    if band not in MACHINE_BANDS:
        raise ValueError(f"Unknown watermark band {band!r} — one of "
                         f"{sorted(MACHINE_BANDS)}")
    legacy_payloads = {decode_codeword(c)[0] for c in legacy_watermark_ids()
                      if decode_codeword(c) is not None}
    band_range = MACHINE_BANDS[band]
    for _ in range(64):
        payload = band_range.start + secrets.randbelow(len(band_range))
        if payload in legacy_payloads:
            continue
        candidate = encode_payload(payload)
        if candidate not in used:
            return candidate
    for payload in band_range:   # dense band: walk deterministically
        if payload in legacy_payloads:
            continue
        candidate = encode_payload(payload)
        if candidate not in used:
            return candidate
    log.error("Watermark band %r exhausted (%d payloads)", band, len(band_range))
    return None


def allocate_watermark_id(used: set[int], issuer: str = "platform") -> int | None:
    """DEPRECATED: per-output allocation is retired — new audio carries its
    model's run ID (allocate_model_watermark_id). Kept for legacy callers."""
    payload_range = PAYLOAD_RANGES[issuer]
    span = len(payload_range)
    for _ in range(64):
        candidate = encode_payload(payload_range.start + secrets.randbelow(span))
        if candidate not in used:
            return candidate
    # dense registry: walk the partition deterministically instead of gambling
    for payload in payload_range:
        candidate = encode_payload(payload)
        if candidate not in used:
            return candidate
    log.error("Watermark partition %r exhausted (%d IDs)", issuer, span)
    return None


# ---- AudioSeal wrapper (heavy deps, all lazy) ----

_SAMPLE_RATE = 16000  # AudioSeal-native

# Model cache is process-global: the models are stateless once loaded.
_MODELS_LOCK = threading.Lock()
_MODELS = None


def _load_models():
    global _MODELS
    with _MODELS_LOCK:
        if _MODELS is None:
            from audioseal import AudioSeal

            log.info("Loading AudioSeal 16-bit generator + detector")
            _MODELS = (
                AudioSeal.load_generator("audioseal_wm_16bits"),
                AudioSeal.load_detector("audioseal_detector_16bits"),
            )
        return _MODELS


class Watermarker:
    """Embed/detect on in-memory audio arrays.

    Arrays are numpy float32 shaped (frames, channels) — the layout
    soundfile produces and consumes. File-based callers load first and
    save after, keeping container/metadata handling on their side.
    """

    @staticmethod
    def available() -> bool:
        try:
            import audioseal  # noqa: F401
            import torchaudio  # noqa: F401
        except ImportError:
            return False
        return True

    # The mark's strengths, tried in order until the id reads back clean: quiet, sparse material can flip two bits
    # of the message at the default strength, one more than the code corrects; 1.5 has read back clean where 1.0
    # did not, and the mark then still sits about 30 dB under the music.
    STRENGTHS = (1.0, 1.5, 2.0)

    @staticmethod
    def embed_array(audio, sample_rate: int, watermark_id: int):
        """Return a watermarked copy of (frames, channels) float32 audio whose id reads back."""
        return Watermarker.embed_verified(audio, sample_rate, watermark_id)[0]

    @staticmethod
    def embed_verified(audio, sample_rate: int, watermark_id: int):
        """(marked audio, strength): the mark at the first strength whose id reads back clean, or the strongest
        with a warning, since a present mark whose id cannot be read is still a mark."""
        import torch
        import torchaudio

        generator, _ = _load_models()
        wav = torch.from_numpy(audio).float().T  # (channels, frames)
        mono = wav.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, T)
        message = torch.tensor([[(watermark_id >> (15 - i)) & 1 for i in range(16)]])
        mono16 = torchaudio.functional.resample(mono, sample_rate, _SAMPLE_RATE)
        with torch.inference_mode():
            delta16 = generator.get_watermark(mono16, _SAMPLE_RATE, message=message)
        delta = torchaudio.functional.resample(delta16, _SAMPLE_RATE, sample_rate)[0, 0][: wav.shape[-1]]
        marked = None
        for strength in Watermarker.STRENGTHS:
            marked = (wav + strength * delta).clamp(-1.0, 1.0).T.contiguous().numpy()
            _, read, corrected = Watermarker.detect_array(marked, sample_rate)
            if read == watermark_id and not corrected:
                return marked, strength
        log.warning("watermark %d does not read back clean at strength %s; the strongest mark is kept",
                    watermark_id, Watermarker.STRENGTHS[-1])
        return marked, Watermarker.STRENGTHS[-1]

    @staticmethod
    def decode_consistent(audio, sample_rate: int, *, window_s: float = 6.0,
                          hop_s: float = 3.0) -> tuple[float, int | None]:
        """DEPRECATED: windowed agreement does not discriminate excerpts from
        full files. Use cross-evidence (a fingerprint match whose recorded ID
        confirms or contradicts the decoded payload) instead. Kept only so
        pinned consumers keep importing."""
        import numpy as np  # noqa: PLC0415

        whole_prob, whole_id, _ = Watermarker.detect_array(audio, sample_rate)
        if whole_id is None:
            return whole_prob, None
        window = int(window_s * sample_rate)
        hop = int(hop_s * sample_rate)
        agree = disagree = 0
        for start in range(0, max(1, len(audio) - window + 1), hop):
            segment = np.ascontiguousarray(audio[start:start + window])
            if len(segment) < sample_rate:
                continue
            _, got, _ = Watermarker.detect_array(segment, sample_rate)
            if got == whole_id:
                agree += 1
            elif got is not None:
                disagree += 1
        # Short clips may yield a single window; consistency then means
        # "nothing contradicted the whole-clip decode".
        if disagree == 0 and agree >= 1:
            return whole_prob, whole_id
        return whole_prob, None

    @staticmethod
    def detect_array(audio, sample_rate: int) -> tuple[float, int | None, bool]:
        """(probability, clean codeword or None, corrected?) for audio
        shaped (frames, channels) float32."""
        import torch
        import torchaudio

        _, detector = _load_models()
        wav = torch.from_numpy(audio).float().T
        mono = wav.mean(dim=0, keepdim=True).unsqueeze(0)
        mono16 = torchaudio.functional.resample(mono, sample_rate, _SAMPLE_RATE)
        with torch.inference_mode():
            probability, message = detector.detect_watermark(mono16, _SAMPLE_RATE)
        raw = 0
        for bit in message[0].tolist():
            raw = (raw << 1) | int(bit)
        decoded = decode_codeword(raw)
        if decoded is None:
            return float(probability), None, False
        _, codeword, corrected = decoded
        return float(probability), codeword, corrected


# ---- retired model IDs ----
# Allocation scans the cards on disk, so removing an artist would otherwise
# free its codewords. Removal retires them into a record under ``home`` that
# allocators read alongside the cards. Append-only, never pruned — an ID is
# spent forever.

RETIRED_IDS_FILENAME = ".retired_watermark_ids.json"


def retired_watermark_ids(home) -> set[int]:
    """Codewords retired under ``home`` (an artists directory). Missing or
    unreadable record → empty set (the cards themselves still count)."""
    import json
    from pathlib import Path

    path = Path(home) / RETIRED_IDS_FILENAME
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    out: set[int] = set()
    for entry in doc.get("retired", []) if isinstance(doc, dict) else []:
        try:
            out.add(int(entry["watermark_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def retire_watermark_ids(home, entries) -> int:
    """Append ``entries`` (dicts with at least ``watermark_id``; ``artist_id``,
    ``adapter_version``/``run_id``, ``reason`` welcome) to the record under
    ``home``, stamping ``retired_at_utc``. Returns how many were new.
    Atomic write; an existing unreadable record is preserved as
    ``.corrupt-<timestamp>`` rather than overwritten."""
    import json
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    home = Path(home)
    path = home / RETIRED_IDS_FILENAME
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    doc = {"schema_version": "1.0.0", "retired": []}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("retired"), list):
                doc = loaded
            else:
                raise ValueError("not a retired-ids record")
        except (OSError, ValueError):
            stamp = now.replace(":", "").replace("+0000", "Z")
            path.replace(path.with_name(f"{path.name}.corrupt-{stamp}"))
    known: set[int] = set()
    for e in doc["retired"]:
        try:
            known.add(int(float(e["watermark_id"])))
        except (KeyError, TypeError, ValueError):
            continue
    added = 0
    for entry in entries:
        try:
            wm = int(entry["watermark_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if wm in known:
            continue
        record = {k: v for k, v in entry.items() if k != "watermark_id"}
        record.update(watermark_id=wm, retired_at_utc=now)
        doc["retired"].append(record)
        known.add(wm)
        added += 1
    home.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return added


# ---- resets ----
# The record of every deliberate freeing of IDs (package data), so verifying
# a freed codeword can say "issued before the reset of …" rather than point
# at whatever it is re-issued to later.

_RESETS = None


def reset_history() -> list[dict]:
    """Every reset on record (package data), oldest first."""
    global _RESETS
    if _RESETS is None:
        import json  # noqa: PLC0415
        from importlib.resources import files  # noqa: PLC0415
        try:
            _RESETS = json.loads(files("khaos_attribution")
                                 .joinpath("watermark_resets.json").read_text())
        except (OSError, ValueError) as exc:
            # Without the record every freed ID would quietly read as "unknown".
            log.warning("watermark_resets.json unreadable (%s) — reset history unavailable", exc)
            _RESETS = []
    import copy  # noqa: PLC0415
    return copy.deepcopy(_RESETS)


def reset_before(codeword: int) -> dict | None:
    """The most recent reset that freed ``codeword`` (it was in play before
    that date and nothing carrying it survived), or None. A codeword kept by
    an active run across a reset is NOT reported here."""
    latest = None
    for entry in reset_history():
        try:
            freed = {int(c) for c in entry.get("freed", [])}
        except (TypeError, ValueError):
            continue
        if int(codeword) in freed:
            latest = entry
    return latest
