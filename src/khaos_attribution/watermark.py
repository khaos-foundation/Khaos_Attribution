"""The watermark every Khaos issuer shares: AudioSeal's 16-bit message carrying a (16,11) SECDED codeword, one bit
corrected and two refused. A model's payload is a bucket derived from its machine key and run id, shared by many
models; the issuer partition is the earlier scheme, kept to read what it marked."""

from __future__ import annotations

import logging
import os
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


def _framed(audio):
    """The array as embed and detect take it, or ValueError: a guess at a caller's layout would mark the wrong axis."""
    if getattr(audio, "ndim", 0) != 2 or not len(audio):
        raise ValueError(f"audio must be shaped (frames, channels) with at least one frame, got {getattr(audio, 'shape', None)}")
    return audio


class WatermarkUnreadable(RuntimeError):
    """The id read back at no strength, and the caller asked for a mark that does."""


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
    def embed_array(audio, sample_rate: int, watermark_id: int, *, require_readback: bool = False):
        """The marked copy embed_verified returns for (frames, channels) float32 audio, without its strength."""
        return Watermarker.embed_verified(audio, sample_rate, watermark_id, require_readback=require_readback)[0]

    @staticmethod
    def embed_verified(audio, sample_rate: int, watermark_id: int, *, require_readback: bool = False):
        """(marked audio, strength): the first strength whose id reads back clean, else the strongest whose id reads
        back corrected, else the strongest with a warning, since a present mark whose id cannot be read is still a
        mark; with require_readback that last case raises WatermarkUnreadable instead."""
        audio = _framed(audio)
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
        marked = corrected_read = None
        for strength in Watermarker.STRENGTHS:
            marked = (wav + strength * delta).clamp(-1.0, 1.0).T.contiguous().numpy()
            _, read, corrected = Watermarker.detect_array(marked, sample_rate)
            if read == watermark_id and not corrected:
                return marked, strength
            if read == watermark_id:
                corrected_read = (marked, strength)
        if corrected_read:
            return corrected_read
        if require_readback:
            raise WatermarkUnreadable(f"watermark {watermark_id} does not read back at any strength up to "
                                      f"{Watermarker.STRENGTHS[-1]}")
        log.warning("watermark %d does not read back at strength %s; the strongest mark is kept",
                    watermark_id, Watermarker.STRENGTHS[-1])
        return marked, Watermarker.STRENGTHS[-1]

    @staticmethod
    def detect_array(audio, sample_rate: int) -> tuple[float, int | None, bool]:
        """(probability, clean codeword or None, corrected?) of the whole recording, as AudioSeal reads a whole array:
        read window by window so memory does not grow with length, every window's frames counted together."""
        return Watermarker.detect_whole_and_window(audio, sample_rate)[0]

    @staticmethod
    def strongest_window(audio, sample_rate: int) -> tuple[float, int | None, bool]:
        """The reading of the window most likely marked, preferring a clean codeword, then a corrected one, among windows
        at LIKELY_MARKED or above, else the most likely window: a mark inside a longer recording reads as its own."""
        return Watermarker.detect_whole_and_window(audio, sample_rate)[1]

    @staticmethod
    def detect_whole_and_window(audio, sample_rate: int) -> tuple[tuple[float, int | None, bool], tuple[float, int | None, bool]]:
        """(detect_array's reading, strongest_window's reading) from one pass over the windows."""
        audio = _framed(audio)
        windows = [_window_frames(audio[start:end], sample_rate) for start, end in detect_windows(len(audio), sample_rate)]
        marked, frames = sum(w[0] for w in windows), sum(w[1] for w in windows)
        whole = (marked / frames, *_decoded(sum(w[2] for w in windows)))
        readings = [(w[0] / w[1], *_decoded(w[2])) for w in windows]
        for readable in (lambda r: r[1] is not None and not r[2], lambda r: r[1] is not None):
            likely = [r for r in readings if readable(r) and r[0] >= LIKELY_MARKED]
            if likely:
                return whole, max(likely, key=lambda r: r[0])
        return whole, max(readings, key=lambda r: r[0])


# A recording is read in windows of this length, so memory does not grow with its length; a tail shorter than
# DETECT_MIN_TAIL_S joins the window before it.
DETECT_WINDOW_S = 30
DETECT_MIN_TAIL_S = 5
# The probability at which a reading counts as likely marked, as Verify hears it (verify.WATERMARK_THRESHOLD).
LIKELY_MARKED = 0.5


def detect_windows(n_frames: int, sample_rate: int) -> list[tuple[int, int]]:
    """The (start, end) frames of each window a recording of ``n_frames`` is read in."""
    window, tail = int(DETECT_WINDOW_S * sample_rate), int(DETECT_MIN_TAIL_S * sample_rate)
    if n_frames <= window + tail:
        return [(0, n_frames)]
    starts = list(range(0, n_frames, window))
    if n_frames - starts[-1] < tail:
        starts.pop()
    return [(start, starts[i + 1] if i + 1 < len(starts) else n_frames) for i, start in enumerate(starts)]


def _window_frames(audio, sample_rate: int):
    """(frames read as marked, frames, each bit's summed message logit) of one window, as AudioSeal counts a reading."""
    import torch
    import torchaudio

    _, detector = _load_models()
    wav = torch.from_numpy(audio).float().T
    mono = wav.mean(dim=0, keepdim=True).unsqueeze(0)
    x = torchaudio.functional.resample(mono, sample_rate, _SAMPLE_RATE)
    with torch.inference_mode():
        if detector.normalizer is not None:
            x = detector.normalizer.loudness_normalization(x)
        out = detector.detector(x)
        marked = int((torch.softmax(out[:, :2, :], dim=1)[0, 1, :] > 0.5).sum())
        return marked, int(out.shape[-1]), out[0, 2:, :].double().sum(dim=-1).numpy()


def _decoded(bit_logits) -> tuple[int | None, bool]:
    """(clean codeword or None, corrected?) of a message whose bit is 1 where its summed logit is above zero."""
    raw = 0
    for value in bit_logits:
        raw = (raw << 1) | int(value > 0)
    decoded = decode_codeword(raw)
    if decoded is None:
        return None, False
    return decoded[1], decoded[2]


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


def issued_codewords() -> frozenset:
    """Every codeword on record as embedded before derived buckets: the frozen legacy list and every reset's IDs."""
    out = set(legacy_watermark_ids())
    for entry in reset_history():
        for value in [*entry.get("freed", []), *entry.get("kept_active", {}), *(c for v in entry.get("sources", {}).values() for c in v)]:
            try:
                out.add(int(value))
            except (TypeError, ValueError):
                continue
    return frozenset(out)


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
