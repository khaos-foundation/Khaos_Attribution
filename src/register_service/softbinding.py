"""The C2PA Soft Binding Resolution API on the register: from a piece's audio, or its fingerprint, to the Content
Credentials manifest it was released with, for files whose own record was stripped.

A match is always by fingerprint. A Khaos watermark names a group of models, never one piece, so it is accepted only
as the hint that narrows a content search, never as a binding on its own.
"""

from __future__ import annotations

import base64
import binascii
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

from khaos_attribution.credentials import WATERMARK_ALG
from khaos_attribution.watermark import decode_codeword
from register_service.index import phases_of

PREFIX = "/c2pa/v1"
FINGERPRINT_ALG = "dev.khaos.landmarks.1"
# The register's front door (Caddy) takes 10 MB a request: a minute of WAV, ten of MP3, and a short clip is enough.
CONTENT_MAX_BYTES = 10_000_000
# A clip is enough to name a piece, and the peak finder holds several times the decoded audio in memory.
CONTENT_MAX_SECONDS = 120
# A small file can still describe hours of many-channel audio; what is decoded is bounded, whatever the file claims.
DECODE_BUDGET_BYTES = 200_000_000
MAX_RATE = 96_000
MAX_CHANNELS = 2
_decoding = threading.BoundedSemaphore(2)
BINDING_MAX_CHARS = 8_000_000
LIMITS = {f"{PREFIX}/matches/byContent": 10, f"{PREFIX}/matches/byBinding": 30, f"{PREFIX}/manifests": 120}
WATERMARK_NOT_A_BINDING = ("a Khaos watermark names a group of models, not one piece; send the audio to "
                           "matches/byContent with this watermark as hintAlg and hintValue")


def limit_key(path: str) -> str | None:
    """The rate-limit bucket a soft-binding path falls in, or None for a path outside the API."""
    if path.startswith(f"{PREFIX}/manifests/"):
        return f"{PREFIX}/manifests"
    return path if path in LIMITS else None


def _phases(value: str) -> list:
    """A fingerprint binding's value: base64 of the JSON list of four phases (or one fingerprint) of [hash, offset_ms]."""
    if len(value) > BINDING_MAX_CHARS:
        raise HTTPException(414, "the binding is too long for a URL; POST it to matches/byBinding")
    try:
        data = json.loads(base64.b64decode(value, validate=True))
    except (binascii.Error, ValueError):
        raise HTTPException(400, "value is not base64 of a JSON fingerprint") from None
    if isinstance(data, list) and data and all(isinstance(x, list) and len(x) == 2 and not isinstance(x[0], list) for x in data):
        data = [data]
    try:
        return phases_of(data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


def _hint_payload(hint_alg: str | None, hint_value: str | None) -> int | None:
    if not hint_alg and not hint_value:
        return None
    if hint_alg != WATERMARK_ALG or hint_value is None:
        raise HTTPException(400, f"the only hint is {WATERMARK_ALG}, with the watermark ID in decimal as its value")
    try:
        decoded = decode_codeword(int(hint_value)) if hint_value.isdigit() else None
    except ValueError:
        decoded = None
    if decoded is None:
        raise HTTPException(400, "hintValue is not a Khaos watermark ID")
    return decoded[0]


def decode(path: Path):
    """(audio, sample rate) of an upload, within the decode budget; a format soundfile cannot open goes through FFmpeg
    as mono 22.05 kHz, local file only."""
    import soundfile as sf  # noqa: PLC0415
    try:
        info = sf.info(str(path))
    except Exception:  # noqa: BLE001 — not a format soundfile reads
        info = None
    if info is not None:
        if info.samplerate > MAX_RATE or info.channels > MAX_CHANNELS:
            raise HTTPException(415, f"at most {MAX_RATE // 1000} kHz and {MAX_CHANNELS} channels")
        frames = min(CONTENT_MAX_SECONDS * info.samplerate, DECODE_BUDGET_BYTES // (4 * info.channels))
        audio, sr = sf.read(str(path), frames=frames, dtype="float32", always_2d=True)
        return audio, sr
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise HTTPException(415, "send WAV, FLAC, MP3 or Ogg audio")
    out = path.with_name("decoded.wav")
    try:
        proc = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-protocol_whitelist", "file", "-i", f"file:{path}",
                               "-vn", "-sn", "-dn", "-ac", "1", "-ar", "22050", "-t", str(CONTENT_MAX_SECONDS),
                               "-f", "wav", str(out)], capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise HTTPException(415, "the audio took too long to decode") from None
    if proc.returncode != 0 or not out.is_file():
        raise HTTPException(415, "the audio could not be read")
    audio, sr = sf.read(str(out), dtype="float32", always_2d=True)
    return audio, sr


def _max_results(n: int | None) -> int:
    if n is not None and n < 1:
        raise HTTPException(400, "maxResults is at least 1")
    return n or 10


def add_routes(app: FastAPI, index, public_url: str) -> None:
    endpoint = public_url.rstrip("/") + PREFIX

    def matches(phases: list, payload: int | None, max_results: int) -> dict:
        """The released piece the fingerprint names, when its manifest is held; never a guess."""
        answer = index.verify(phases, payload)
        if answer["grade"] != "output" or max_results < 1:
            return {"matches": []}
        gid = answer["output"]["generation_id"]
        if not index.has_manifest(gid) or not index.offered(gid):
            return {"matches": []}
        query = max(1, len(phases[0]) if phases else 1)
        return {"matches": [{"manifestId": gid, "endpoint": endpoint,
                             "similarityScore": max(0, min(100, round(100 * answer["votes"] / query)))}]}

    def by_binding(alg: str | None, value: str | None, max_results: int) -> dict:
        if alg == WATERMARK_ALG:
            raise HTTPException(400, WATERMARK_NOT_A_BINDING)
        if alg != FINGERPRINT_ALG:
            raise HTTPException(400, f"unknown alg; supported: {FINGERPRINT_ALG}")
        if not value:
            raise HTTPException(400, "value is required")
        return matches(_phases(value), None, max_results)

    @app.get(f"{PREFIX}/services/supportedAlgorithms")
    async def supported():
        return {"watermarks": [{"alg": WATERMARK_ALG}], "fingerprints": [{"alg": FINGERPRINT_ALG}]}

    @app.get(f"{PREFIX}/matches/byBinding")
    async def query_by_binding(alg: str | None = None, value: str | None = None, maxResults: int | None = None):  # noqa: N803
        n = _max_results(maxResults)
        return await run_in_threadpool(by_binding, alg, value, n)

    @app.post(f"{PREFIX}/matches/byBinding")
    async def query_by_large_binding(request: Request, maxResults: int | None = None):  # noqa: N803
        n = _max_results(maxResults)
        raw = await request.body()
        if len(raw) > BINDING_MAX_CHARS + 1000:
            raise HTTPException(413, "too large")
        try:
            body = json.loads(raw)
        except ValueError:
            raise HTTPException(400, "not JSON") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "the body is {alg, value}")
        return await run_in_threadpool(by_binding, body.get("alg"), body.get("value"), n)

    @app.post(f"{PREFIX}/matches/byContent")
    async def query_by_content(request: Request, alg: str | None = None, maxResults: int | None = None,  # noqa: N803
                               hintAlg: str | None = None, hintValue: str | None = None):  # noqa: N803
        n = _max_results(maxResults)
        if alg not in (None, FINGERPRINT_ALG):
            raise HTTPException(400, f"content is matched by {FINGERPRINT_ALG} only")
        if not request.headers.get("content-type", "").lower().startswith("audio/"):
            raise HTTPException(415, "only audio is matched here")
        payload = _hint_payload(hintAlg, hintValue)
        raw = await request.body()
        if not raw:
            raise HTTPException(400, "no content")
        if len(raw) > CONTENT_MAX_BYTES:
            raise HTTPException(413, f"at most {CONTENT_MAX_BYTES // 1_000_000} MB")

        def work():
            from khaos_attribution.fingerprint import query_phases  # noqa: PLC0415
            with _decoding, tempfile.TemporaryDirectory(prefix="register-softbinding-") as tmp:
                path = Path(tmp) / "content"
                path.write_bytes(raw)
                audio, sr = decode(path)
                phases = query_phases(audio, sr)
            return matches(phases, payload, n)

        return await run_in_threadpool(work)

    @app.get(f"{PREFIX}/manifests/{{manifest_id}}")
    async def get_manifest(manifest_id: str, returnActiveManifest: bool = False):  # noqa: N803, ARG001
        # One manifest store is held per piece and its active manifest is the piece's own, so both answers are it.
        if not manifest_id or len(manifest_id) > 120:
            raise HTTPException(400, "manifestId is a short string")
        held = await run_in_threadpool(index.manifest, manifest_id)
        if held is None:
            raise HTTPException(404, "no manifest is held for that id")
        if not await run_in_threadpool(index.offered, manifest_id):
            raise HTTPException(410, "this piece is no longer offered: its model was withdrawn or its maker took it down")
        return Response(content=held, media_type="application/c2pa")
