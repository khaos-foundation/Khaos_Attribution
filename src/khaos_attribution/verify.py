"""Did Khaos make this audio? One verifier for the command line, both rooms and their agents.

It reads the file's Content Credentials, listens for the watermark and, when given a Register, asks its fingerprint
index. Origin is confirmed only by a content match in an index: a self-issued signature proves the record is
intact, not who made it, and a watermark alone can be a false positive or forged. A negative never proves a person
made the audio.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

VERIFY_VERSION = 2
FORMATS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".mp4")
WATERMARK_THRESHOLD = 0.5
REGISTER_TIMEOUT_S = 20.0
# A small compressed file can decode to hours of audio; nothing longer than this is decoded for a check.
MAX_VERIFY_SECONDS = 20 * 60
# The Guild Register anyone may ask; a verifier needs no enrolment to ask it.
REGISTER_PUBLIC_URL = "https://register.khaos.dev"

VERDICTS = {
    "confirmed": "Khaos origin confirmed: the audio matches a released Khaos output.",
    "signed": ("Carries an intact signed record that claims to come from Khaos. The signing machine is not yet "
               "certified, so this alone does not prove Khaos made it."),
    "tampered": "Carries a Khaos record whose signature no longer holds: the file was changed after it was signed.",
    "marked": "A Khaos watermark is present. A watermark alone does not prove Khaos made it.",
    "none": "No Khaos mark found. That is not proof a person made it: marks can be removed or never added.",
}
# The verdict in five words. DETECTED means a Khaos mark was found, never that Khaos is proven to have made the audio.
STATUSES = ("DETECTED", "NOT_DETECTED", "INCONCLUSIVE", "NOT_SUPPORTED", "ERROR")


class AudioUnreadable(ValueError):
    """The file is not audio this verifier can decode."""


def audio_seconds(path: Path | str) -> float | None:
    """The file's length from its header (soundfile, else ffprobe), or None when neither can tell."""
    import soundfile as sf  # noqa: PLC0415
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate if info.samplerate else None
    except Exception:  # noqa: BLE001 — containers soundfile cannot open are asked of ffprobe
        pass
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        proc = subprocess.run([ffprobe, "-v", "error", "-protocol_whitelist", "file", "-show_entries", "format=duration",
                               "-of", "default=nw=1:nk=1", f"file:{Path(path).resolve()}"],
                              capture_output=True, text=True, timeout=30)
        return float(proc.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def load_audio(path: Path | str, max_seconds: float | None = None):
    """(frames x channels float32, sample rate); WAV, FLAC, MP3 and OGG directly, M4A/AAC through FFmpeg.
    With max_seconds, a longer file is refused before it is decoded."""
    import soundfile as sf  # noqa: PLC0415
    path = Path(path)
    if max_seconds is not None:
        seconds = audio_seconds(path)
        if seconds is not None and seconds > max_seconds:
            raise AudioUnreadable(f"longer than {max_seconds / 60:g} minutes ({seconds / 60:.0f} min)")
    try:
        return sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as first:  # noqa: BLE001 — containers soundfile cannot open go through FFmpeg
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise AudioUnreadable(f"{path.suffix or 'this'} audio needs FFmpeg to decode ({first})") from None
        # Local files only, audio only, and never more than the cap decoded.
        cap = ["-t", str(int(max_seconds) + 1)] if max_seconds is not None else []
        with tempfile.TemporaryDirectory(prefix="khaos-verify-") as tmp:
            wav = Path(tmp) / "decoded.wav"
            try:
                proc = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-protocol_whitelist", "file",
                                       "-i", f"file:{path.resolve()}", "-vn", "-sn", "-dn", *cap, "-f", "wav", str(wav)],
                                      capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                raise AudioUnreadable(f"decoding {path.name} took too long") from None
            if proc.returncode != 0 or not wav.is_file():
                raise AudioUnreadable(f"could not decode {path.name}: {(proc.stderr or '').strip()[:200]}") from None
            audio, sample_rate = sf.read(str(wav), dtype="float32", always_2d=True)
        if max_seconds is not None and len(audio) > max_seconds * sample_rate:
            raise AudioUnreadable(f"longer than {max_seconds / 60:g} minutes")
        return audio, sample_rate


def read_credentials(path: Path | str) -> dict:
    """What the file's Content Credentials say, in the verifier's terms."""
    from khaos_attribution import credentials as C  # noqa: PLC0415
    got = C.read_file(Path(path))
    khaos = got.get("khaos") or {}
    return {"present": got["present"], "state": got["state"], "intact": got["state"] in C.SIGNED_STATES,
            "ai_generated": got["ai_generated"], "khaos": bool(khaos), "record": khaos or None,
            "signer": got["signer"], "timestamped": got["timestamped"], "trusted": got["trusted"],
            "checked": got.get("checked", True),
            **({"reason": got["reason"]} if got.get("reason") and not got["present"] else {}),
            **({"error": got["error"]} if got.get("error") else {})}


def _watermark(audio, sample_rate: int) -> dict:
    from khaos_attribution.watermark import Watermarker, decode_codeword  # noqa: PLC0415
    if not Watermarker.available():
        return {"checked": False, "reason": "the watermark detector (audioseal) is not installed"}
    if not len(audio):
        return {"checked": False, "reason": "the file holds no audio"}
    probability, codeword, corrected = Watermarker.detect_array(audio, sample_rate)
    detected = probability >= WATERMARK_THRESHOLD
    decoded = decode_codeword(codeword) if detected and codeword is not None else None
    return {"checked": True, "detected": detected, "confidence": round(float(probability), 3),
            "watermark_id": codeword if detected else None, "corrected": bool(corrected) if detected else False,
            "payload": decoded[0] if decoded else None}


def ask_register(url: str, audio, sample_rate: int, codeword: int | None, *, key: str | None = None) -> dict:
    """The Register's fingerprint index asked; an answer it could not give is an error in the answer."""
    from khaos_attribution.fingerprint import query_phases  # noqa: PLC0415
    phases = query_phases(audio, sample_rate)
    if not any(phases):
        return {"grade": None, "url": url, "error": "too short or too quiet to fingerprint"}
    body = json.dumps({"phases": [[list(p) for p in ph] for ph in phases], "watermark_id": codeword}).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-Khaos-Verify-Key"] = key
    req = urllib.request.Request(url.rstrip("/") + "/api/verify", data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=REGISTER_TIMEOUT_S) as resp:  # noqa: S310 — a configured https URL
            answer = json.loads(resp.read())
        if not isinstance(answer, dict):
            raise ValueError("the Register's answer is not a JSON object")
    except Exception as exc:  # noqa: BLE001
        return {"grade": None, "url": url, "error": str(exc)[:200]}
    answer["url"] = url
    return answer


def verdict(credentials: dict, watermark: dict, register: dict | None) -> str:
    if register and register.get("grade") == "output":
        return "confirmed"
    if credentials.get("khaos") and credentials.get("intact"):
        return "signed"
    if credentials.get("khaos") and credentials.get("present"):
        return "tampered"
    if watermark.get("detected"):
        return "marked"
    return "none"


def status(kind: str, credentials: dict, watermark: dict, register: dict | None) -> str:
    """The verdict in one of STATUSES; first match wins, and a negative needs every check to have run."""
    if kind == "confirmed":
        return "DETECTED"
    if credentials.get("error"):
        return "ERROR"
    if kind == "signed":
        return "INCONCLUSIVE" if watermark.get("matches_record") is False else "DETECTED"
    if kind in ("tampered", "marked"):
        return "INCONCLUSIVE"
    if not credentials.get("checked", True):
        return "NOT_SUPPORTED"
    if not watermark.get("checked") or (register is not None and register.get("grade") is None):
        return "INCONCLUSIVE"
    return "NOT_DETECTED"


def message(kind: str, credentials: dict) -> str:
    """The verdict in words; a signed record's AI-generated statement is said only when the record makes it."""
    text = VERDICTS[kind]
    if kind == "signed" and credentials.get("ai_generated"):
        text = text.replace("claims to come from Khaos.", "claims to come from Khaos and says the audio is AI-generated.")
    return text


def verify_file(path: Path | str, *, register_url: str | None = None, register_key: str | None = None,
                watermark: bool = True) -> dict:
    """Every check this machine can make on one file, and the verdict they support."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    creds = read_credentials(path)
    audio, sample_rate = load_audio(path, max_seconds=MAX_VERIFY_SECONDS)
    mark = _watermark(audio, sample_rate) if watermark else {"checked": False, "reason": "not asked"}
    register = (ask_register(register_url, audio, sample_rate, mark.get("watermark_id"), key=register_key)
                if register_url else None)
    record = creds.get("record") or {}
    if mark.get("detected") and record.get("watermark_id") is not None:
        mark["matches_record"] = mark.get("watermark_id") == record.get("watermark_id")
    kind = verdict(creds, mark, register)
    return {"verify_version": VERIFY_VERSION, "file": path.name, "verdict": kind,
            "status": status(kind, creds, mark, register), "message": message(kind, creds),
            "content_credentials": creds, "watermark": mark, "register": register}


def _line(result: dict) -> str:
    creds, mark, reg = result["content_credentials"], result["watermark"], result["register"]
    parts = [f"{result['file']}: {result['message']}", f"  Status: {result['status']}"]
    if creds["present"]:
        rec = creds.get("record") or {}
        who = rec.get("artist_name") or rec.get("artist_id")
        parts.append(f"  Content Credentials: {'intact' if creds['intact'] else 'BROKEN'}"
                     f"{', AI-generated' if creds['ai_generated'] else ''}"
                     f"{', time-stamped' if creds['timestamped'] else ''}"
                     f"{f', artist {who}' if who else ''}"
                     f"{', model ' + str(rec['adapter_version']) if rec.get('adapter_version') else ''}")
    else:
        parts.append("  Content Credentials: none")
    if mark.get("checked"):
        parts.append(f"  Watermark: {'present' if mark['detected'] else 'not found'} (confidence {mark['confidence']})"
                     + (f", ID {mark['watermark_id']}" if mark.get("watermark_id") is not None else "")
                     + ({True: ", matches the record", False: ", DOES NOT match the record"}
                        .get(mark.get("matches_record"), "")))
    else:
        parts.append(f"  Watermark: not checked ({mark.get('reason')})")
    if reg is not None:
        if reg.get("grade") == "output":
            model = reg.get("model") or {}
            parts.append(f"  Register: matches a released output of {model.get('artist_name') or model.get('artist_id')}"
                         + (f", {model['page']}" if model.get("page") else ""))
        elif reg.get("grade") is None:
            parts.append(f"  Register: could not be asked ({reg.get('error')})")
        else:
            parts.append(f"  Register: no released output matches ({reg.get('grade')})")
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="khaos-verify", description="Check whether Khaos made an audio file. "
                                 "Origin is confirmed only by a match in the Register's index.")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--register", default=os.environ.get("KHAOS_REGISTER_URL") or None,
                    help="the Register to ask (default KHAOS_REGISTER_URL; none asks nothing beyond this machine)")
    ap.add_argument("--key", default=os.environ.get("KHAOS_VERIFY_KEY") or None,
                    help="a verify key for the Register's unlimited lane (authorities, researchers, media)")
    ap.add_argument("--no-watermark", action="store_true", help="skip the watermark detector")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)
    results, failed = [], False
    for f in args.files:
        try:
            results.append(verify_file(f, register_url=args.register, register_key=args.key,
                                       watermark=not args.no_watermark))
        except Exception as exc:  # noqa: BLE001 — one bad file never stops the rest
            failed = True
            results.append({"file": str(f), "status": "ERROR", "error": str(exc) or type(exc).__name__})
    for r in results:
        if args.json:
            continue
        print(f"{r['file']}: {r['error']}" if "error" in r else _line(r))
    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=1))
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
