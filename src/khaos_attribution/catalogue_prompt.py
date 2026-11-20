"""Every prompt said in the artist's catalogue words: the catalogue wins over what was asked.

`catalogue_prompt` holds the request to the catalogue (`catalogue_guard.hold`), then asks the translator for one
caption in the artist's own vocabulary, linted by `lint_catalogue`: no instrument, genre or voice the captions never
name, and at least `FITS_FLOOR` of its words from the captions. When the translator is absent, busy or fails, the
caption is stitched from two of the nearest captions of different tracks after the request's own catalogue words.
The listener's own words are never served as they are. Pure apart from the injected ``llm``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable

from khaos_attribution.caption_fit import fit, profile, tokens
from khaos_attribution.catalogue_guard import MIN_CORPUS_CAPTIONS, hold
from khaos_attribution.prompt_expansion import tidy_caption
from khaos_attribution.translation import (_DIGITS, _SOUND_ALIKE, _THEORY, DEFAULT_TIMEOUT_SEC, LINT_WORDS, MAX_ATTEMPTS,
                                           RETRY_SEED, RETRY_TEMPERATURE, TRANSLATION_WORDS, TranslatorUnavailable, _names,
                                           _WORD, clean_request, parse_answer, vocal_hits)

__all__ = ["CATALOGUE_PROMPT_VERSION", "FITS_FLOOR", "catalogue_prompt", "lint_catalogue", "stitched", "build_prompt"]

CATALOGUE_PROMPT_VERSION = "0.1.0"
# The share of a served prompt's words that must come from the artist's captions (the page's "Fits" meter).
FITS_FLOOR = 0.60
WORDS = TRANSLATION_WORDS

_SYSTEM = (
    "You write one description of a piece of music by this artist, in the words this artist's music is "
    "described in. You have NOT heard any recording. The artist's catalogue always wins over the request: "
    "name only instruments, genres, voices and production that the artist's descriptions name, and take most "
    "of your words from those descriptions. From the request keep only what fits this artist: its mood, energy "
    "and any instrument or genre the descriptions also name; leave out anything else it asks for, without "
    "saying so. Never name a song, artist, band, film or album, never compare to anyone, and never state "
    "tempo, BPM, key, time signature, duration, timestamps or numbers. Write in English. Answer with one line:\n"
    "Caption: <the description>"
)


def build_prompt(request: str, shots: list[str], instrumental: bool = True) -> str:
    """Qwen ChatML with thinking off: the catalogue-first rule, the artist's descriptions, the request."""
    vocals = ("Instrumental: do not mention vocals, voices, singing or words at all."
              if instrumental else "Sung: the music has vocals.")
    described = "\n".join(f"- {s}" for s in shots) or "- (none)"
    user = (f"How this artist's music is described:\n{described}\n\nRequest: {request.strip() or '(none)'}\n{vocals}\n\n"
            f"Write one flowing description of {WORDS[0]}-{WORDS[1]} words in this artist's words. /no_think")
    return (f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _fits(text: str, prof: dict) -> float:
    reading = fit(text, prof)["in_vocab_tokens"]
    return float(reading) if reading is not None else 0.0


def lint_catalogue(text: str, pool: list[str], *, instrumental: bool, prof: dict | None = None) -> list[str]:
    """Why ``text`` may not be served for this catalogue; empty when it may."""
    text = (text or "").strip()
    if not text:
        return ["empty"]
    reasons: list[str] = []
    outside = hold(text, pool)["dropped"]
    reasons += [f"outside:{d['word'].lower()}" for d in outside]
    prof = prof or profile(pool)
    share = _fits(text, prof)
    if share < FITS_FLOOR:
        reasons.append(f"fits:{share:.2f}")
    if instrumental and vocal_hits(text):
        reasons.append(f"vocal_on_instrumental:{vocal_hits(text)[0]}")
    for rx, name in ((_THEORY, "theory"), (_SOUND_ALIKE, "sound_alike")):
        hit = rx.search(text)
        if hit:
            reasons.append(f"{name}:{hit.group(0).strip().lower()}")
    if _DIGITS.search(text):
        reasons.append("number")
    names = _names(text, set(prof["counts"]))
    if names:
        reasons.append(f"name:{names[0]}")
    n = len(_WORD.findall(text))
    if not LINT_WORDS[0] <= n <= LINT_WORDS[1]:
        reasons.append(f"length:{n}_words")
    return reasons


_SENTENCE = re.compile(r"[^.!?]+[.!?]?")


def _first_sentence(text: str) -> str:
    m = _SENTENCE.search(" ".join(text.split()))
    return m.group(0).strip() if m else text.strip()


def stitched(request: str, ranked: list[dict], pool: list[str], prof: dict | None = None) -> str:
    """The request's own catalogue words, then the first sentence of each of the two nearest captions of different
    tracks: never one training caption as it is, and every word from the catalogue."""
    prof = prof or profile(pool)
    seen = set(prof["counts"])
    kept = [w for w in dict.fromkeys(tokens(request)) if w in seen][:6]
    picked, tracks = [], set()
    for entry in ranked:
        track = entry.get("track")
        if track in tracks and track is not None:
            continue
        picked.append(_first_sentence(entry["text"]))
        tracks.add(track)
        if len(picked) == 2:
            break
    lead = (", ".join(kept).capitalize() + ". ") if kept else ""
    return tidy_caption(lead + " ".join(picked))


def catalogue_prompt(request: str, *, pool: Iterable[str], ranked: list[dict], instrumental: bool, llm=None,
                     shots_per_attempt: int = 3, deadline: float | None = None, cancelled=lambda: False) -> dict:
    """``{caption, rewriter: translator | stitched, fits, dropped, attempts, rejects}`` for a request; ``ranked`` are
    the pool's exemplars (``{text, track}``), nearest first. Raises ValueError for a pool too small to hold."""
    pool = [p for p in pool if isinstance(p, str) and p.strip()]
    if len(pool) < MIN_CORPUS_CAPTIONS:
        raise ValueError(f"{len(pool)} captions are too few to say this artist's words")
    prof = profile(pool)
    held = hold(clean_request(request or "", instrumental), pool)
    asked = held["prompt"]
    rejects: list[dict] = []
    attempts = 0
    if llm is not None:
        deadline = time.monotonic() + DEFAULT_TIMEOUT_SEC if deadline is None else deadline
        for n in range(MAX_ATTEMPTS):
            if cancelled() or deadline - time.monotonic() <= 0.5:
                break
            window = ranked[n * shots_per_attempt:(n + 1) * shots_per_attempt] or ranked[:shots_per_attempt]
            temperature, seed = (0.0, None) if n == 0 else (RETRY_TEMPERATURE, RETRY_SEED + n)
            attempts += 1
            try:
                raw = llm(build_prompt(asked, [e["text"] for e in window], instrumental), temperature=temperature,
                          seed=seed, timeout=deadline - time.monotonic())
            except (TranslatorUnavailable, OSError, ValueError, RuntimeError) as exc:
                rejects.append({"attempt": n + 1, "reasons": [f"unavailable:{exc}"[:200]], "error": True})
                break
            caption = parse_answer(raw)["caption"]
            reasons = lint_catalogue(caption, pool, instrumental=instrumental, prof=prof)
            if not reasons:
                return {"caption": caption, "rewriter": "translator", "fits": round(_fits(caption, prof), 4),
                        "dropped": held["dropped"], "attempts": attempts, "rejects": rejects}
            rejects.append({"attempt": n + 1, "reasons": reasons, "text": caption[:400]})
    caption = stitched(asked, ranked, pool, prof)
    return {"caption": caption, "rewriter": "stitched", "fits": round(_fits(caption, prof), 4),
            "dropped": held["dropped"], "attempts": attempts, "rejects": rejects}
