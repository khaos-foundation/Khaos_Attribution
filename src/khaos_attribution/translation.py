"""Expand's translator: a listener's request restated in the words this artist's music is described in.

A local instruction model rewrites TEXT only; it never hears audio. Its answer is linted in code
(claim nouns from the request or the artist's caption pool, nothing the request denies, no vocals on
an instrumental, no names, numbers, theory or clock times) and the same request check holds the 5Hz LM
to the request when the translator is not there. The contract loads no model: the rooms inject an
``llm(prompt, temperature=, seed=)`` callable, or reach llama.cpp's server at ``KHAOS_TRANSLATOR_URL``.
When nothing passes, the listener's own request is the caption, said so, never an invented one.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

from khaos_attribution.caption_fit import answered, tokens
from khaos_attribution.caption_vocabulary import content_nouns, generalises
from khaos_attribution.prompt_expansion import MIN_CAPTION_CHARS, tidy_caption

__all__ = ["TRANSLATOR_URL_ENV", "TRANSLATOR_TIMEOUT_ENV", "MODEL_REPO", "MODEL_FILE", "MODEL_REVISION",
           "DEFAULT_LABEL", "MAX_ATTEMPTS", "ANSWERED_FLOOR", "FALLBACK_NOTE", "TranslatorUnavailable",
           "build_prompt", "extract_answer", "check_caption", "lint_translation", "translate",
           "expand_checked", "request_fallback", "model_label", "http_llm", "translator_from_env"]

TRANSLATOR_URL_ENV = "KHAOS_TRANSLATOR_URL"
TRANSLATOR_TIMEOUT_ENV = "KHAOS_TRANSLATOR_TIMEOUT"

# The model both rooms run; the Workshop's caption rewrite stage pins the same file.
MODEL_REPO = "Qwen/Qwen3-8B-GGUF"
MODEL_FILE = "Qwen3-8B-Q4_K_M.gguf"
MODEL_REVISION = "7c41481f57cb95916b40956ab2f0b139b296d974"
DEFAULT_LABEL = "qwen3-8b"

# One try at temperature 0, then two seeded, warmer tries on the next windows; more only spends the listener's time.
MAX_ATTEMPTS = 3
RETRY_TEMPERATURE = 0.7
RETRY_SEED = 1000
MAX_TOKENS = 160
# Shorter than a training caption: past this, a restatement can only grow by inventing.
TRANSLATION_WORDS = (25, 55)

# Seconds for one translator call: on six CPU cores one call took up to 14 s; overridden per machine by env.
DEFAULT_TIMEOUT_SEC = 20.0

# Every 5Hz LM answer that changed the request kept at most 0.4 of its words; a restatement keeps at least half.
ANSWERED_FLOOR = 0.5
# Under two content words there is too little to measure a share of.
FLOOR_MIN_WORDS = 2

FALLBACK_NOTE = ("Could not say this in the artist's words without changing what you asked; "
                 "the nearest descriptions are below.")

# An instrumental request names no voice, so any of these in its caption is an invented vocal.
VOCAL_WORDS = frozenset("""
vocal vocals vocalist vocalists vocalise vocalised vocalize vocalized voice voices voiced
sing sings singing singer singers sung song-like lyric lyrics spoken speaking speech narration
narrator narrated rap rapping rapper rappers choir choirs chant chants chanting hum hums humming
hummed whisper whispers whispered whispering falsetto croon crooning crooned acapella
""".split())

_CLOCK_TIME = re.compile(
    r"\b\d{1,2}:\d{2}\b"
    r"|\b[\w-]+[- ](?:second|minute)s?[- ]mark\b"
    r"|\b(?:at|after|around|by|until|from|before)\s+(?:the\s+|about\s+)?(?:\d+|one|two|three|four|five|"
    r"ten|thirty|forty|sixty|ninety)[- ](?:seconds?|minutes?)\b"
    r"|\b(?:the\s+)?(?:first|last|final|opening|closing)\s+(?:\d+\s+|few\s+)?(?:seconds?|minutes?)\b",
    re.IGNORECASE)
_KEY = re.compile(r"\b[A-G](?:#|♯|b|♭)?\s*(?:major|minor|maj|min)\b|\btime signature\b|\b\d+/\d+\b")
_TEMPO = re.compile(r"\b(?:bpm|beats per minute)\b", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
_SOUND_ALIKE = re.compile(r"\b(?:in the style of|reminiscent of|sounds? like|à la|inspired by|in the vein of)\b",
                          re.IGNORECASE)
_DENIAL = re.compile(r"\b(?:no|not|without|never|zero|free of|absent|lacking|minus)\b[^.;,!?]*", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|['’-])*")
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)$")


class TranslatorUnavailable(RuntimeError):
    """The translator did not answer (unreachable, failing or too slow); the caller falls back to the 5Hz LM."""


# ── the prompt ───────────────────────────────────────────────────────────

_SYSTEM = (
    "You restate a listener's request for music in the words this artist's music is described in. "
    "You have NOT heard any recording. Keep everything the request asks for and nothing it does not: "
    "never add an instrument, genre, vocal, effect or mood the request does not imply, never drop an "
    "instrument, genre or mood it names, and never contradict what it rules out. The artist's "
    "descriptions are vocabulary, not content: borrow their words for textures, moods and production "
    "only where they say what the request asks, and never copy their instruments or genres unless the "
    "request names them. Keep the request's own words for what it names. Never name a song, artist, "
    "band, film or album, never compare to anyone, and never state tempo, BPM, key, time signature, "
    "duration, timestamps or numbers. Output the caption only."
)


def build_prompt(request: str, shots: list[str], instrumental: bool = True) -> str:
    """Qwen ChatML with thinking off: the system rule, then the request, its vocal stance and the descriptions."""
    vocals = ("Instrumental: do not mention vocals, voices, singing or words at all."
              if instrumental else "Sung: the music has vocals.")
    described = "\n".join(f"- {s}" for s in shots) or "- (none)"
    user = (f"How this artist's music is described (vocabulary only):\n{described}\n\n"
            f"Request: {request.strip()}\n{vocals}\n\n"
            f"Restate the request as one flowing caption of {TRANSLATION_WORDS[0]}-{TRANSLATION_WORDS[1]} words "
            "that begins with the request's own words and keeps every instrument, genre and mood it names. /no_think")
    return (f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
_CHATML_TAIL = re.compile(r"<\|im_end\|>.*$", re.DOTALL)
_PREAMBLE = re.compile(r"^(?:caption|here(?:'s| is)[^:]*|restated[^:]*)\s*:\s*", re.IGNORECASE)


def extract_answer(raw: str) -> str:
    """The caption from a raw completion: thinking and chat tails removed, the longest line, tidied."""
    text = _THINK.sub("", (raw or "").replace("[end of text]", ""))
    text = _CHATML_TAIL.sub("", text).strip()
    lines = [_PREAMBLE.sub("", ln.strip()) for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.endswith(":")]
    return tidy_caption(max(lines, key=len)) if lines else ""


# ── the checks ───────────────────────────────────────────────────────────

def _asserted(text: str) -> str:
    """The text with denied phrases ("no drums", "without vocals") removed: what it claims is there."""
    return _DENIAL.sub(" ", text or "")


def _words(text: str) -> list[str]:
    """Lower-case words with hyphenated compounds split, so "spoken-word" reads as "spoken"."""
    text = unicodedata.normalize("NFC", text or "").replace("-", " ")
    return [w.lower() for w in _WORD.findall(text)]


def check_caption(request: str, caption: str, *, instrumental: bool) -> list[str]:
    """Why a caption does not answer the request, empty when it does: a vocal on an instrumental, a
    clock time, or under ``ANSWERED_FLOOR`` of a request of at least two content words."""
    reasons: list[str] = []
    if not (caption or "").strip():
        return ["empty"]
    if instrumental:
        sung = sorted({w for w in _words(_asserted(caption)) if w in VOCAL_WORDS})
        if sung:
            reasons.append(f"vocal_on_instrumental:{sung[0]}")
    clock = _CLOCK_TIME.search(caption)
    if clock:
        reasons.append(f"clock_time:{clock.group(0).strip()}")
    asked = set(tokens(request))
    if len(asked) >= FLOOR_MIN_WORDS:
        share = answered(request, caption)["answered"]
        if share is not None and share < ANSWERED_FLOOR:
            reasons.append(f"answered:{share:.2f}")
    return reasons


def _nouns(text: str) -> set[str]:
    """Claim nouns read with hyphens both kept and split, so "synth-pop" and "synth pop" are the same claim."""
    return content_nouns(text) | content_nouns(text.replace("-", " "))


def _singular(noun: str) -> str:
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in noun.split())


def _allowed(noun: str, allowed: set[str]) -> bool:
    """`generalises`, reading "string" and "strings" as one noun."""
    return generalises(noun, allowed) or generalises(_singular(noun), {_singular(a) for a in allowed})


# A denied family rules out its members: "no drums" rules out the kit, the machine and the percussion.
_FAMILIES = {"drum": ("drum", "percussion", "kick", "snare", "hi-hat", "hihat", "cymbal", "tom", "breakbeat",
                      "drum machine", "drum kit", "drum loop", "808"),
             "vocal": tuple(sorted(VOCAL_WORDS))}


def _denies(noun: str, denied: set[str]) -> bool:
    words = set(_singular(noun).split())
    for d in denied:
        d1 = _singular(d)
        if noun == d or d1 in words or _singular(noun) in d1.split():
            return True
        for family, members in _FAMILIES.items():
            if family in d1.split() and (words & {_singular(m) for m in members} or noun in members):
                return True
    return False


def _capitalised_mid_sentence(text: str) -> list[str]:
    out = []
    for m in _WORD.finditer(text):
        word = m.group(0)
        if word[:1].isupper() and not _SENTENCE_START.search(text[:m.start()]):
            out.append(word)
    return out


def lint_translation(request: str, output: str, *, instrumental: bool,
                     pool: Iterable[str]) -> tuple[str | None, list[str]]:
    """(caption, []) when the translation passes, else (None, reasons): the request check plus the claim
    nouns, denials, names, numbers, theory and length."""
    text = (output or "").strip()
    if not text:
        return None, ["empty"]
    reasons = check_caption(request, text, instrumental=instrumental)
    # The engine reads "no vocals" as a word about vocals: an instrumental translation names none at all.
    said = sorted({w for w in _words(text) if w in VOCAL_WORDS}) if instrumental else []
    if said and not any(r.startswith("vocal_on_instrumental") for r in reasons):
        reasons.append(f"vocal_on_instrumental:{said[0]}")
    pool_text = " ".join(pool)
    allowed = content_nouns(request) | content_nouns(pool_text)
    claimed = _nouns(_asserted(text))
    for noun in sorted(claimed):
        if not _allowed(noun, allowed):
            reasons.append(f"invented_noun:{noun}")
    denied = _nouns(request) - _nouns(_asserted(request))
    for noun in sorted(claimed):
        if _denies(noun, denied):
            reasons.append(f"denied:{noun}")
    request_digits = set(_DIGITS.findall(request))
    if _TEMPO.search(text) or _KEY.search(text):
        reasons.append("theory")
    stray = sorted(set(_DIGITS.findall(text)) - request_digits)
    if stray:
        reasons.append(f"number:{stray[0]}")
    if _SOUND_ALIKE.search(text):
        reasons.append("sound_alike")
    known = set(_words(request)) | set(_words(pool_text))
    names = [w for w in _capitalised_mid_sentence(text) if w.lower() not in known]
    if names:
        reasons.append(f"name:{names[0]}")
    if len(text) < MIN_CAPTION_CHARS:
        reasons.append(f"length:{len(text)}")
    return (None, reasons) if reasons else (text, [])


# ── running it ───────────────────────────────────────────────────────────

LLM = Callable[..., str]


def translate(request: str, shots: list[str] | Callable[[int], list[str]], *, instrumental: bool,
              pool: Iterable[str], llm: LLM, label: str = DEFAULT_LABEL) -> dict:
    """Up to ``MAX_ATTEMPTS`` linted translations, try ``n`` on ``shots(n)`` when it is a callable:
    ``{caption | None, attempts, rejects}``. Raises `TranslatorUnavailable` when the model never answers."""
    pool = list(pool)
    rejects: list[dict] = []
    for n in range(MAX_ATTEMPTS):
        prompt = build_prompt(request, shots(n) if callable(shots) else shots, instrumental)
        temperature, seed = (0.0, None) if n == 0 else (RETRY_TEMPERATURE, RETRY_SEED + n)
        try:
            raw = llm(prompt, temperature=temperature, seed=seed)
        except TranslatorUnavailable:
            if not n:
                raise
            break
        except Exception as exc:  # noqa: BLE001 — any backend failure is the same fallback
            if not n:
                raise TranslatorUnavailable(f"{type(exc).__name__}: {exc}") from exc
            break
        text, reasons = lint_translation(request, extract_answer(raw), instrumental=instrumental, pool=pool)
        if text is not None:
            return {"caption": text, "attempts": n + 1, "rejects": rejects}
        rejects.append({"by": label, "attempt": n + 1, "reasons": reasons})
    return {"caption": None, "attempts": len(rejects), "rejects": rejects}


_EMPTY_HARVEST = {"bpm": None, "keyscale": None, "timesignature": None, "duration": None,
                  "lyrics": None, "language": None}


def request_fallback(request: str, attempts: int, rejects: list[dict]) -> dict:
    """The listener's own words as the caption, said so: never an invented caption."""
    return {**_EMPTY_HARVEST, "caption": " ".join(str(request).split()), "translator": "request",
            "fallback": "request", "fallback_note": FALLBACK_NOTE, "attempts": attempts, "rejects": rejects}


def expand_checked(request: str, *, instrumental: bool, attempt: int, pool: Iterable[str],
                   shots: Callable[[int], list[str]], translator: LLM | None = None,
                   translator_label: str = DEFAULT_LABEL,
                   lm: Callable[[int], dict] | None = None) -> dict:
    """Translator first; when it does not answer, the 5Hz LM (``lm(attempt)`` → a harvested answer) held to
    `check_caption`; else the request itself. ``served_attempt`` names the window that served."""
    pool = list(pool)
    attempts, rejects = 0, []
    if translator is not None:
        try:
            out = translate(request, lambda n: shots(attempt + n), instrumental=instrumental, pool=pool,
                            llm=translator, label=translator_label)
        except TranslatorUnavailable as exc:
            rejects.append({"by": translator_label, "attempt": 1, "reasons": [f"unavailable:{exc}"[:200]]})
        else:
            attempts, rejects = out["attempts"], out["rejects"]
            # A server names its model only once it has answered.
            translator_label = getattr(translator, "label", None) or translator_label
            if out["caption"] is None:
                return {**request_fallback(request, attempts, rejects), "served_attempt": attempt}
            return {**_EMPTY_HARVEST, "caption": out["caption"], "translator": translator_label,
                    "fallback": None, "fallback_note": None, "attempts": attempts, "rejects": rejects,
                    "served_attempt": attempt}
    if lm is not None:
        for n in range(MAX_ATTEMPTS):
            result = dict(lm(attempt + n))
            attempts += 1
            reasons = check_caption(request, result.get("caption") or "", instrumental=instrumental)
            if not reasons:
                result.update(translator="lm", fallback=None, fallback_note=None, attempts=attempts,
                              rejects=rejects, served_attempt=attempt + n)
                return result
            rejects.append({"by": "lm", "attempt": n + 1, "reasons": reasons})
    return {**request_fallback(request, attempts, rejects), "served_attempt": attempt}


# ── llama.cpp's server ───────────────────────────────────────────────────

def model_label(name: str | None) -> str:
    """``qwen3-8b`` from ``Qwen3-8B-Q4_K_M.gguf`` or a path to it; the default when nothing is named."""
    stem = Path(str(name or "")).name
    stem = re.sub(r"\.gguf$", "", stem, flags=re.IGNORECASE).lower()
    stem = re.sub(r"[-_.](?:q\d|iq\d|f16|bf16|f32).*$", "", stem)
    return stem or DEFAULT_LABEL


_HEALTH: dict[str, tuple[float, bool]] = {}
_HEALTH_TTL = 30.0


def _healthy(url: str, timeout: float) -> bool:
    """llama-server's /health, remembered for a short while so a down server costs one probe, not one per Expand."""
    now = time.monotonic()
    cached = _HEALTH.get(url)
    if cached and now - cached[0] < _HEALTH_TTL:
        return cached[1]
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=min(timeout, 2.0)) as answer:
            ok = answer.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        ok = False
    _HEALTH[url] = (now, ok)
    return ok


def http_llm(url: str, *, timeout: float = DEFAULT_TIMEOUT_SEC, max_tokens: int = MAX_TOKENS) -> LLM:
    """A callable for llama.cpp's ``POST /completion``; every failure is `TranslatorUnavailable`."""
    base = url.rstrip("/")

    def run(prompt: str, temperature: float = 0.0, seed: int | None = None) -> str:
        if not _healthy(base, timeout):
            raise TranslatorUnavailable(f"{base} is not answering")
        body = {"prompt": prompt, "n_predict": int(max_tokens), "temperature": float(temperature),
                "cache_prompt": True, "stop": ["<|im_end|>"]}
        if seed is not None:
            body["seed"] = int(seed)
        request = urllib.request.Request(f"{base}/completion", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as answer:
                doc = json.loads(answer.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            _HEALTH.pop(base, None)
            raise TranslatorUnavailable(f"{type(exc).__name__}: {exc}") from exc
        content = doc.get("content") if isinstance(doc, dict) else None
        if not isinstance(content, str):
            raise TranslatorUnavailable("the server's answer has no content")
        run.label = model_label(doc.get("model") or doc.get("generation_settings", {}).get("model"))
        return content

    run.label = DEFAULT_LABEL
    return run


def translator_from_env(environ: dict | None = None) -> LLM | None:
    """The server at ``KHAOS_TRANSLATOR_URL`` as a callable, or None when the variable is unset."""
    env = os.environ if environ is None else environ
    url = (env.get(TRANSLATOR_URL_ENV) or "").strip()
    if not url:
        return None
    try:
        timeout = float(env.get(TRANSLATOR_TIMEOUT_ENV) or DEFAULT_TIMEOUT_SEC)
    except ValueError:
        timeout = DEFAULT_TIMEOUT_SEC
    return http_llm(url, timeout=timeout)
