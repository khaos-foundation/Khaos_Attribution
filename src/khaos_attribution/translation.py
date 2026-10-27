"""Expand's translator: a listener's request restated in the words this artist's music is described in.

A local instruction model rewrites TEXT only; it never hears audio. Its answer is linted in code on the
body it wrote (a copy of the request counts for nothing): what it names must come from the request, the
artist's pool lends only texture, production and mood words, nothing the request asks for may be dropped,
denied or turned into its opposite, and no voice, name, number, theory or clock time is added. The same
request check holds the 5Hz LM when the translator is not there. The contract loads no model: the rooms
inject ``llm(prompt, temperature=, seed=, timeout=)``. When nothing passes, the request is the caption.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

from khaos_attribution.caption_fit import answered, tokens
from khaos_attribution.caption_vocabulary import (GENRES, INSTRUMENTS, MOOD_ADJECTIVES, TIMBRE_ADJECTIVES,
                                                  VOCAL_TERMS, extract_attributes, generalises)
from khaos_attribution.prompt_expansion import MIN_CAPTION_CHARS, tidy_caption

__all__ = ["TRANSLATOR_URL_ENV", "TRANSLATOR_TIMEOUT_ENV", "MODEL_REPO", "MODEL_FILE", "MODEL_REVISION",
           "DEFAULT_LABEL", "MAX_ATTEMPTS", "ANSWERED_FLOOR", "FALLBACK_NOTE", "TranslatorUnavailable",
           "build_prompt", "parse_answer", "extract_answer", "clean_request", "split_denials", "vocal_hits",
           "check_caption", "lint_translation", "translate", "expand_checked", "request_fallback",
           "model_label", "http_llm", "translator_from_env", "deadline_seconds"]

TRANSLATOR_URL_ENV = "KHAOS_TRANSLATOR_URL"
# Seconds one Expand may spend in total; every call gets what is left.
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
MAX_TOKENS = 200
# Shorter than a training caption: past this, a restatement can only grow by inventing.
TRANSLATION_WORDS = (25, 55)
# The lint's band around the asked one: the model drifts a little either way.
LINT_WORDS = (15, 75)
# Scripts written without spaces are measured in characters.
LINT_CHARS_UNSPACED = (20, 200)

# One Expand's whole budget: on six CPU cores one call took up to 14 s, and most pass on the first.
DEFAULT_TIMEOUT_SEC = 20.0
# Requests sent to one server at once (compose: --parallel 1): on CPU two at once took 20 s each, one after the other 6 and 11.
SERVER_PARALLEL = 1
MAX_RESPONSE_BYTES = 1 << 20

# Every 5Hz LM answer that changed the request kept at most 0.4 of its words; a restatement keeps at least half.
ANSWERED_FLOOR = 0.5
# Under two content words there is too little to measure a share of.
FLOOR_MIN_WORDS = 2
# A translation adds the artist's words; under six of its own beside the request it only repeats it.
MIN_NEW_WORDS = 6

FALLBACK_NOTE = ("Could not say this in the artist's words without changing what you asked; "
                 "the nearest descriptions are below.")
TRANSLATED_NOTE = "Said in English from your {language} request."
LYRICS_NOTE = "No lyrics draft: the translator writes the description only."
BUSY_NOTE = "The language models were busy or not answering, so your own words are kept; try again in a moment."


class TranslatorUnavailable(RuntimeError):
    """The translator did not answer (unreachable, failing, busy or too slow); the caller falls back."""

# ── words ────────────────────────────────────────────────────────────────

_WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|['’])*")
_POSSESSIVE = re.compile(r"['’]s\b")

# Any of these, in any form, is a voice: on an instrumental request it is an invented vocal.
_VOCAL = re.compile(
    r"\b(?:vocal\w*|sing|sings|singing|singer\w*|singalong|sung|voice\w*|vox|choir\w*|choral\w*|chant\w*|"
    r"rap|raps|rapp\w*|hum|hums|humm\w*|croon\w*|whisper\w*|lyric\w*|scat|scatting|scatted|yodel\w*|"
    r"a[\s-]?cappella|acapella|beatbox\w*|operatic|spoken|speech|narrat\w*|murmur\w*|ad[\s-]?libs?|"
    r"ooh\w*|aah\w*|falsetto|duet\w*|(?:soprano|alto|tenor|baritone|bass)\s+(?:voice|singer|vocal)\w*)\b",
    re.IGNORECASE)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or ""))


def vocal_hits(text: str) -> list[str]:
    """Every voice word in the text, possessives stripped ("singer's" reads as "singer")."""
    return [m.group(0).lower() for m in _VOCAL.finditer(_POSSESSIVE.sub("", _nfc(text)))]


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(_nfc(text).replace("-", " "))]


# ── denials ──────────────────────────────────────────────────────────────

# A denial runs to the end of its clause: punctuation, or a conjunction that starts the next claim.
_STOPS = r"\b(?:but|and|with|just|only|also|then|instead|while|yet|plus|although|though|whereas)\b"
_ITEM = rf"(?:(?!{_STOPS})[^.;,:!?])"
# A denial runs to the end of its clause, and across a comma list that closes with "or"/"nor" ("no X, Y or Z").
_DENIAL = re.compile(
    r"\b(?:no|not|without|never|zero|none\s+of|free\s+(?:of|from)|devoid\s+of|absence\s+of|absent|lacking|minus|nor|"
    r"instead\s+of|rather\s+than)\b"
    rf"{_ITEM}*?(?:(?:,\s*{_ITEM}+?)*,?\s+(?:or|nor)\s+{_ITEM}+?)?(?=[.;,:!?]|{_STOPS}|$)",
    re.IGNORECASE | re.DOTALL)
_FREE = re.compile(r"\b([^\W\d_][\w]*)-free\b", re.IGNORECASE)
_UNWANTED = re.compile(
    r"\b((?:[^\W\d_]+[\s-]){0,2}[^\W\d_]+)\s+(?:are|is|be)\s+(?:not\s+wanted|not\s+needed|unwanted|not\s+welcome|"
    r"not\s+allowed|left\s+out|banned)\b", re.IGNORECASE)


_GONE = re.compile(
    r"\b((?:[^\W\d_]+[\s-]){0,2}[^\W\d_]+)\s+(?:(?:are|is|were|was|go|goes|went)\s+)?(?:absent|gone|missing|silent|"
    r"fades?\s+(?:out|away)|faded\s+(?:out|away)|fading\s+(?:out|away)|drops?\s+out|dropped\s+out|vanish\w*|"
    r"disappear\w*|cuts?\s+out|falls?\s+silent|fell\s+silent)\b", re.IGNORECASE)


def split_denials(text: str) -> tuple[str, str]:
    """(what the text claims is there, what it rules out): "no X", "without X", "X-free", "X is not wanted",
    "X fades out", "X is absent"."""
    denied: list[str] = []

    def take(m: re.Match) -> str:
        denied.append(m.group(1) if m.re is not _DENIAL else m.group(0))
        return " "
    text = _nfc(text)
    for rx in (_UNWANTED, _GONE, _FREE, _DENIAL):
        text = rx.sub(take, text)
    return text, " ; ".join(denied)


def clean_request(request: str, instrumental: bool) -> str:
    """The request without its "no vocals" phrases on an instrumental: the flag carries that, and the engine
    reads "no vocals" as a word about vocals."""
    text = " ".join(_nfc(request).split())
    if not instrumental:
        return text
    text = _UNWANTED.sub(lambda m: _without_voices(m.group(1), m.group(0)[len(m.group(1)):], lead=False), text)
    text = _FREE.sub(lambda m: " " if vocal_hits(m.group(1)) else m.group(0), text)
    text = _DENIAL.sub(lambda m: _without_voices_denial(m.group(0)), text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(text.split()))
    text = re.sub(r"\b(?:with|and|or|but|plus)\s*(?=[,.;:!?]|$)", "", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(text.split()))
    text = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", text)
    return text.strip(" ,;:") or " ".join(_nfc(request).split())


_LIST_SPLIT = re.compile(r"\s*(?:,|/|&|\bor\b|\bnor\b|\band\b)\s*", re.IGNORECASE)
_DENIAL_LEAD = re.compile(r"^(\S+(?:\s+(?:of|from|than))?(?:\s+any)?)\s+(.*)$", re.IGNORECASE | re.DOTALL)


def _without_voices(items_text: str, tail: str, lead: bool) -> str:
    """A denied list with its voice items taken out and the rest kept: "singing or drums" -> "drums"."""
    items = [i for i in _LIST_SPLIT.split(items_text) if i.strip()]
    keep = [i for i in items if not vocal_hits(i)]
    if len(keep) == len(items):
        return items_text + tail
    return f"{' or '.join(keep)}{tail}" if keep else " "


def _without_voices_denial(span: str) -> str:
    m = _DENIAL_LEAD.match(span.strip())
    if not m or not vocal_hits(span):
        return span
    rest = _without_voices(m.group(2), "", lead=True)
    return f" {m.group(1)} {rest.strip()} " if rest.strip() else " "


# ── claims ───────────────────────────────────────────────────────────────

_NAMING = ("instrument", "genre", "vocal")


# "ambient production", "a cinematic swell": these genre words also describe a character, so they claim none.
_CHARACTER = (set(MOOD_ADJECTIVES) | set(TIMBRE_ADJECTIVES)) & set(GENRES)


# Genres the caption vocabulary does not list but a translation names: claims all the same.
_MORE_GENRES = re.compile(r"\b(?:raw[\s-])?(?:hardstyle|hard\s+dance|gabber|phonk|hyperpop|footwork|juke|bhangra|"
                          r"amapiano|afro\s+house|breakcore|drumstep|neurofunk|future\s+bass|moombahton|"
                          r"hardcore\s+techno|big\s+room|dubstep|brostep)\b", re.IGNORECASE)


# Sound sources the caption vocabulary does not list, as the instrument each one names: claims all the same.
_SOURCES = {
    "beat": r"beats?", "keys": r"keys", "low end": r"low[\s-]end", "lead": r"leads?", "reese": r"reese",
    "wobble": r"wobbles?|wobbling", "electronics": r"electronics", "saw": r"saws?|sawtooth", "square wave": r"square\s+waves?",
    "arpeggio": r"arps?", "hi-hat": r"hats", "synth": r"synthy", "sub bass": r"subs?",
    "drum": r"drumming", "bleep": r"bleeps?|blips?",
}
_SOURCE_RE = [(name, re.compile(rf"(?<![\w-])(?:{rx})(?![\w-])", re.IGNORECASE)) for name, rx in _SOURCES.items()]


def _claims(text: str) -> dict[str, set[str]]:
    """Vocabulary hits by category, hyphens read both kept and split; `_SOURCES` read as instruments."""
    out: dict[str, set[str]] = {}
    for variant in (text, text.replace("-", " ")):
        for cat, hits in extract_attributes(variant).items():
            out.setdefault(cat, set()).update(hits)
        out["genre"].update(" ".join(m.group(0).lower().split()) for m in _MORE_GENRES.finditer(variant))
        out["instrument"].update(name for name, rx in _SOURCE_RE if rx.search(variant))
    out["genre"] = out.get("genre", set()) - _CHARACTER
    return out


def _singular(noun: str) -> str:
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in noun.split())


# Naming a family names its members: "drums" asks for a kick and a snare, "strings" for a cello.
_FAMILIES = {
    "drum": {"drum", "beat", "drum kit", "kick", "kick drum", "snare", "snare drum", "hi-hat", "hihat", "cymbal",
             "ride cymbal", "crash cymbal", "tom", "percussion", "breakbeat", "drum machine", "drum loop",
             "808", "clap", "handclap", "shaker", "tambourine", "conga", "bongo"},
    "string": {"string", "string section", "violin", "viola", "cello", "double bass", "orchestral strings",
               "fiddle", "harp"},
    "brass": {"brass", "trumpet", "trombone", "horn", "french horn", "tuba"},
    "keys": {"keys", "piano", "grand piano", "upright piano", "electric piano", "keyboard", "organ", "hammond organ",
             "rhodes", "wurlitzer", "clavinet", "harpsichord"},
    "bass": {"bass", "bass guitar", "sub bass", "sub-bass", "synth bass", "bassline", "low end", "reese", "wobble"},
    "synth": {"synth", "synthesizer", "synth pad", "pad", "synth lead", "lead synth", "lead", "arpeggio",
              "arpeggiator", "modular synth", "analog synth", "electronics", "saw", "square wave", "bleep"},
}
_FAMILIES["percussion"] = _FAMILIES["drum"]
# Words, not vocabulary terms, that still play a denied kit: "hats", "a four on the floor".
_DRUM_WORDS = re.compile(r"\b(?:drum\w*|kicks?|snares?|hats?|hi-?hats?|cymbals?|toms?|percussi\w*|"
                         r"breakbeats?|808s?|four[\s-]on[\s-]the[\s-]floor|backbeat)\b", re.IGNORECASE)


def _family_of(noun: str) -> str | None:
    s = _singular(noun)
    for name, members in _FAMILIES.items():
        if s == name or s in {_singular(m) for m in members}:
            return name
    return None


def _from(noun: str, named: set[str]) -> bool:
    """The noun is one of ``named``, its generalisation, its specialisation ("electric guitar" of "guitar"),
    or a member of a family ``named`` asks for."""
    if not named:
        return False
    singular = {_singular(a) for a in named}
    s = _singular(noun)
    if generalises(noun, named) or generalises(s, singular) or s.split()[-1] in singular:
        return True
    # A request word says the genre it is made from: "funky" asks for funk, "jazzy" for jazz.
    if len(s) >= 4 and any(w.startswith(s) and len(w) - len(s) <= 3 for w in words_of_request(named)):
        return True
    families = {_family_of(a) for a in named} - {None}
    return _family_of(noun) in families


def words_of_request(named: set[str]) -> set[str]:
    return getattr(named, "words", set())


class _Named(set):
    """The request's naming nouns, carrying the request's own words for `_from`."""

    def __init__(self, nouns, words=()):
        super().__init__(nouns)
        self.words = set(words)


def _denies(noun: str, denied: set[str]) -> bool:
    s = _singular(noun)
    for d in denied:
        d1 = _singular(d)
        if s == d1 or d1 in s.split() or s in d1.split():
            return True
        if _family_of(d) and _family_of(d) == _family_of(noun):
            return True
    return False


# ── moods ────────────────────────────────────────────────────────────────

# A body that says the opposite of a mood the request asked for has changed the request.
_OPPOSED = (
    ({"sad", "melancholic", "melancholy", "mournful", "somber", "sombre", "gloomy", "wistful", "bleak", "sorrowful",
      "sorrow", "grief", "grieving", "forlorn", "despairing", "heartbroken", "doleful"},
     {"happy", "joyful", "cheerful", "upbeat", "uplifting", "euphoric", "playful", "bouncy", "sunny", "celebratory",
      "triumphant", "ecstatic", "jubilant", "merry", "lighthearted", "light-hearted", "blissful", "glee", "gleeful",
      "sunshine", "carefree", "elated", "exuberant"}),
    ({"calm", "peaceful", "serene", "relaxed", "tranquil", "chill", "mellow", "gentle", "laid-back", "soft",
      "quiet", "meditative", "placid", "still"},
     {"aggressive", "frantic", "chaotic", "intense", "urgent", "restless", "pounding", "harsh", "loud",
      "relentless", "violent", "ferocious", "explosive", "furious", "stormy", "savage", "brutal"}),
    ({"tense", "anxious", "uneasy", "ominous", "menacing"}, {"relaxed", "calm", "peaceful", "serene", "laid-back"}),
    ({"dark", "brooding"}, {"bright", "sunny", "cheerful", "luminous", "sparkling", "radiant", "glittering"}),
    ({"slow", "glacial", "unhurried", "languid"}, {"fast", "high-energy", "frantic", "upbeat", "brisk", "rapid",
                                                   "breakneck", "racing"}),
    ({"triumphant", "celebratory", "victorious", "heroic", "glorious", "majestic"},
     {"aggressive", "mechanical", "menacing", "sinister", "defeated", "bleak"}),
)
# Words a restatement may say for a request word; they count toward what it kept.
_KIN_WORDS = {
    "victory": {"triumphant", "victorious", "celebratory", "winning", "triumph"},
    "night": {"nocturnal", "nighttime", "midnight", "night's"}, "rainy": {"rain", "rainfall", "rain-soaked"},
    "stealth": {"stealthy", "covert", "sneaking", "furtive"}, "big": {"grand", "huge", "massive", "epic", "bold"},
    "morning": {"dawn", "sunrise", "daybreak"}, "home": {"homecoming", "homeward"},
    "happy": {"joyful", "cheerful"}, "sunny": {"bright", "sunlit"},
}
# A request word that names a mood without being one: its opposite is a changed request too.
_IMPLIED_MOODS = {"victory": "triumphant", "victorious": "triumphant", "celebration": "celebratory",
                  "party": "celebratory", "funeral": "mournful", "lullaby": "gentle", "rainy": "melancholic"}
_MOOD_WORDS = {w for pair in _OPPOSED for side in pair for w in side} | set(MOOD_ADJECTIVES)


def _moods(text: str) -> set[str]:
    """The text's mood words (a hyphenated one whole too)."""
    words = set(_tokens(text)) | {m.group(0).lower() for m in re.finditer(r"\b\w+-\w+\b", _nfc(text))}
    return {w for w in words if w in _MOOD_WORDS}


def _kin(mood: str) -> set[str]:
    """The words on the mood's own side of a pair: "melancholic" keeps "sad"."""
    out: set[str] = set()
    for side in (s for pair in _OPPOSED for s in pair):
        if mood in side:
            out |= side
    return out


def _opposites(mood: str) -> set[str]:
    out: set[str] = set()
    for a, b in _OPPOSED:
        if mood in a:
            out |= b
        if mood in b:
            out |= a
    return out


# ── numbers, names, time ─────────────────────────────────────────────────

_NUM = (r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
        r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
        r"a\s+hundred|half|a\s+few|several|couple)")
_ORDINAL = r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final|opening|closing)"
_CLOCK_TIME = re.compile(
    r"\b\d{1,2}:\d{2}\b"
    r"|\b[\w-]+[- ](?:second|minute)s?[- ]mark\b"
    rf"|\b{_NUM}(?:[\s-]+(?:and\s+)?{_NUM})*[\s-]+(?:seconds?|secs?|minutes?|mins?)\b"
    rf"|\b{_ORDINAL}\s+(?:few\s+)?(?:seconds?|minutes?)\b",
    re.IGNORECASE)
_THEORY = re.compile(
    r"\b[A-G](?:#|♯|b|♭)?\s*(?:major|minor|maj|min)\b"
    r"|\b(?:in|of)\s+(?:the\s+key\s+of\s+)?[a-g](?:#|♯|b|♭|\s+(?:sharp|flat))?\s+(?:major|minor)\b"
    r"|\bkey\s+of\b|\btime\s+signature\b|\b\d+/\d+\b"
    rf"|\b{_NUM}(?:[\s-]+{_NUM})?[\s-]+time\b"
    rf"|\b{_NUM}(?:[\s-]+(?:and\s+)?{_NUM})*[\s-]+(?:beats?|bpm)\b"
    r"|\bbpm|\bbeats\s+(?:per|a)\s+min\w*|\btempo\s+of\b",
    re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
# Comparisons and titles name somebody whatever the case: "like burial", "channeling X", "a track called X".
_SOUND_ALIKE = re.compile(
    r"\b(?:in the style of|reminiscent of|sounds? like|à la|a la|inspired by|in the vein of|channell?ing|"
    r"a nod to|homage|straight out of|would approve|called|titled|named)\b|\w-esque\b",
    re.IGNORECASE)
_SIMILE = re.compile(r"\b(?:like|echoing|evoking|recalling)\s+(?!a\b|an\b|the\b|its\b|their\b|this\b)([^\W\d_][\w'’-]*)",
                     re.IGNORECASE)
_SENTENCE_START = re.compile(r"(?:^|[.!?:;]\s+|[\"“(]\s*)$")


_ENGLISH_GLUE = frozenset("the a an and with of in for over to is its on by at from into under".split())
_OTHER_GLUE = frozenset("""der die das und mit ein eine für im el la los las y con una uno para por une le les et avec
pour du des il di de""".split())


def _english(text: str) -> bool:
    """English by its function words: only there does a capital mid-sentence suggest a name."""
    words = _tokens(text)
    english = sum(w in _ENGLISH_GLUE for w in words)
    return english >= 2 and english > sum(w in _OTHER_GLUE for w in words)


def _reads_english(text: str) -> bool:
    """A request reads as English unless another language's function words outnumber English ones or its
    letters are not plain ASCII; short requests ("sad solo piano") have none of either."""
    words = _tokens(text)
    letters = [c for c in _nfc(text) if c.isalpha()]
    plain = not letters or sum(c.isascii() for c in letters) / len(letters) >= 0.97
    return plain and sum(w in _OTHER_GLUE for w in words) <= sum(w in _ENGLISH_GLUE for w in words)


# Common images a simile reaches for; any other word after "like" may be a name.
_IMAGERY = frozenset("""rain tide tides wave waves wind breath breathing water light smoke glass fog mist dust star stars
snow ice fire thunder cloud clouds dream dreams ghost ghosts heartbeat clockwork machine machines rust silk velvet
honey sunlight moonlight shadow shadows ocean sea storm storms river rivers memory memories echo echoes whisper
lullaby hymn prayer march heartbeat pulse candle candles drizzle bells""".split())
# A sentence that opens on a name usually says what that person would do: "Burial would make this".
_LEADING_NAME = re.compile(r"(?:^|[.!?]\s+)([A-Z][\w'’-]+)\s+(?:would|could|might|will|meets|does|did|made|makes|"
                           r"wrote|writes|produced|produces|says|said|approves?)\b")


def _names(text: str, known: set[str]) -> list[str]:
    """Likely names: capitals mid-sentence in English, a sentence opened by one doing something, and the
    word after "like" when it is capitalised or neither known nor a common image."""
    out = []
    if _english(text):
        for m in _WORD.finditer(text):
            word = m.group(0)
            if word[:1].isupper() and not _SENTENCE_START.search(text[:m.start()]) and word.lower() not in known:
                out.append(word)
        out += [m.group(1) for m in _LEADING_NAME.finditer(text) if m.group(1).lower() not in known]
    for m in _SIMILE.finditer(text):
        word = m.group(1).strip("'’")
        lower = word.lower()
        if word[:1].isupper() or (lower not in known and lower not in _IMAGERY and _singular(lower) not in _IMAGERY):
            out.append(word)
    return out


def _spaced(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    unspaced = sum(1 for c in letters if unicodedata.name(c, "").split()[0] in {"CJK", "HIRAGANA", "KATAKANA", "THAI"})
    return not letters or unspaced / len(letters) < 0.5


# ── the prompt and the answer ────────────────────────────────────────────

_SYSTEM = (
    "You restate a listener's request for music in the words this artist's music is described in. "
    "You have NOT heard any recording. Keep everything the request asks for and nothing it does not: "
    "never add an instrument, genre, vocal, effect or mood the request does not imply, never drop an "
    "instrument, genre or mood it names, and never contradict what it rules out. Use the request's own "
    "words for every instrument, genre and mood it names, inside your own sentence; do not repeat the "
    "request as a phrase. Name no instrument, genre or voice the request does not name; when the request "
    "names an instrument or a genre, describe everything else with texture, production and mood words. The "
    "artist's descriptions are vocabulary, not content: borrow their texture, production and mood words, "
    "and their instruments or genres only when the request names none. "
    "Never name a song, artist, band, film or album, never compare to anyone, and never state tempo, BPM, "
    "key, time signature, duration, timestamps or numbers. Write in English whatever language the request "
    "is in. Answer in exactly three lines:\n"
    "Language: <the request's language>\nEnglish request: <the request in English>\nCaption: <the caption>"
)


def build_prompt(request: str, shots: list[str], instrumental: bool = True) -> str:
    """Qwen ChatML with thinking off: the system rule, then the descriptions, the request and its vocal stance."""
    vocals = ("Instrumental: do not mention vocals, voices, singing or words at all."
              if instrumental else "Sung: the music has vocals.")
    described = "\n".join(f"- {s}" for s in shots) or "- (none)"
    user = (f"How this artist's music is described (vocabulary only):\n{described}\n\n"
            f"Request: {request.strip()}\n{vocals}\n\n"
            f"Restate the request as one flowing caption of {TRANSLATION_WORDS[0]}-{TRANSLATION_WORDS[1]} words "
            "that keeps every instrument, genre and mood it names. /no_think")
    return (f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
_CHATML_TAIL = re.compile(r"<\|im_end\|>.*$", re.DOTALL)
_PREAMBLE = re.compile(r"^(?:caption|here(?:'s| is)[^:]*|restated[^:]*)\s*:\s*", re.IGNORECASE)
_FIELD = re.compile(r"^\s*(language|english request|caption)\s*:\s*(.*)$", re.IGNORECASE)


def parse_answer(raw: str) -> dict:
    """``{language, english_request, caption}`` from a raw completion; a bare caption is read as one."""
    text = _CHATML_TAIL.sub("", _THINK.sub("", (raw or "").replace("[end of text]", ""))).strip()
    fields: dict[str, str] = {}
    rest: list[str] = []
    for line in text.splitlines():
        m = _FIELD.match(line)
        if m:
            fields[m.group(1).lower()] = m.group(2).strip().strip('"“”')
        elif line.strip() and not line.strip().endswith(":"):
            rest.append(_PREAMBLE.sub("", line.strip()))
    caption = fields.get("caption") or (max(rest, key=len) if rest else "")
    return {"language": fields.get("language") or None, "english_request": fields.get("english request") or None,
            "caption": tidy_caption(caption)}


def extract_answer(raw: str) -> str:
    """The caption alone from a raw completion."""
    return parse_answer(raw)["caption"]


def _rendered(request: str, parsed: dict) -> tuple[str, str | None]:
    """The English request the checks read, and the language it was translated from (None for English).
    A request that reads as English is checked as written, whatever language the model claims."""
    language, english = parsed.get("language"), parsed.get("english_request")
    if (not english or not language or language.strip().lower().startswith("english")
            or _reads_english(request)):
        return request, None
    asked = set(tokens(request))
    if asked and len(asked & set(tokens(english))) / len(asked) >= 0.5:
        return request, None
    return english, language.strip()


def _fold(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", word.lower()) if not unicodedata.combining(c))


def _cognate(a: str, b: str) -> bool:
    """Spelled alike across languages: "guitarra" and "guitar", "violín" and "violin", "piano" and "piano"."""
    a, b = _fold(a), _fold(b)
    if a == b:
        return True
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= 5 or (n >= 4 and n >= min(len(a), len(b)) - 1)


_VOCAB_WORDS = sorted({w for t in (*INSTRUMENTS, *GENRES, *VOCAL_TERMS) for w in _tokens(t) if len(w) >= 4})


def _rendering_reasons(request: str, english: str) -> list[str]:
    """Why an English rendering of a foreign request cannot be read for it: a claim noun the original has no
    cognate for, or a cognate of a vocabulary noun in the original that the rendering lost."""
    words = [w for w in _tokens(request) if len(w) >= 4]
    claims = {n for cat in _NAMING for n in _claims(english).get(cat, set())}
    reasons = [f"rendering_adds:{n}" for n in sorted(claims)
               if not any(_cognate(w, part) for w in words for part in n.split())]
    rendered = set(_tokens(english))
    for w in words:
        kin = [v for v in _VOCAB_WORDS if _cognate(w, v)]
        if kin and not any(_cognate(k, r) for k in kin for r in rendered):
            reasons.append(f"rendering_drops:{w}")
    return reasons


# ── the checks ───────────────────────────────────────────────────────────

def check_caption(request: str, caption: str, *, instrumental: bool) -> list[str]:
    """Why a caption does not answer the request, empty when it does: any voice word on an instrumental, a
    clock time, or under ``ANSWERED_FLOOR`` of a request of at least two content words."""
    reasons: list[str] = []
    if not (caption or "").strip():
        return ["empty"]
    if instrumental:
        sung = vocal_hits(caption)
        if sung:
            reasons.append(f"vocal_on_instrumental:{sung[0]}")
    clock = _CLOCK_TIME.search(caption)
    if clock:
        reasons.append(f"clock_time:{clock.group(0).strip()}")
    asked = _asked_words(request)
    if len(asked) >= FLOOR_MIN_WORDS:
        share = answered(" ".join(asked), caption.replace("-", " "))["answered"]
        if share is not None and share < ANSWERED_FLOOR:
            reasons.append(f"answered:{share:.2f}")
    return reasons


# Words a request leans on without asking for anything: dropping them changes nothing asked.
_FILLER = frozenset("""something anything music musical track tune song piece sound sounds like kind sort vibe
vibes please want need make give feel feeling style""".split())


def _asked_words(request: str) -> list[str]:
    """The request's content words, hyphens split, filler out."""
    return [w for w in dict.fromkeys(tokens(request.replace("-", " "))) if w not in _FILLER]


def _with_kin(request: str, body: str) -> str:
    """The body with each request word it says in a kindred word appended, so the share counts it kept."""
    said = set(_tokens(body))
    extra = [m for m in _moods(request) if m not in said and _kin(m) & said]
    extra += [w for w in _tokens(request) if w not in said and w in _KIN_WORDS and _KIN_WORDS[w] & said]
    return f"{body} {' '.join(extra)}" if extra else body


def _body(request: str, output: str) -> str:
    """The output without a leading copy of the request: only what the model wrote itself is measured."""
    words = _WORD.findall(_nfc(request))
    if not words:
        return output
    copy = re.compile(r"^[\W_]*(?:(?:a|an|the)\s+)?" + r"[\W_]*".join(re.escape(w) for w in words), re.IGNORECASE)
    return re.sub(r"^[\s,.;:!?—–-]+", "", copy.sub(" ", _nfc(output).strip())).strip()


def lint_translation(request: str, output: str, *, instrumental: bool,
                     pool: Iterable[str]) -> tuple[str | None, list[str]]:
    """(caption, []) when the translation passes, else (None, reasons), all read on the body the model wrote:
    the request check, what it names, what it denies, the request's moods, names, numbers, theory, length."""
    text = (output or "").strip()
    if not text:
        return None, ["empty"]
    if instrumental:
        text = clean_request(text, True)
    body = _body(request, text)
    reasons = check_caption(request, _with_kin(request, body), instrumental=instrumental)
    if instrumental and not any(r.startswith("vocal_on_instrumental") for r in reasons) and vocal_hits(text):
        reasons.append(f"vocal_on_instrumental:{vocal_hits(text)[0]}")
    pool_text = " ".join(pool)
    asked_text, ruled_out = split_denials(request)
    asked, denied_by_request = _claims(asked_text), _claims(ruled_out)
    said_text, said_denied = split_denials(body)
    said, negated = _claims(said_text), _claims(said_denied)
    pool_claims = _claims(pool_text)
    named = _Named({n for cat in _NAMING for n in asked.get(cat, set())}, _tokens(asked_text))
    names_nothing = not (asked.get("instrument") or asked.get("genre"))
    for cat in ("instrument", "genre", "vocal", "production"):
        for noun in sorted(said.get(cat, set())):
            if _from(noun, _Named(named | asked.get("production", set()), named.words)):
                continue
            if cat == "production" and _from(noun, pool_claims.get("production", set())):
                continue
            if cat in ("instrument", "genre") and names_nothing and _from(noun, pool_claims.get(cat, set())):
                continue
            if cat == "vocal" and not instrumental and not asked.get("vocal"):
                continue
            reasons.append(f"invented_noun:{noun}")
    ruled = {n for cat in ("instrument", "genre", "vocal", "production") for n in denied_by_request.get(cat, set())}
    for noun in sorted({n for cat in ("instrument", "genre", "vocal", "production") for n in said.get(cat, set())}):
        if _denies(noun, ruled):
            reasons.append(f"denied:{noun}")
    if any(_family_of(d) == "drum" for d in ruled):
        hit = _DRUM_WORDS.search(said_text)
        if hit and f"denied:{hit.group(0).lower()}" not in reasons:
            reasons.append(f"denied:{hit.group(0).lower()}")
    for noun in sorted(named):
        if any(_denies(noun, {n}) for cat in _NAMING for n in negated.get(cat, set())):
            reasons.append(f"negated_request:{noun}")
    new = {_singular(w) for w in _tokens(body) if len(w) >= 3} - {_singular(w) for w in _tokens(request)} - _ENGLISH_GLUE
    if len(new) < MIN_NEW_WORDS:
        reasons.append(f"repeats_request:{len(new)}_new_words")
    body_words = set(_tokens(said_text)) | _moods(said_text)
    implied = {_IMPLIED_MOODS[w] for w in _tokens(asked_text) if w in _IMPLIED_MOODS}
    for mood in sorted(_moods(asked_text) | implied):
        if mood not in implied and not ({mood, _singular(mood)} | _kin(mood)) & body_words:
            reasons.append(f"mood_dropped:{mood}")
        flipped = sorted((_opposites(mood) - _kin(mood)) & _moods(said_text))
        if flipped:
            reasons.append(f"mood_opposite:{mood}->{flipped[0]}")
    theory = _THEORY.search(text)
    if theory:
        reasons.append(f"theory:{theory.group(0).strip()}")
    stray = sorted(set(_DIGITS.findall(text)) - set(_DIGITS.findall(request)))
    if stray:
        reasons.append(f"number:{stray[0]}")
    alike = _SOUND_ALIKE.search(text)
    if alike:
        reasons.append(f"sound_alike:{alike.group(0).lower()}")
    known = (set(_tokens(request)) | set(_tokens(pool_text)) | _MOOD_WORDS
             | {w for t in (*INSTRUMENTS, *GENRES, *VOCAL_TERMS) for w in _tokens(t)})
    names = _names(text, known)
    if names:
        reasons.append(f"name:{names[0]}")
    if _spaced(text):
        n = len(_WORD.findall(text))
        if not LINT_WORDS[0] <= n <= LINT_WORDS[1]:
            reasons.append(f"length:{n}_words")
    else:
        n = sum(1 for c in text if c.isalpha())
        if not LINT_CHARS_UNSPACED[0] <= n <= LINT_CHARS_UNSPACED[1]:
            reasons.append(f"length:{n}_chars")
    if len(text) < MIN_CAPTION_CHARS and _spaced(text):
        reasons.append(f"length:{len(text)}_chars")
    reasons = list(dict.fromkeys(reasons))
    return (None, reasons) if reasons else (text, [])


# ── running it ───────────────────────────────────────────────────────────

LLM = Callable[..., str]


def deadline_seconds(environ: dict | None = None) -> float:
    """``KHAOS_TRANSLATOR_TIMEOUT`` as seconds, else the default."""
    env = os.environ if environ is None else environ
    try:
        value = float(env.get(TRANSLATOR_TIMEOUT_ENV) or DEFAULT_TIMEOUT_SEC)
    except ValueError:
        return DEFAULT_TIMEOUT_SEC
    return value if value > 0 else DEFAULT_TIMEOUT_SEC


def _label(llm, fallback: str) -> str:
    return getattr(llm, "label", None) or fallback


def translate(request: str, shots: list[str] | Callable[[int], list[str]], *, instrumental: bool,
              pool: Iterable[str], llm: LLM, label: str = DEFAULT_LABEL, deadline: float | None = None) -> dict:
    """Up to ``MAX_ATTEMPTS`` linted translations before the monotonic ``deadline``, try ``n`` on ``shots(n)``:
    ``{caption | None, attempts, rejects, label, translated_from}``. Raises `TranslatorUnavailable` when none answers."""
    pool = list(pool)
    deadline = time.monotonic() + DEFAULT_TIMEOUT_SEC if deadline is None else deadline
    rejects: list[dict] = []
    for n in range(MAX_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining <= 0.5:
            if not n:
                raise TranslatorUnavailable("no time left")
            break
        prompt = build_prompt(request, shots(n) if callable(shots) else shots, instrumental)
        temperature, seed = (0.0, None) if n == 0 else (RETRY_TEMPERATURE, RETRY_SEED + n)
        try:
            raw = llm(prompt, temperature=temperature, seed=seed, timeout=remaining)
        except Exception as exc:  # noqa: BLE001 — any backend failure is the same fallback
            if not n:
                raise exc if isinstance(exc, TranslatorUnavailable) else \
                    TranslatorUnavailable(f"{type(exc).__name__}: {exc}") from exc
            rejects.append({"by": _label(llm, label), "attempt": n + 1, "reasons": [f"unavailable:{exc}"[:200]],
                            "error": True})
            break
        parsed = parse_answer(raw)
        english, language = _rendered(request, parsed)
        text, reasons = lint_translation(english, parsed["caption"], instrumental=instrumental, pool=pool)
        if language:
            more = _rendering_reasons(request, english)
            text, reasons = (None, more + reasons) if more else (text, reasons)
        if text is not None:
            return {"caption": text, "attempts": n + 1, "rejects": rejects, "label": _label(llm, label),
                    "translated_from": language, "english_request": english if language else None}
        rejects.append({"by": _label(llm, label), "attempt": n + 1, "reasons": reasons,
                        "text": parsed["caption"][:400]})
    return {"caption": None, "attempts": len(rejects), "rejects": rejects, "label": _label(llm, label),
            "translated_from": None, "english_request": None}


_EMPTY_HARVEST = {"bpm": None, "keyscale": None, "timesignature": None, "duration": None,
                  "lyrics": None, "language": None}


def request_fallback(request: str, attempts: int, rejects: list[dict]) -> dict:
    """The listener's own words as the caption, said so: never an invented caption. When no model wrote
    anything at all the note says they were busy, not that the request could not be kept."""
    wrote = any(not r.get("error") for r in rejects)
    return {**_EMPTY_HARVEST, "caption": " ".join(str(request).split()), "translator": "request",
            "fallback": "request", "fallback_note": FALLBACK_NOTE if wrote else BUSY_NOTE,
            "attempts": attempts, "rejects": rejects,
            "translated_from": None, "translated_note": None, "lyrics_note": None}


def expand_checked(request: str, *, instrumental: bool, attempt: int, pool: Iterable[str],
                   shots: Callable[[int], list[str]], translator: LLM | None = None,
                   translator_label: str = DEFAULT_LABEL, lm: Callable[[int], dict] | None = None,
                   deadline: float | None = None) -> dict:
    """Translator first; when it does not answer, the 5Hz LM (``lm(attempt)`` → a harvested answer) held to
    `check_caption`; else the request itself. One monotonic deadline covers all of it; the LM's first try
    runs whatever is left, so a cold model still answers. ``served_attempt`` names the window that served."""
    pool = list(pool)
    request = clean_request(request, instrumental)
    deadline = time.monotonic() + deadline_seconds() if deadline is None else deadline
    attempts, rejects = 0, []
    if translator is not None:
        try:
            out = translate(request, lambda n: shots(attempt + n), instrumental=instrumental, pool=pool,
                            llm=translator, label=translator_label, deadline=deadline)
        except TranslatorUnavailable as exc:
            rejects.append({"by": _label(translator, translator_label), "attempt": 1,
                            "reasons": [f"unavailable:{exc}"[:200]], "error": True})
        else:
            attempts, rejects = out["attempts"], out["rejects"]
            if out["caption"] is None:
                return {**request_fallback(request, attempts, rejects), "served_attempt": attempt}
            language = out["translated_from"]
            return {**_EMPTY_HARVEST, "caption": out["caption"], "translator": out["label"],
                    "fallback": None, "fallback_note": None, "attempts": attempts, "rejects": rejects,
                    "translated_from": language, "english_request": out["english_request"],
                    "translated_note": TRANSLATED_NOTE.format(language=language) if language else None,
                    "lyrics_note": None if instrumental else LYRICS_NOTE, "served_attempt": attempt}
    if lm is not None:
        for n in range(MAX_ATTEMPTS):
            if n and time.monotonic() >= deadline:
                rejects.append({"by": "lm", "attempt": n + 1, "reasons": ["no time left"], "error": True})
                break
            attempts += 1
            try:
                result = dict(lm(attempt + n))
            except Exception as exc:  # noqa: BLE001 — a failing or busy LM is a refusal, never a 500
                rejects.append({"by": "lm", "attempt": n + 1, "reasons": [f"{type(exc).__name__}: {exc}"[:200]],
                                "error": True})
                continue
            reasons = check_caption(request, result.get("caption") or "", instrumental=instrumental)
            if not reasons:
                result.update(translator="lm", fallback=None, fallback_note=None, attempts=attempts,
                              rejects=rejects, served_attempt=attempt + n, translated_from=None,
                              translated_note=None, lyrics_note=None)
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
_SLOTS: dict[str, threading.BoundedSemaphore] = {}
_SLOTS_LOCK = threading.Lock()


def _slots(url: str) -> threading.BoundedSemaphore:
    with _SLOTS_LOCK:
        return _SLOTS.setdefault(url, threading.BoundedSemaphore(SERVER_PARALLEL))


def _healthy(url: str, timeout: float) -> bool:
    """llama-server's /health, remembered for a short while so a down server costs one probe, not one per Expand."""
    now = time.monotonic()
    cached = _HEALTH.get(url)
    if cached and now - cached[0] < _HEALTH_TTL:
        return cached[1]
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=max(0.1, min(timeout, 2.0))) as answer:
            ok = answer.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        ok = False
    _HEALTH[url] = (now, ok)
    return ok


def _read_capped(answer, deadline: float) -> bytes:
    """The body, refused past ``MAX_RESPONSE_BYTES`` or the deadline, however slowly it trickles."""
    chunks, size = [], 0
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TranslatorUnavailable("the answer took longer than the time left")
        try:
            answer.fp.raw._sock.settimeout(left)  # type: ignore[attr-defined]
        except AttributeError:
            pass
        chunk = answer.read1(8192) if hasattr(answer, "read1") else answer.read(8192)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise TranslatorUnavailable("the answer is larger than any caption")
        chunks.append(chunk)


def http_llm(url: str, *, timeout: float = DEFAULT_TIMEOUT_SEC, max_tokens: int = MAX_TOKENS) -> LLM:
    """A callable for llama.cpp's ``POST /completion``, at most `SERVER_PARALLEL` at once; every failure is
    `TranslatorUnavailable`; ``timeout`` per call is what the caller has left."""
    base = url.rstrip("/")

    def run(prompt: str, temperature: float = 0.0, seed: int | None = None, timeout: float = timeout) -> str:
        deadline = time.monotonic() + max(0.1, float(timeout))
        if not _healthy(base, deadline - time.monotonic()):
            raise TranslatorUnavailable(f"{base} is not answering")
        slots = _slots(base)
        if not slots.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise TranslatorUnavailable("every translator slot is busy")
        try:
            body = {"prompt": prompt, "n_predict": int(max_tokens), "temperature": float(temperature),
                    "cache_prompt": True, "stop": ["<|im_end|>"]}
            if seed is not None:
                body["seed"] = int(seed)
            request = urllib.request.Request(f"{base}/completion", data=json.dumps(body).encode("utf-8"),
                                             headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=max(0.1, deadline - time.monotonic())) as answer:
                    doc = json.loads(_read_capped(answer, deadline).decode("utf-8"))
            except TranslatorUnavailable:
                raise
            except (urllib.error.URLError, OSError, ValueError) as exc:
                _HEALTH.pop(base, None)
                raise TranslatorUnavailable(f"{type(exc).__name__}: {exc}") from exc
        finally:
            slots.release()
        content = doc.get("content") if isinstance(doc, dict) else None
        if not isinstance(content, str):
            raise TranslatorUnavailable("the server's answer has no content")
        settings = doc.get("generation_settings") if isinstance(doc.get("generation_settings"), dict) else {}
        run.label = model_label(doc.get("model") or settings.get("model"))
        return content

    run.label = DEFAULT_LABEL
    return run


def translator_from_env(environ: dict | None = None) -> LLM | None:
    """The server at ``KHAOS_TRANSLATOR_URL`` as a callable, or None when the variable is unset."""
    env = os.environ if environ is None else environ
    url = (env.get(TRANSLATOR_URL_ENV) or "").strip()
    if not url:
        return None
    return http_llm(url, timeout=deadline_seconds(env))
