"""The catalogue guard: a prompt may name only the instruments, genres and kinds of voice this artist's captions name.

`hold` drops what the catalogue never held and returns what it dropped; the words left are the listener's own.
It reads named sounds from the caption vocabulary, so a word in none of its lists passes: a guard on what is
named, not a list of every word allowed. Pure, no model.
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from collections.abc import Iterable

from khaos_attribution.caption_fit import MIN_CORPUS_CAPTIONS
from khaos_attribution.caption_vocabulary import GENRES, INSTRUMENTS, VOCAL_TERMS, canonical_term, generalises
from khaos_attribution.translation import (_CHARACTER, _FAMILIES, _MORE_GENRES, _SOURCES, _claims, _denial_spans,
                                           _family_of, _singular)

__all__ = ["GUARD_VERSION", "KINDS", "catalogue_claims", "hold"]

# Readings compare only within a version: a change to what is dropped is a new one.
GUARD_VERSION = "0.1.2"
KINDS = ("instrument", "genre", "vocal")

# Named sounds the caption vocabulary does not list; a catalogue's captions never hold them, so they are outside.
_MORE_INSTRUMENTS = ("didgeridoo", "kazoo", "bagpipes", "hurdy-gurdy", "hurdy gurdy", "lute", "piccolo", "kalimba",
                     "mbira", "kora", "pipa", "guzheng", "duduk", "ocarina", "melodica", "concertina", "washboard",
                     "jaw harp", "pedal steel", "lap steel", "dobro", "sarod", "santoor", "veena", "bansuri",
                     "shakuhachi", "taiko", "dhol", "darbuka", "castanets", "cowbell", "pipe organ",
                     "church organ", "hang drum", "handpan", "tin whistle", "penny whistle", "euphonium", "cornet",
                     "flugelhorn", "cor anglais", "cittern", "charango", "cuatro", "berimbau", "cuica")
_MORE_GENRE_WORDS = ("hoedown", "sea shanty", "shanty", "mariachi", "klezmer", "zydeco", "calypso", "soca", "fado",
                     "polka", "ragtime", "barbershop", "yodel", "yodelling", "yodeling", "raga", "gamelan", "qawwali",
                     "enka", "schlager", "chiptune", "8-bit", "8 bit", "skiffle", "honky-tonk", "honky tonk",
                     "rockabilly", "doo-wop", "doo wop", "dixieland", "merengue", "bachata", "kizomba", "gqom",
                     "kuduro", "baile funk", "city pop", "krautrock", "britpop", "surf rock", "yacht rock", "nu-metal",
                     "djent", "grindcore", "crust punk", "horrorcore", "cloud rap", "mumble rap", "g-funk")

# Words that are more often a description than a named sound: never dropped.
_EVERYDAY = frozenset("""soul world march sample loop lead whisper whispered harmonies harmony acoustic romantic
experimental holiday noise scratch swing fusion chamber garage spoken voice voices beat keys saw sub bell chime clap
kick tom pad lounge dub minimalism minimalist electro drone baroque
children's easy listening string strings choir chorus vocals vocal singer singing sung duet chant chanting humming
falsetto belting crooning shout shouting ad-lib ad-libs classical orchestral cinematic soundtrack film score""".split()) \
    | {"spoken word", "easy listening", "film score", "chorus vocals", "new age", "space music", "found sound",
       "field recording", "vocal harmonies", "harmony vocals", "backing vocal", "backing vocals", "background vocals",
       "lead vocal", "lead vocals"}

# The kinds of voice that are a different singer or source, not a way of singing: held to the catalogue.
_VOICE_SOURCES = frozenset({"rap", "rapping", "rapper", "growl", "growling", "scream", "screams", "screaming",
                            "vocoder", "autotune", "auto-tune", "vocal chop", "vocal chops", "vocal sample"})

# One sound source under several words: a catalogue that holds one holds its siblings.
_SAME_SOURCE = ("drum", "percussion", "bass", "synth")
_REEDS = {"saxophone", "sax", "clarinet", "oboe", "bassoon"}


def _term_regex(terms: Iterable[str]) -> re.Pattern:
    """The terms with plurals and hyphen/space variants; a hyphen joins two words and each is read on its own."""
    ordered = sorted({t for t in terms}, key=lambda t: (-len(t), t))
    alts = "|".join(re.escape(t).replace(r"\ ", r"[\s-]").replace(r"\-", r"[\s-]") for t in ordered)
    return re.compile(rf"(?<!\w)(?P<term>{alts})(?:s|es)?(?!\w)", re.IGNORECASE)


_GENRE_TERMS = frozenset(GENRES) - _CHARACTER
_FINDERS: tuple[tuple[str, re.Pattern, frozenset[str]], ...] = (
    ("instrument", _term_regex((*INSTRUMENTS, *_MORE_INSTRUMENTS)), frozenset((*INSTRUMENTS, *_MORE_INSTRUMENTS))),
    ("genre", _term_regex((*_GENRE_TERMS, *_MORE_GENRE_WORDS)), frozenset((*_GENRE_TERMS, *_MORE_GENRE_WORDS))),
    ("vocal", _term_regex(VOCAL_TERMS), frozenset(VOCAL_TERMS)),
)
_SOURCE_FINDERS = [(name, re.compile(rf"(?<!\w)(?:{rx})(?!\w)", re.IGNORECASE))
                   for name, rx in _SOURCES.items()]


def _found(text: str) -> list[tuple[int, int, str, str]]:
    """Every named sound in ``text`` as (start, end, canonical term, kind); spans of different kinds may overlap."""
    out = []
    for kind, rx, terms in _FINDERS:
        for m in rx.finditer(text):
            out.append((m.start(), m.end(), canonical_term(" ".join(m.group("term").split()), terms), kind))
    for m in _MORE_GENRES.finditer(text):
        out.append((m.start(), m.end(), " ".join(m.group(0).lower().split()), "genre"))
    for name, rx in _SOURCE_FINDERS:
        for m in rx.finditer(text):
            out.append((m.start(), m.end(), name, "instrument"))
    return out


@functools.lru_cache(maxsize=64)
def _pool(captions: tuple[str, ...]) -> tuple[dict[str, frozenset[str]], tuple[str, ...]]:
    """(the pool's claims by kind, its commonest genres and instruments for a prompt left with nothing)."""
    claims = {kind: set() for kind in KINDS}
    counts: dict[str, Counter] = {"genre": Counter(), "instrument": Counter()}
    for caption in captions:
        said = _claims(caption)
        for kind in KINDS:
            claims[kind] |= said.get(kind, set())
        for kind in counts:
            counts[kind].update(said.get(kind, set()))
    for _, _, term, kind in _found(" ".join(captions)):
        claims[kind].add(term)
    # "electronic" describes a character in a caption, so it claims no genre there; it still holds its own genre.
    if any(re.search(r"(?<![\w-])electronic", c, re.IGNORECASE) for c in captions):
        claims["genre"].add("electronic")
    common = [t for kind in ("genre", "instrument")
              for t, _ in sorted(counts[kind].items(), key=lambda kv: (-kv[1], kv[0]))[:2]]
    return {k: frozenset(v) for k, v in claims.items()}, tuple(dict.fromkeys(common))


def catalogue_claims(pool: Iterable[str]) -> dict[str, frozenset[str]]:
    """The instruments, genres and voices this artist's captions name."""
    return dict(_pool(tuple(pool))[0])


# One sound under two spellings.
_ALIASES = {"sax": "saxophone", "synthpop": "synth pop", "lofi": "lo fi", "hihat": "hi hat", "dnb": "drum and bass",
            "drum & bass": "drum and bass", "rnb": "r&b", "rapping": "rap", "rapper": "rap", "screaming": "scream",
            "growling": "growl", "electronica": "electronic", "edm": "electronic", "synthesizer": "synth",
            "hip-hop": "hip hop", "trip-hop": "trip hop"}
_ENDINGS = ("s", "es", "y", "al", "ic", "ing", "er", "ed")


def _words(term: str) -> list[str]:
    term = _ALIASES.get(term, _ALIASES.get(_singular(term), term))
    return _singular(term.replace("-", " ")).split()


def _same_word(a: str, b: str) -> bool:
    """The same word, or one with a plain ending on the other ("funk", "funky"); "bass" is not "bassoon"."""
    short, long = sorted((a, b), key=len)
    return a == b or (len(short) >= 3 and long.startswith(short) and long[len(short):] in _ENDINGS)


def _inside(term: str, kind: str, have: frozenset[str]) -> bool:
    """Whether the catalogue holds ``term``: itself, its head or its kind, the family it names or belongs to."""
    if term in _EVERYDAY or _singular(term) in _EVERYDAY:
        return True
    if kind == "vocal":
        return term not in _VOICE_SOURCES or any(all(any(_same_word(w, x) for x in _words(h)) for w in _words(term)) for h in have)
    if not have:
        return False
    words = _words(term)
    held = [_words(h) for h in have]
    # Its head ("guitar" for a held "electric guitar"), or a held head made more exact ("indie rock" for "rock").
    if any(_same_word(h[-1], words[-1]) and (all(any(_same_word(w, x) for x in h) for w in words)
                                             or all(any(_same_word(x, w) for w in words) for x in h)) for h in held):
        return True
    if kind != "instrument":
        return False
    if generalises(term, set(have)):
        return True
    family = _family_of(term)
    families = {_family_of(h) for h in have} - {None}
    if family in _SAME_SOURCE and family in families:
        return True
    names = {_singular(h) for h in have}
    # A family's own name and its members stand for each other; two members do not.
    if family and (_singular(term) == family or family in names or family + "s" in have):
        return family in families
    if _singular(term) == "horn":
        return bool(names & (_REEDS | {_singular(m) for m in _FAMILIES["brass"]}))
    return False


_GLUE = r"(?:with|and|or|plus|featuring|feat\.?|ft\.?|on|over|of|by|for|to|from|in|&)"
_END_ARTICLE = re.compile(r"(?:^|(?<=\s))(?:a|an|the|some|An|The|Some)$|^A$")
_END_GLUE = re.compile(rf"(?:^|(?<=\s)){_GLUE}$", re.IGNORECASE)
_END_CUE = re.compile(r"(?:^|(?<=\s))(?:no|not|without|never|minus|sans)$", re.IGNORECASE)
_START_GLUE = re.compile(rf"{_GLUE}(?!\w)", re.IGNORECASE)
_START_JOIN = re.compile(r"(?:and|or|plus|&)(?!\w)\s*", re.IGNORECASE)
_END_A = re.compile(r"(?:^|(?<=\s))(an?)$|^(An?)$")
_STOPS = ".,;!?"
_LETTERS = re.compile(r"[^\W\d_]{3,}")
_FILLER = frozenset("""the and with for but not any some something anything music musical track tune song piece sound
sounds like kind sort vibe style feel mood make made give want please really very just only more most""".split())


def _join(left: str, right: str) -> str:
    """The two sides of a cut put back together: only the article, connective, comma or hyphen the dropped word
    left hanging is cleared, and nothing else in the prompt is touched."""
    left, right = left.rstrip(" \t-"), right.lstrip(" \t-")
    for _ in range(4):
        before = (left, right)
        closes = not right or right[0] in _STOPS + "\n"
        if closes or _START_GLUE.match(right):
            left = _END_ARTICLE.sub("", left).rstrip(" \t")
        if closes:
            left = _END_CUE.sub("", _END_GLUE.sub("", left).rstrip(" \t")).rstrip(" \t")
            if not right or right[0] in _STOPS:
                left = left.rstrip(" \t,;")
        opens = not left or left[-1] in _STOPS + "\n"
        if opens or _END_GLUE.search(left):
            right = _START_JOIN.sub("", right, count=1) if _START_JOIN.match(right) else right
        if opens:
            right = right.lstrip(" \t,;" if left else " \t,;.!?")
        if (left, right) == before:
            break
    article = _END_A.search(left)
    if article and right[:1].isalpha() and right[0].lower() != "u":
        want = "an" if right[0].lower() in "aeio" else "a"
        left = left[:article.start()] + (want if article.group(1) else want.capitalize())
    gap = " " if left and right and left[-1] != "\n" and right[0] not in _STOPS + "\n" else ""
    return left + gap + right


def hold(prompt: str, pool: Iterable[str]) -> dict:
    """``{prompt, dropped: [{word, kind}], changed, emptied}``: the prompt less every instrument, genre and kind of
    voice the pool's captions never name, byte for byte the same when nothing is outside. A pool under
    `MIN_CORPUS_CAPTIONS` captions holds nothing; a prompt left with nothing becomes the pool's commonest words."""
    text = str(prompt or "")
    captions = tuple(c for c in pool if isinstance(c, str) and c.strip())
    same = {"prompt": text, "dropped": [], "changed": False, "emptied": False}
    if len(captions) < MIN_CORPUS_CAPTIONS or not text.strip():
        return same
    have, common = _pool(captions)
    # The longest span wins where two overlap; one span read as two kinds is inside when either kind is.
    spans: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for start, end, term, kind in _found(text):
        spans.setdefault((start, end), []).append((term, kind))
    kept: list[tuple[int, int]] = []
    for span in sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0])):
        if not any(span[0] < e and s < span[1] for s, e in kept):
            kept.append(span)
    outside = {span: spans[span][0] for span in kept
               if not any(_inside(term, kind, have[kind]) for term, kind in spans[span])}
    if not outside:
        return same
    denials = _denial_spans(text)
    cuts: list[tuple[int, int]] = []
    dropped = []
    for (start, end), (term, kind) in sorted(outside.items()):
        clause = next(((a, max(b, end)) for a, b in denials if a <= start < b), None)
        # "no vocals" stays as written; a ruled-out instrument or genre goes with its clause, since the engine
        # reads the noun and not the "no".
        if clause and kind == "vocal":
            continue
        cuts.append(clause or (start, end))
        dropped.append({"word": " ".join(text[start:end].split()), "kind": kind})
    if not cuts:
        return same
    merged: list[list[int]] = []
    for a, b in sorted(set(cuts)):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    # Each kept stretch is the listener's own text; only the seam between two of them is mended.
    served, last = text[:merged[0][0]], merged[0][1]
    for a, b in merged[1:]:
        served, last = _join(served, text[last:a]), b
    served = _join(served, text[last:])
    served = served.strip(" \t")
    emptied = not [w for w in _LETTERS.findall(served.lower()) if w not in _FILLER]
    if emptied:
        served = ", ".join(common)
    return {"prompt": served, "dropped": dropped, "changed": True, "emptied": emptied}
