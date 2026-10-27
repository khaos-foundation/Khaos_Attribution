"""The caption vocabulary: the claim words (instrument, genre, vocal, production) and the mood and timbre words.

One list for every room: the Workshop's captions, gates and rewrite lint read it, and the translator's
lint (`translation`) holds a rewrite of a listener's request to it. Canonical singular forms; the
regexes match plurals and hyphen/space variants. Pure, no model.
"""

from __future__ import annotations

import functools
import re

__all__ = ["INSTRUMENTS", "GENRES", "VOCAL_TERMS", "PRODUCTION_TERMS", "MOOD_ADJECTIVES",
           "TIMBRE_ADJECTIVES", "CATEGORIES", "NOUN_CATEGORIES", "canonical_term",
           "extract_attributes", "content_nouns", "generalises"]

INSTRUMENTS: frozenset[str] = frozenset({
    "acoustic guitar", "electric guitar", "bass guitar", "slide guitar", "steel guitar",
    "nylon guitar", "twelve-string guitar", "guitar", "bass", "double bass", "upright bass",
    "synth bass", "sub-bass", "sub bass", "bassline",
    "drum", "drum kit", "drum machine", "kick", "kick drum", "snare", "snare drum", "hi-hat",
    "hihat", "cymbal", "ride cymbal", "crash cymbal", "tom", "percussion", "conga", "bongo",
    "tabla", "djembe", "cajon", "shaker", "tambourine", "clap", "handclap", "808", "808s",
    "piano", "grand piano", "upright piano", "electric piano", "rhodes", "wurlitzer",
    "keyboard", "organ", "hammond organ", "harpsichord", "clavinet", "celesta",
    "synth", "synthesizer", "synth pad", "pad", "synth lead", "lead synth", "arpeggio",
    "arpeggiator", "sequencer", "modular synth", "analog synth", "sampler", "sample",
    "string", "strings", "string section", "violin", "viola", "cello", "fiddle", "harp",
    "orchestra", "orchestral strings",
    "brass", "trumpet", "trombone", "horn", "french horn", "tuba", "saxophone", "sax",
    "clarinet", "flute", "oboe", "bassoon", "recorder", "pan flute", "harmonica",
    "accordion", "bagpipe", "banjo", "mandolin", "ukulele", "sitar", "oud", "koto",
    "shamisen", "erhu", "dulcimer", "zither", "bouzouki", "balalaika", "marimba",
    "xylophone", "vibraphone", "glockenspiel", "bell", "chime", "tubular bell",
    "steel drum", "steelpan", "gong", "timpani", "theremin", "vocoder", "talkbox",
    "mellotron", "beatbox", "turntable", "scratch", "loop", "breakbeat", "drum loop",
    "field recording", "found sound", "noise", "drone", "choir", "chorus vocals",
})

GENRES: frozenset[str] = frozenset({
    "rock", "indie rock", "alt rock", "alternative rock", "post-rock", "post rock",
    "prog rock", "progressive rock", "psychedelic rock", "garage rock", "hard rock",
    "classic rock", "soft rock", "punk", "punk rock", "post-punk", "pop punk", "hardcore",
    "metal", "heavy metal", "doom metal", "black metal", "death metal", "thrash metal",
    "metalcore", "nu metal", "grunge", "shoegaze", "dream pop", "noise rock", "emo",
    "pop", "synth-pop", "synthpop", "synth pop", "electropop", "indie pop", "art pop",
    "power pop", "k-pop", "j-pop", "dance-pop", "dance pop", "bubblegum pop",
    "electronic", "electronica", "edm", "house", "deep house", "tech house",
    "progressive house", "techno", "minimal techno", "trance", "psytrance",
    "drum and bass", "drum & bass", "dnb", "jungle", "dubstep", "garage", "uk garage",
    "2-step", "breakbeat", "big beat", "idm", "glitch", "downtempo", "trip hop", "trip-hop",
    "chillout", "chillwave", "lo-fi", "lofi", "lo-fi hip hop", "vaporwave", "synthwave",
    "retrowave", "darkwave", "coldwave", "industrial", "ebm", "ambient", "dark ambient",
    "drone", "new age", "space music", "electro", "eurodance", "italo disco", "disco",
    "nu-disco", "nu disco", "funk", "soul", "neo-soul", "neo soul", "r&b", "rnb",
    "contemporary r&b", "motown", "gospel", "blues", "delta blues", "blues rock",
    "jazz", "smooth jazz", "bebop", "cool jazz", "free jazz", "jazz fusion", "fusion",
    "acid jazz", "nu jazz", "swing", "big band", "bossa nova", "samba", "latin", "salsa",
    "reggaeton", "cumbia", "tango", "flamenco", "afrobeat", "afrobeats", "highlife",
    "reggae", "dub", "ska", "dancehall", "hip hop", "hip-hop", "rap", "trap", "drill",
    "boom bap", "grime", "country", "folk", "indie folk", "folk rock", "americana",
    "bluegrass", "singer-songwriter", "singer songwriter", "acoustic", "classical",
    "baroque", "romantic", "contemporary classical", "minimalism", "minimalist",
    "neoclassical", "film score", "soundtrack", "cinematic", "orchestral", "chamber",
    "opera", "choral", "world", "celtic", "middle eastern", "indian classical",
    "bollywood", "experimental", "avant-garde", "avant garde", "noise", "musique concrete",
    "spoken word", "lounge", "easy listening", "exotica", "children's", "holiday",
    "march", "polka", "waltz",
})

VOCAL_TERMS: frozenset[str] = frozenset({
    "vocal", "vocals", "lead vocal", "lead vocals", "backing vocal", "backing vocals",
    "background vocals", "harmony vocals", "vocal harmonies", "harmonies", "singer",
    "singing", "sung", "voice", "voices", "female vocal", "female vocals", "female voice",
    "female singer", "female lead vocal", "female lead vocals", "male vocal", "male vocals",
    "male voice", "male singer", "male lead vocal", "male lead vocals", "duet",
    "choir", "chant", "chanting", "rap", "rapping", "rapper", "spoken word", "spoken",
    "falsetto", "belting", "crooning", "whisper", "whispered", "screaming", "screams",
    "growl", "growling", "scream", "shout", "shouting", "humming", "vocoder", "autotune",
    "auto-tune", "vocal chop", "vocal chops", "vocal sample", "ad-lib", "ad-libs",
})

PRODUCTION_TERMS: frozenset[str] = frozenset({
    "reverb", "delay", "echo", "distortion", "overdrive", "fuzz", "saturation",
    "tape saturation", "compression", "sidechain", "sidechain compression", "chorus effect",
    "phaser", "flanger", "tremolo", "vibrato", "wah", "filter", "filter sweep", "low-pass",
    "lowpass", "high-pass", "highpass", "eq", "stereo", "wide stereo", "mono", "panning",
    "lo-fi production", "lo-fi", "hi-fi", "tape", "vinyl crackle", "crackle", "hiss",
    "bitcrush", "bitcrushed", "glitch", "granular", "layered", "layering", "double-tracked",
    "double tracked", "gated reverb", "spring reverb", "plate reverb", "hall reverb",
    "room reverb", "dry", "wet", "polished", "raw", "live recording", "studio",
    "sample-based", "sampled", "programmed drums", "live drums", "quantized", "swing",
    "mix", "mixed", "mastered", "production", "produced", "processing", "processed",
    "effects", "fx", "ambience", "atmosphere", "texture", "textures", "soundscape",
})

MOOD_ADJECTIVES: frozenset[str] = frozenset({
    "melancholic", "melancholy", "sad", "sombre", "somber", "mournful", "wistful",
    "nostalgic", "bittersweet", "dark", "brooding", "ominous", "menacing", "tense",
    "anxious", "eerie", "haunting", "mysterious", "hypnotic", "dreamy", "ethereal",
    "serene", "calm", "peaceful", "tranquil", "relaxed", "laid-back", "mellow", "gentle",
    "soft", "tender", "intimate", "warm", "cosy", "cozy", "romantic", "sensual", "sultry",
    "happy", "joyful", "uplifting", "euphoric", "triumphant", "hopeful", "optimistic",
    "bright", "cheerful", "playful", "quirky", "whimsical", "fun", "energetic", "upbeat",
    "driving", "urgent", "intense", "aggressive", "angry", "raw", "gritty", "heavy",
    "powerful", "epic", "anthemic", "cinematic", "dramatic", "majestic", "grand",
    "soaring", "expansive", "spacious", "atmospheric", "ambient", "meditative",
    "introspective", "reflective", "contemplative", "spiritual", "groovy", "funky",
    "danceable", "bouncy", "swaggering", "confident", "cool", "chill", "lazy", "hazy",
    "woozy", "psychedelic", "trippy", "restless", "frantic", "chaotic", "noisy",
    "abrasive", "harsh", "cold", "icy", "clinical", "sterile", "lush", "rich", "sparse",
    "minimal", "stripped-back", "stripped back", "bare", "delicate", "fragile",
    "emotional", "passionate", "yearning", "longing", "solemn", "reverent", "sacred",
    "festive", "celebratory", "mischievous", "sinister", "unsettling", "uneasy",
})

TIMBRE_ADJECTIVES: frozenset[str] = frozenset({
    "warm", "bright", "dark", "muted", "muffled", "crisp", "clean", "distorted", "fuzzy",
    "crunchy", "gritty", "smooth", "silky", "velvety", "airy", "breathy", "raspy",
    "husky", "gravelly", "nasal", "thin", "thick", "fat", "punchy", "tight", "loose",
    "boomy", "deep", "low", "high", "shimmering", "glassy", "metallic", "woody", "hollow",
    "resonant", "ringing", "sustained", "staccato", "plucked", "strummed", "fingerpicked",
    "picked", "bowed", "muted", "palm-muted", "tremolo-picked", "arpeggiated", "pulsing",
    "throbbing", "pounding", "thumping", "rolling", "shuffling", "syncopated",
    "swung", "steady", "insistent", "hypnotic", "swirling", "washed-out", "washed out",
    "saturated", "overdriven", "crushed", "lo-fi", "hi-fi", "polished", "glossy",
    "grainy", "dusty", "vintage", "retro", "modern", "futuristic", "analog", "analogue",
    "digital", "acoustic", "electric", "electronic", "organic", "synthetic", "layered",
    "dense", "sparse", "wide", "narrow", "spacious", "close", "distant", "reverberant",
    "echoing", "dry", "wet", "soft", "loud", "quiet", "gentle", "heavy", "light",
    "delicate", "aggressive", "mellow", "soulful", "bluesy", "jazzy", "funky", "groovy",
    "melodic", "harmonic", "rhythmic", "percussive", "droning", "buzzing", "humming",
})


def _vocab_regex(terms: frozenset[str]) -> re.Pattern:
    ordered = sorted(terms, key=lambda t: (-len(t), t))
    alts = "|".join(re.escape(t).replace(r"\ ", r"[\s-]") for t in ordered)
    return re.compile(rf"(?<![\w-])(?:{alts})(?:s|es)?(?![\w-])", re.IGNORECASE)


_INSTRUMENT_RE = _vocab_regex(INSTRUMENTS)
_GENRE_RE = _vocab_regex(GENRES)
_VOCAL_RE = _vocab_regex(VOCAL_TERMS)
_PRODUCTION_RE = _vocab_regex(PRODUCTION_TERMS)
_MOOD_RE = _vocab_regex(MOOD_ADJECTIVES)
_TIMBRE_RE = _vocab_regex(TIMBRE_ADJECTIVES)

CATEGORIES: tuple[tuple[str, re.Pattern, frozenset[str]], ...] = (
    ("genre", _GENRE_RE, GENRES),
    ("mood", _MOOD_RE, MOOD_ADJECTIVES),
    ("instrument", _INSTRUMENT_RE, INSTRUMENTS),
    ("vocal", _VOCAL_RE, VOCAL_TERMS),
    ("timbre", _TIMBRE_RE, TIMBRE_ADJECTIVES),
    ("production", _PRODUCTION_RE, PRODUCTION_TERMS),
)

# Categories whose terms are CLAIMS about what is audible — the lint set.
NOUN_CATEGORIES: tuple[str, ...] = ("instrument", "genre", "vocal", "production")


def canonical_term(match_text: str, terms: frozenset[str]) -> str:
    """Map a surface match ("Synths", "Hi Hats") to its canonical vocabulary key."""
    t = match_text.lower().replace("-", " ").strip()
    candidates = [t, t[:-1] if t.endswith("s") else t, t[:-2] if t.endswith("es") else t]
    for c in candidates:
        for term in (c, c.replace(" ", "-")):
            if term in terms:
                return term
    for term in terms:
        if term.replace("-", " ") == t or term.replace("-", " ") + "s" == t:
            return term
    return t


def extract_attributes(text: str) -> dict[str, list[str]]:
    """Vocabulary hits by category, in order of first appearance, canonical
    singular form; a word in two categories appears in both."""
    return {name: list(hits) for name, hits in _extract_attributes(text or "")}


@functools.lru_cache(maxsize=8192)
def _extract_attributes(text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    # Cached: the gates ask this twice per segment on every spot-check poll.
    out = []
    for name, rx, terms in CATEGORIES:
        seen: list[str] = []
        for m in rx.finditer(text):
            term = canonical_term(m.group(0), terms)
            if term not in seen:
                seen.append(term)
        out.append((name, tuple(seen)))
    return tuple(out)


def content_nouns(text: str) -> set[str]:
    """The claim-bearing terms of a caption: instrument, genre, vocal and
    production hits (canonical form); what the rewrite lint and MIR gate read."""
    attrs = extract_attributes(text)
    return {t for cat in NOUN_CATEGORIES for t in attrs[cat]}


def generalises(noun: str, allowed: set[str]) -> bool:
    """Whether ``noun`` is in ``allowed`` or is the head of an allowed compound ("guitar" for "electric guitar")."""
    if noun in allowed:
        return True
    for a in allowed:
        if a.endswith(" " + noun) or a.endswith("-" + noun):
            return True
    return noun in INSTRUMENTS and any(noun in a.split() for a in allowed)
