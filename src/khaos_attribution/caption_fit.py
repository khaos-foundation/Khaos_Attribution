"""Caption fit: does a text read like this artist's captions, and did it keep the request.

Pure functions, no model: a smoothed unigram profile of the artist's captions,
``fit`` and ``margin`` against it, ``answered`` for what survived of a request,
and ``meters`` for the two readouts the rooms show beside a prompt. All are claims
about text, never about audio or weights. Latin script only; ``script_coverage``
measures what the tokeniser missed and a profile under the guards claims no fit.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from khaos_attribution.prompt_expansion import MAX_CAPTION_CHARS, MIN_CAPTION_CHARS

__all__ = ["CAPTION_FIT_VERSION", "MIN_CORPUS_CAPTIONS", "MIN_SCRIPT_COVERAGE",
           "SMOOTHING", "ANSWERED_LIMITS", "NoFitClaimed", "CorpusTooSmall",
           "CorpusUnreadable", "tokens", "script_coverage", "profile",
           "require_claimable", "fit", "margin", "answered", "meters"]

# 0.2.0: `margin` normalises both corpora to a common pseudo-total and support.
CAPTION_FIT_VERSION = "0.2.0"

# Below this many captions no fit is claimed (one track's worth of segments); a guard, not a calibration.
MIN_CORPUS_CAPTIONS = 8

# Below this share of the corpus's letters read by the tokeniser, no fit is claimed.
MIN_SCRIPT_COVERAGE = 0.8

# Jeffreys rather than add-one: with corpora this small, add-one flattens the differences being read.
SMOOTHING = 0.5

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

# Frozen: the measure's own stopwords, not the selection tokeniser's; a change needs a version bump.
_STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from
had has have her here his how into is it its more most much no nor not of
off on once only or other our out over own same she should so some such
than that the their them then there these they this those through to too
under until very was we were what when where which while who whom why
will with would you your
""".split())

ANSWERED_LIMITS = (
    "answered is exact-token bag coverage: it cannot see negation ('no drums' "
    "and 'pounding drums' share every token it reads), it scores a paraphrase "
    "('harpsichord' -> 'plucked keyboard') as a miss, and numbers are invisible "
    "('190', '808s' contribute nothing; 'bpm' counts). Read the missing list, "
    "not the share alone.")


def tokens(text: str) -> list[str]:
    """The text's content words in order, with repeats; no stemming, so "drum" and "drums" stay apart."""
    out: list[str] = []
    for word in _WORD_RE.findall(str(text).lower()):
        word = word.strip("'-")
        if len(word) < 3 or word in _STOPWORDS:
            continue
        out.append(word)
    return out


def script_coverage(texts: list[str]) -> dict:
    """The share of the corpus's letters that are Latin A–Z, and the scripts that were skipped."""
    letters = readable = 0
    unread: Counter[str] = Counter()
    for text in texts:
        for ch in str(text):
            if not ch.isalpha():
                continue
            letters += 1
            if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
                readable += 1
            else:
                unread[unicodedata.name(ch, "UNKNOWN").split()[0]] += 1
    return {
        "letters": letters,
        "readable_letters": readable,
        "coverage": round(readable / letters, 4) if letters else None,
        "unread_scripts": [name for name, _ in unread.most_common(3)],
    }


class NoFitClaimed(RuntimeError):
    """The corpus cannot carry a fit claim."""


class CorpusTooSmall(NoFitClaimed):
    """Fewer than MIN_CORPUS_CAPTIONS usable captions."""


class CorpusUnreadable(NoFitClaimed):
    """The tokeniser reads under MIN_SCRIPT_COVERAGE of the corpus's letters."""


def profile(captions: list[str], *, label: str = "") -> dict:
    """The distribution: word counts, caption lengths and the captions; ``words_sd`` is full precision
    because `fit` divides by it."""
    texts = [" ".join(str(c).split()) for c in captions if str(c).strip()]
    counts: Counter[str] = Counter()
    lengths: list[int] = []
    for text in texts:
        toks = tokens(text)
        counts.update(toks)
        lengths.append(len(toks))
    mean = sum(lengths) / len(lengths) if lengths else 0.0
    var = (sum((n - mean) ** 2 for n in lengths) / (len(lengths) - 1)
           if len(lengths) > 1 else 0.0)
    return {
        "label": label,
        "captions": texts,
        "n_captions": len(texts),
        "counts": counts,
        "vocab_size": len(counts),
        "total_tokens": sum(counts.values()),
        "words_mean": mean,
        "words_sd": math.sqrt(var),
        "top_terms": [w for w, _ in counts.most_common(20)],
        "script": script_coverage(texts),
        "caption_chars": {"min": MIN_CAPTION_CHARS, "max": MAX_CAPTION_CHARS},
        "caption_fit_version": CAPTION_FIT_VERSION,
    }


def require_claimable(prof: dict) -> None:
    """Raise `CorpusTooSmall` or `CorpusUnreadable` when this profile cannot carry a fit claim."""
    if prof["n_captions"] < MIN_CORPUS_CAPTIONS:
        raise CorpusTooSmall(
            f"{prof['label'] or 'corpus'}: {prof['n_captions']} usable captions, "
            f"under MIN_CORPUS_CAPTIONS={MIN_CORPUS_CAPTIONS} — no fit claimed")
    script = prof["script"]
    coverage = script["coverage"]
    if coverage is None or coverage < MIN_SCRIPT_COVERAGE:
        seen = ", ".join(script["unread_scripts"]) or "none"
        raise CorpusUnreadable(
            f"{prof['label'] or 'corpus'}: the tokeniser reads "
            f"{'no' if coverage is None else f'{coverage:.0%} of'} this corpus's "
            f"letters (Latin only; unread: {seen}), under "
            f"MIN_SCRIPT_COVERAGE={MIN_SCRIPT_COVERAGE:.0%} — no fit claimed")


def _mean_logp(toks: list[str], counts: Counter, *, support: int,
               pseudo_total: float | None = None) -> float | None:
    """Mean log P(word) under an add-k unigram model of ``counts``, scaled to ``pseudo_total`` when given;
    None, never NaN, when there is nothing to score."""
    total = sum(counts.values())
    if not toks or not total or support <= 0:
        return None
    scale = float(total if pseudo_total is None else pseudo_total)
    denom = scale + SMOOTHING * support
    acc = 0.0
    for word in toks:
        expected = counts.get(word, 0) / total * scale
        acc += math.log((expected + SMOOTHING) / denom)
    return acc / len(toks)


def fit(text: str, prof: dict) -> dict:
    """How much this text reads like a caption from ``prof``; ``typicality`` compares only before/after
    against ONE profile, never between artists (``margin`` is the comparable form)."""
    toks = tokens(text)
    types = set(toks)
    seen = set(prof["counts"])
    # fixed support: the corpus's vocabulary plus ONE bucket for everything unseen
    support = len(seen) + 1
    sd = prof["words_sd"]
    typicality = _mean_logp(toks, prof["counts"], support=support)
    return {
        "words": len(toks),
        "length_z": (round((len(toks) - prof["words_mean"]) / sd, 2)
                     if sd else None),
        "in_vocab": (round(len(types & seen) / len(types), 4) if types else None),
        "in_vocab_tokens": (round(sum(1 for w in toks if w in seen) / len(toks), 4)
                            if toks else None),
        "novel_terms": sorted(types - seen)[:20],
        "typicality": round(typicality, 4) if typicality is not None else None,
        "against": prof["label"],
        "corpus_captions": prof["n_captions"],
        "caption_fit_version": CAPTION_FIT_VERSION,
    }


def margin(text: str, prof: dict, control: dict) -> dict:
    """Nats per word by which the text is likelier under the artist's caption model than a control's,
    both corpora scaled to the smaller token total over the shared support; both guarded."""
    require_claimable(prof)
    require_claimable(control)
    toks = tokens(text)
    a_counts, b_counts = prof["counts"], control["counts"]
    pseudo_total = float(min(sum(a_counts.values()), sum(b_counts.values())))
    support = len(set(a_counts) | set(b_counts)) + 1
    a = _mean_logp(toks, a_counts, support=support, pseudo_total=pseudo_total)
    b = _mean_logp(toks, b_counts, support=support, pseudo_total=pseudo_total)
    both = a is not None and b is not None
    return {
        "margin": round(a - b, 4) if both else None,
        "artist_logp": round(a, 4) if a is not None else None,
        "control_logp": round(b, 4) if b is not None else None,
        "artist": prof["label"],
        "control": control["label"],
        "balanced_on_tokens": pseudo_total,
        "support": support,
        "artist_captions": prof["n_captions"],
        "control_captions": control["n_captions"],
        "caption_fit_version": CAPTION_FIT_VERSION,
    }


def answered(request: str, text: str) -> dict:
    """The share of the request's content words that survived into the text, and the missing ones by name;
    a bag of words, so `ANSWERED_LIMITS` travels in the result."""
    asked = set(tokens(request))
    got = set(tokens(text))
    if not asked:
        return {"answered": None, "asked": 0, "missing": [],
                "answered_limits": ANSWERED_LIMITS}
    kept = asked & got
    return {"answered": round(len(kept) / len(asked), 4),
            "asked": len(asked),
            "missing": sorted(asked - got),
            "answered_limits": ANSWERED_LIMITS}


def meters(text: str, prof: dict, request: str | None = None) -> dict:
    """The two readouts beside a prompt: ``fits`` (share of its words the artist's captions use) and
    ``answered`` (share of ``request`` kept); each None when there is nothing to claim."""
    out = {"fits": None, "answered": None, "missing": [], "novel_terms": [],
           "no_fit_reason": None, "corpus_captions": prof["n_captions"],
           "caption_fit_version": CAPTION_FIT_VERSION}
    try:
        require_claimable(prof)
    except NoFitClaimed as exc:
        out["no_fit_reason"] = str(exc)
    else:
        reading = fit(text, prof)
        out["fits"] = reading["in_vocab_tokens"]
        out["novel_terms"] = reading["novel_terms"]
    if request and " ".join(str(request).split()) != " ".join(str(text).split()):
        kept = answered(request, text)
        out["answered"] = kept["answered"]
        out["missing"] = kept["missing"]
    return out
