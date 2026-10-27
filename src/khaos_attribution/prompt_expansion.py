"""Prompt expansion — one idea in, the adapter's caption dialect out.

Shared by the Workshop and the Listening Space: the exemplar file shape,
exemplar selection, the query template, the tidy-up and the trigger rule.
The exemplar file (``prompt_exemplars.json``, written by the Workshop at
training completion) is ``{"exemplars_version", "artist_id",
"trigger_phrase", "created_at_utc", "embedding": {"model", "dim"} | null,
"exemplars": [{"text", "section", "track", "vector" | null}, ...]}``;
0.1.0 files carry plain strings and no vectors, and both are read.
Heavy imports stay inside functions.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

EXEMPLARS_VERSION = "0.2.0"
EXEMPLARS_FILENAME = "prompt_exemplars.json"
EMBEDDING_MODEL = "Qwen3-Embedding-0.6B"

# More exemplars or a higher temperature make the LM copy the examples'
# musical content rather than their dialect.
FEW_SHOT_COUNT = 3
TEMPERATURE = 0.5

# How many of the nearest catalogue captions a listener is offered as starting points.
NEAREST_COUNT = 3

# The band training captions sit in; the result is clipped to the exporter's
# own limit so nothing longer than a training caption reaches the engine.
CAPTION_WORDS = (45, 70)
MIN_CAPTION_CHARS = 80
MAX_CAPTION_CHARS = 420

_STOPWORDS = frozenset("""
a an and the of for to in on at with by from into over under as is are be
it its this that these those some something very really bit kind sort more
most like music track piece song sound sounds feel feels feeling mood
""".split())
_WORD_RE = re.compile(r"[a-z']+")


# ── exemplars ────────────────────────────────────────────────────────────

def normalise_exemplars(payload: dict | None) -> list[dict]:
    """The file's exemplars as dicts ``{text, section, track, vector}``
    whatever version wrote them. Empty texts are dropped."""
    out: list[dict] = []
    for item in (payload or {}).get("exemplars") or []:
        if isinstance(item, str):
            entry = {"text": item, "section": None, "track": None, "vector": None}
        elif isinstance(item, dict):
            entry = {"text": str(item.get("text") or ""),
                     "section": item.get("section"),
                     "track": item.get("track"),
                     "vector": item.get("vector")}
        else:
            continue
        entry["text"] = " ".join(entry["text"].split())
        if entry["text"]:
            out.append(entry)
    return out


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for w in _WORD_RE.findall(text.lower()):
        w = w.strip("'")
        if len(w) < 3 or w in _STOPWORDS:
            continue
        # Crude suffix strip so "drums" meets "drum"; a tie-breaker, not a stemmer.
        for suffix in ("ing", "ed", "es", "s"):
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        out.add(w)
    return out


def lexical_score(request: str, text: str) -> float:
    """How much of the request's vocabulary the caption covers, with a small
    credit for the caption being about that and little else."""
    q, t = _tokens(request), _tokens(text)
    if not q or not t:
        return 0.0
    shared = len(q & t)
    return shared / len(q) + 0.25 * shared / len(t)


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rank_exemplars(request: str, exemplars: list[dict],
                   request_vector: Iterable[float] | None = None) -> tuple[list[int], str]:
    """Exemplar indices, most relevant first, and the method used:
    ``"embedding"`` when every exemplar carries a vector and the request's
    vector is given, else ``"lexical"``. Stable: ties keep file order."""
    vec = list(request_vector) if request_vector is not None else None
    if vec and exemplars and all(e.get("vector") for e in exemplars) \
            and all(len(e["vector"]) == len(vec) for e in exemplars):
        # A dimension mismatch would score every exemplar 0 under the
        # "embedding" label, so it ranks lexically instead.
        scores = [_cosine(vec, e["vector"]) for e in exemplars]
        method = "embedding"
    else:
        scores = [lexical_score(request, e["text"]) for e in exemplars]
        method = "lexical"
    order = sorted(range(len(exemplars)), key=lambda i: -scores[i])
    return order, method


def select_exemplars(request: str, exemplars: list[dict], attempt: int = 0,
                     count: int = FEW_SHOT_COUNT,
                     request_vector: Iterable[float] | None = None) -> tuple[list[int], str]:
    """The ``count`` exemplars to show the LM for this ``attempt``: the top
    of the ranking first, then the next window for each retry, wrapping —
    every attempt sees a different set while the catalogue allows it."""
    order, method = rank_exemplars(request, exemplars, request_vector)
    if not order:
        return [], method
    n = len(order)
    count = max(1, min(count, n))
    # With fewer exemplars than a window, rotate which one leads instead.
    start = (max(0, attempt) * count) % n if n > count else max(0, attempt) % n
    chosen = [order[(start + i) % n] for i in range(count)]
    return chosen, method


def nearest_captions(exemplars: list[dict], picks: list[int], track_names: dict | None = None,
                     count: int = NEAREST_COUNT) -> list[dict]:
    """The first ``count`` of ``picks`` (pool indices, most relevant first) as ``{text, section, track,
    track_name}`` for a listener to start from; ``track_name`` only where the artist named the track."""
    names = track_names or {}
    out: list[dict] = []
    for index in picks[:max(0, count)]:
        entry = exemplars[index]
        track = entry.get("track")
        name = names.get(track) if isinstance(track, str) else None
        out.append({"text": entry["text"], "section": entry.get("section"),
                    "track": track if isinstance(track, str) else None,
                    "track_name": name if isinstance(name, str) and name.strip() else None})
    return out


# ── the query ────────────────────────────────────────────────────────────

# "translate" restates the request in the artist's own caption words; "elaborate" is the earlier
# instruction (copy the style, not the content), kept so the judge can render both.
QUERY_STYLES = ("translate", "elaborate")
DEFAULT_QUERY_STYLE = "translate"

_NO_METADATA = ("Do not mention BPM, key, time signature, duration, timestamps or clock times, "
                "or any artist, band or song names.")


def build_query(request: str, shots: list[str], instrumental: bool = True, *,
                style: str = DEFAULT_QUERY_STYLE) -> str:
    """The text handed to the 5Hz LM's Simple-mode entry: by default the request's meaning kept and
    restated in the examples' own vocabulary, nothing added that the request does not imply."""
    if style not in QUERY_STYLES:
        raise ValueError(f"unknown query style {style!r}; expected one of {QUERY_STYLES}")
    lo, hi = CAPTION_WORDS
    examples = "\n".join(f"- {s}" for s in shots)
    vocals = ("The music is instrumental — no vocals, no singing."
              if instrumental else "The music has sung vocals.")
    if style == "elaborate":
        return (
            "Here are example descriptions of this artist's music. Copy their "
            "writing STYLE — vocabulary, sentence rhythm, level of detail — but "
            "NOT their musical content:\n"
            f"{examples}\n\n"
            f"Request: {request.strip()}\n\n"
            "Write one flowing prose music caption describing music that matches "
            "the request's mood and character, in exactly the writing style of the "
            f"examples: a single paragraph of roughly {lo}–{hi} words. {vocals} "
            f"Every musical detail must serve the request. {_NO_METADATA}"
        )
    return (
        "Here are descriptions of this artist's music, in the artist's own words:\n"
        f"{examples}\n\n"
        f"Request: {request.strip()}\n\n"
        "Translate the request into this artist's words. Keep exactly what the request "
        "asks for — its mood, instruments, energy and character — and restate it the way "
        "these descriptions would. Wherever they fit what was asked, reuse the examples' "
        "own words for instruments, textures, moods and production. Add no musical detail "
        "the request does not imply. Write one flowing prose music caption: a single "
        f"paragraph of roughly {lo}–{hi} words. {vocals} {_NO_METADATA}"
    )


# ── tidying what comes back ──────────────────────────────────────────────

# The LM leaks clock-time anchors ("At 1:26, …") despite the instruction.
# Longer alternatives first, or "At" would match inside "At around".
_TIMESTAMP = re.compile(
    r"(?:,\s*)?\b(?:[Aa]t\s+around|[Aa]round|[Aa]t|[Aa]fter|[Bb]y|[Nn]ear)\s+"
    r"(?:the\s+)?\d{1,2}:\d{2}(?:\s+mark)?,?\s*"
    r"|\(\s*\d{1,2}:\d{2}\s*\)"
)
_TIDY_SPACES = re.compile(r"\s{2,}")
_TIDY_PUNCT = re.compile(r"\s+([.,;!?])")


def strip_timestamps(text: str) -> str:
    text = _TIMESTAMP.sub("", text)
    text = _TIDY_SPACES.sub(" ", text)
    return _TIDY_PUNCT.sub(r"\1", text).strip(" ,;")


def clip_caption(text: str, limit: int = MAX_CAPTION_CHARS) -> str:
    """Clip to the last full sentence inside ``limit`` (the whole caption
    when it fits) — the exporter's rule, applied to the LM's output too."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = cut.rfind(". ")
    return cut[: dot + 1] if dot > MIN_CAPTION_CHARS else cut.rstrip() + "…"


_WRAPPING_QUOTES = "'\"`\u2018\u2019\u201c\u201d"


def strip_wrapping_quotes(text: str) -> str:
    """Strip quote marks wrapping the whole caption; apostrophes inside the
    prose stay."""
    text = (text or "").strip()
    while len(text) > 1 and text[0] in _WRAPPING_QUOTES and (text[-1] in _WRAPPING_QUOTES
                                                             or text.count(text[0]) == 1):
        text = text[1:].strip()
        if text and text[-1] in _WRAPPING_QUOTES:
            text = text[:-1].strip()
    if text and text[-1] in _WRAPPING_QUOTES and text.count(text[-1]) == 1:
        text = text[:-1].strip()
    return text


def tidy_caption(text: str) -> str:
    return clip_caption(strip_timestamps(strip_wrapping_quotes(text or "")))


def with_trigger(caption: str, trigger: str | None) -> str:
    """``trigger`` in front of ``caption`` unless it is already there —
    a rework's base caption may carry the trigger (even a longer, per-track
    one) from the generation it derives from."""
    caption = (caption or "").strip()
    trigger = (trigger or "").strip()
    if not trigger:
        return caption
    artist = trigger.split()[0]
    if caption == artist or caption.startswith(artist + " "):
        return caption
    return f"{trigger} {caption}".strip()


# ── harvesting the LM's whole answer ─────────────────────────────────────

_FLAT_TO_SHARP = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#",
                  "CB": "B", "FB": "E"}
_KEYSCALE = re.compile(r"^\s*([A-Ga-g])\s*([#♯b♭]?)\s*[- ]?\s*(major|minor|maj|min|m)?\s*$", re.IGNORECASE)


def normalise_keyscale(value) -> str | None:
    """``"A minor"`` form — the form the apps' key pickers list (sharps
    only) — from the LM's spellings (``a min``, ``Bb Major``, ``F#m``…)."""
    if not isinstance(value, str):
        return None
    m = _KEYSCALE.match(value)
    if not m:
        return None
    note = m.group(1).upper()
    accidental = m.group(2)
    if accidental in ("#", "♯"):
        note = {"E#": "F", "B#": "C"}.get(note + "#", note + "#")  # the pickers list no E#/B#
    elif accidental in ("b", "♭"):
        note = _FLAT_TO_SHARP.get(note + "B", note + "b")
    mode = (m.group(3) or "major").lower()
    mode = "minor" if mode in ("minor", "min", "m") else "major"
    return f"{note} {mode}"


def normalise_timesignature(value) -> str | None:
    """Beats per bar as the engines take it (``"4"``), from ``"4/4"``,
    ``"6/8"``, ``4``…"""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        beats = int(value)
    elif isinstance(value, str):
        m = re.match(r"^\s*(\d{1,2})(?:\s*/\s*\d{1,2})?\s*$", value)
        if not m:
            return None
        beats = int(m.group(1))
    else:
        return None
    return str(beats) if 2 <= beats <= 12 else None


def _int_in(value, lo: int, hi: int) -> int | None:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return n if lo <= n <= hi else None


def harvest(metadata: dict | None, instrumental: bool = True) -> dict:
    """Everything usable from the LM's answer, normalised: the caption
    tidied; tempo, key and time signature in the apps' own forms (``None``
    when absent or implausible); a lyrics draft only for a sung request
    (the LM's ``[Instrumental]`` placeholder is not a draft)."""
    md = metadata or {}
    lyrics = md.get("lyrics")
    if instrumental or not isinstance(lyrics, str) or not lyrics.strip() \
            or lyrics.strip().lower() in ("[instrumental]", "instrumental"):
        lyrics = None
    language = md.get("language")
    return {
        "caption": tidy_caption(str(md.get("caption") or "")),
        "bpm": _int_in(md.get("bpm"), 30, 300),
        "keyscale": normalise_keyscale(md.get("keyscale")),
        "timesignature": normalise_timesignature(md.get("timesignature")),
        "duration": _int_in(md.get("duration"), 10, 600),
        "lyrics": lyrics.strip() if lyrics else None,
        "language": language if isinstance(language, str) and language.strip()
                    and language.strip().lower() != "unknown" else None,
    }


# ── embeddings (optional; torch + transformers) ──────────────────────────

def embed_texts(model, tokenizer, texts: list[str], max_length: int = 256):
    """L2-normalised mean-pooled embeddings from a loaded Qwen3-Embedding
    encoder, as a ``(len(texts), dim)`` numpy array. Loads nothing."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    device = next(model.parameters()).device
    rows = []
    with torch.inference_mode():
        for text in texts:
            enc = tokenizer(text, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            ids = enc.input_ids.to(device)
            mask = enc.attention_mask.to(device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            m = mask.unsqueeze(-1).float()
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            rows.append(torch.nn.functional.normalize(pooled, dim=-1)[0].cpu().numpy())
    return np.stack(rows).astype(np.float32)


class TextEmbedder:
    """Lazy owner of a Qwen3-Embedding encoder. ``available()`` says whether
    the weights and libraries are present without loading them."""

    def __init__(self, model_dir, device: str | None = None):
        from pathlib import Path  # noqa: PLC0415
        import threading  # noqa: PLC0415
        self.model_dir = Path(model_dir)
        self.device = device
        self._pair = None
        self._load_lock = threading.Lock()

    def available(self) -> bool:
        if not (self.model_dir / "config.json").exists():
            return False
        try:
            import torch  # noqa: F401, PLC0415
            import transformers  # noqa: F401, PLC0415
        except Exception:
            return False
        return True

    def _load(self):
        with self._load_lock:  # two first calls must not both load 1.2 GB
            if self._pair is None:
                import torch  # noqa: PLC0415
                from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
                device = self.device
                if device is None:
                    device = ("cuda" if torch.cuda.is_available()
                              else "mps" if getattr(torch.backends, "mps", None)
                              and torch.backends.mps.is_available() else "cpu")
                tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                model = AutoModel.from_pretrained(str(self.model_dir)).to(device).eval()
                self._pair = (model, tokenizer)
            return self._pair

    def embed(self, texts: list[str]):
        model, tokenizer = self._load()
        return embed_texts(model, tokenizer, texts)
