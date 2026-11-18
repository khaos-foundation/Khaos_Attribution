"""Every prompt is said in the catalogue's words: the translator's caption is served only when it names nothing the
captions never name and reads at least FITS_FLOOR from them; otherwise the caption is stitched from the nearest
captions of two tracks, and the listener's own words are never served as they are."""

import pytest

from khaos_attribution import catalogue_prompt as C
from khaos_attribution.caption_fit import fit, profile

POOL = [
    "Driving four on the floor kick with a deep rolling sub bass and bright synth stabs, an energetic house groove.",
    "Hypnotic arpeggiated synth lead over a pounding kick and crisp hi-hats, a dark techno build.",
    "Warm analog pads swell under a punchy drum machine beat, a dreamy electronic breakdown.",
    "Glitchy vocal chops bounce over a heavy trap beat with booming 808 bass and sharp snares.",
    "A gritty distorted bassline drives a fast techno groove with clattering percussion.",
    "Shimmering synth arpeggios and a steady kick, an uplifting trance drop with wide pads.",
    "Sparse minimal techno with a dry kick, ticking hi-hats and a filtered synth stab.",
    "A bouncy electro house groove with a funky synth bass and handclaps on the backbeat.",
]
RANKED = [{"text": t, "track": f"t{i % 4}"} for i, t in enumerate(POOL)]
GOOD = ("An energetic house groove with a driving kick, a deep rolling sub bass and bright synth stabs, warm analog "
        "pads swelling into a dreamy electronic breakdown.")


def _llm(answer):
    calls = []

    def run(prompt, **kw):
        calls.append(prompt)
        if isinstance(answer, Exception):
            raise answer
        return f"Caption: {answer}"
    run.calls = calls
    return run


def test_a_caption_in_the_catalogues_words_is_served_as_the_translator_wrote_it():
    llm = _llm(GOOD)
    out = C.catalogue_prompt("upbeat dance track", pool=POOL, ranked=RANKED, instrumental=True, llm=llm)
    assert out["rewriter"] == "translator" and out["caption"] == GOOD and out["fits"] >= C.FITS_FLOOR
    assert "upbeat dance track" in llm.calls[0], "the request reaches the translator, after the guard"


def test_an_outside_instrument_in_the_translators_caption_is_never_served():
    bad = GOOD.replace("bright synth stabs", "a twangy banjo")
    out = C.catalogue_prompt("banjo country", pool=POOL, ranked=RANKED, instrumental=True, llm=_llm(bad))
    assert out["rewriter"] == "stitched" and "banjo" not in out["caption"].lower()
    assert any(r.startswith("outside:banjo") for rej in out["rejects"] for r in rej["reasons"])


def test_a_caption_under_the_floor_of_catalogue_words_is_never_served():
    off = "A tender whimsical lullaby about seashells and clouds drifting quietly over meadows at sunrise, gently swaying in moonlit breezes."
    reasons = C.lint_catalogue(off, POOL, instrumental=True)
    assert any(r.startswith("fits:") for r in reasons)


@pytest.mark.parametrize("llm", [None, _llm(TimeoutError("busy"))])
def test_with_no_translator_the_caption_is_stitched_from_two_tracks_and_reads_as_the_catalogue(llm):
    out = C.catalogue_prompt("banjo country hoedown, fast and happy", pool=POOL, ranked=RANKED, instrumental=True,
                             llm=llm)
    caption = out["caption"]
    assert out["rewriter"] == "stitched" and "banjo" not in caption.lower() and "country" not in caption.lower()
    assert caption not in POOL and out["fits"] >= C.FITS_FLOOR
    assert fit(caption, profile(POOL))["in_vocab_tokens"] == out["fits"]


def test_the_listeners_words_are_never_served_as_they_are():
    for request in ("banjo country", "classical piano", "x"):
        out = C.catalogue_prompt(request, pool=POOL, ranked=RANKED, instrumental=True)
        assert out["caption"] != request and out["fits"] >= C.FITS_FLOOR


def test_a_catalogue_too_small_to_hold_refuses():
    with pytest.raises(ValueError):
        C.catalogue_prompt("x", pool=POOL[:3], ranked=RANKED[:3], instrumental=True)


def test_a_cancel_stops_before_the_translator_is_asked():
    llm = _llm(GOOD)
    out = C.catalogue_prompt("x", pool=POOL, ranked=RANKED, instrumental=True, llm=llm, cancelled=lambda: True)
    assert llm.calls == [] and out["rewriter"] == "stitched"
