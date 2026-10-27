"""The shared caption vocabulary: claim nouns, canonical forms and the generalisation rule."""

from khaos_attribution import caption_vocabulary as V


def test_content_nouns_are_the_claim_categories_in_canonical_form():
    assert V.content_nouns("Synths and a drum machine, female vocals, lush reverb, sad mood") == {
        "synth", "drum machine", "female vocals", "reverb"}


def test_a_head_noun_generalises_its_compound():
    """'guitar' for 'electric guitar' is not an invention; 'banjo' is."""
    assert V.generalises("guitar", {"electric guitar"})
    assert not V.generalises("banjo", {"electric guitar"})


def test_the_lists_hold_their_words():
    """The lists the Workshop's captions and gates read; a removal changes what every lint accepts."""
    assert {"piano", "drum machine", "strings"} <= V.INSTRUMENTS and "techno" in V.GENRES
    assert {"voice", "spoken word"} <= V.VOCAL_TERMS and "reverb" in V.PRODUCTION_TERMS
    assert [c for c, _, _ in V.CATEGORIES] == ["genre", "mood", "instrument", "vocal", "timbre", "production"]
