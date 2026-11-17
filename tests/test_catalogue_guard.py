"""The catalogue guard drops the instruments, genres and kinds of voice an artist's captions never name, leaves
everything else as the listener wrote it, and holds nothing for a catalogue too small to say what it lacks."""

import pytest

from khaos_attribution import catalogue_guard as G
from khaos_attribution.caption_fit import MIN_CORPUS_CAPTIONS

POOL = [
    "A slow piano piece with warm strings and a soft synth pad, calm and spacious.",
    "Driving drums and a deep bass guitar under bright electric guitar, an indie rock chorus.",
    "A solo cello line over grand piano, classical and intimate.",
    "Punchy kick drum and hi-hat with a rolling bassline, a house groove.",
    "Female vocals over acoustic guitar, a quiet folk verse.",
    "Layered synth leads and arpeggios, a bright synth pop hook.",
    "Brushed snare and upright bass with a muted trumpet, a late jazz mood.",
    "Orchestral strings swell under a lone violin, cinematic and wide.",
]


def held(prompt, pool=POOL):
    return G.hold(prompt, pool)


@pytest.mark.parametrize("prompt, served, words", [
    ("upbeat banjo tune", "upbeat tune", ["banjo"]),
    ("a banjo and piano duet, hopeful", "piano duet, hopeful", ["banjo"]),
    ("Banjo-driven stomp with warm piano", "driven stomp with warm piano", ["Banjo"]),
    ("a piano-banjo duet", "a piano duet", ["banjo"]),
    ("rock-reggae with drums", "rock with drums", ["reggae"]),
    ("alt-country ballad on piano", "alt ballad on piano", ["country"]),
    ("warm piano with sitar and tabla", "warm piano", ["sitar", "tabla"]),
    ("a slow piano piece, no banjo", "a slow piano piece", ["banjo"]),
    ("piano over a reggae groove", "piano over a groove", ["reggae"]),
    ("a didgeridoo drone under strings", "a drone under strings", ["didgeridoo"]),
    ("an accordion and piano waltz", "piano", ["accordion", "waltz"]),
    ("rap over drums and bass", "over drums and bass", ["rap"]),
])
def test_what_the_catalogue_never_held_is_dropped(prompt, served, words):
    out = held(prompt)
    assert out["prompt"] == served and [d["word"] for d in out["dropped"]] == words and out["changed"]


@pytest.mark.parametrize("prompt", [
    "a slow piano piece with warm strings", "driving drums and electric guitar, indie rock",
    "guitar and bass", "a rock chorus with snare and cymbals", "deep house with a sub bass",
    "violin and viola over a string section", "synth pads and an arpeggiator", "keys and brass, a jazz mood",
    "female vocals, whispered, with harmonies", "heavy, dark and cinematic with no vocals",
    "something with soul and a big lead", "a calm ambient piece, lo-fi and dusty", "",
])
def test_a_prompt_inside_the_catalogue_comes_back_byte_for_byte(prompt):
    assert held(prompt) == {"prompt": prompt, "dropped": [], "changed": False, "emptied": False}


def test_the_catalogues_own_captions_are_never_changed():
    assert [c for c in POOL if held(c)["changed"]] == []


def test_a_sibling_is_not_the_instrument_the_catalogue_holds():
    """A catalogue with a piano has not trained a harpsichord, and one with a trumpet has not trained a tuba."""
    assert [d["word"] for d in held("harpsichord and tuba")["dropped"]] == ["harpsichord", "tuba"]


def test_one_source_under_several_words_is_held_together():
    """Drums, bass and synths are one source each: a kit that was trained is not refused its toms or its 808."""
    assert not held("808s, toms and a reese bass under a square wave lead")["changed"]


def test_a_prompt_left_with_nothing_becomes_the_catalogues_commonest_words():
    out = held("banjo")
    assert out["emptied"] and out["dropped"] == [{"word": "banjo", "kind": "instrument"}]
    assert out["prompt"] and not held(out["prompt"])["changed"]


def test_no_vocals_is_left_as_written_and_a_voice_the_catalogue_lacks_is_dropped():
    assert not held("piano, no rapping, no vocals")["changed"]
    assert held("piano with screaming")["prompt"] == "piano"


def test_a_small_catalogue_holds_nothing():
    small = POOL[:MIN_CORPUS_CAPTIONS - 1]
    assert held("a banjo hoedown", small) == {"prompt": "a banjo hoedown", "dropped": [], "changed": False, "emptied": False}


@pytest.mark.parametrize("prompt", [
    "a banjo hoedown with fiddle", "Banjo-driven country stomp, no accordion, with warm piano", "banjo",
    "sitar raga with tabla and a soft pad", "rap verse over a trap beat with 808s", "an accordion waltz",
    "piano and banjo duet, hopeful", "heavy metal riff, no vocals", "lo-fi hip hop beat with harpsichord",
])
def test_holding_twice_changes_nothing_more(prompt):
    once = held(prompt)["prompt"]
    assert held(once)["prompt"] == once


def test_the_claims_are_the_pools_own():
    claims = G.catalogue_claims(POOL)
    assert {"piano", "cello", "trumpet"} <= claims["instrument"] and "banjo" not in claims["instrument"]
    assert {"jazz", "folk", "house"} <= claims["genre"] and set(claims) == set(G.KINDS)


def test_only_the_seam_of_a_cut_is_mended():
    """Words nobody dropped are the listener's: a repeated word, a key, an article and a line break all stay."""
    asked = "On and on, in the key of A, for an hour:\na unique piano line with banjo, then on and on"
    assert held(asked)["prompt"] == "On and on, in the key of A, for an hour:\na unique piano line, then on and on"


def test_two_spellings_of_one_sound_are_one_sound_and_a_longer_word_is_another():
    pool = [*POOL, "A smoky saxophone solo over rapping and a synthesizer drone."]
    assert not held("sax and synths with a rap verse", pool)["changed"]
    assert [d["word"] for d in held("bass and bassoon")["dropped"]] == ["bassoon"]


@pytest.mark.parametrize("prompt, served", [
    ("piano, no banjo or drum and bass", "piano"),
    ("piano, no drum and bass", "piano"),
    ("banjo sitar piano", "piano"),
    ("piano with banjo, sitar and tabla", "piano"),
])
def test_cuts_that_touch_or_overlap_leave_none_of_their_words(prompt, served):
    """A denial clause and a named sound that runs past its end are one cut: nothing listed as dropped is served."""
    out = held(prompt)
    assert out["prompt"] == served
    assert not [d for d in out["dropped"] if d["word"].lower() in out["prompt"].lower()]


def test_a_word_inside_a_held_name_is_not_that_name():
    """A catalogue with folk rock has trained rock of a kind, not folk; and a key of A is not an article."""
    pool = [*POOL[:4], *POOL[5:], "A stomping folk rock chorus with drums.", "An electronic pulse under piano."]
    assert [d["word"] for d in held("a folk tune", pool)["dropped"]] == ["folk"]
    assert not held("edm with a rock chorus", pool)["changed"]
    assert held("in the key of A banjo opening", pool)["prompt"] == "in the key of A opening"
