"""Caption fit in the contract: the frozen tokeniser, the guards, fit/answered/margin and the two meters."""

import pytest

from khaos_attribution import caption_fit as CF

CORPUS = [f"A soothing ambient synth pad drifts over a gentle pulse, take {i}." for i in range(10)]


def test_stopwords_are_frozen():
    """The measure's stopwords are the frozen 85; moving them moves every historical number."""
    assert len(CF._STOPWORDS) == 85
    assert {"no", "nor", "not", "the", "with"} <= CF._STOPWORDS


def test_tokens_keep_repeats_and_skip_short_and_stop_words():
    """Content words in order with repeats, no stemming."""
    assert CF.tokens("The drums and the drum, a pad") == ["drums", "drum", "pad"]


def test_profile_guards_refuse_small_and_unreadable_corpora():
    """Under eight captions or under 80% Latin letters no fit is claimed."""
    with pytest.raises(CF.CorpusTooSmall):
        CF.require_claimable(CF.profile(CORPUS[:3]))
    with pytest.raises(CF.CorpusUnreadable):
        CF.require_claimable(CF.profile(["静かなピアノの曲です"] * 10))
    CF.require_claimable(CF.profile(CORPUS))


def test_fit_and_answered_read_the_text():
    """In-vocabulary share and the request's surviving words, with the missing ones named."""
    prof = CF.profile(CORPUS)
    reading = CF.fit("ambient synth pad with harpsichord", prof)
    assert reading["in_vocab_tokens"] == 0.75 and reading["novel_terms"] == ["harpsichord"]
    kept = CF.answered("sad harpsichord", "ambient synth pad with harpsichord")
    assert kept["answered"] == 0.5 and kept["missing"] == ["sad"]


def test_margin_is_zero_for_identical_corpora():
    """Two identical corpora give no margin to any text."""
    prof = CF.profile(CORPUS, label="a")
    assert CF.margin("ambient synth pad", prof, CF.profile(CORPUS, label="b"))["margin"] == 0.0


def test_meters_give_both_readouts_and_say_why_one_is_missing():
    """fits reads the artist's words, answered compares to the request only when they differ."""
    prof = CF.profile(CORPUS)
    m = CF.meters("ambient synth pad with harpsichord", prof, request="sad harpsichord")
    assert m["fits"] == 0.75 and m["answered"] == 0.5 and m["missing"] == ["sad"]
    same = CF.meters("sad harpsichord", prof, request=" sad  harpsichord ")
    assert same["answered"] is None
    small = CF.meters("ambient synth", CF.profile(CORPUS[:2]))
    assert small["fits"] is None and "MIN_CORPUS_CAPTIONS" in small["no_fit_reason"]


FROZEN_STOPWORDS = (
    "a an and are as at be been being but by can could did do does for "
    "from had has have her here his how into is it its more most much no "
    "nor not of off on once only or other our out over own same she "
    "should so some such than that the their them then there these they "
    "this those through to too under until very was we were what when "
    "where which while who whom why will with would you your").split()


def test_the_stopword_list_is_pinned_word_for_word():
    """The one pin of the frozen list: adding or removing a word re-scores every stored number."""
    assert sorted(CF._STOPWORDS) == sorted(FROZEN_STOPWORDS)


def test_accented_words_stay_whole():
    """A word with an accent is one token, never a fragment shown as 'dropped from what you asked'."""
    assert CF.tokens("música alegre con guitarra") == ["música", "alegre", "con", "guitarra"]
    assert CF.tokens("Café crème, naïve Ångström") == ["café", "crème", "naïve", "ångström"]
    kept = CF.answered("música alegre", "warm guitar")
    assert kept["missing"] == ["alegre", "música"]


def test_decomposed_and_composed_accents_are_one_word():
    """NFC first, so an accent typed as a combining mark matches the precomposed letter."""
    assert CF.tokens("música") == CF.tokens("música") == ["música"]


def test_numbers_stay_invisible():
    """Digits are not letters: '190' and '808s' still contribute nothing, as ANSWERED_LIMITS says."""
    assert CF.tokens("190 bpm 808s") == ["bpm"]


def test_accented_latin_counts_as_readable():
    """Script coverage reads Latin letters with accents; another script is still unread."""
    assert CF.script_coverage(["música"])["coverage"] == 1.0
    assert CF.script_coverage(["静か"])["coverage"] == 0.0


def test_version_moves_with_the_tokeniser():
    """Stored readings name the measure's version; the Unicode tokeniser is 0.3.0."""
    assert CF.CAPTION_FIT_VERSION == "0.3.0"
