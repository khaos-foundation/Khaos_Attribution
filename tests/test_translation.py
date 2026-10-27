"""The translator's prompt, lint and fallback chain, with a fake model; nothing is loaded."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from khaos_attribution import translation as T
from khaos_attribution.prompt_expansion import tidy_caption

POOL = ["Melancholic ambient piano over a soft synth pad, warm tape hiss and a slow drum machine pulse.",
        "Warm strings swell over a gentle electric piano with lush reverb and a dreamy texture.",
        "Dark techno with a driving kick, distorted bass and metallic percussion under filtered synth pads."]
REQ = "sad solo piano for a rainy night"
GOOD = ("A melancholic solo piano for a rainy night, sad and intimate, with warm tape hiss and lush reverb "
        "around every note.")
TAIL = " with warm tape hiss and lush reverb around every slow, intimate note of the night."


def fake(*answers):
    calls = []

    def llm(prompt, temperature=0.0, seed=None, timeout=None):
        calls.append({"prompt": prompt, "temperature": temperature, "seed": seed, "timeout": timeout})
        return answers[min(len(calls), len(answers)) - 1]
    llm.calls = calls
    return llm


def lint(request, text, instrumental=True):
    return T.lint_translation(request, text, instrumental=instrumental, pool=POOL)[1]


# ── the request check, shared by the translator and the 5Hz LM ──

def test_an_instrumental_caption_with_any_voice_is_rejected():
    """Every form of a voice, even denied ("no vocals"), is a vocal on an instrumental caption."""
    for words in ("vocals", "a voice", "singing", "spoken words", "lyrics", "ethereal vox", "a cappella layers",
                  "ooh-ing", "murmured words", "beatboxed rhythm", "chanted syllables", "a choral hymn",
                  "an operatic swell", "a crooner", "rapped verses", "a vocalist's breath", "whispery lines",
                  "the singer's tone", "scat", "yodel", "ad-libs", "no vocals at all"):
        reasons = T.check_caption("sad piano", f"sad piano with {words}", instrumental=True)
        assert any(r.startswith("vocal_on_instrumental") for r in reasons), words
    assert T.check_caption("sad piano", "sad piano with soft vocals", instrumental=False) == []
    assert T.vocal_hits("an alto saxophone and a tenor sax, single notes") == []


def test_a_clock_time_the_strip_missed_is_rejected():
    """Clock times, marks, spelled-out durations and ordinal minutes are timestamps."""
    for caption in ("sad piano. At the two-minute mark it swells.", "sad piano, swelling at 1:26",
                    "sad piano returning after thirty seconds", "sad piano lasting ten minutes",
                    "sad piano around the second minute", "sad piano at ninety seconds it explodes"):
        assert any(r.startswith("clock_time") for r in T.check_caption("sad piano", caption, instrumental=True)), caption


def test_a_caption_that_drops_the_request_is_rejected():
    """Under the floor of a two-word request; filler words are not asked for; one word is not measured."""
    assert T.check_caption(REQ, "ambient techno with a four on the floor kick", instrumental=True) == ["answered:0.00"]
    assert T.ANSWERED_FLOOR == 0.5
    assert T.check_caption("something tense for a stealth level", "a tense stealth-level cue", instrumental=True) == []
    assert T.check_caption("piano", "ambient techno with a kick drum", instrumental=True) == []


# ── the lint: what the body names ──

def test_lint_passes_a_faithful_translation():
    assert lint(REQ, GOOD) == []


def test_a_copied_request_counts_for_nothing():
    """A leading copy of the request is taken out before the body is measured."""
    assert any(r.startswith("answered") for r in lint(REQ, "Sad solo piano for a rainy night." + TAIL))


def test_the_pool_lends_no_instrument_or_genre_when_the_request_names_one():
    """Asked for strings and a harp, a drone and dark ambient from the artist's pool are replacements."""
    reasons = lint("warm strings and a soft harp", "Warm, soft strings beside a harp give way to a dark ambient "
                   "synth pad and distorted bass, slow and lush with reverb around the long warm notes.")
    assert "invented_noun:synth pad" in reasons and "invented_noun:bass" in reasons


def test_the_pool_lends_instruments_when_the_request_names_none():
    """"Something tense" names nothing: the artist's own instruments are the translation."""
    text = ("A tense, uneasy stealth level cue of filtered synth pads and metallic percussion, sparse and dark, "
            "with lush reverb and a dreamy texture around every hit.")
    assert lint("something tense for a stealth level", text) == []


def test_naming_a_family_names_its_members_and_a_specialisation_is_no_invention():
    text = ("Funky bass locked to tight drums, a crisp snare and a kick drum, an electric bass line and "
            "funk swagger with lush reverb around every hit of the groove.")
    assert lint("funky bass and tight drums", text) == []


def test_lint_rejects_a_claim_noun_from_nowhere():
    assert "invented_noun:banjo" in lint(REQ, GOOD.replace("warm tape hiss", "a banjo"))


def test_negating_what_was_asked_is_rejected():
    """A request noun inside the body's own denial is the request turned down."""
    reasons = lint("piano and cello", "A piano and cello duo, slow and lush, with no cello at all and "
                   "warm tape hiss and lush reverb around every soft note of the night.")
    assert "negated_request:cello" in reasons


def test_what_the_request_rules_out_stays_out_however_it_is_said():
    """'no X', 'without X', 'X-free' and 'X are not wanted' deny the whole family, to the end of the clause."""
    body = (", then a driving kick and metallic percussion arrive with hats ticking over warm strings,"
            " lush reverb and a dreamy texture.")
    for request in ("warm strings, no drums", "warm strings without any percussion at all",
                    "warm strings, but drums are not wanted", "warm strings, drum-free"):
        reasons = lint(request, "Warm strings swell" + body)
        assert any(r.startswith("denied:") for r in reasons), request
    assert "denied:cello" in lint("piano, no strings", "A piano, slow and bare, then a lone cello enters with warm tape "
                                  "hiss and lush reverb around every slow note of the long night.")
    assert lint("ambient pads, drum-free", "Soft ambient synth pads with a dreamy texture and lush reverb, devoid of "
                "any drum or percussive elements, warm and slow across the long night.") == []
    assert "invented_noun:hardstyle" in lint("solo cello", "The cello plays solo over a hardstyle pulse, warm and "
                                              "slow, with lush reverb around every note of the night.")
    asserted, denied = T.split_denials("no drums but a big kick, and strings")
    assert "kick" in asserted and "strings" in asserted and "drums" in denied


def test_the_request_s_moods_stay_and_never_flip():
    """A mood asked for is kept (its own word or a kindred one); its listed opposite is a changed request."""
    assert "mood_dropped:sad" in lint(REQ, "A solo piano for a rainy night, intimate and bare, with warm tape "
                                      "hiss and lush reverb around every slow note.")
    assert lint(REQ, GOOD.replace("sad and intimate", "mournful and intimate")) == []
    assert "mood_opposite:sad->joyful" in lint(REQ, GOOD.replace("intimate", "joyful"))


def test_lint_rejects_names_numbers_theory_and_comparisons():
    for reason, text in {"theory:In A minor": GOOD + " In A minor.",
                         "number:120": GOOD.replace("sad and", "120 sad and"),
                         "theory:one hundred and twenty beats": GOOD + " At one hundred and twenty beats a minute.",
                         "theory:three-four time": GOOD + " In three-four time.",
                         "theory:key of": GOOD + " In the key of d.",
                         "sound_alike:reminiscent of": GOOD + " Reminiscent of rain.",
                         "sound_alike:called": GOOD + " The track is called Midnight.",
                         "name:burial": GOOD.replace("intimate", "intimate like burial"),
                         "name:Satie": GOOD.replace("solo piano", "solo piano Satie")}.items():
        assert reason in lint(REQ, text), reason


def test_capitals_are_names_only_in_english():
    """A capitalised word mid-sentence is a name in English text; German capitalises every noun."""
    assert T._names("Ein trauriges Klavier für eine Regennacht mit Hall", set()) == []
    assert T._names("A sad piano for Burial fans", set()) == ["Burial"]


def test_length_is_words_for_spaced_scripts_and_characters_for_others():
    assert any(r.startswith("length") for r in lint("sad piano", "A sad piano, slow and soft."))
    assert not any(r.startswith("length") for r in lint("静かなピアノ", "静かなピアノの曲、雨の夜に柔らかく響く温かい残響。"))


# ── the request and the fallback ──

def test_no_vocals_leaves_an_instrumental_request():
    """The flag carries 'instrumental'; the engine reads 'no vocals' as a word about vocals."""
    assert T.clean_request("solo cello, no vocals", True) == "solo cello"
    assert T.clean_request("ambient pads, vocal-free", True) == "ambient pads"
    assert T.clean_request("solo cello, no vocals", False) == "solo cello, no vocals"
    out = T.expand_checked("solo cello, no vocals", instrumental=True, attempt=0, pool=POOL,
                           shots=lambda a: POOL, translator=fake("A dull drone."))
    assert out["caption"] == "solo cello" and out["fallback"] == "request"


# ── the prompt and the answer ──

def test_prompt_is_chatml_with_thinking_off_and_the_rule():
    p = T.build_prompt("sad piano", POOL, instrumental=True)
    assert p.startswith("<|im_start|>system\n") and "/no_think" in p and p.endswith("<think>\n\n</think>\n\n")
    assert "You have NOT heard any recording" in p and "never add an instrument" in p
    assert "Name no instrument, genre or voice the request does not name" in p and "Write in English" in p
    assert "Instrumental: do not mention vocals" in p and POOL[0] in p
    assert "Sung" in T.build_prompt("sad piano", POOL, instrumental=False)


def test_answer_extraction_reads_the_three_lines_or_a_bare_caption():
    raw = f"<think>hmm</think>\nLanguage: English\nEnglish request: {REQ}\nCaption: \"{GOOD}\"<|im_end|>junk"
    assert T.parse_answer(raw) == {"language": "English", "english_request": REQ, "caption": tidy_caption(GOOD)}
    assert T.extract_answer("Here is the caption:\n" + GOOD) == tidy_caption(GOOD)


def test_a_foreign_request_is_checked_in_its_english_rendering_and_said_so():
    """The checks read the model's English rendering; the answer names the language it came from."""
    raw = ("Language: Spanish\nEnglish request: happy music with guitar\nCaption: A happy, bright guitar melody "
           "over a lively groove, warm and joyful, with lush reverb and a dreamy texture around every note.")
    out = T.expand_checked("música alegre con guitarra", instrumental=True, attempt=0, pool=POOL,
                           shots=lambda a: POOL, translator=fake(raw))
    assert out["fallback"] is None and out["translated_from"] == "Spanish"
    assert out["translated_note"] == "Said in English from your Spanish request."
    claimed = raw.replace("Language: Spanish", "Language: French").replace("happy music with guitar", REQ)
    assert T._rendered(REQ, T.parse_answer(claimed)) == (REQ, None)


# ── attempts, the deadline and the fallback chain ──

def test_translate_retries_warmer_with_a_seed_on_the_next_window_then_passes():
    llm = fake("ambient techno with a kick drum and a robotic voice", GOOD)
    windows = []
    out = T.translate(REQ, lambda n: windows.append(n) or POOL[:1], instrumental=True, pool=POOL, llm=llm)
    assert out["caption"] == GOOD and out["attempts"] == 2 and len(out["rejects"]) == 1 and windows == [0, 1]
    assert out["rejects"][0]["text"].startswith("ambient techno")
    assert llm.calls[0]["temperature"] == 0.0 and llm.calls[0]["seed"] is None
    assert llm.calls[1]["temperature"] > 0 and llm.calls[1]["seed"] is not None


def test_three_failed_translations_return_the_request_never_an_invention():
    lm_calls = []
    out = T.expand_checked(REQ, instrumental=True, attempt=0, pool=POOL, shots=lambda a: POOL,
                           translator=fake("ambient techno with a kick"),
                           lm=lambda a: lm_calls.append(a) or {"caption": GOOD})
    assert out["caption"] == REQ and out["fallback"] == "request" and out["translator"] == "request"
    assert out["attempts"] == 3 and len(out["rejects"]) == 3 and out["fallback_note"] == T.FALLBACK_NOTE
    assert lm_calls == []


def test_one_deadline_covers_every_call():
    """Each call gets what is left of one budget; nothing starts after it."""
    clock = {"t": 100.0}
    llm = fake("ambient techno with a kick")

    def slow(prompt, **kw):
        clock["t"] += 8.0
        return llm(prompt, **kw)
    real = time.monotonic
    T.time.monotonic = lambda: clock["t"]
    try:
        out = T.translate(REQ, POOL, instrumental=True, pool=POOL, llm=slow, deadline=120.0)
    finally:
        T.time.monotonic = real
    assert [round(c["timeout"]) for c in llm.calls] == [20, 12, 4]
    assert out["attempts"] == 3
    T.time.monotonic = lambda: clock["t"]
    try:
        out = T.translate(REQ, POOL, instrumental=True, pool=POOL, llm=slow, deadline=clock["t"] + 8.4)
    finally:
        T.time.monotonic = real
    assert len(out["rejects"]) == 1


def test_an_unreachable_translator_falls_back_to_the_checked_lm():
    def down(prompt, **kw):
        raise OSError("connection refused")
    seen = []

    def lm(a):
        seen.append(a)
        return {"caption": "a voice sings over ambient techno" if a == 0 else GOOD, "bpm": 80}
    out = T.expand_checked(REQ, instrumental=True, attempt=0, pool=POOL, shots=lambda a: POOL, translator=down, lm=lm)
    assert out["translator"] == "lm" and out["caption"] == GOOD and out["bpm"] == 80
    assert seen == [0, 1] and out["served_attempt"] == 1 and out["attempts"] == 2
    assert out["rejects"][0]["reasons"][0].startswith("unavailable") and out["rejects"][1]["by"] == "lm"


def test_an_lm_that_raises_is_a_refusal_never_an_error():
    """A busy or failing LM attempt is recorded and the chain goes on to the request."""
    class Busy(RuntimeError):
        pass

    def lm(a):
        raise Busy("an expansion is already running")
    out = T.expand_checked(REQ, instrumental=True, attempt=0, pool=POOL, shots=lambda a: POOL, lm=lm)
    assert out["fallback"] == "request" and [r["by"] for r in out["rejects"]] == ["lm"] * 3
    assert out["rejects"][0]["reasons"][0].startswith("Busy")


def test_the_lm_failing_three_times_returns_the_request():
    out = T.expand_checked("no drums, just warm strings", instrumental=True, attempt=2, pool=POOL,
                           shots=lambda a: POOL, lm=lambda a: {"caption": "a spoken intro over warm strings"})
    assert out["fallback"] == "request" and out["attempts"] == 3 and out["caption"] == "no drums, just warm strings"


def test_the_label_is_the_one_the_server_reports_on_passes_and_rejects():
    def llm(prompt, **kw):
        llm.label = "qwen3-4b"
        return "ambient techno with a kick" if len(llm.seen) < 1 else GOOD
    llm.seen = []
    llm.label = "qwen3-8b"
    real = llm

    def counting(prompt, **kw):
        out = real(prompt, **kw)
        real.seen.append(1)
        counting.label = real.label
        return out
    counting.label = "qwen3-8b"
    out = T.expand_checked(REQ, instrumental=True, attempt=0, pool=POOL, shots=lambda a: POOL, translator=counting)
    assert out["translator"] == "qwen3-4b" and out["rejects"][0]["by"] == "qwen3-4b"


def test_model_label_names_the_model_not_the_quantisation():
    assert T.model_label("/models/Qwen3-8B-Q4_K_M.gguf") == "qwen3-8b"
    assert T.model_label("Qwen3-4B-Q4_K_M.gguf") == "qwen3-4b"
    assert T.model_label(None) == T.DEFAULT_LABEL


# ── the server ──

class _Server(BaseHTTPRequestHandler):
    bodies = []
    mode = "ok"

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Server.bodies.append(body)
        if _Server.mode == "trickle":
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            for _ in range(20):
                self.wfile.write(b" ")
                self.wfile.flush()
                time.sleep(0.3)
            return
        if _Server.mode == "huge":
            payload = json.dumps({"content": "x " * (T.MAX_RESPONSE_BYTES // 2 + 10)}).encode()
        elif _Server.mode == "slow":
            time.sleep(0.6)
            payload = json.dumps({"content": GOOD, "model": "/m/Qwen3-8B-Q4_K_M.gguf"}).encode()
        else:
            payload = json.dumps({"content": GOOD, "model": "/m/Qwen3-8B-Q4_K_M.gguf"}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Server.mode = "ok"
    T._HEALTH.clear()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_http_llm_speaks_llama_servers_completion(server):
    llm = T.translator_from_env({T.TRANSLATOR_URL_ENV: server})
    assert llm("p", temperature=0.4, seed=7) == GOOD and llm.label == "qwen3-8b"
    assert _Server.bodies[-1] == {"prompt": "p", "n_predict": T.MAX_TOKENS, "temperature": 0.4, "seed": 7,
                                  "cache_prompt": True, "stop": ["<|im_end|>"]}


def test_a_trickling_or_oversized_answer_is_cut_off(server):
    """The body is read against the time left and a size cap, however slowly it arrives."""
    llm = T.http_llm(server)
    _Server.mode = "trickle"
    started = time.monotonic()
    with pytest.raises(T.TranslatorUnavailable):
        llm("p", timeout=1.0)
    assert time.monotonic() - started < 2.5
    T._HEALTH.clear()
    _Server.mode = "huge"
    with pytest.raises(T.TranslatorUnavailable):
        llm("p", timeout=5.0)


def test_at_most_the_servers_slots_are_used_at_once(server):
    """Calls beyond the server's slots wait for one in the client, inside their own budget."""
    _Server.mode = "slow"
    llm = T.http_llm(server)
    results = []
    threads = [threading.Thread(target=lambda: results.append(_try(llm))) for _ in range(3)]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 3 and time.monotonic() - started >= 1.7
    assert T.SERVER_PARALLEL == 1


def _try(llm):
    try:
        llm("p", timeout=5.0)
        return "ok"
    except T.TranslatorUnavailable:
        return "busy"


def test_http_llm_unreachable_is_translator_unavailable():
    T._HEALTH.clear()
    llm = T.http_llm("http://127.0.0.1:9", timeout=0.5)
    with pytest.raises(T.TranslatorUnavailable):
        llm("p")


def test_no_url_means_no_translator_and_the_budget_reads_from_env():
    assert T.translator_from_env({}) is None
    assert T.deadline_seconds({}) == 20.0 and T.deadline_seconds({T.TRANSLATOR_TIMEOUT_ENV: "45"}) == 45.0


def test_a_sung_request_served_by_the_translator_says_no_lyrics_were_drafted():
    sung = GOOD
    out = T.expand_checked(REQ, instrumental=False, attempt=0, pool=POOL, shots=lambda a: POOL, translator=fake(sung))
    assert out["lyrics"] is None and out["lyrics_note"] == T.LYRICS_NOTE
    out = T.expand_checked(REQ, instrumental=True, attempt=0, pool=POOL, shots=lambda a: POOL, translator=fake(sung))
    assert out["lyrics_note"] is None


# ── the third pass: renderings, voices in lists, sound sources, moods, absences, repeats, names ──

def test_an_english_request_is_never_swapped_for_a_rendering():
    """A model claiming French with an unrelated rendering does not change what an English request is checked against."""
    raw = ("Language: French\nEnglish request: aggressive techno with a pounding kick\nCaption: Aggressive techno with "
           "a pounding kick, distorted bass and metallic percussion under filtered synth pads, driving all night.")
    out = T.translate("warm strings and a soft harp", POOL, instrumental=True, pool=POOL, llm=fake(raw))
    assert out["caption"] is None and out["translated_from"] is None
    assert T._reads_english("sad solo piano") and not T._reads_english("música alegre con guitarra")
    swap = ("Language: French\nEnglish request: happy piano party\nCaption: A happy, joyful piano party, bright and "
            "bouncy, with lush reverb and a dreamy texture around every note.")
    out = T.translate("sad solo piano", POOL, instrumental=True, pool=POOL, llm=fake(swap))
    assert out["caption"] is None and "mood_dropped:sad" in out["rejects"][0]["reasons"]


def test_a_foreign_rendering_keeps_cognates_and_adds_no_claim():
    """'guitarra' must come back as a guitar, and the rendering may not bring drums the original never named."""
    assert T._rendering_reasons("música alegre con guitarra", "happy music with guitar") == []
    assert T._rendering_reasons("música alegre con guitarra", "happy music with drums and guitar") == ["rendering_adds:drum"]
    assert T._rendering_reasons("música alegre con guitarra", "happy music") == ["rendering_drops:guitarra"]
    raw = ("Language: Spanish\nEnglish request: happy music with drums and guitar\nCaption: A happy, bright guitar "
           "melody over lively drums, warm and joyful, with lush reverb and a dreamy texture around every note.")
    out = T.translate("música alegre con guitarra", POOL, instrumental=True, pool=POOL, llm=fake(raw))
    assert out["caption"] is None and "rendering_adds:drum" in out["rejects"][0]["reasons"]


def test_a_voice_leaves_a_denial_list_and_the_rest_stays():
    assert T.clean_request("ambient pads without singing or drums", True) == "ambient pads without drums"
    assert T.clean_request("strings, vocals and drums are not wanted", True) == "strings, drums are not wanted"
    assert T.clean_request("piano with no voice, choir or percussion", True) == "piano with no percussion"
    assert T.clean_request("no vocals, no drums, just pads", True) == "no drums, just pads"


def test_sound_sources_outside_the_vocabulary_are_claims():
    """Beats, keys, a low end, leads, a reese, a wobble, electronics and saws name instruments too."""
    body = ("Warm strings beside a soft harp drift through a gentle, intimate space, lush reverb and tape hiss around "
            "them, tender and unhurried, {}.")
    for extra, noun in (("with glitchy beats", "beat"), ("over bright plinking keys", "keys"),
                        ("over a thumping low end", "low end"), ("with buzzy electronics", "electronics"),
                        ("soon replaced by sawtooth leads", "lead"), ("and a wobbling reese", "reese")):
        assert f"invented_noun:{noun}" in lint("warm strings and a soft harp", body.format(extra)), extra
    assert lint("warm strings and a soft harp", body.format("the harp's glassy plucks ringing")) == []


def test_wider_mood_opposites_and_kindred_words():
    """'ecstatic', 'blissful', 'breakneck', 'luminous' flip their opposites; 'triumphant' answers 'victory'."""
    assert "mood_opposite:sad->ecstatic" in lint(REQ, GOOD.replace("intimate", "ecstatic"))
    for mood, body in (("calm", "violent"), ("slow", "breakneck"), ("dark", "luminous")):
        reasons = lint(f"{mood} solo piano", f"A {mood} solo piano piece, soft and intimate, {body} at heart, with warm "
                       "tape hiss and lush reverb around every note of the night.")
        assert f"mood_opposite:{mood}->{body}" in reasons, mood
    fanfare = ("A triumphant brass fanfare, bold and celebratory, rich resonant brass and lush reverb around a grand, "
               "uplifting swell.")
    assert lint("a big brass fanfare for a victory screen", fanfare) == []
    assert "mood_opposite:triumphant->aggressive" in lint(
        "a big brass fanfare for a victory screen", "A big brass fanfare for the victory screen, aggressive and "
        "mechanical, with lush reverb and relentless drive across every bar.")


def test_a_requested_noun_that_fades_out_or_is_absent_is_turned_down():
    body = ("Warm strings beside a soft harp drift through a gentle, intimate space with lush reverb and tape hiss, "
            "{}.")
    for tail in ("until the strings fade out and vanish entirely", "although the strings are absent here"):
        assert "negated_request:strings" in lint("warm strings and a soft harp", body.format(tail)), tail


def test_a_body_that_only_repeats_the_request_is_not_a_translation():
    assert T.MIN_NEW_WORDS == 6
    reasons = lint("warm strings and a soft harp", "Warm strings and soft harp, warm strings and a soft harp, warm "
                   "strings and soft harps, warm strings and the soft harp, gentle.")
    assert any(r.startswith("repeats_request") for r in reasons)


def test_similes_reach_for_images_not_names():
    """'recede like tides' is an image; 'like Burial' or 'like burial' is a name; a sentence opened by one too."""
    assert T._names("Pads swell and recede like tides over the night", set()) == []
    assert T._names("Pads swell like burial over the night", set()) == ["burial"]
    assert T._names("Pads swell for the night. Burial would make this.", set()) == ["Burial"]


def test_a_busy_model_is_said_to_be_busy():
    """When nothing was written at all the note says the models were busy, not that the request was changed."""
    def lm(a):
        raise RuntimeError("An expansion is already running")
    out = T.expand_checked(REQ, instrumental=True, attempt=0, pool=POOL, shots=lambda a: POOL, lm=lm)
    assert out["fallback_note"] == T.BUSY_NOTE
    out = T.expand_checked(REQ, instrumental=True, attempt=0, pool=POOL, shots=lambda a: POOL,
                           lm=lambda a: {"caption": "ambient techno"})
    assert out["fallback_note"] == T.FALLBACK_NOTE
