"""The translator's prompt, lint and fallback chain, with a fake model; nothing is loaded."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from khaos_attribution import translation as T
from khaos_attribution.prompt_expansion import tidy_caption

POOL = ["Melancholic ambient piano over a soft synth pad, warm tape hiss and a slow drum machine pulse.",
        "Warm strings swell over a gentle electric piano with lush reverb and a dreamy texture."]


def fake(*answers):
    calls = []

    def llm(prompt, temperature=0.0, seed=None):
        calls.append({"prompt": prompt, "temperature": temperature, "seed": seed})
        return answers[min(len(calls), len(answers)) - 1]
    llm.calls = calls
    return llm


GOOD = ("A sad, melancholic solo piano piece for a rainy night, soft and intimate, warm tape hiss "
        "and lush reverb around every note.")


# ── item 1: the request check, shared by the translator and the 5Hz LM ──

def test_an_instrumental_caption_with_a_voice_is_rejected():
    """Vocals, voice, singing, spoken words or lyrics on an instrumental request are an invention."""
    for word in ("vocals", "voice", "singing", "spoken", "lyrics"):
        reasons = T.check_caption("sad piano", f"sad piano with a {word} line", instrumental=True)
        assert any(r.startswith("vocal_on_instrumental") for r in reasons), word
    assert T.check_caption("sad piano", "sad piano with soft vocals", instrumental=False) == []


def test_a_denied_vocal_is_not_a_claim():
    """'no vocals' in an instrumental caption says what is absent, not what is there."""
    assert T.check_caption("sad piano", "sad piano, no vocals at all", instrumental=True) == []


def test_a_clock_time_the_strip_missed_is_rejected():
    """'At the two-minute mark' and '1:26' are timestamps whatever the strip caught."""
    for caption in ("sad piano. At the two-minute mark the piano swells.", "sad piano, swelling at 1:26",
                    "sad piano that returns after 30 seconds"):
        assert any(r.startswith("clock_time") for r in T.check_caption("sad piano", caption, instrumental=True))


def test_a_caption_that_drops_the_request_is_rejected():
    """Under the floor of a two-word request; a one-word request is not measured."""
    reasons = T.check_caption("sad solo piano for a rainy night", "ambient techno with a four on the floor kick",
                              instrumental=True)
    assert reasons == ["answered:0.00"]
    assert T.ANSWERED_FLOOR == 0.5
    assert T.check_caption("piano", "ambient techno with a kick drum", instrumental=True) == []


# ── the lint ──

def test_lint_passes_a_faithful_translation():
    text, reasons = T.lint_translation("sad solo piano for a rainy night", GOOD, instrumental=True, pool=POOL)
    assert reasons == [] and text == GOOD


def test_lint_rejects_a_claim_noun_from_nowhere():
    """An instrument neither asked for nor in the artist's captions is invented."""
    out = GOOD.replace("warm tape hiss", "a banjo")
    assert "invented_noun:banjo" in T.lint_translation("sad solo piano for a rainy night", out,
                                                       instrumental=True, pool=POOL)[1]


def test_lint_rejects_what_the_request_rules_out():
    """'no drums' then a drum machine is a contradiction even though the pool has drums."""
    out = "Warm strings swell over a slow drum machine pulse with lush reverb and a dreamy, gentle texture."
    reasons = T.lint_translation("no drums, just warm strings", out, instrumental=True, pool=POOL)[1]
    assert "denied:drum machine" in reasons


def test_lint_rejects_names_numbers_theory_and_comparisons():
    base = "sad solo piano for a rainy night"
    cases = {"theory": GOOD + " In A minor.", "number:120": GOOD.replace("soft", "120 soft"),
             "sound_alike": GOOD + " Reminiscent of rain.", "name:Satie": GOOD.replace("piano piece", "piano piece Satie")}
    for reason, text in cases.items():
        assert reason in T.lint_translation(base, text, instrumental=True, pool=POOL)[1], reason


def test_lint_rejects_a_caption_too_short_to_be_one():
    assert any(r.startswith("length") for r in T.lint_translation("sad piano", "sad piano.", instrumental=True,
                                                                  pool=POOL)[1])


# ── the prompt and the answer ──

def test_prompt_is_chatml_with_thinking_off_and_the_rule():
    p = T.build_prompt("sad piano", POOL, instrumental=True)
    assert p.startswith("<|im_start|>system\n") and "/no_think" in p and p.endswith("<think>\n\n</think>\n\n")
    assert "You have NOT heard any recording" in p and "never add an instrument" in p
    assert "Instrumental: do not mention vocals" in p and POOL[0] in p
    assert "Sung" in T.build_prompt("sad piano", POOL, instrumental=False)


def test_answer_extraction_drops_thinking_preambles_and_tails():
    raw = "<think>hmm</think>\nHere is the caption:\n\"" + GOOD + "\"<|im_end|>junk"
    assert T.extract_answer(raw) == tidy_caption(GOOD)


# ── attempts and the fallback chain ──

def test_translate_retries_warmer_with_a_seed_then_passes():
    llm = fake("ambient techno with a kick drum and a robotic voice", GOOD)
    out = T.translate("sad solo piano for a rainy night", POOL[:1], instrumental=True, pool=POOL, llm=llm)
    assert out["caption"] == GOOD and out["attempts"] == 2 and len(out["rejects"]) == 1
    assert llm.calls[0]["temperature"] == 0.0 and llm.calls[0]["seed"] is None
    assert llm.calls[1]["temperature"] > 0 and llm.calls[1]["seed"] is not None


def test_three_failed_translations_return_the_request_never_an_invention():
    """After MAX_ATTEMPTS the caption is the fan's own request, marked as such, and the LM is not tried."""
    lm_calls = []
    out = T.expand_checked("sad solo piano for a rainy night", instrumental=True, attempt=0, pool=POOL,
                           shots=lambda a: POOL, translator=fake("ambient techno with a kick"),
                           lm=lambda a: lm_calls.append(a) or {"caption": GOOD})
    assert out["caption"] == "sad solo piano for a rainy night" and out["fallback"] == "request"
    assert out["translator"] == "request" and out["attempts"] == 3 and len(out["rejects"]) == 3
    assert out["fallback_note"] == T.FALLBACK_NOTE and lm_calls == []


def test_an_unreachable_translator_falls_back_to_the_checked_lm():
    """A translator that raises hands over to the 5Hz LM, whose answers meet the same check."""
    def down(prompt, **kw):
        raise OSError("connection refused")
    seen = []

    def lm(a):
        seen.append(a)
        return {"caption": "a voice sings over ambient techno" if a == 0 else GOOD, "bpm": 80}
    out = T.expand_checked("sad solo piano for a rainy night", instrumental=True, attempt=0, pool=POOL,
                           shots=lambda a: POOL, translator=down, lm=lm)
    assert out["translator"] == "lm" and out["caption"] == GOOD and out["bpm"] == 80
    assert seen == [0, 1] and out["served_attempt"] == 1 and out["attempts"] == 2
    assert out["rejects"][0]["reasons"][0].startswith("unavailable")
    assert out["rejects"][1]["by"] == "lm"


def test_the_lm_failing_three_times_returns_the_request():
    out = T.expand_checked("no drums, just warm strings", instrumental=True, attempt=2, pool=POOL,
                           shots=lambda a: POOL, lm=lambda a: {"caption": "a spoken intro over warm strings"})
    assert out["fallback"] == "request" and out["attempts"] == 3 and out["caption"] == "no drums, just warm strings"


def test_a_passing_translation_serves_with_its_label():
    out = T.expand_checked("sad solo piano for a rainy night", instrumental=True, attempt=0, pool=POOL,
                           shots=lambda a: POOL, translator=fake(GOOD), translator_label="qwen3-4b")
    assert out["translator"] == "qwen3-4b" and out["caption"] == GOOD and out["fallback"] is None


def test_model_label_names_the_model_not_the_quantisation():
    assert T.model_label("/models/Qwen3-8B-Q4_K_M.gguf") == "qwen3-8b"
    assert T.model_label("Qwen3-4B-Q4_K_M.gguf") == "qwen3-4b"
    assert T.model_label(None) == T.DEFAULT_LABEL


# ── the server ──

class _Server(BaseHTTPRequestHandler):
    bodies = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Server.bodies.append(body)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"content": GOOD, "model": "/m/Qwen3-8B-Q4_K_M.gguf"}).encode())


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Server)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_http_llm_speaks_llama_servers_completion(server):
    llm = T.translator_from_env({T.TRANSLATOR_URL_ENV: server})
    assert llm("p", temperature=0.4, seed=7) == GOOD and llm.label == "qwen3-8b"
    assert _Server.bodies[-1] | {} == {"prompt": "p", "n_predict": T.MAX_TOKENS, "temperature": 0.4, "seed": 7,
                                        "cache_prompt": True, "stop": ["<|im_end|>"]}


def test_http_llm_unreachable_is_translator_unavailable():
    T._HEALTH.clear()
    llm = T.http_llm("http://127.0.0.1:9", timeout=0.5)
    with pytest.raises(T.TranslatorUnavailable):
        llm("p")


def test_no_url_means_no_translator():
    assert T.translator_from_env({}) is None


def test_the_label_is_read_after_the_server_answers():
    """A server's model name is known only from its answer; the record names that model."""
    def llm(prompt, **kw):
        llm.label = "qwen3-4b"
        return GOOD
    llm.label = "qwen3-8b"
    out = T.expand_checked("sad solo piano for a rainy night", instrumental=True, attempt=0, pool=POOL,
                           shots=lambda a: POOL, translator=llm)
    assert out["translator"] == "qwen3-4b"
