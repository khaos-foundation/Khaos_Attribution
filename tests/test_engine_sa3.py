"""Stable Audio 3's render slot behind a fake worker: the request reaches the worker in the engine's words with the
descriptor's row filling what was not asked, the reply becomes a RenderResult, and the refusals are the socket's."""
import json
import sys
import wave

import pytest

from khaos_attribution.engines.descriptor import packaged_descriptor
from khaos_attribution.engines.render import RenderCancelled, RenderFailed, RenderRequest
from khaos_attribution.engines.sa3 import render as R

FAKE_WORKER = '''
import json, sys, wave
req = json.loads(sys.stdin.read())
if req["prompt"] == "fail":
    sys.stdout.write(json.dumps({"error": "the model said no"}) + "\\n"); sys.exit(1)
with wave.open(req["out"], "wb") as fh:
    fh.setnchannels(2); fh.setsampwidth(2); fh.setframerate(44100)
    fh.writeframes(b"\\0\\0\\0\\0" * int(req["seconds"] * 44100))
(sys.stdout.write("noise from the model\\n"))
sys.stdout.write(json.dumps({"seed": req["seed"] if req["seed"] is not None else 99, "sample_rate": 44100,
                             "dropped": [], "echo": req}) + "\\n")
'''


@pytest.fixture
def fake_engine(tmp_path, monkeypatch):
    root = tmp_path / "stable-audio-3"
    root.mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    monkeypatch.setattr(R, "WORKER", worker)
    monkeypatch.setattr(R, "python_for", lambda root, runtime: sys.executable)
    return root


def test_the_descriptor_says_what_the_engine_cannot_do():
    d = packaged_descriptor("khaos_attribution.engines.sa3")
    assert d.id == "stable-audio-3" and not d.sings and not d.conditioning["lyrics"] and not d.conditioning["tempo"]
    assert d.sample_rate == 44100 and d.default_variant == "base" and d.render_defaults() == {"steps": 50, "guidance_scale": 7.0, "apg_scale": 1.0}
    assert d.supports("mps", "train") and d.supports("cuda", "render") and not d.supports("cuda", "measure")
    assert not [w for w in d.weights if w.get("gated")], "no account is needed: the encoder is read from the open base repository"
    assert d.adapter["type"] == "lora" and d.train["adapter_type"] == "lora" and d.licence["name"].startswith("Stability AI")


def test_a_render_reaches_the_worker_in_the_engines_words_and_comes_back_as_a_result(fake_engine, tmp_path):
    r = R.Renderer(repo_dir=fake_engine, device="cpu")
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"w")
    req = RenderRequest(prompt="a bright sting", duration_seconds=300, seed=7, adapter_scale=0.85, steps=None)
    out = r.generate(req, adapter, tmp_path / "out.wav")
    assert out.audio_path == tmp_path / "out.wav" and out.seed == 7 and out.variant == "base"
    assert out.applied_scale == 0.85 and out.scale_applied_via == "lora_strength"
    assert out.params["steps"] == 50 and out.params["guidance_scale"] == 7.0 and out.params["duration"] == 120, \
        "the descriptor's row fills what was not asked, and the duration is clamped to the engine's bound"
    assert out.params["runtime"] == "torch" and out.params["sample_rate"] == 44100
    with wave.open(str(out.audio_path), "rb") as fh:
        assert fh.getframerate() == 44100 and fh.getnframes() == 120 * 44100


def test_the_worker_request_carries_the_adapter_and_the_callers_settings(fake_engine, tmp_path):
    r = R.Renderer(repo_dir=fake_engine, device="mps")
    assert r.runtime == "mlx"
    r.apply_adapter(tmp_path / "a.safetensors", None)
    req = RenderRequest(prompt="p", duration_seconds=30, seed=1, steps=20, guidance_scale=3.0, negative_prompt="no drums")
    w = r.worker_request(req, tmp_path / "o.wav")
    assert w["runtime"] == "mlx" and w["model"] == "medium-base" and w["steps"] == 20 and w["cfg_scale"] == 3.0
    assert w["adapter"] == str((tmp_path / "a.safetensors").resolve()) and w["adapter_scale"] == 0.85
    assert w["negative_prompt"] == "no drums" and w["seconds"] == 30


def test_refusals_are_the_sockets(fake_engine, tmp_path):
    r = R.Renderer(repo_dir=fake_engine, device="cpu")
    with pytest.raises(RenderFailed, match="cannot sing"):
        r.generate(RenderRequest(prompt="p", duration_seconds=10, lyrics="la la", instrumental=False), None, tmp_path / "x.wav")
    with pytest.raises(RenderCancelled):
        r.generate(RenderRequest(prompt="p", duration_seconds=10), None, tmp_path / "x.wav", should_stop=lambda: True)
    with pytest.raises(RenderFailed, match="the model said no"):
        r.generate(RenderRequest(prompt="fail", duration_seconds=10), None, tmp_path / "x.wav")
    with pytest.raises(RenderFailed, match="no variant"):
        r.load("turbo")


def test_an_unprovisioned_checkout_is_refused_with_the_path(tmp_path, monkeypatch):
    monkeypatch.delenv("SA3_PYTHON", raising=False)
    r = R.Renderer(repo_dir=tmp_path / "nowhere", device="cpu")
    with pytest.raises(RenderFailed, match="not found"):
        r.load()
    (tmp_path / "here").mkdir()
    with pytest.raises(RenderFailed, match="not provisioned"):
        R.Renderer(repo_dir=tmp_path / "here", device="cuda").load()
    assert R.runtime_for("cuda") == "torch" and R.runtime_for("mps") == "mlx"
    monkeypatch.setenv("SA3_PYTHON", str(tmp_path / "elsewhere" / "python"))
    with pytest.raises(RenderFailed, match="SA3_PYTHON names no interpreter"):
        R.python_for(tmp_path / "here", "torch")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "python").write_text("")
    assert R.python_for(tmp_path / "here", "torch") == tmp_path / "elsewhere" / "python"


STEPPING_WORKER = '''
import json, os, sys, time, wave
req = json.loads(sys.stdin.read())
open(os.environ["SA3_FAKE_PID"], "w").write(str(os.getpid()))
for i in range(4):
    print("khaos-row " + json.dumps({"step": i + 1, "of": 4}), flush=True)
    time.sleep(float(os.environ.get("SA3_FAKE_STEP", "0.01")))
with wave.open(req["out"], "wb") as fh:
    fh.setnchannels(2); fh.setsampwidth(2); fh.setframerate(44100); fh.writeframes(b"\\0\\0\\0\\0" * 4410)
print(json.dumps({"seed": 3, "sample_rate": 44100, "dropped": []}))
'''


@pytest.fixture
def stepping(tmp_path, monkeypatch):
    root = tmp_path / "stable-audio-3"
    root.mkdir()
    worker = tmp_path / "stepping.py"
    worker.write_text(STEPPING_WORKER)
    monkeypatch.setattr(R, "WORKER", worker)
    monkeypatch.setattr(R, "python_for", lambda root, runtime: sys.executable)
    monkeypatch.setenv("SA3_FAKE_PID", str(tmp_path / "pid"))
    return root


def test_a_render_reports_each_step_as_the_worker_takes_it(stepping, tmp_path):
    """The listener's card moves while the model samples, not only before and after."""
    seen = []
    R.Renderer(repo_dir=stepping, device="cpu").generate(RenderRequest(prompt="p", duration_seconds=5, seed=3), None,
                                                         tmp_path / "o.wav", on_progress=lambda d, t: seen.append(d / t))
    assert seen == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]


def test_a_render_is_stopped_while_its_steps_still_come(stepping, tmp_path, monkeypatch):
    """A stop asked between two steps ends the worker then, not when the render would have finished."""
    import os
    import time
    monkeypatch.setenv("SA3_FAKE_STEP", "5")
    asked = []
    start = time.monotonic()
    with pytest.raises(RenderCancelled):
        R.Renderer(repo_dir=stepping, device="cpu").generate(
            RenderRequest(prompt="p", duration_seconds=5, seed=3), None, tmp_path / "o.wav",
            on_progress=lambda d, t: asked.append(d), should_stop=lambda: bool(asked) and asked[-1] > 0)
    assert time.monotonic() - start < 15
    pid = int((tmp_path / "pid").read_text())
    for _ in range(80):
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("the worker outlived the stop")


def test_a_render_with_no_output_path_is_refused_never_written_where_the_process_stands(fake_engine):
    with pytest.raises(RenderFailed, match="path to write"):
        R.Renderer(repo_dir=fake_engine, device="cpu").generate(RenderRequest(prompt="p", duration_seconds=5), None, None)


def test_the_torch_worker_prints_a_row_at_each_step_the_sampler_reports(monkeypatch, capsys, tmp_path):
    """The step rows come from the sampler's own callback, handed through Stability's generate."""
    import types
    import numpy as np
    from khaos_attribution.engines.sa3 import loading
    from khaos_attribution.engines.sa3 import worker as WK

    class Audio:
        def __init__(self, a):
            self.a = a

        def __getitem__(self, i):
            return Audio(self.a[i])

        @property
        def T(self):
            return Audio(self.a.T)

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    class Model:
        model = types.SimpleNamespace(sample_rate=44100)

        def generate(self, callback=None, steps=0, **kw):
            for i in range(steps):
                if callback is not None:
                    callback({"i": i, "x": None})
            return Audio(np.zeros((1, 2, 441), np.float32))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(from_numpy=lambda a: a))
    monkeypatch.setattr(loading, "load_model", lambda name: Model())
    req = {"model": "medium-base", "prompt": "p", "seconds": 1.0, "steps": 3, "cfg_scale": 7.0, "seed": 1,
           "out": str(tmp_path / "o.wav")}
    assert WK.render_torch(req)["seed"] == 1
    rows = [json.loads(line[len("khaos-row "):]) for line in capsys.readouterr().out.splitlines() if line.startswith("khaos-row ")]
    assert rows == [{"step": 1, "of": 3}, {"step": 2, "of": 3}, {"step": 3, "of": 3}]
