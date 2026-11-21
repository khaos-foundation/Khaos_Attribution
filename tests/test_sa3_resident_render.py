"""A resident Stable Audio 3 render worker, behind a descriptor switch that ships off: off, every render is a worker
of its own as before; on, renders share one worker per adapter, an adapter change starts another, a stop ends it
and the next render starts fresh, and yielding the memory ends it."""
import faulthandler
import os
import sys
import time
from pathlib import Path

import pytest

from khaos_attribution.engines import contract_guard
from khaos_attribution.engines import worker_process as W
from khaos_attribution.engines.render import RenderCancelled, RenderRequest
from khaos_attribution.engines.sa3 import DESCRIPTOR
from khaos_attribution.engines.sa3 import render as R

# One JSON request per line; a row, then the audio, then the reply naming the process that rendered it.
SERVING = r'''
import json, os, sys, time, wave
assert "--serve" in sys.argv
for line in sys.stdin:
    req = json.loads(line)
    print("khaos-row " + json.dumps({"step": 1, "of": 2}), flush=True)
    time.sleep(float(os.environ.get("SA3_FAKE_STEP", "0.01")))
    print("khaos-row " + json.dumps({"step": 2, "of": 2}), flush=True)
    with wave.open(req["out"], "wb") as fh:
        fh.setnchannels(2); fh.setsampwidth(2); fh.setframerate(44100); fh.writeframes(b"\0\0\0\0" * 441)
    print("khaos-reply " + json.dumps({"seed": 3, "sample_rate": 44100, "dropped": [], "pid": os.getpid(),
                                        "adapter": req.get("adapter"), "scale": req.get("adapter_scale")}), flush=True)
'''


@pytest.fixture(autouse=True)
def bounded():
    """No resident test waits on a hung worker for longer than a minute."""
    faulthandler.dump_traceback_later(60, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def engine(tmp_path, monkeypatch):
    root = tmp_path / "stable-audio-3"
    root.mkdir()
    worker = tmp_path / "serving.py"
    worker.write_text(SERVING)
    monkeypatch.setattr(R, "WORKER", worker)
    monkeypatch.setattr(R, "python_for", lambda root, runtime: Path(sys.executable))
    monkeypatch.setattr(contract_guard, "refusal", lambda *a, **k: None)

    made = []

    def make(on: bool):
        monkeypatch.setitem(DESCRIPTOR.doc, "runtime", {**DESCRIPTOR.runtime, "resident_render": on})
        made.append(R.Renderer(repo_dir=root, device="cpu"))
        return made[-1]
    yield make
    for renderer in made:
        renderer._end_resident()


def _render(renderer, tmp_path, name, **kw):
    return renderer.generate(RenderRequest(prompt="p", duration_seconds=5, seed=3), None, tmp_path / f"{name}.wav", **kw)


def _gone(pid: int) -> bool:
    for _ in range(80):
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.1)
    return False


def test_the_switch_ships_off_and_off_every_render_is_its_own_worker(engine, tmp_path, monkeypatch):
    assert DESCRIPTOR.runtime.get("resident_render") is False
    renderer = engine(False)
    assert renderer.resident is False
    asked = []
    monkeypatch.setattr(R, "run_worker", lambda *a, **k: asked.append(a[2]["out"]) or
                        (Path(a[2]["out"]).write_bytes(b"x"), {"seed": 3, "sample_rate": 44100})[1])
    _render(renderer, tmp_path, "a")
    _render(renderer, tmp_path, "b")
    assert len(asked) == 2 and renderer._resident is None


def test_on_renders_share_one_worker_until_the_adapter_changes(engine, tmp_path):
    renderer = engine(True)
    assert renderer.resident is True
    seen = []
    first = _render(renderer, tmp_path, "a", on_progress=lambda d, t=1.0: seen.append(d / t))
    pid = renderer._resident.proc.pid
    _render(renderer, tmp_path, "b")
    assert renderer._resident.proc.pid == pid and renderer._resident.asked == 2
    assert first.audio_path.is_file() and seen == sorted(seen) and seen[-1] == 1.0
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"a")
    renderer.apply_adapter(adapter, 0.5)
    _render(renderer, tmp_path, "c")
    assert renderer._resident.proc.pid != pid and _gone(pid), "another adapter, another worker; the old one ends"
    renderer.yield_memory()
    assert renderer._resident is None


def test_on_a_stop_ends_the_worker_and_the_next_render_starts_fresh(engine, tmp_path, monkeypatch):
    renderer = engine(True)
    _render(renderer, tmp_path, "a")
    pid = renderer._resident.proc.pid
    monkeypatch.setenv("SA3_FAKE_STEP", "5")
    renderer._end_resident()
    assert _gone(pid)
    rows = []
    with pytest.raises(RenderCancelled):
        _render(renderer, tmp_path, "stopped", on_progress=lambda d, t=1.0: rows.append(d),
                should_stop=lambda: bool(rows) and rows[-1] > 0)
    assert renderer._resident is None
    monkeypatch.setenv("SA3_FAKE_STEP", "0.01")
    assert _render(renderer, tmp_path, "after").audio_path.is_file()


def test_a_resident_worker_that_dies_is_said_and_replaced(engine, tmp_path):
    renderer = engine(True)
    _render(renderer, tmp_path, "a")
    W.stop_tree(renderer._resident.proc)
    assert _render(renderer, tmp_path, "b").audio_path.is_file()


def test_the_worker_serves_one_model_and_adapter_and_answers_each_request():
    """The SA3 worker's resident mode refuses another model or adapter, so a change always means a new worker."""
    from khaos_attribution.engines.sa3 import worker as WK
    loaded = []
    served = WK._Resident()
    WK_loaded, WK_render = WK._loaded_torch, WK.render_torch
    try:
        WK._loaded_torch = lambda req: loaded.append(req["adapter"]) or object()
        WK.render_torch = lambda req, model=None: {"seed": 1, "model": model}
        a = served({"model": "m", "adapter": None, "out": "o"})
        b = served({"model": "m", "adapter": None, "out": "o"})
        assert loaded == [None] and a["model"] is b["model"]
        with pytest.raises(RuntimeError, match="started with"):
            served({"model": "m", "adapter": "/x.safetensors", "out": "o"})
        with pytest.raises(RuntimeError, match="torch only"):
            served({"model": "m", "runtime": "mlx"})
    finally:
        WK._loaded_torch, WK.render_torch = WK_loaded, WK_render


def test_a_progress_callback_that_raises_ends_the_worker_so_no_later_render_reads_its_reply(engine, tmp_path):
    renderer = engine(True)
    _render(renderer, tmp_path, "a")
    pid = renderer._resident.proc.pid

    def broken(done, total=1.0):
        if done > 0:
            raise RuntimeError("the room's progress handler failed")
    with pytest.raises(RuntimeError, match="progress handler"):
        _render(renderer, tmp_path, "b", on_progress=broken)
    assert renderer._resident is None and _gone(pid)
    after = _render(renderer, tmp_path, "c")
    assert after.audio_path.name == "c.wav" and after.audio_path.is_file()


def test_an_end_asked_mid_render_waits_for_the_render_and_never_breaks_it(engine, tmp_path, monkeypatch):
    """The Space forgets an adapter outside its engine lock: the render finishes whole, then the worker ends."""
    import threading
    renderer = engine(True)
    _render(renderer, tmp_path, "warm")
    monkeypatch.setenv("SA3_FAKE_STEP", "1.5")
    renderer._end_resident()
    started, outcome = threading.Event(), {}

    def render():
        try:
            outcome["result"] = _render(renderer, tmp_path, "long", on_progress=lambda d, t=1.0: d > 0 and started.set())
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc
    thread = threading.Thread(target=render)
    thread.start()
    assert started.wait(30)
    renderer.forget_adapter()
    thread.join(30)
    assert "error" not in outcome, outcome.get("error")
    assert outcome["result"].audio_path.is_file() and renderer._resident is None
