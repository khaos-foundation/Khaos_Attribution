"""An engine environment holding another contract than this room's is refused in a sentence where its process
would start, once per interpreter for the life of the process; the running interpreter is never checked."""
import subprocess
import sys
from pathlib import Path

import pytest

from khaos_attribution import __version__
from khaos_attribution.engines import contract_guard as CG
from khaos_attribution.engines.render import RenderFailed
from khaos_attribution.engines.train import TrainingUnavailable
from khaos_attribution.gradient import GradientUnavailable

OURS = f"{__version__} 0123abcd"


@pytest.fixture
def fingerprints(monkeypatch):
    """The fingerprint subprocess replaced: ours for this process, the dict's entry for any other path, counted."""
    seen = {None: OURS}
    calls = []

    def fake(python):
        calls.append(python)
        return seen.get(None if python is None else str(python))

    monkeypatch.setattr(CG, "_fingerprint", fake)
    monkeypatch.setattr(CG, "_CHECKED", {})
    return seen, calls


def test_a_stale_environment_is_refused_with_the_sentence_and_a_current_one_is_checked_once(tmp_path, fingerprints):
    seen, calls = fingerprints
    stale, current = tmp_path / "stale" / "python", tmp_path / "current" / "python"
    seen[str(stale)] = "0.41.0 fedcba98"
    seen[str(current)] = OURS
    why = CG.refusal(stale, engine="Stable Audio 3", engine_id="stable-audio-3")
    assert why == (f"the Stable Audio 3 environment at {stale} holds contract 0.41.0, this room runs {__version__}: "
                   "run the installer with --engines stable-audio-3")
    assert CG.refusal(current, engine="mimetic-a0", engine_id="mimetic-a0") is None
    n = len(calls)
    assert CG.refusal(current, engine="mimetic-a0", engine_id="mimetic-a0") is None
    assert CG.refusal(stale, engine="Stable Audio 3", engine_id="stable-audio-3") == why
    assert len(calls) == n, "a second call reads the answer remembered for that interpreter"
    with pytest.raises(TrainingUnavailable, match="run the installer with --engines stable-audio-3"):
        CG.require(stale, engine="Stable Audio 3", engine_id="stable-audio-3", error=TrainingUnavailable)


def test_other_code_at_the_same_version_and_no_contract_at_all_are_said_so(tmp_path, fingerprints):
    seen, _calls = fingerprints
    seen[str(tmp_path / "a")] = f"{__version__} 89abcdef"
    assert "built from other code than this room's: run the installer" in CG.refusal(tmp_path / "a", engine="x", engine_id="x")
    assert "holds no contract, or one older than 0.41.1: run the installer" in CG.refusal(tmp_path / "b", engine="x", engine_id="x")


def test_the_running_interpreter_is_never_checked(fingerprints):
    _seen, calls = fingerprints
    assert CG.refusal(sys.executable, engine="x", engine_id="x") is None
    assert CG.refusal(Path(sys.executable), engine="x", engine_id="x") is None
    assert calls == []


@pytest.fixture
def stale(tmp_path, fingerprints, monkeypatch):
    """A fake engine interpreter holding an older contract, and a Popen that must never be reached."""
    seen, _calls = fingerprints
    python = tmp_path / "engine" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    seen[str(python)] = "0.41.0 fedcba98"

    def never(*a, **k):
        raise AssertionError("the engine process was started on a stale environment")

    monkeypatch.setattr(subprocess, "Popen", never)
    return python


SENTENCE = r"holds contract 0\.41\.0, this room runs .*: run the installer with --engines "


def test_the_sa3_measure_render_and_trainer_refuse_a_stale_environment_before_starting_it(tmp_path, stale, monkeypatch):
    from khaos_attribution.engines.sa3 import measure as M
    from khaos_attribution.engines.sa3 import render as R
    from khaos_attribution.engines.sa3 import train as T
    handle = M.Handle(root=tmp_path, python=stale, adapter=tmp_path / "a.safetensors")
    with pytest.raises(GradientUnavailable, match="the Stable Audio 3 environment at .*" + SENTENCE + "stable-audio-3"):
        M._ask(handle, {"command": "parameters"})
    monkeypatch.setattr(R, "python_for", lambda root, runtime: stale)
    with pytest.raises(RenderFailed, match=SENTENCE + "stable-audio-3"):
        R.run_worker(tmp_path, "torch", {"prompt": "x"})
    api = T.TrainingApi(root=tmp_path, runtime="torch", python=stale)
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "a_seg_000.wav").write_bytes(b"")
    monkeypatch.setattr(T, "TORCH_MIN_PAIRS", 1)
    session = T.open_session(api, {"adapter_rank": 8, "adapter_alpha": 8, "max_epochs": 1, "window_sec": 5.0},
                             checkpoint_root=tmp_path, tensor_dir=clips, output_dir=tmp_path / "run", device="cuda",
                             precision="bf16", seed=1)
    with pytest.raises(TrainingUnavailable, match=SENTENCE + "stable-audio-3"):
        list(session.train())
    assert not (tmp_path / "run" / T.TRAINER_DIRNAME / "train_command.json").exists(), "no record of a run that never started"


def test_the_mimetic_measure_render_and_trainer_refuse_a_stale_environment_before_starting_it(tmp_path, stale, monkeypatch):
    from khaos_attribution.engines.mimetic import measure as M
    from khaos_attribution.engines.mimetic import render as R
    from khaos_attribution.engines.mimetic import train as T
    handle = M.Handle(root=tmp_path, python=stale, adapter=tmp_path / "a.safetensors")
    with pytest.raises(GradientUnavailable, match="the mimetic-a0 environment at .*" + SENTENCE + "mimetic-a0"):
        M._ask(handle, {"command": "parameters"})
    monkeypatch.setattr(R, "python_for", lambda root: stale)
    with pytest.raises(RenderFailed, match=SENTENCE + "mimetic-a0"):
        R.run_worker(tmp_path, {"prompt": "x"})
    api = T.TrainingApi(root=tmp_path, runtime="torch", python=stale)
    clips = tmp_path / "clips"
    clips.mkdir()
    for i in range(T.MIN_PAIRS):
        (clips / f"a_seg_{i:03d}.wav").write_bytes(b"")
    session = T.open_session(api, {"adapter_rank": 8, "adapter_alpha": 8, "max_epochs": 1}, checkpoint_root=tmp_path,
                             tensor_dir=clips, output_dir=tmp_path / "run", device="cuda", precision="32", seed=1)
    with pytest.raises(TrainingUnavailable, match=SENTENCE + "mimetic-a0"):
        list(session.train())
    assert not (tmp_path / "run" / T.TRAINER_DIRNAME / "train_command.json").exists()


def test_a_missing_interpreter_is_refused_in_a_sentence_not_an_os_error(tmp_path):
    from khaos_attribution.engines import contract_guard as g
    why = g.refusal(tmp_path / "venv" / "bin" / "python", engine="Stable Audio 3", engine_id="stable-audio-3")
    assert why.startswith("the Stable Audio 3 environment has no interpreter at") and "--engines stable-audio-3" in why
