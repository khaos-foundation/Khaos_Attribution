"""An engine's interpreter is refused in a sentence only when it is missing, read again at every start; whatever
contract copy its environment holds is never read, and provisioning gives it the contract's base requirements at
pyproject's floors while leaving any older copy where it is."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from khaos_attribution.engines import contract_guard as CG
from khaos_attribution.engines import environment
from khaos_attribution.engines.render import RenderFailed
from khaos_attribution.engines.worker_process import POSIX_ONLY
from khaos_attribution.engines.train import TrainingUnavailable
from khaos_attribution.gradient import GradientUnavailable

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def no_probe(monkeypatch):
    """Nothing is started to check an interpreter that is there."""
    def never(*a, **k):
        raise AssertionError("the guard started a process")
    monkeypatch.setattr(environment, "run", never)


def test_an_interpreter_that_is_there_passes_whatever_contract_its_environment_holds(tmp_path, no_probe):
    python = tmp_path / "engine" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    assert CG.refusal(python, engine="Stable Audio 3", engine_id="stable-audio-3") is None


def test_a_missing_interpreter_is_refused_in_a_sentence_and_read_again_at_every_start(tmp_path, no_probe):
    python = tmp_path / "venv" / "bin" / "python"
    why = CG.refusal(python, engine="Stable Audio 3", engine_id="stable-audio-3")
    assert why == f"the Stable Audio 3 environment has no interpreter at {python}: run the installer with --engines stable-audio-3"
    with pytest.raises(TrainingUnavailable, match="run the installer with --engines stable-audio-3"):
        CG.require(python, engine="Stable Audio 3", engine_id="stable-audio-3", error=TrainingUnavailable)
    python.parent.mkdir(parents=True)
    python.write_text("")
    assert CG.refusal(python, engine="Stable Audio 3", engine_id="stable-audio-3") is None, "a repaired one needs no restart"


def test_the_running_interpreter_is_never_checked(no_probe):
    assert CG.refusal(sys.executable, engine="x", engine_id="x") is None
    assert CG.refusal(Path(sys.executable), engine="x", engine_id="x") is None


@pytest.fixture
def missing(tmp_path, monkeypatch):
    """An engine interpreter that is not there, and a Popen that must never be reached."""
    def never(*a, **k):
        raise AssertionError("the engine process was started without its interpreter")
    monkeypatch.setattr(subprocess, "Popen", never)
    return tmp_path / "engine" / "bin" / "python"


SENTENCE = r"environment has no interpreter at .*: run the installer with --engines "


def test_the_sa3_measure_render_and_trainer_refuse_a_missing_interpreter_before_starting(tmp_path, missing, monkeypatch):
    from khaos_attribution.engines.sa3 import measure as M
    from khaos_attribution.engines.sa3 import render as R
    from khaos_attribution.engines.sa3 import train as T
    handle = M.Handle(root=tmp_path, python=missing, adapter=tmp_path / "a.safetensors")
    with pytest.raises(GradientUnavailable, match="the Stable Audio 3 " + SENTENCE + "stable-audio-3"):
        M._ask(handle, {"command": "parameters"})
    monkeypatch.setattr(R, "python_for", lambda root, runtime: missing)
    with pytest.raises(RenderFailed, match=SENTENCE + "stable-audio-3"):
        R.run_worker(tmp_path, "torch", {"prompt": "x"})
    api = T.TrainingApi(root=tmp_path, runtime="torch", python=missing)
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


def test_the_mimetic_measure_render_and_trainer_refuse_a_missing_interpreter_before_starting(tmp_path, missing, monkeypatch):
    from khaos_attribution.engines.mimetic import measure as M
    from khaos_attribution.engines.mimetic import render as R
    from khaos_attribution.engines.mimetic import train as T
    handle = M.Handle(root=tmp_path, python=missing, adapter=tmp_path / "a.safetensors")
    with pytest.raises(GradientUnavailable, match="the mimetic-a0 " + SENTENCE + "mimetic-a0"):
        M._ask(handle, {"command": "parameters"})
    monkeypatch.setattr(R, "python_for", lambda root: missing)
    with pytest.raises(RenderFailed, match=SENTENCE + "mimetic-a0"):
        R.run_worker(tmp_path, {"prompt": "x"})
    api = T.TrainingApi(root=tmp_path, runtime="torch", python=missing)
    clips = tmp_path / "clips"
    clips.mkdir()
    for i in range(T.MIN_PAIRS):
        (clips / f"a_seg_{i:03d}.wav").write_bytes(b"")
    session = T.open_session(api, {"adapter_rank": 8, "adapter_alpha": 8, "max_epochs": 1}, checkpoint_root=tmp_path,
                             tensor_dir=clips, output_dir=tmp_path / "run", device="cuda", precision="32", seed=1)
    with pytest.raises(TrainingUnavailable, match=SENTENCE + "mimetic-a0"):
        list(session.train())
    assert not (tmp_path / "run" / T.TRAINER_DIRNAME / "train_command.json").exists()


def test_the_requirements_are_the_contracts_own_dependencies_at_their_floors():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text[text.index("dependencies = ["):text.index("]", text.index("dependencies = ["))]
    assert tuple(re.findall(r'^\s*"([^"]+)"', block, re.M)) == CG.REQUIREMENTS


@pytest.mark.skipif(os.name == "nt", reason=POSIX_ONLY)
def test_an_environment_that_cannot_load_the_contract_is_said_so(tmp_path):
    """The probe runs in the engine's interpreter; a base requirement that will not import is named."""
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "jsonschema.py").write_text("raise ImportError('no jsonschema here')\n")
    engine = tmp_path / "python"
    engine.write_text(f'#!/bin/sh\nPYTHONPATH={shadow} exec {sys.executable} "$@"\n')
    engine.chmod(0o755)
    assert CG.requirements_missing(engine) == "cannot load the room's contract (ImportError: no jsonschema here)"
    assert CG.requirements_missing(sys.executable) is None


def test_provisioning_installs_the_requirements_and_never_touches_the_contract_copy(tmp_path, monkeypatch):
    """Only the requirements at their floors are installed; an older contract copy stays for a rollback image."""
    calls, state = [], {"present": False}

    def fake_run(cmd, cwd=None, env=None):
        calls.append([str(c) for c in cmd])
        if cmd[1:3] == ["-c", CG.REQUIREMENTS_PROBE]:
            return subprocess.CompletedProcess(cmd, 0 if state["present"] else 1, "", "ModuleNotFoundError: jsonschema")
        state["present"] = True
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(environment, "run", fake_run)
    monkeypatch.setattr(environment, "uv_path", lambda: "uv")
    ok, note = CG.requirements_into(tmp_path / "python")
    assert ok and "requirements installed" in note
    installs = [c for c in calls if c[:2] == ["uv", "pip"]]
    assert installs == [["uv", "pip", "install", "-q", "--python", str(tmp_path / "python"), *CG.REQUIREMENTS]]
    assert not any("khaos-attribution" in part or "khaos_attribution" in part for c in calls for part in c)
    calls.clear()
    assert CG.requirements_into(tmp_path / "python")[0] and not [c for c in calls if c[0] == "uv"], "present: nothing installed"
