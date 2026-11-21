"""The checklist: the mock engine passes it live, ACE-Step passes its surface without its weights, and an engine
that lacks a slot or a name fails with the finding named."""
import types

import pytest

from khaos_attribution import engines as E
from khaos_attribution.engines import checklist as C


def test_the_mock_engine_passes_the_whole_checklist_live(tmp_path):
    result = C.check("mock", tmp_path, live=True)
    assert result.passed, result.report()
    names = {f.name for f in result.findings}
    assert {"live.deterministic", "live.seed_matters", "live.sample_rate", "live.duration", "live.cancel",
            "render.words", "measure.memory_plan", "render.Renderer.calls", "live.progress"} <= names


@pytest.mark.parametrize("engine_id", ["acestep-1.5", "stable-audio-3"])
def test_a_real_engine_passes_the_surface_checks_without_its_weights(tmp_path, engine_id):
    result = C.check(engine_id, tmp_path, live=False)
    assert result.passed, result.report()
    assert not any(f.name.startswith("live.") for f in result.findings)


def test_an_unknown_or_broken_engine_fails_with_the_finding_named(tmp_path, monkeypatch):
    assert not C.check("nope", tmp_path).passed

    class Engine:
        descriptor = E.descriptor("mock")

        @property
        def render(self):
            return E.engine("mock").render

        @property
        def train(self):
            raise ImportError("no trainer here")

        @property
        def measure(self):
            return types.SimpleNamespace()
    monkeypatch.setitem(E.KNOWN_ENGINES, "broken", "khaos_attribution.engines.broken")
    monkeypatch.setattr(C, "descriptor", lambda engine_id: E.descriptor("mock"))
    monkeypatch.setattr(C, "engine", lambda engine_id: Engine())
    result = C.check("broken", tmp_path)
    failed = {f.name for f in result.failed}
    assert "slot.train" in failed and "descriptor.id" in failed
    assert "no trainer here" in result.report()
    assert not any(f.name.startswith(("render.", "measure.")) for f in result.findings), "a missing slot stops the walk"


def test_the_command_line_reports_and_exits_by_the_verdict(tmp_path, capsys):
    assert C.main(["mock", "--live", "--workdir", str(tmp_path)]) == 0
    assert "engine mock: PASS" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        C.main(["nope"])


@pytest.mark.parametrize("engine_id", sorted(E.KNOWN_ENGINES))
def test_every_known_engine_carries_every_name_the_rooms_call(tmp_path, engine_id):
    result = C.check(engine_id, tmp_path, live=False)
    assert result.passed, result.report()
    names = {f.name for f in result.findings}
    assert {"measure.names", "train.Session.methods"} <= names


def _engine_like_mock(measure=None, session=None):
    mock = E.engine("mock")

    class Engine:
        descriptor = mock.descriptor
        render = mock.render

        @property
        def train(self):
            if session is None:
                return mock.train
            return types.SimpleNamespace(**{n: getattr(mock.train, n) for n in C.TRAIN_NAMES if n != "Session"},
                                         Session=session)

        @property
        def measure(self):
            return measure or mock.measure
    return Engine()


def test_an_engine_whose_measure_lacks_what_the_rooms_call_fails_the_checklist(tmp_path, monkeypatch):
    """The Workshop calls index_rows, output_feature, paired_loss and parameter_count; the checklist asks for them."""
    mock = E.engine("mock").measure
    lacking = types.SimpleNamespace(**{n: getattr(mock, n) for n in C.MEASURE_NAMES if n != "paired_loss"})
    monkeypatch.setattr(C, "engine", lambda engine_id: _engine_like_mock(measure=lacking))
    result = C.check("mock", tmp_path)
    assert "measure.names" in {f.name for f in result.failed} and "paired_loss" in result.report()


def test_an_engine_whose_session_lacks_a_method_the_rooms_call_fails_the_checklist(tmp_path, monkeypatch):
    class Session:
        def install_hooks(self, **kw):
            return None

        def train(self):
            return iter(())

        def heldout_evaluator(self, *a, **kw):
            return None
    monkeypatch.setattr(C, "engine", lambda engine_id: _engine_like_mock(session=Session))
    result = C.check("mock", tmp_path)
    assert "train.Session.methods" in {f.name for f in result.failed} and "save_final" in result.report()


class _Descriptor:
    """The mock's descriptor with other platforms."""

    def __init__(self, platforms):
        self._d, self.platforms = E.descriptor("mock"), platforms

    def __getattr__(self, name):
        return getattr(self._d, name)


def test_the_memory_plan_is_probed_on_a_device_the_descriptor_measures_on(tmp_path, monkeypatch):
    """An engine measured on CUDA alone, whose plan refuses every other device, passes; the probe asks CUDA."""
    asked = []
    mock = E.engine("mock").measure

    def plan(device_type, duration_sec, **kw):
        asked.append(device_type)
        if device_type != "cuda":
            return {"fits": None, "checkpointed": False, "need_gb": None, "total_gb": None, "refusal": "not measured here"}
        return mock.memory_plan(device_type, duration_sec, **kw)
    measure = types.SimpleNamespace(**{**{n: getattr(mock, n) for n in C.MEASURE_NAMES}, "memory_plan": plan})
    monkeypatch.setattr(C, "descriptor", lambda engine_id: _Descriptor({"cuda": ["train", "render", "measure"]}))
    monkeypatch.setattr(C, "engine", lambda engine_id: _engine_like_mock(measure=measure))
    result = C.check("mock", tmp_path)
    assert result.passed, result.report()
    assert asked == ["cuda"]


def test_the_mocks_measure_path_runs_end_to_end(tmp_path):
    """The rooms' tests can drive a whole measurement on the mock: an index, an output's feature and its losses."""
    from khaos_attribution.gradient import make_projection
    M = E.engine("mock").measure
    files = [tmp_path / f"t_seg_00{i}.pt" for i in range(3)]
    handle = object()
    n, tensors, signature = M.parameter_count(handle)
    matrix, rows = M.index_rows(handle, files, projection=make_projection(n, 8, 1), timesteps=[0.5], noise_seed=7)
    assert matrix.shape == (3, 8) and str(matrix.dtype) == "float32" and [r["pair_id"] for r in rows] == [f.stem for f in files]
    assert rows[0]["track_id"] == "t" and all(r["gradient_norm"] > 0 for r in rows)
    again, _ = M.index_rows(handle, files, projection=make_projection(n, 8, 1), timesteps=[0.5], noise_seed=7)
    assert (again == matrix).all(), "the same files and seed give the same rows"
    meta = {"timesteps": [0.5], "noise_seed": 7, "projection": {"dim": 8, "seed": 1, "n_params": n}}
    assert M.output_feature(handle, meta, files[:1]).shape == (8,)
    losses = M.paired_loss(handle, meta, files)
    assert set(losses) == {"with", "without"} and M.output_loss(handle, meta, files) == losses["with"]
    with pytest.raises(Exception, match="block sketches"):
        M.index_rows(handle, files, projection=make_projection(n, 8, 1), timesteps=[0.5], noise_seed=7,
                     param_groups="x", groups_out=tmp_path / "g.npy")


@pytest.mark.parametrize("engine_id", sorted(E.KNOWN_ENGINES))
def test_every_renderer_takes_every_call_the_rooms_make(tmp_path, engine_id):
    """One written Renderer: each engine's methods take the rooms' calls, forget_adapter with or without a name."""
    from khaos_attribution.engines import render as R
    result = C.check(engine_id, tmp_path, live=False)
    assert "render.Renderer.calls" in {f.name for f in result.findings if f.ok}, result.report()
    assert set(R.RENDERER_METHODS) <= {n for n in dir(R.Renderer) if not n.startswith("_")}


def test_a_renderer_whose_forget_takes_only_a_name_fails_the_checklist(tmp_path, monkeypatch):
    mock = E.engine("mock")

    class Strict(mock.render.Renderer):
        def forget_adapter(self, name):
            return None
    slot = types.SimpleNamespace(**{**{n: getattr(mock.render, n) for n in C.RENDER_NAMES}, "Renderer": Strict})

    class Engine:
        descriptor = mock.descriptor
        render = slot
        train = mock.train
        measure = mock.measure
    monkeypatch.setattr(C, "engine", lambda engine_id: Engine())
    result = C.check("mock", tmp_path)
    assert "render.Renderer.calls" in {f.name for f in result.failed} and "forget_adapter" in result.report()


def test_the_mocks_progress_is_a_rising_fraction_that_ends_whole(tmp_path):
    from khaos_attribution.engines.render import RenderRequest
    seen = []
    renderer = E.engine("mock").render.Renderer()
    renderer.generate(RenderRequest(prompt="p", duration_seconds=1.0, seed=1), None, tmp_path / "o.wav",
                      on_progress=lambda done, total=1.0: seen.append(done / total))
    assert seen and seen == sorted(seen) and seen[-1] == 1.0 and all(0.0 <= f <= 1.0 for f in seen)
