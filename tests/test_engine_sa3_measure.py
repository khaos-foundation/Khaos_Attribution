"""Stable Audio 3's measure slot behind a fake worker: the handle names where the engine and the adapter are, the
index rows and an output's feature and paired loss come back from the worker in the contract's shapes, and a
worker's refusal is the socket's."""
import json
import sys

import numpy as np
import pytest

from khaos_attribution.engines.sa3 import measure as M
from khaos_attribution.engines.sa3 import train as T
from khaos_attribution.gradient import GradientUnavailable, make_projection

FAKE_WORKER = '''
import json, sys
import numpy as np
req = json.loads(sys.stdin.read())
c = req["command"]
if req["adapter"].endswith("bad.safetensors"):
    sys.stdout.write(json.dumps({"error": "a weight-decomposed adapter cannot be switched off for a measurement"}) + "\\n"); sys.exit(1)
if c == "parameters":
    out = {"n_params": 12, "n_tensors": 3, "signature": "abc", "device": "cuda", "compute_dtype": "fp32"}
elif c == "index":
    dim = req["projection"]["dim"]
    m = np.zeros((len(req["files"]), dim), np.float32); m[:, 0] = 1.0
    np.save(req["out"], m)
    out = {"rows": [{"row": i, "pair_id": f.split("/")[-1][:-4], "track_id": "t", "gradient_norm": 2.0} for i, f in enumerate(req["files"])],
           "matrix": req["out"], "n_params": 12, "n_tensors": 3, "signature": "abc", "device": "cuda", "compute_dtype": "fp32"}
elif c == "feature":
    out = {"feature": [0.5] * req["index_meta"]["projection"]["dim"], "n": len(req["files"])}
elif c == "loss":
    out = {"with": 0.3, "without": 0.9, "n": len(req["files"])}
sys.stdout.write("noise\\n" + json.dumps(out) + "\\n")
'''


@pytest.fixture
def handle(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    monkeypatch.setattr(M, "MEASURE_WORKER", worker)
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"w")
    return M.Handle(root=tmp_path, python=sys.executable, adapter=adapter)


def test_open_module_gives_a_handle_for_each_runtime(tmp_path):
    api = T.TrainingApi(root=tmp_path, runtime="torch", python=sys.executable)
    adapter = tmp_path / "a.safetensors"
    with pytest.raises(T.TrainingUnavailable, match="no adapter file"):
        T.open_module(api, {}, adapter, device="cuda:0")
    adapter.write_bytes(b"w")
    h = T.open_module(api, {}, adapter, device="cuda:0")
    assert isinstance(h, M.Handle) and h.device_type == "cuda" and h.adapter == adapter.resolve() and h.dtype == "torch.float32"
    assert h.runtime == "torch" and h.base_npz is None
    mlx = T.TrainingApi(root=tmp_path, runtime="mlx", python=sys.executable)
    with pytest.raises(T.TrainingUnavailable, match="base weights"):
        T.open_module(mlx, {}, adapter, checkpoint_root=tmp_path / "ckpt")
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / T.BASE_NPZ).write_bytes(b"npz")
    hm = T.open_module(mlx, {}, adapter, checkpoint_root=tmp_path / "ckpt", device="mps")
    assert hm.runtime == "mlx" and hm.device_type == "mps" and hm.base_npz == (tmp_path / "ckpt" / T.BASE_NPZ).absolute()
    assert hm.dtype == "float16", "recorded as fp16: the trainer's half-precision base"


def test_an_mlx_handle_asks_the_mlx_worker_and_names_the_base_weights(handle, tmp_path, monkeypatch):
    seen = tmp_path / "seen.json"
    mlx_worker = tmp_path / "mlx_worker.py"
    mlx_worker.write_text("import json, sys\nreq = json.loads(sys.stdin.read())\n"
                          f"open({str(seen)!r}, 'w').write(json.dumps(req))\n"
                          "print(json.dumps({'n_params': 5, 'n_tensors': 2, 'signature': 'mlx'}))\n")
    monkeypatch.setattr(M, "MEASURE_WORKER_MLX", mlx_worker)
    handle.runtime, handle.base_npz, handle.device_type = "mlx", tmp_path / "base.npz", "mps"
    assert M.parameter_count(handle) == (5, 2, "mlx")
    req = json.loads(seen.read_text())
    assert req["runtime"] == "mlx" and req["base_npz"] == str(tmp_path / "base.npz")


def test_the_index_rows_and_features_come_back_in_the_contracts_shapes(handle, tmp_path):
    assert M.parameter_count(handle) == (12, 3, "abc")
    files = [tmp_path / f"t_seg_00{i}.wav" for i in range(3)]
    for f in files:
        f.write_bytes(b"x")
    seen = []
    matrix, rows = M.index_rows(handle, files, projection=make_projection(12, 8, 1), timesteps=[0.5], noise_seed=7,
                                on_row=lambda i, p, n: seen.append((i, p.name, n)))
    assert matrix.shape == (3, 8) and matrix.dtype == np.float32 and rows[1]["pair_id"] == "t_seg_001"
    assert seen == [(0, "t_seg_000.wav", 2.0), (1, "t_seg_001.wav", 2.0), (2, "t_seg_002.wav", 2.0)]
    assert not (tmp_path / ".khaos_index_rows.tmp.npy").exists(), "the worker's matrix is read and removed"
    meta = {"timesteps": [0.5], "noise_seed": 7, "target": "vnorm", "feature_version": M.DESCRIPTOR.feature_version,
            "projection": {"kind": "count_sketch", "dim": 8, "seed": 1, "n_params": 12}}
    feat = M.output_feature(handle, meta, files[:1])
    assert feat.shape == (8,) and feat.dtype == np.float32 and float(feat[0]) == 0.5
    assert M.paired_loss(handle, meta, files) == {"with": 0.3, "without": 0.9} and M.output_loss(handle, meta, files) == 0.3
    with pytest.raises(NotImplementedError):
        M.index_rows(handle, files, projection=make_projection(12, 8, 1), timesteps=[0.5], noise_seed=7, timestep_norm=True)


def test_the_rows_reach_the_room_while_the_worker_runs(handle, tmp_path, monkeypatch):
    """Each row line the worker prints reaches on_row as it comes, once, and a stray line never breaks the reply."""
    worker = tmp_path / "rows_worker.py"
    worker.write_text("import json, sys, numpy as np\nreq = json.loads(sys.stdin.read())\n"
                      "rows = []\n"
                      "for i, f in enumerate(req['files']):\n"
                      "    rows.append({'row': i, 'pair_id': 'p', 'track_id': 't', 'gradient_norm': 1.0 + i})\n"
                      "    print('khaos-row ' + json.dumps(rows[-1]), flush=True)\n"
                      "print('khaos-row not json', flush=True)\n"
                      "sys.stderr.write('x' * 200000)\n"
                      "np.save(req['out'], np.zeros((len(rows), req['projection']['dim']), np.float32))\n"
                      "print(json.dumps({'rows': rows, 'matrix': req['out']}))\n")
    monkeypatch.setattr(M, "MEASURE_WORKER", worker)
    files = [tmp_path / f"t_seg_00{i}.wav" for i in range(2)]
    for f in files:
        f.write_bytes(b"x")
    seen = []
    M.index_rows(handle, files, projection=make_projection(12, 8, 1), timesteps=[0.5], noise_seed=7,
                 on_row=lambda i, p, n: seen.append((i, n)))
    assert seen == [(0, 1.0), (1, 2.0)]


def test_a_workers_refusal_is_the_sockets(handle, tmp_path):
    handle.adapter = tmp_path / "bad.safetensors"
    handle.adapter.write_bytes(b"w")
    with pytest.raises(GradientUnavailable, match="cannot be switched off"):
        M.paired_loss(handle, {"timesteps": [0.5], "noise_seed": 1}, [tmp_path / "x.wav"])
    assert M.memory_plan("cuda", 30.0, total_gb=96.0, env={})["fits"] == "resident"
    assert M.memory_plan("cuda", 30.0, total_gb=8.0, env={})["fits"] is None
    assert "not been measured on cpu" in M.memory_plan("cpu", 30.0, total_gb=96.0, env={})["refusal"]
    assert M.memory_plan("mps", 56.0, total_gb=32.0, env={})["fits"] == "resident", "the MLX worker fits a 32 GB Mac"
    assert M.memory_plan("mps", 56.0, total_gb=16.0, env={})["fits"] is None
    assert M.scorer_need_gb(56, device_type="mps") == 21.9
    assert M.scorer_need_gb(300) == M.scorer_need_gb(5) == 9.2, "one measured number, not fitted by length"


def test_an_index_from_another_measure_version_or_layout_is_refused(handle, tmp_path):
    """Noise and parameter order are part of what a row means: an index built by another version of the measure,
    or on another layout of the adapter's parameters, is never read against today's features."""
    f = tmp_path / "t_seg_000.wav"
    f.write_bytes(b"x")
    meta = {"timesteps": [0.5], "noise_seed": 7, "projection": {"kind": "count_sketch", "dim": 8, "seed": 1, "n_params": 12}}
    with pytest.raises(GradientUnavailable, match="before it was versioned"):
        M.output_feature(handle, meta, [f])
    with pytest.raises(GradientUnavailable, match="rebuild the index"):
        M.output_feature(handle, {**meta, "feature_version": "sa3-vnorm/1"}, [f])
    ok = {**meta, "feature_version": M.DESCRIPTOR.feature_version}
    assert M.output_feature(handle, ok, [f]).shape == (8,), "a worker that names no signature is read"


def test_a_worker_left_behind_is_stopped_when_its_caller_fails(handle, tmp_path, monkeypatch):
    import time
    worker = tmp_path / "slow.py"
    worker.write_text("import json, sys, time\nsys.stdin.read()\n"
                      "print('khaos-row ' + json.dumps({'row': 0, 'gradient_norm': 1.0}), flush=True)\ntime.sleep(60)\n")
    monkeypatch.setattr(M, "MEASURE_WORKER", worker)

    def boom(row):
        raise RuntimeError("the caller stopped")
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="caller stopped"):
        M._ask(handle, {"command": "index"}, on_progress=boom)
    assert time.monotonic() - start < 10, "the worker is killed, not waited on for its minute"
