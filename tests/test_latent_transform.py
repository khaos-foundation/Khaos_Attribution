"""The research latent transforms: statistics measured once over the training latents give one linear map over the
channels, whitening scales each channel by 1/std, PCA keeps the top directions; the index carries the map so an
output's feature is taken in the same space, and engines that do not build it refuse."""
import json
import sys

import numpy as np
import pytest

from khaos_attribution import gradient as G


def _moments(frames: np.ndarray) -> tuple:
    """(n, channel sums, channel outer sums) of frames laid out channels x time."""
    x = frames.astype(np.float64)
    return x.shape[1], x.sum(axis=1), x @ x.T


def test_whitening_scales_each_channel_to_unit_std_over_the_training_frames():
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(3, 4000)) * np.array([[10.0], [1.0], [0.1]]) + 5.0
    t = G.latent_transform_from_moments("whiten", *_moments(frames))
    assert t["kind"] == "whiten" and t["channels"] == 3 and t["frames"] == 4000
    mapped = G.latent_map(t) @ frames
    assert np.allclose(mapped.std(axis=1), 1.0, atol=0.02), "every channel at unit spread after the map"


def test_pca_keeps_the_fewest_directions_reaching_the_variance_target_and_records_it():
    rng = np.random.default_rng(1)
    frames = rng.normal(size=(4, 5000)) * np.array([[10.0], [3.0], [0.5], [0.1]])
    t = G.latent_transform_from_moments("pca", *_moments(frames), variance=0.9)
    assert t["k"] == 1 and t["explained"] >= 0.9 and t["variance_target"] == 0.9
    t95 = G.latent_transform_from_moments("pca", *_moments(frames), variance=0.95)
    assert t95["k"] == 2 and G.latent_map(t95).shape == (2, 4)
    m = G.latent_map(t95)
    assert np.allclose(m @ m.T, np.eye(2), atol=1e-8), "orthonormal directions: a projection, not a rescaling"


def test_a_transform_refuses_an_unknown_kind_or_no_frames():
    with pytest.raises(G.GradientUnavailable, match="latent transform"):
        G.latent_transform_from_moments("blur", 10, np.zeros(2), np.zeros((2, 2)))
    with pytest.raises(G.GradientUnavailable, match="no latent frames"):
        G.latent_transform_from_moments("whiten", 0, np.zeros(2), np.zeros((2, 2)))


def test_an_ace_step_feature_refuses_an_index_built_in_a_transformed_latent_space():
    with pytest.raises(G.GradientUnavailable, match="latent transform"):
        G.output_feature(object(), {"projection": {"dim": 8, "seed": 0, "n_params": 4}, "timesteps": [0.5],
                                    "noise_seed": 0, "latent_transform": {"kind": "whiten", "scale": [1.0]}}, ["x.pt"])


def test_ace_step_and_mimetic_refuse_to_build_a_transformed_index():
    from khaos_attribution.engines.acestep import measure as ACE
    from khaos_attribution.engines.mimetic import measure as MIM
    for measure in (ACE, MIM):
        assert not hasattr(measure, "latent_stats"), "no statistics on an engine that does not build the transform"
        with pytest.raises((G.GradientUnavailable, NotImplementedError), match="latent transform"):
            measure.index_rows(object(), ["x.pt"], projection=G.make_projection(4, 2, 0), timesteps=[0.5], noise_seed=0,
                               latent_transform={"kind": "whiten", "scale": [1.0]})


def test_the_mlx_worker_refuses_a_latent_transform_before_loading_anything():
    from khaos_attribution.engines.sa3 import measure_worker_mlx as W
    with pytest.raises(RuntimeError, match="latent_transform"):
        W.run({"command": "feature", "timesteps": [0.5], "noise_seed": 0, "files": [],
               "latent_transform": {"kind": "whiten", "scale": [1.0]}})


def test_the_sa3_measure_asks_the_worker_for_statistics_and_sends_them_with_the_index_and_features(tmp_path, monkeypatch):
    from khaos_attribution.engines.sa3 import measure as M
    seen = tmp_path / "seen.jsonl"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys\nimport numpy as np\nreq = json.loads(sys.stdin.read())\n"
        f"open({str(seen)!r}, 'a').write(json.dumps(req) + '\\n')\n"
        "c = req['command']\n"
        "if c == 'latent_stats':\n"
        "    out = {'transform': {'kind': req['latent_kind'], 'channels': 2, 'frames': 9, 'scale': [1.0, 2.0]}}\n"
        "elif c == 'index':\n"
        "    np.save(req['out'], np.ones((len(req['files']), req['projection']['dim']), np.float32))\n"
        "    out = {'rows': [{'row': i, 'pair_id': 'p', 'track_id': 't', 'gradient_norm': 1.0} for i, _ in enumerate(req['files'])],\n"
        "           'matrix': req['out']}\n"
        "else:\n"
        "    out = {'feature': [0.0] * req['index_meta']['projection']['dim']}\n"
        "print(json.dumps(out))\n")
    monkeypatch.setattr(M, "MEASURE_WORKER", worker)
    adapter = tmp_path / "a.safetensors"
    adapter.write_bytes(b"w")
    handle = M.Handle(root=tmp_path, python=sys.executable, adapter=adapter, dtype="float32")
    files = [tmp_path / "t_seg_000.wav"]
    files[0].write_bytes(b"x")
    stats = M.latent_stats(handle, files, kind="whiten")
    assert stats["kind"] == "whiten" and stats["scale"] == [1.0, 2.0]
    M.index_rows(handle, files, projection=G.make_projection(4, 2, 0), timesteps=[0.5], noise_seed=0, latent_transform=stats)
    meta = {"timesteps": [0.5], "noise_seed": 0, "feature_version": M.DESCRIPTOR.feature_version,
            "projection": {"dim": 2, "seed": 0, "n_params": 4}, "latent_transform": stats}
    M.output_feature(handle, meta, files)
    M.paired_loss.__doc__  # the loss command is the truth's and is never transformed (checked in the worker)
    asked = [json.loads(line) for line in seen.read_text().splitlines()]
    assert [a["command"] for a in asked] == ["latent_stats", "index", "feature"]
    assert asked[0]["latent_kind"] == "whiten" and asked[0]["latent_variance"] == 0.9
    assert asked[1]["latent_transform"] == stats and asked[2]["latent_transform"] == stats
    with pytest.raises(G.GradientUnavailable, match="latent transform"):
        M.latent_stats(handle, files, kind="blur")
    handle.runtime = "mlx"
    with pytest.raises(G.GradientUnavailable, match="MLX"):
        M.latent_stats(handle, files, kind="whiten")


def test_the_worker_squares_the_loss_in_the_transformed_space_and_leaves_the_truths_loss_alone():
    """The map's place in the worker: one helper applies it, the truth's loss path never calls it."""
    from pathlib import Path
    src = (Path(G.__file__).parent / "engines" / "sa3" / "measure_worker.py").read_text()
    loss_fn = src[src.index("def _loss("):src.index("def _base_dtype(")]
    assert "_squared(" in src[src.index("def _gradient("):src.index("def _loss(")]
    assert "lmap" not in loss_fn and "_squared" not in loss_fn
