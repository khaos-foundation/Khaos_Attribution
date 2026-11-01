"""Shared noise and residual rows on the SA3 measure: one seeded draw per timestep shared by every clip and output,
cut to each clip's frames; a training residual per index row, normalised over the timesteps; the signed-mean output
functions; every other measure refuses before working."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from khaos_attribution import gradient as G


def test_shared_noise_depends_on_the_timestep_not_the_file_and_agrees_on_common_frames():
    a = G.shared_noise(7, 0, (1, 4, 100))
    b = G.shared_noise(7, 0, (1, 4, 60))
    assert a.shape == (1, 4, 100) and b.shape == (1, 4, 60)
    assert np.array_equal(a[:, :, :60], b), "two clips of different lengths share every frame they both have"
    assert not np.array_equal(G.shared_noise(7, 1, (1, 4, 60)), b), "another timestep is another draw"
    assert not np.array_equal(G.shared_noise(8, 0, (1, 4, 60)), b), "another run seed is another draw"
    assert G.shared_noise_seed(7, 3) == G.pair_noise_seed(7, "shared-noise:3")
    with pytest.raises(G.GradientUnavailable, match="longer than the shared noise"):
        G.shared_noise(7, 0, (1, 4, G.SHARED_NOISE_FRAMES + 1))


def test_a_pairs_residual_is_the_mean_of_its_timesteps_residuals_at_unit_norm():
    assert G.residual_over_timesteps([3.0, 4.0]) == pytest.approx((0.6 + 0.8) / 2)
    assert G.residual_over_timesteps([-1.0, 1.0]) == pytest.approx(0.0)
    assert G.residual_over_timesteps([0.0, 0.0]) == 0.0
    with pytest.raises(G.GradientUnavailable):
        G.residual_over_timesteps([])


def test_the_worker_writes_a_residual_per_row_and_shares_noise_per_timestep_when_asked():
    src = (Path(G.__file__).parent / "engines" / "sa3" / "measure_worker.py").read_text()
    grad = src[src.index("def _gradient("):src.index("def _shared_noises(")]
    assert "out.detach() - (noise - latents)" in grad and "residual_over_timesteps" in grad
    assert '"vmean"' in grad and '"dmean"' in grad and "_signed_mean" in grad
    loop = src[src.index('    if command in ("index", "feature"):'):src.index('    if command == "loss":')]
    assert '"residual": round(residual, 8)' in loop
    assert "_shared_noises(torch, noise_seed, d, timesteps, clip[0].shape, device) if noise_shared else None" in loop
    loss_fn = src[src.index("def _loss("):src.index("def _base_dtype(")]
    assert "shared" not in loss_fn and "residual" not in loss_fn, "the truth's loss is untouched"
    assert "vmean" in src[src.index("TARGETS = ("):src.index("TARGETS = (") + 80]


def test_the_other_measures_refuse_shared_noise_before_working():
    from khaos_attribution.engines.acestep import measure as ACE
    from khaos_attribution.engines.mimetic import measure as MIM
    from khaos_attribution.engines.sa3 import measure_worker_mlx as MLX
    for measure in (ACE, MIM):
        with pytest.raises(G.GradientUnavailable, match="[Ss]hared noise"):
            measure.index_rows(object(), ["x.pt"], projection=G.make_projection(4, 2, 0), timesteps=[0.5],
                               noise_seed=0, noise_shared=True)
    with pytest.raises(G.GradientUnavailable, match="[Ss]hared noise"):
        G.output_feature(object(), {"projection": {"dim": 8, "seed": 0, "n_params": 4}, "timesteps": [0.5],
                                    "noise_seed": 0, "noise_shared": True}, ["x.pt"])
    with pytest.raises(RuntimeError, match="noise_shared"):
        MLX.run({"command": "feature", "timesteps": [0.5], "noise_seed": 0, "files": [], "noise_shared": True})
    with pytest.raises(RuntimeError, match="signed-mean"):
        MLX.run({"command": "feature", "timesteps": [0.5], "noise_seed": 0, "files": [], "target": "dmean"})


def test_the_sa3_measure_sends_shared_noise_with_the_index_and_the_features(tmp_path, monkeypatch):
    from khaos_attribution.engines.sa3 import measure as M
    seen = tmp_path / "seen.jsonl"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys\nimport numpy as np\nreq = json.loads(sys.stdin.read())\n"
        f"open({str(seen)!r}, 'a').write(json.dumps(req) + '\\n')\n"
        "if req['command'] == 'index':\n"
        "    np.save(req['out'], np.ones((len(req['files']), req['projection']['dim']), np.float32))\n"
        "    out = {'rows': [{'row': i, 'pair_id': 'p', 'track_id': 't', 'gradient_norm': 1.0, 'residual': 0.1} for i, _ in enumerate(req['files'])],\n"
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
    _, rows = M.index_rows(handle, files, projection=G.make_projection(4, 2, 0), timesteps=[0.5], noise_seed=0,
                           noise_shared=True)
    assert rows[0]["residual"] == 0.1, "the worker's residual rides on the row"
    meta = {"timesteps": [0.5], "noise_seed": 0, "feature_version": M.DESCRIPTOR.feature_version,
            "projection": {"dim": 2, "seed": 0, "n_params": 4}, "noise_shared": True}
    M.output_feature(handle, meta, files)
    asked = [json.loads(line) for line in seen.read_text().splitlines()]
    assert [a["noise_shared"] for a in asked] == [True, True]
