"""The research options latent_pool and noise_draws: pooling averages the latent over blocks of frames before the
square, one draw reads as no option at all, and every measure that does not build them refuses before working."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from khaos_attribution import gradient as G

torch = pytest.importorskip("torch")


def _worker():
    from khaos_attribution.engines.sa3 import measure_worker as W
    return W


def test_pooling_averages_blocks_of_valid_frames_before_the_square():
    W = _worker()
    diff = torch.tensor([[[1.0, -1.0, 3.0, 5.0, 9.0, 0.0]]])
    where = torch.zeros(1, 1, 6, dtype=torch.bool)
    where[..., :5] = True
    # Five valid frames in blocks of two: (1,-1) and (3,5); the fifth is a short tail and the padding never counts.
    assert float(W._squared(torch, diff, where, None, 2)) == pytest.approx((0.0 ** 2 + 4.0 ** 2) / 2)
    assert float(W._squared(torch, diff, where, None, 1)) == pytest.approx((1 + 1 + 9 + 25 + 81) / 5)
    # Fewer valid frames than one block: the mean over what there is.
    short = torch.zeros(1, 1, 6, dtype=torch.bool)
    short[..., :3] = True
    assert float(W._squared(torch, diff, short, None, 8)) == pytest.approx(1.0)


def test_research_counts_are_whole_numbers_of_at_least_one():
    assert G.research_int(None, "x") == 1 and G.research_int(1, "x") == 1 and G.research_int(4, "x") == 4
    for bad in (-2, 2.5, True):
        with pytest.raises(G.GradientUnavailable, match="whole number"):
            G.research_int(bad, "x")


def test_the_worker_takes_one_gradient_per_draw_and_draw_zero_is_the_index_noise():
    src = (Path(G.__file__).parent / "engines" / "sa3" / "measure_worker.py").read_text()
    body = src[src.index('    if command in ("index", "feature"):'):src.index('    if command == "loss":')]
    assert "for d in range(draws)" in body and "pair_noise_seed(noise_seed + d, wav.name)" in body
    assert "flats[0] if draws == 1" in body, "one draw is exactly the index without the option"
    loss_fn = src[src.index("def _loss("):src.index("def _base_dtype(")]
    assert "pool" not in loss_fn, "the truth's loss is never pooled"


def test_the_other_measures_refuse_pooling_and_draws_before_working():
    from khaos_attribution.engines.acestep import measure as ACE
    from khaos_attribution.engines.mimetic import measure as MIM
    from khaos_attribution.engines.sa3 import measure_worker_mlx as MLX
    for measure in (ACE, MIM):
        for kw in ({"latent_pool": 4}, {"noise_draws": 2}):
            with pytest.raises(G.GradientUnavailable, match="pooling and noise draws"):
                measure.index_rows(object(), ["x.pt"], projection=G.make_projection(4, 2, 0), timesteps=[0.5],
                                   noise_seed=0, **kw)
    with pytest.raises(G.GradientUnavailable, match="pooling and noise draws"):
        G.output_feature(object(), {"projection": {"dim": 8, "seed": 0, "n_params": 4}, "timesteps": [0.5],
                                    "noise_seed": 0, "latent_pool": 4}, ["x.pt"])
    with pytest.raises(RuntimeError, match="noise_draws"):
        MLX.run({"command": "feature", "timesteps": [0.5], "noise_seed": 0, "files": [], "noise_draws": 2})


def test_the_sa3_measure_sends_pooling_and_draws_with_the_index_and_the_features(tmp_path, monkeypatch):
    from khaos_attribution.engines.sa3 import measure as M
    seen = tmp_path / "seen.jsonl"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys\nimport numpy as np\nreq = json.loads(sys.stdin.read())\n"
        f"open({str(seen)!r}, 'a').write(json.dumps(req) + '\\n')\n"
        "if req['command'] == 'index':\n"
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
    M.index_rows(handle, files, projection=G.make_projection(4, 2, 0), timesteps=[0.5], noise_seed=0,
                 latent_pool=8, noise_draws=4)
    meta = {"timesteps": [0.5], "noise_seed": 0, "feature_version": M.DESCRIPTOR.feature_version,
            "projection": {"dim": 2, "seed": 0, "n_params": 4}, "latent_pool": 8, "noise_draws": 4}
    M.output_feature(handle, meta, files)
    asked = [json.loads(line) for line in seen.read_text().splitlines()]
    assert [(a["latent_pool"], a["noise_draws"]) for a in asked] == [(8, 4), (8, 4)]
