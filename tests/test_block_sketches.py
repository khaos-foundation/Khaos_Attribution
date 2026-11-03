"""Block sketches on the SA3 measure: each adapter tensor named by a research pattern, each row's sketch cut into
its blocks so a reader can weight them; the blocks add up to the row; every other measure refuses before working."""
import json
import sys

import numpy as np
import pytest

from khaos_attribution import gradient as G

PATTERN = r"layers\.(\d+)\.(attn|ff)\."
NAMES = ["emb.lora_A", "layers.0.attn.lora_A", "layers.0.ff.lora_A", "layers.1.attn.lora_A", "layers.0.attn.lora_B",
         "out.lora_B"]


def test_each_tensor_is_named_by_the_patterns_captured_parts_or_other():
    assert G.param_groups(NAMES, PATTERN) == ["other", "0|attn", "0|ff", "1|attn", "0|attn", "other"]
    with pytest.raises(G.GradientUnavailable, match="matches none"):
        G.param_groups(NAMES, r"blocks\.(\d+)\.")
    with pytest.raises(G.GradientUnavailable, match="captures nothing"):
        G.param_groups(NAMES, r"layers\.")


def test_the_blocks_add_up_to_the_rows_sketch_even_when_a_blocks_tensors_are_apart():
    rng = np.random.default_rng(3)
    sizes = [5, 40, 17, 33, 9, 12]
    labels = G.param_groups(NAMES, PATTERN)
    proj = G.make_projection(sum(sizes), 16, 11)
    sketch = G.group_sketch(proj, sizes, labels)
    assert sketch.labels == ["other", "0|attn", "0|ff", "1|attn"], "groups keep their first-seen order"
    v = rng.normal(size=sum(sizes)).astype(np.float32)
    parts = G.project_groups(v, sketch)
    assert parts.shape == (4, 16)
    assert np.allclose(parts.sum(axis=0), G.project(v, proj), rtol=1e-5, atol=1e-6)
    # Block "0|attn" holds tensors 1 and 4 only: zeroing every other coordinate must leave its sketch alone.
    keep = np.zeros_like(v)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    for i in (1, 4):
        keep[starts[i]:starts[i] + sizes[i]] = v[starts[i]:starts[i] + sizes[i]]
    assert np.allclose(parts[1], G.project(keep, proj), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        G.group_sketch(proj, sizes[:-1], labels[:-1])


def test_the_worker_writes_each_rows_blocks_and_each_outputs_mean_blocks():
    from pathlib import Path
    src = (Path(G.__file__).parent / "engines" / "sa3" / "measure_worker.py").read_text()
    loop = src[src.index('    if command in ("index", "feature"):'):src.index('    if command == "loss":')]
    assert "group_rows[i] = parts" in loop and "G.project_groups(flat, sketch)" in loop
    assert "np.mean(np.stack(group_feats), axis=0)" in loop
    assert 'extra["group_labels"] = sketch.labels' in loop
    assert loop.index("G.project_groups(flat, sketch)") < loop.index("projected = G.project(flat, projection)"), \
        "the blocks are cut from the same vector the row is"


def test_the_other_measures_refuse_block_sketches_before_working():
    from khaos_attribution.engines.acestep import measure as ACE
    from khaos_attribution.engines.mimetic import measure as MIM
    from khaos_attribution.engines.sa3 import measure_worker_mlx as MLX
    for measure in (ACE, MIM):
        with pytest.raises(G.GradientUnavailable, match="block sketches"):
            measure.index_rows(object(), ["x.pt"], projection=G.make_projection(4, 2, 0), timesteps=[0.5],
                               noise_seed=0, param_groups=PATTERN, groups_out="g.npy")
        with pytest.raises(G.GradientUnavailable, match="block sketches"):
            measure.output_feature(object(), {"projection": {"dim": 2, "seed": 0, "n_params": 4}, "timesteps": [0.5],
                                              "noise_seed": 0}, ["x.pt"], groups_out="g.npy")
    with pytest.raises(G.GradientUnavailable, match="block sketches"):
        G.output_feature(object(), {"projection": {"dim": 8, "seed": 0, "n_params": 4}, "timesteps": [0.5],
                                    "noise_seed": 0, "param_groups": PATTERN}, ["x.pt"])
    with pytest.raises(RuntimeError, match="param_groups"):
        MLX.run({"command": "feature", "timesteps": [0.5], "noise_seed": 0, "files": [], "param_groups": PATTERN})


def _fake_worker(tmp_path):
    seen = tmp_path / "seen.jsonl"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys\nimport numpy as np\nreq = json.loads(sys.stdin.read())\n"
        f"open({str(seen)!r}, 'a').write(json.dumps(req) + '\\n')\n"
        "dim = req['projection']['dim'] if req['command'] == 'index' else req['index_meta']['projection']['dim']\n"
        "out = {}\n"
        "if req.get('groups_out'):\n"
        "    shape = (len(req['files']), 2, dim) if req['command'] == 'index' else (2, dim)\n"
        "    np.save(req['groups_out'], np.ones(shape, np.float32))\n"
        "    out['group_labels'] = ['0|attn', 'other']\n"
        "if req['command'] == 'index':\n"
        "    np.save(req['out'], np.ones((len(req['files']), dim), np.float32))\n"
        "    out.update({'rows': [{'row': i, 'pair_id': 'p', 'track_id': 't', 'gradient_norm': 1.0, 'residual': 0.1}\n"
        "                         for i, _ in enumerate(req['files'])], 'matrix': req['out']})\n"
        "else:\n"
        "    out['feature'] = [0.0] * dim\n"
        "print(json.dumps(out))\n")
    return seen, worker


def test_the_sa3_measure_sends_the_pattern_and_keeps_the_block_names_beside_the_sketches(tmp_path, monkeypatch):
    from khaos_attribution.engines.sa3 import measure as M
    seen, worker = _fake_worker(tmp_path)
    monkeypatch.setattr(M, "MEASURE_WORKER", worker)
    adapter = tmp_path / "a.safetensors"
    adapter.write_bytes(b"w")
    handle = M.Handle(root=tmp_path, python=sys.executable, adapter=adapter, dtype="float32")
    files = [tmp_path / "t_seg_000.wav"]
    files[0].write_bytes(b"x")
    groups = tmp_path / "index_groups.npy"
    M.index_rows(handle, files, projection=G.make_projection(4, 2, 0), timesteps=[0.5], noise_seed=0,
                 param_groups=PATTERN, groups_out=groups)
    assert np.load(groups).shape == (1, 2, 2)
    assert json.loads(groups.with_suffix(".labels.json").read_text()) == ["0|attn", "other"]
    meta = {"timesteps": [0.5], "noise_seed": 0, "feature_version": M.DESCRIPTOR.feature_version,
            "projection": {"dim": 2, "seed": 0, "n_params": 4}, "param_groups": PATTERN}
    out = tmp_path / "o.npy"
    M.output_feature(handle, meta, files, groups_out=out)
    assert np.load(out).shape == (2, 2)
    M.output_feature(handle, meta, files)
    asked = [json.loads(line) for line in seen.read_text().splitlines()]
    assert [a["param_groups"] for a in asked] == [PATTERN] * 3
    assert [bool(a.get("groups_out")) for a in asked] == [True, True, False], "blocks only where a file is named"


def test_the_sa3_measure_refuses_half_a_block_request(tmp_path):
    from khaos_attribution.engines.sa3 import measure as M
    handle = M.Handle(root=tmp_path, python=sys.executable, adapter=tmp_path / "a", dtype="float32")
    with pytest.raises(G.GradientUnavailable, match="needs groups_out"):
        M.index_rows(handle, [tmp_path / "x.wav"], projection=G.make_projection(4, 2, 0), timesteps=[0.5],
                     noise_seed=0, param_groups=PATTERN)
    with pytest.raises(G.GradientUnavailable, match="needs param_groups"):
        M.index_rows(handle, [tmp_path / "x.wav"], projection=G.make_projection(4, 2, 0), timesteps=[0.5],
                     noise_seed=0, groups_out=tmp_path / "g.npy")
    meta = {"timesteps": [0.5], "noise_seed": 0, "feature_version": M.DESCRIPTOR.feature_version,
            "projection": {"dim": 2, "seed": 0, "n_params": 4}}
    with pytest.raises(G.GradientUnavailable, match="without block sketches"):
        M.output_feature(handle, meta, [tmp_path / "x.wav"], groups_out=tmp_path / "o.npy")
