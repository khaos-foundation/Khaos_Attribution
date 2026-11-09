"""Stable Audio 3's reworks: regions on the latent grid, the noise each strength starts from, extend chunks inside
the window, the splice that leaves the source's samples untouched outside the parts and their crossfades, the worker
request per mode on both runtimes, and the MLX driver's every-part mask."""
import json
import sys

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from importlib import resources  # noqa: E402

from khaos_attribution.engines.descriptor import DescriptorError, packaged_descriptor, validate_descriptor  # noqa: E402
from khaos_attribution.engines.render import RenderFailed, RenderRequest  # noqa: E402
from khaos_attribution.engines.sa3 import render as R  # noqa: E402
from khaos_attribution.engines.sa3 import render_driver_mlx as M  # noqa: E402
from khaos_attribution.engines.sa3 import rework as RW  # noqa: E402
from khaos_attribution.engines.sa3 import worker as WK  # noqa: E402

SR, FRAME = 44100, 4096

FAKE_WORKER = '''
import json, os, sys
import numpy as np, soundfile as sf
req = json.loads(sys.stdin.read())
with open(os.environ["SA3_FAKE_LOG"], "a") as fh:
    fh.write(json.dumps(req) + "\\n")
n = int(req["seconds"] * 44100)
t = np.arange(n) / 44100.0
made = 0.4 * np.sin(2 * np.pi * 220 * t + 0.7)
if req.get("source_audio"):
    src, _ = sf.read(req["source_audio"], dtype="float64", always_2d=True)
    keep = np.ones(n)
    for s, e in req.get("regions") or []:
        keep[int(round(s * 44100)):int(round(e * 44100))] = 0
    src = np.concatenate([src[:n, 0], np.zeros(max(0, n - len(src)))])
    made = np.where(keep > 0, 0.98 * src + 0.02 * made, made)
sf.write(req["out"], np.stack([made, made], axis=1), 44100, subtype="PCM_16")
sys.stdout.write(json.dumps({"seed": req["seed"] if req["seed"] is not None else 5, "sample_rate": 44100, "dropped": []}) + "\\n")
'''


def tone(seconds: float, freq: float = 220.0, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(round(seconds * SR))) / SR
    x = 0.5 * np.sin(2 * np.pi * freq * t + phase)
    return np.stack([x, x], axis=1)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    root = tmp_path / "stable-audio-3"
    root.mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    log = tmp_path / "requests.jsonl"
    monkeypatch.setattr(R, "WORKER", worker)
    monkeypatch.setattr(R, "python_for", lambda root, runtime: sys.executable)
    monkeypatch.setenv("SA3_FAKE_LOG", str(log))
    renderer = R.Renderer(repo_dir=root, device="cpu")
    renderer.requests = lambda: [json.loads(line) for line in log.read_text().splitlines()]
    return renderer


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.wav"
    sf.write(str(path), tone(30.0), SR, subtype="PCM_16")
    return path


def test_regions_widen_outward_to_whole_latent_frames_clip_to_the_piece_and_merge():
    n = 30 * SR
    regions = RW.plan_regions([(10.0, 12.0), (11.9, 13.0), (29.95, None)], n, SR, FRAME)
    assert regions[0][0] == (10 * SR // FRAME) * FRAME and regions[0][1] == -(-13 * SR // FRAME) * FRAME
    assert all(s % FRAME == 0 for s, _ in regions) and regions[-1][1] == n, "the end clips to the piece"
    assert len(regions) == 2, "parts that meet after widening are one part"
    with pytest.raises(RenderFailed, match="must end after it starts"):
        RW.plan_regions([(5.0, 4.0)], n, SR, FRAME)


def test_the_noise_each_strength_starts_from_is_the_descriptors_and_follow_runs_the_right_way():
    table = packaged_descriptor("khaos_attribution.engines.sa3").rework
    level = lambda mode, **kw: RW.noise_level(mode, RenderRequest(prompt="p", duration_seconds=30, **kw), table)  # noqa: E731
    assert [level("repaint", repaint_mode=m) for m in RW.REPAINT_MODES] == [0.3, 0.6, 1.0]
    assert level("repaint") == 0.6, "a fix with no strength named is a Rework"
    assert level("retake", retake_variance=0.2) == 0.2 and level("retake") == 0.5
    assert level("audio2audio", cover_strength=0.8) == pytest.approx(0.2), "following more is less noise"
    assert level("audio2audio") == 0.75 and level("audio2audio", cover_strength=1.0) == 0.01, "never below the floor"


def test_the_splice_leaves_every_sample_outside_the_parts_and_their_crossfades_bit_identical(tmp_path):
    path = tmp_path / "s.wav"
    sf.write(str(path), tone(30.0), SR, subtype="PCM_16")
    other = tmp_path / "r.wav"
    sf.write(str(other), tone(30.0, freq=330.0), SR, subtype="PCM_16")
    src, _, subtype, step = RW.read_audio(path)
    regen = RW.read_as(other, src)
    regions = RW.plan_regions([(8.0, 10.0), (20.0, 21.0)], len(src), SR, FRAME)
    xf = int(0.05 * SR)
    out = RW.splice(src, regen, regions, xf, 0, step)
    RW.write_audio(tmp_path / "o.wav", out, SR, subtype)
    back, _ = sf.read(str(tmp_path / "o.wav"), dtype="int16", always_2d=True)
    orig, _ = sf.read(str(path), dtype="int16", always_2d=True)
    w = RW.weights(regions, len(src), xf)
    assert np.array_equal(back[w == 0], orig[w == 0]), "outside the parts and crossfades the source leaves unchanged"
    assert np.array_equal(out[w == 1], regen[w == 1]), "inside, away from the seams, the part is the model's"
    assert ((w > 0) & (w < 1)).sum() == 4 * 2 * (xf // 2), "four seams, each a crossfade centred on its edge"


def test_a_seam_has_no_step_and_the_piece_edges_are_never_faded():
    a, b = tone(10.0), tone(10.0, phase=2.0)
    regions = [(4 * FRAME, 40 * FRAME)]
    out = RW.splice(a, b, regions, int(0.05 * SR))
    biggest = max(np.abs(np.diff(a[:, 0])).max(), np.abs(np.diff(b[:, 0])).max())
    assert np.abs(np.diff(out[:, 0])).max() <= biggest * 1.5, "no click at either seam"
    w = RW.weights([(0, 8 * FRAME), (len(a) - 8 * FRAME, len(a))], len(a), 2000)
    assert w[0] == 1.0 and w[-1] == 1.0, "a part at the very start or end is not faded against nothing"


def test_an_extend_past_the_window_is_cut_into_chunks_that_each_fit_it():
    window, context = 120 * SR - 1, 30 * SR
    plans = RW.extend_plan(120 * SR, 30 * SR, window, context, None, FRAME)
    assert len(plans) == 2 and sum(new for _, new in plans) == 120 * SR
    assert all(ctx % FRAME == 0 and ctx <= context and ctx + new <= window for ctx, new in plans)
    head = RW.extend_plan(10 * SR, 30 * SR, window, context, None, FRAME, head=True)
    assert head == [((30 * SR // FRAME) * FRAME, 10 * SR)]
    assert len(RW.extend_plan(50 * SR, 30 * SR, window, context, 20 * SR, FRAME)) == 3, "a chunk size caps each pass"


def test_the_joins_keep_the_source_and_cross_into_the_new_audio_without_a_step():
    cur, xf = tone(30.0), 2205
    ctx = (10 * SR // FRAME) * FRAME
    gen = np.concatenate([cur[-ctx:], tone(20.0, phase=0.3)])
    tail = RW.join_tail(cur, gen, ctx, xf)
    assert len(tail) == len(cur) + 20 * SR and np.array_equal(tail[:len(cur) - xf], cur[:len(cur) - xf])
    pad = -(-5 * SR // FRAME) * FRAME
    gen = np.concatenate([tone(pad / SR, phase=1.0), cur[:ctx]])
    head = RW.join_head(cur, gen, pad, 5 * SR, xf)
    assert len(head) == len(cur) + 5 * SR and np.array_equal(head[5 * SR + xf:], cur[xf:])


def test_a_long_source_is_fixed_through_a_window_that_holds_every_part():
    n, window, margin = 200 * SR, 120 * SR - 1, 2205
    regions = RW.plan_regions([(150.0, 160.0)], n, SR, FRAME)
    offset, length = RW.repaint_window(regions, n, window, FRAME, margin, SR)
    assert offset % FRAME == 0 and offset <= regions[0][0] - margin and offset + length >= regions[0][1] + margin
    with pytest.raises(RenderFailed, match="separate passes"):
        RW.repaint_window(RW.plan_regions([(10.0, 12.0), (180.0, 182.0)], n, SR, FRAME), n, window, FRAME, margin, SR)


def test_fix_a_section_is_one_worker_pass_over_every_part_spliced_into_the_source(engine, source, tmp_path):
    req = RenderRequest(prompt="p", duration_seconds=30, mode="repaint", source_audio=source, repaint_mode="balanced",
                        repaint_regions=[[8.0, 10.0], [20.0, 21.0]], seed=3)
    out = engine.generate(req, None, tmp_path / "fixed.wav")
    (w,) = engine.requests()
    assert w["mode"] == "repaint" and w["noise_level"] == 0.6 and len(w["regions"]) == 2 and w["seed"] == 3
    assert all(abs(s * SR / FRAME - round(s * SR / FRAME)) < 1e-6 for s, _ in w["regions"]), "parts sit on the grid"
    assert int(w["seconds"] * SR) == 30 * SR
    got, _ = sf.read(str(out.audio_path), dtype="int16", always_2d=True)
    orig, _ = sf.read(str(source), dtype="int16", always_2d=True)
    weights = RW.weights([(int(round(s * SR)), int(round(e * SR))) for s, e in out.params["regions"]], len(orig), 2205)
    assert len(got) == len(orig) and np.array_equal(got[weights == 0], orig[weights == 0])
    assert not np.array_equal(got, orig)
    p = out.params
    assert p["mode"] == "repaint" and p["khaos_seed"] == 3 and p["repaint_mode"] == "balanced" and p["duration"] == 30
    assert p["requested_regions"] == [[8.0, 10.0], [20.0, 21.0]] and p["crossfade_seconds"] == 0.05


def test_another_take_noises_the_whole_source_with_a_new_seed_per_candidate(engine, source, tmp_path):
    a = engine.generate(RenderRequest(prompt="p", duration_seconds=30, mode="retake", source_audio=source,
                                      retake_variance=0.3, seed=3), None, tmp_path / "a.wav")
    b = engine.generate(RenderRequest(prompt="p", duration_seconds=30, mode="retake", source_audio=source,
                                      retake_variance=0.3, seed=3), None, tmp_path / "b.wav")
    first, second = engine.requests()
    assert first["noise_level"] == 0.3 and first["regions"] == [] and first["source_audio"] == str(source.resolve())
    assert a.seed != b.seed and a.params["khaos_seed"] == a.seed, "the source's seed is not the candidate's"
    assert a.params["duration"] == 30, "the record keeps the piece's length, not the worker's headroom"
    c = engine.generate(RenderRequest(prompt="p", duration_seconds=30, mode="retake", source_audio=source,
                                      retake_seed=42), None, tmp_path / "c.wav")
    assert c.seed == 42 and engine.requests()[-1]["noise_level"] == 0.5


def test_take_it_somewhere_new_follows_the_source_by_its_strength_with_the_new_prompt(engine, source, tmp_path):
    out = engine.generate(RenderRequest(prompt="a new direction", duration_seconds=30, mode="audio2audio",
                                        source_audio=source, cover_strength=0.25), None, tmp_path / "n.wav")
    (w,) = engine.requests()
    assert w["prompt"] == "a new direction" and w["noise_level"] == pytest.approx(0.75) and w["regions"] == []
    assert out.params["cover_strength"] == 0.25 and out.params["mode"] == "audio2audio"


def test_make_it_longer_past_the_window_renders_chunks_and_keeps_the_source(engine, source, tmp_path):
    out = engine.generate(RenderRequest(prompt="p", duration_seconds=150, mode="extend", source_audio=source, seed=9),
                          None, tmp_path / "long.wav")
    requests = engine.requests()
    assert len(requests) == 2 and all(r["seconds"] <= 120 and r["noise_level"] == 1.0 for r in requests)
    assert [r["seed"] for r in requests] == [9, 10]
    got, _ = sf.read(str(out.audio_path), dtype="int16", always_2d=True)
    orig, _ = sf.read(str(source), dtype="int16", always_2d=True)
    assert len(got) == 150 * SR and np.array_equal(got[:len(orig) - 2205], orig[:len(orig) - 2205])
    assert out.params["duration"] == 150 and len(out.params["chunks"]) == 2 and out.params["extend_side"] == "tail"


def test_make_it_longer_at_the_start_puts_new_audio_before_the_source(engine, source, tmp_path):
    out = engine.generate(RenderRequest(prompt="p", duration_seconds=40, mode="extend", source_audio=source,
                                        extend_head=10.0), None, tmp_path / "head.wav")
    (w,) = engine.requests()
    assert w["regions"][0][0] == 0.0 and (w["regions"][0][1] * SR) % FRAME == pytest.approx(0, abs=1e-6)
    got, _ = sf.read(str(out.audio_path), dtype="int16", always_2d=True)
    orig, _ = sf.read(str(source), dtype="int16", always_2d=True)
    assert len(got) == 40 * SR and np.array_equal(got[10 * SR + 2205:], orig[2205:])


def test_a_rework_without_its_source_or_too_long_to_take_whole_is_refused(engine, tmp_path):
    with pytest.raises(RenderFailed, match="needs the source audio"):
        engine.generate(RenderRequest(prompt="p", duration_seconds=30, mode="retake"), None, tmp_path / "x.wav")
    long = tmp_path / "long.wav"
    sf.write(str(long), tone(130.0), SR, subtype="PCM_16")
    with pytest.raises(RenderFailed, match="not take it again"):
        engine.generate(RenderRequest(prompt="p", duration_seconds=130, mode="retake", source_audio=long), None,
                        tmp_path / "x.wav")


def test_the_mlx_worker_runs_a_fix_under_the_driver_and_a_take_through_the_script(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(WK, "_run_forwarding_stop", lambda cmd, cwd: seen.append(cmd) or type("D", (), {"returncode": 0})())
    base = {"prompt": "p", "seconds": 30.0, "seed": 1, "steps": 50, "cfg_scale": 7.0, "out": str(tmp_path / "o.wav"),
            "model": "medium-base"}
    WK.render_mlx({**base, "mode": "repaint", "source_audio": "s.wav", "noise_level": 0.6,
                   "regions": [[8.0, 10.0], [20.0, 21.0]]})
    WK.render_mlx({**base, "mode": "retake", "source_audio": "s.wav", "noise_level": 0.3, "regions": []})
    WK.render_mlx({**base, "mode": "text2music"})
    WK.render_mlx({**base, "model": "medium"})
    fix, take, fresh, distilled = seen
    assert all(c[1].endswith("render_driver_mlx.py") and c[2].endswith("sa3_mlx.py") for c in seen)
    assert fix[fix.index("--khaos-regions") + 1] == "8.0,10.0;20.0,21.0"
    assert fix[fix.index("--inpaint-range") + 1] == "8.0,21.0", "the script's own separator, over the parts' span"
    assert fix[fix.index("--init-noise-level") + 1] == "0.6"
    assert "--inpaint-range" not in take and take[take.index("--init-noise-level") + 1] == "0.3"
    assert "--init-audio" not in fresh
    assert all("--khaos-base" in c for c in (fix, take, fresh)), "an adapter renders on the base it trained on"
    assert "--khaos-base" not in distilled


class _NumpyMx:
    """Enough of mlx.core, in numpy, to drive the driver's wrappers."""
    float32 = np.float32
    array = staticmethod(np.asarray)
    ones = staticmethod(lambda shape, dtype=None: np.ones(shape, dtype=dtype))
    eval = staticmethod(lambda *a: None)
    concatenate = staticmethod(lambda xs, axis=0: np.concatenate(xs, axis=axis))
    broadcast_to = staticmethod(np.broadcast_to)


def test_the_mlx_driver_masks_every_part_and_mixes_the_source_into_the_start():
    args, regions, base = M.split_khaos_flags(["--prompt", "p", "--khaos-regions", "0.0,0.2786;0.9288,1.0", "--khaos-base",
                                                "--seconds", "1"])
    assert args == ["--prompt", "p", "--seconds", "1"] and regions == [(0.0, 0.2786), (0.9288, 1.0)] and base
    assert M.latent_ranges(regions, 11) == [(0, 3), (10, 11)]
    got, state = {}, {}
    sampler = M.masked_sampler(lambda fn, x, s, **kw: got.update(x=x, **kw), regions, 0.6, state, _NumpyMx, np)
    init, noise = np.ones((1, 256, 11), np.float32), np.zeros((1, 256, 11), np.float32)
    sampler(None, noise, np.array([0.6, 0.0]), seed=2, paste_back=(init, np.ones((1, 1, 11))))
    keep = got["paste_back"][1][0, 0]
    assert keep.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0], "every part regenerates, the rest is pasted back"
    assert np.allclose(got["x"], 0.4), "below pure noise the source is mixed into the start"
    assert state["local_add_cond"].shape == (1, 11, 257)
    seen = {}
    call = M.masked_call(lambda self, x, t, c, g, local_add_cond=None: seen.update(lac=local_add_cond), state, _NumpyMx)
    call(None, np.zeros((2, 256, 11)), None, None, None, local_add_cond=np.zeros((2, 11, 257), np.float16))
    assert seen["lac"].shape == (2, 11, 257) and seen["lac"].dtype == np.float16 and seen["lac"][:, 3, 0].tolist() == [1, 1]


def test_a_rework_block_must_name_every_strength_the_rooms_offer():
    doc = json.loads(resources.files("khaos_attribution.engines.sa3").joinpath("engine.json").read_text())
    validate_descriptor(doc)
    del doc["rework"]["repaint_noise"]["aggressive"]
    with pytest.raises(DescriptorError, match="aggressive"):
        validate_descriptor(doc)


def test_a_short_chunk_the_worker_renders_long_is_trimmed_to_its_plan(engine, tmp_path):
    short = tmp_path / "short.wav"
    sf.write(str(short), tone(2.0), SR, subtype="PCM_16")
    out = engine.generate(RenderRequest(prompt="p", duration_seconds=3.0, mode="extend", source_audio=short), None,
                          tmp_path / "x.wav")
    assert engine.requests()[0]["seconds"] == 5, "the worker is held to the engine's floor"
    assert sf.info(str(out.audio_path)).frames == 3 * SR, "what comes back is cut to what was asked"


def test_a_piece_of_the_full_window_can_be_taken_again_and_fixed_at_both_ends(engine, tmp_path):
    full = tmp_path / "full.wav"
    sf.write(str(full), tone(120.0), SR, subtype="PCM_16")
    engine.generate(RenderRequest(prompt="p", duration_seconds=120, mode="retake", source_audio=full), None,
                    tmp_path / "t.wav")
    out = engine.generate(RenderRequest(prompt="p", duration_seconds=120, mode="repaint", source_audio=full,
                                        repaint_regions=[[0.0, 5.0], [115.0, 120.0]]), None, tmp_path / "f.wav")
    assert [r["seconds"] for r in engine.requests()] == [120, 120], "the worker is held at the bound, not past it"
    assert sf.info(str(out.audio_path)).frames == 120 * SR and out.params["window"] == [0.0, 120.0]


def test_the_worker_reads_a_16_bit_copy_and_the_output_keeps_the_sources_own_format(engine, tmp_path):
    deep = tmp_path / "deep.wav"
    sf.write(str(deep), tone(30.0), SR, subtype="PCM_24")
    out = engine.generate(RenderRequest(prompt="p", duration_seconds=30, mode="repaint", source_audio=deep,
                                        repaint_start=8.0, repaint_end=10.0), None, tmp_path / "f.wav")
    (w,) = engine.requests()
    assert w["source_audio"] != str(deep.resolve()), "both runtimes read 16-bit WAV natively"
    assert sf.info(str(out.audio_path)).subtype == "PCM_24"
    got, _ = sf.read(str(out.audio_path), dtype="int32", always_2d=True)
    orig, _ = sf.read(str(deep), dtype="int32", always_2d=True)
    weights = RW.weights([(int(round(s * SR)), int(round(e * SR))) for s, e in out.params["regions"]], len(orig), 2205)
    assert np.array_equal(got[weights == 0], orig[weights == 0])


def test_a_head_extend_stops_at_the_length_the_room_asked_for(engine, source, tmp_path):
    out = engine.generate(RenderRequest(prompt="p", duration_seconds=35, mode="extend", source_audio=source,
                                        extend_head=10.0), None, tmp_path / "h.wav")
    assert sf.info(str(out.audio_path)).frames == 35 * SR and out.params["added_seconds"] == 5


def test_the_noise_floor_cannot_go_below_what_the_mlx_script_accepts():
    doc = json.loads(resources.files("khaos_attribution.engines.sa3").joinpath("engine.json").read_text())
    doc["rework"]["noise_floor"] = 0.001
    with pytest.raises(DescriptorError, match="noise_floor|0.001"):
        validate_descriptor(doc)


def test_the_mlx_driver_renders_on_the_base_checkpoint_with_euler_and_keeps_the_scripts_hooks():
    """The distilled file the script names is answered by the base one; the base is integrated x += dt * v."""
    asked = []
    ensure, present = M.base_checkpoint(lambda rel: asked.append(rel) or rel, lambda rel: rel,
                                        "models/mlx/dit_medium_f16.npz", "models/mlx/dit_medium-base_f16.npz")
    assert ensure("models/mlx/dit_medium_f16.npz") == "models/mlx/dit_medium-base_f16.npz"
    assert present("models/mlx/dit_medium_f16.npz") == "models/mlx/dit_medium-base_f16.npz"
    assert ensure("models/mlx/t5gemma_f16.npz") == "models/mlx/t5gemma_f16.npz", "nothing else is redirected"
    sample = M.euler_sampler(_NumpyMx)
    steps, seen = [], []
    x = sample(lambda x, t: np.full_like(x, 2.0), np.zeros((1, 2, 4), np.float32), np.array([1.0, 0.5, 0.0], np.float32),
               on_step=lambda i, n: steps.append((i, n)), before_step=seen.append,
               paste_back=(np.full((1, 2, 4), 9.0, np.float32), np.array([[[1, 0, 0, 1]]], np.float32)))
    assert np.allclose(x[0, 0], [9, -2, -2, 9]), "two Euler steps of dt -0.5 at v 2, then the kept frames pasted back"
    assert steps == [(1, 2), (2, 2)] and seen == [0, 1]
