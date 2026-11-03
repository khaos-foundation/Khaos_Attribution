"""The SA3 worker's TID path on the CPU: its hooks give autograd's LoRA gradients, its factor pass, inversion and
per-draw kernels equal a brute-force Kronecker reference on a small parametrized model, every clip of a draw shares
its noise and time, and the guards and refusals fire before any work."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from khaos_attribution import tid as T
from khaos_attribution.engines.sa3 import measure as M
from khaos_attribution.engines.sa3 import measure_worker as W
from khaos_attribution.gradient import GradientUnavailable


def _torch():
    return pytest.importorskip("torch")


def _adapt(torch, layer, *, rank=3, alpha=6.0, seed=0, adapter_type="lora"):
    """Stability's plain-LoRA parametrization, W + scaling·strength·(B @ A), on ``layer``'s frozen weight."""
    from torch.nn.utils import parametrize

    class Lora(torch.nn.Module):
        def __init__(self, fan_in, fan_out):
            super().__init__()
            gen = torch.Generator().manual_seed(seed)
            self.scaling = alpha / rank
            self.adapter_type = adapter_type
            self.register_buffer("lora_strength", torch.tensor(1.0, dtype=torch.float64))
            self.lora_A = torch.nn.Parameter(torch.randn(rank, fan_in, generator=gen, dtype=torch.float64) * 0.5)
            self.lora_B = torch.nn.Parameter(torch.randn(fan_out, rank, generator=gen, dtype=torch.float64) * 0.5)

        def forward(self, W0):
            return W0 + (self.scaling * self.lora_strength * (self.lora_B @ self.lora_A)).view(W0.shape)

    layer.double()
    for p in layer.parameters():
        p.requires_grad_(False)
    parametrize.register_parametrization(layer, "weight", Lora(layer.weight[0].numel(), layer.weight.shape[0]), unsafe=True)
    return layer


def _toy(torch, *, rank=2, alpha=4.0, twice=False):
    """A token model in float64 with LoRA on a 1-wide Conv1d and four Linears; ``zero`` reads a zero input and
    ``unused`` never runs, as the SA3 local-conditioning and conditioner layers do. It records each layer's
    input and output so the reference reads them without hooks."""
    nn = torch.nn
    torch.manual_seed(1)
    c, d, e = 3, 6, 5

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = _adapt(torch, nn.Conv1d(c, c, 1, bias=False), rank=rank, alpha=alpha, seed=11)
            self.inp = _adapt(torch, nn.Linear(c, d), rank=rank, alpha=alpha, seed=12)
            self.zero = _adapt(torch, nn.Linear(c, d), rank=rank, alpha=alpha, seed=13)
            self.mid = _adapt(torch, nn.Linear(d, e), rank=rank, alpha=alpha, seed=14)
            self.out = _adapt(torch, nn.Linear(e, c, bias=False), rank=rank, alpha=alpha, seed=15)
            self.unused = _adapt(torch, nn.Linear(c, c), rank=rank, alpha=alpha, seed=16)
            self.trace, self.kwargs = {}, {}

        def forward(self, x, t, cond=None, cfg_dropout_prob=0.0, padding_mask=None, **kwargs):
            self.kwargs = kwargs
            rec = self.trace = {}
            y = self.pre(x)
            rec["pre"] = (x, y)
            h = (y + x).transpose(1, 2)
            y_in = self.inp(h)
            if twice:
                y_in = y_in + self.inp(h)
            rec["inp"] = (h, y_in)
            z = torch.zeros_like(h)
            y_zero = self.zero(z)
            rec["zero"] = (z, y_zero)
            h = torch.tanh(y_in + y_zero + cond["emb"] + t.to(h.dtype).view(1, 1, 1))
            y_mid = self.mid(h)
            rec["mid"] = (h, y_mid)
            y_out = self.out(torch.tanh(y_mid))
            rec["out"] = (torch.tanh(y_mid), y_out)
            return y_out.transpose(1, 2)

    model = Toy()
    names = sorted(n for n, _p in model.named_parameters() if n.endswith(("lora_A", "lora_B")))
    params = [dict(model.named_parameters())[n] for n in names]

    class Wrapped:
        def set_lora_strength(self, value):
            for m in model.modules():
                if hasattr(m, "lora_strength"):
                    m.lora_strength.fill_(float(value))

    return model, Wrapped(), names, params


def _clips(torch, lengths, seed):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for n in lengths:
        latents = torch.randn(1, 3, n, generator=gen, dtype=torch.float64)
        mask = torch.ones(1, n, dtype=torch.bool)
        mask[:, -1] = False
        out.append((latents, {"emb": torch.randn(1, 1, 6, generator=gen, dtype=torch.float64)}, mask, None))
    return out


def _tokens_ref(t, conv):
    return t[0] if conv else t[0].T


def test_the_hooks_lora_factor_matrices_give_autogradss_gradients():
    torch = _torch()
    for layer, x in ((torch.nn.Linear(7, 5), lambda g: torch.randn(2, 9, 7, generator=g, dtype=torch.float64)),
                     (torch.nn.Conv1d(6, 4, 1, bias=False), lambda g: torch.randn(2, 6, 9, generator=g, dtype=torch.float64))):
        _adapt(torch, layer, rank=3, alpha=6.0, seed=2)
        lora = layer.parametrizations.weight[0]
        entries = W._lora_modules(layer, ["parametrizations.weight.0.lora_A"])
        assert [e[0] for e in entries] == [""] and W._lora_scale(lora) == 2.0
        seen = {}

        def sink(key, lo, X, G_out):
            a_A, b_A, a_B, b_B = T.lora_factor_matrices(X, G_out, lo.lora_A.detach(), lo.lora_B.detach(), W._lora_scale(lo))
            seen["A"], seen["B"] = b_A @ a_A.T, b_B @ a_B.T

        gen = torch.Generator().manual_seed(9)
        inp = x(gen)
        with W._TidHooks(entries, sink):
            y = layer(inp)
            weights = torch.randn(y.shape, generator=gen, dtype=torch.float64)
            gA, gB = torch.autograd.grad((y * weights).sum(), [lora.lora_A, lora.lora_B])
        assert torch.allclose(seen["A"], gA, rtol=0, atol=1e-12), type(layer).__name__
        assert torch.allclose(seen["B"], gB, rtol=0, atol=1e-12), type(layer).__name__


def _reference(torch, model, wrapped, names, params, train, queries, grid, seed, target, live):
    """TID by brute force: per-sample gradients from autograd, factor matrices from the recorded activations, each
    tensor's preconditioner as the explicit damped Kronecker product, solved directly."""
    by_name = dict(zip(names, params))
    layers = sorted({n.rpartition(".parametrizations.weight.0.")[0] for n in live})

    def sample(clip, m, t):
        latents, cond, mask, _ = clip
        noise = torch.from_numpy(T.draw_noise(seed, m, tuple(latents.shape))).to(latents.dtype)
        base = None
        if target == "vdelta":
            wrapped.set_lora_strength(0.0)
            with torch.no_grad():
                base = W._forward(model, torch, latents, cond, mask, noise, t, checkpointing=False)
            wrapped.set_lora_strength(1.0)
        out = W._forward(model, torch, latents, cond, mask, noise, t, checkpointing=False)
        where = mask.unsqueeze(1)
        loss = W._signed_mean(torch, out, where) if target == "vmean" else W._squared(torch, out - base, where)
        ys = [model.trace[k][1] for k in layers]
        lora_params = [by_name[f"{k}.parametrizations.weight.0.lora_{w}"] for k in layers for w in "AB"]
        grads = torch.autograd.grad(loss, ys + lora_params)
        result = {}
        for j, key in enumerate(layers):
            conv = key == "pre"
            X, G_out = _tokens_ref(model.trace[key][0].detach(), conv), _tokens_ref(grads[j], conv)
            lora = dict(model.named_modules())[key].parametrizations.weight[0]
            s = lora.scaling * float(lora.lora_strength)
            A, B = lora.lora_A.detach(), lora.lora_B.detach()
            mats = {"U_A": X, "V_A": s * B.T @ G_out, "U_B": s * A @ X, "V_B": G_out}
            gA, gB = grads[len(layers) + 2 * j], grads[len(layers) + 2 * j + 1]
            result[key] = {"mats": mats,
                           "A": gA / (mats["U_A"].norm() * mats["V_A"].norm()),
                           "B": gB / (mats["U_B"].norm() * mats["V_B"].norm())}
        return result

    factors, count = {}, 0
    for m, t in enumerate(grid):
        for clip in train:
            for key, r in sample(clip, m, t).items():
                f = factors.setdefault(key, {n: 0.0 for n in T.FACTORS})
                for n, mat in r["mats"].items():
                    u = mat / mat.norm()
                    f[n] = f[n] + u @ u.T
            count += 1
    eye = lambda n: torch.eye(n, dtype=torch.float64)  # noqa: E731
    damped = {key: {n: f[n] / count + 0.1 * float(torch.trace(f[n] / count)) / f[n].shape[0] * eye(f[n].shape[0])
                    for n in T.FACTORS} for key, f in factors.items()}
    K = np.zeros((len(grid), len(queries), len(train)))
    for m, t in enumerate(grid):
        rows = [sample(clip, m, t) for clip in train]
        for c, clip in enumerate(queries):
            q = sample(clip, m, t)
            for i, r in enumerate(rows):
                total = 0.0
                for key in layers:
                    for w in "AB":
                        full = torch.kron(damped[key][f"U_{w}"], damped[key][f"V_{w}"])
                        total += float(q[key][w].T.reshape(-1) @ torch.linalg.solve(full, r[key][w].T.reshape(-1)))
                K[m, c, i] = total
    return K


@pytest.mark.parametrize("target", ["vmean", "vdelta"])
def test_the_worker_path_equals_a_brute_force_kronecker_reference(tmp_path, target):
    torch = _torch()
    model, wrapped, names, params = _toy(torch)
    train, queries = _clips(torch, [7, 9, 5, 8], seed=21), _clips(torch, [6, 9], seed=22)
    grid, seed = [0.85, 0.4, 0.15], 1000
    entries = W._lora_modules(model, names)
    acc = T.FactorAccumulator()
    W._tid_factor_pass(model, torch, train, grid, seed, target, entries, acc, wrapped=wrapped, params=params)
    modules = acc.save(tmp_path)
    live, dropped = W._tid_live(names, modules)
    assert sorted(dropped) == sorted(n for n in names if n.startswith(("zero.", "unused."))), \
        "a layer on a zero input and a layer never run carry no gradient"
    (tmp_path / T.FACTORS_RECORD).write_text(json.dumps({"tid_version": T.TID_VERSION, "modules": modules}))
    summary = W.run({"command": "tid_invert", "factors_dir": str(tmp_path), "alpha": 0.1})
    assert summary["factors"] == 16 and summary["max_trace_deviation"] < 1e-12
    layout, bounds, order, n_live = W._tid_layout(names, params, live)
    assert order == live and n_live == sum(int(p.numel()) for n, p in zip(names, params) if n in set(live))
    inverses = W._tid_inverses(tmp_path, list(layout), pin=False)
    sink = W._RowSink(layout)
    kept = [e for e in entries if e[0] in layout]
    K = np.zeros((len(grid), len(queries), len(train)))
    for m, t in enumerate(grid):
        k_m, l_m = W._tid_draw(model, torch, train, queries, m, t, seed, target, kept, layout, inverses, sink,
                               n_live=n_live, wrapped=wrapped, params=params, train_chunk=3, bounds=bounds)
        assert torch.allclose(l_m.sum(dim=1), k_m, rtol=0, atol=1e-12), "the tensors' shares sum to the kernel"
        K[m] = k_m.numpy()
    ref = _reference(torch, model, wrapped, names, params, train, queries, grid, seed, target, live)
    assert np.abs(K - ref).max() <= 1e-9 * np.abs(ref).max()
    assert np.allclose(T.tau_from_kernels(K), (ref ** 2).mean(axis=0), rtol=1e-8, atol=0)
    check = sink.check
    # Two chunks of training rows, and every query clip is read once against each.
    assert check["samples"] == len(grid) * (len(train) + 2 * len(queries)) and check["missing"] == 0
    assert check["min_cosine"] > 1 - 1e-12 and check["max_norm_ratio_dev"] < 1e-10 and check["dropped_max_abs"] == 0.0
    assert sink.zero == 0


def test_every_clip_of_a_draw_is_noised_at_that_draws_time_with_its_shared_noise(monkeypatch):
    torch = _torch()
    model, wrapped, names, params = _toy(torch)
    train, queries = _clips(torch, [7, 9], seed=23), _clips(torch, [5], seed=24)
    grid, seed = [0.7, 0.3], 1000
    seen = []
    forward = W._forward

    def spy(model_, torch_, latents, cond, mask, noise, t, **kw):
        seen.append((t, noise.detach().clone()))
        return forward(model_, torch_, latents, cond, mask, noise, t, **kw)

    monkeypatch.setattr(W, "_forward", spy)
    entries = W._lora_modules(model, names)
    W._tid_factor_pass(model, torch, train + queries, grid, seed, "vmean", entries, T.FactorAccumulator(),
                       wrapped=wrapped, params=params)
    assert len(seen) == len(grid) * 3
    for t, noise in seen:
        m = grid.index(t)
        want = torch.from_numpy(T.draw_noise(seed, m, tuple(noise.shape))).to(noise.dtype)
        assert torch.equal(noise, want), "draw m pairs grid[m] with draw_noise(seed, m) for every clip"
    src = inspect.getsource(W)
    for fn in ("_tid_factor_pass", "_tid_draw"):
        body = src[src.index(f"def {fn}("):]
        body = body[:body.index("\ndef ", 1)]
        assert "_tid_noise(torch, noise_seed, m, clip[0])" in body and "pair_noise_seed" not in body
    assert "for m, t in enumerate(grid):" in src[src.index("def _tid_kernels("):]


def test_the_forward_turns_checkpointing_off_only_when_asked():
    torch = _torch()
    model, _wrapped, _names, _params = _toy(torch)
    latents, cond, mask, _ = _clips(torch, [4], seed=25)[0]
    W._forward(model, torch, latents, cond, mask, torch.zeros_like(latents), 0.5, checkpointing=False)
    assert model.kwargs == {"use_checkpointing": False}
    W._forward(model, torch, latents, cond, mask, torch.zeros_like(latents), 0.5)
    assert model.kwargs == {}, "every other measure keeps the engine's own checkpointing"
    src = inspect.getsource(W._tid_backward)
    assert src.count("checkpointing=False") == 2


def test_a_layer_run_twice_in_one_forward_is_refused():
    torch = _torch()
    model, wrapped, names, params = _toy(torch, twice=True)
    entries = W._lora_modules(model, names)
    with pytest.raises(RuntimeError, match="ran twice in one forward"):
        W._tid_factor_pass(model, torch, _clips(torch, [5], seed=26), [0.5], 1000, "vmean", entries,
                           T.FactorAccumulator(), wrapped=wrapped, params=params)


def test_compiled_blocks_and_live_dropout_are_refused(monkeypatch):
    entries = [("layer", None, SimpleNamespace(training=False), "linear")]
    W._tid_guard(entries)
    monkeypatch.setenv("ENABLE_TORCH_COMPILE", "1")
    with pytest.raises(RuntimeError, match="ENABLE_TORCH_COMPILE"):
        W._tid_guard(entries)
    monkeypatch.delenv("ENABLE_TORCH_COMPILE")
    with pytest.raises(RuntimeError, match="training mode"):
        W._tid_guard([("layer", None, SimpleNamespace(training=True), "linear")])


def test_a_weight_decomposed_or_stacked_adapter_is_refused():
    torch = _torch()
    layer = _adapt(torch, torch.nn.Linear(4, 3), adapter_type="dora-rows")
    with pytest.raises(RuntimeError, match="plain LoRA"):
        W._lora_modules(layer, ["parametrizations.weight.0.lora_A"])
    stacked = _adapt(torch, torch.nn.Linear(4, 3))
    _adapt(torch, stacked)
    with pytest.raises(RuntimeError, match="2 weight parametrizations"):
        W._lora_modules(stacked, ["parametrizations.weight.0.lora_A"])
    with pytest.raises(RuntimeError, match="no LoRA layer"):
        W._lora_modules(_adapt(torch, torch.nn.Linear(4, 3)), ["parametrizations.weight.0.lora_A"], pattern="nothing")


def test_tid_requests_are_refused_on_mlx_and_when_malformed():
    from khaos_attribution.engines.sa3 import measure_worker_mlx as MLX
    for command in ("tid_factors", "tid_invert", "tid_kernels"):
        with pytest.raises(RuntimeError, match="CUDA worker only"):
            MLX.run({"command": command})
    handle = M.Handle(root=Path("."), python=Path("python"), adapter=Path("a.safetensors"), runtime="mlx")
    with pytest.raises(GradientUnavailable, match="CUDA worker only"):
        M.tid_factors(handle, [], grid=[0.5], noise_seed=0, target="vmean", out_dir="f")
    with pytest.raises(GradientUnavailable, match="CUDA worker only"):
        M.tid_invert(handle, "f")
    with pytest.raises(GradientUnavailable, match="CUDA worker only"):
        M.tid_kernels(handle, [], [], [], grid=[0.5], noise_seed=0, target="vmean", factors_dir="f", out="k.npy")
    cuda = M.Handle(root=Path("."), python=Path("python"), adapter=Path("a.safetensors"))
    with pytest.raises(GradientUnavailable, match="query groups"):
        M.tid_kernels(cuda, [], ["q.wav"], [], grid=[0.5], noise_seed=0, target="vmean", factors_dir="f", out="k.npy")
    with pytest.raises(GradientUnavailable, match=".npy"):
        M.tid_kernels(cuda, [], [], [], grid=[0.5], noise_seed=0, target="vmean", factors_dir="f", out="k.bin")
    with pytest.raises(RuntimeError, match="strictly inside"):
        W._tid_settings({"grid": [1.0], "noise_seed": 0})
    with pytest.raises(RuntimeError, match="target"):
        W._tid_settings({"grid": [0.5], "noise_seed": 0, "target": "vnorm"})


def test_the_host_functions_take_the_workshops_keywords_and_check_the_replys_keys():
    def params(fn):
        return [(p.name, p.kind.name, p.default) for p in inspect.signature(fn).parameters.values()]
    kw = "KEYWORD_ONLY"
    assert params(M.tid_kernels)[4:] == [("grid", kw, inspect._empty), ("noise_seed", kw, inspect._empty),
                                         ("target", kw, inspect._empty), ("factors_dir", kw, inspect._empty),
                                         ("out", kw, inspect._empty), ("train_chunk", kw, 1),
                                         ("keep_layer_kernels", kw, False), ("resume", kw, True), ("on_row", kw, None)]
    assert [p[0] for p in params(M.tid_kernels)[:4]] == ["handle", "train_files", "query_files", "query_groups"]
    assert params(M.tid_factors)[2:] == [("grid", kw, inspect._empty), ("noise_seed", kw, inspect._empty),
                                         ("target", kw, inspect._empty), ("out_dir", kw, inspect._empty),
                                         ("factor_groups", kw, 1), ("modules", kw, None), ("on_row", kw, None)]
    assert params(M.tid_invert)[2:] == [("alpha", kw, 0.1)]
    for key in ("kernels", "draws_done", "grid", "live_tensors", "dropped_tensors", "normaliser_zero", "seconds"):
        assert key in M._REPLY_KEYS["tid_kernels"]


def test_a_resumed_kernels_file_keeps_its_draws_and_refuses_other_settings(tmp_path):
    out = tmp_path / "kernels.npy"
    config = {"grid": [0.5, 0.25], "noise_seed": 1000}
    K, done = W._tid_store(out, (2, 3, 4), config, resume=True)
    assert done == [] and K.shape == (2, 3, 4) and K.dtype == np.float32
    K[1] = 7.0
    K.flush()
    W._tid_mark(out, config, [1])
    del K
    again, done = W._tid_store(out, (2, 3, 4), config, resume=True)
    assert done == [1] and float(again[1, 0, 0]) == 7.0, "a resumed run skips the draws already written"
    del again
    with pytest.raises(RuntimeError, match="other settings"):
        W._tid_store(out, (2, 3, 4), {**config, "noise_seed": 1}, resume=True)
    with pytest.raises(RuntimeError, match="not"):
        W._tid_store(out, (2, 3, 5), config, resume=True)
    fresh, done = W._tid_store(out, (2, 3, 4), config, resume=False)
    assert done == [] and not np.asarray(fresh).any()


def test_the_factor_pass_can_hold_half_the_transformer_at_a_time():
    keys = [f"model.model.transformer.layers.{i}.ff.ff.2" for i in range(4)] + ["model.model.to_cond_embed.0"]
    assert W._factor_groups(keys, 1) == [keys]
    first, second = W._factor_groups(keys, 2)
    assert first == keys[:2] and second == keys[2:]
    with pytest.raises(RuntimeError):
        W._factor_groups(keys, 3)
    with pytest.raises(RuntimeError, match="name none"):
        W._factor_groups(["model.model.to_cond_embed.0"], 2)


@pytest.fixture
def _restore_determinism(monkeypatch):
    import os
    import warnings
    torch = _torch()
    monkeypatch.setattr(W, "_DETERMINISM", {})
    shown, had = warnings.showwarning, os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    yield torch
    torch.use_deterministic_algorithms(False)
    warnings.showwarning = shown
    if had is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = had


def test_flash_attention_runs_its_deterministic_backward_and_is_wrapped_once(_restore_determinism):
    """The engine's flash attention accumulates its backward in no fixed order unless asked; TID asks, once."""
    calls = []

    def fake(q, k, v, *, deterministic=False):
        calls.append(deterministic)
        return q
    module = SimpleNamespace(flash_attn_func=fake, flash_attn_kvpacked_func=None, flash_attn_varlen_func=fake)
    assert W._deterministic_attention(module) == ["flash_attn_func", "flash_attn_varlen_func"]
    first = module.flash_attn_func
    assert W._deterministic_attention(module) == ["flash_attn_func", "flash_attn_varlen_func"]
    assert module.flash_attn_func is first, "a second call must not wrap the wrapper"
    module.flash_attn_func(1, 2, 3)
    module.flash_attn_varlen_func(1, 2, 3)
    assert calls == [True, True]
    assert W._deterministic_attention(None) == []


def test_a_tid_pass_is_deterministic_before_the_model_loads(_restore_determinism, monkeypatch):
    """cuBLAS reads its workspace setting when CUDA first runs, so TID sets it, and torch's flag, before loading."""
    import os
    import warnings
    torch = _restore_determinism
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    seen = {}

    class Loaded(Exception):
        pass

    def load(req):
        seen.update(env=os.environ.get("CUBLAS_WORKSPACE_CONFIG"), flag=torch.are_deterministic_algorithms_enabled())
        raise Loaded
    monkeypatch.setattr(W, "_load", load)
    monkeypatch.setattr(warnings, "showwarning", lambda *a, **k: None)
    with pytest.raises(Loaded):
        W._tid_setup({})
    assert seen == {"env": W.CUBLAS_WORKSPACE, "flag": True}
    warnings.showwarning("an op does not have a deterministic implementation", UserWarning, "x.py", 1)
    warnings.showwarning("an unrelated warning", UserWarning, "x.py", 2)
    assert W._DETERMINISM["warned"] == ["an op does not have a deterministic implementation"]


def test_the_d_trak_commands_do_not_take_the_tid_settings(_restore_determinism, monkeypatch):
    """Only TID's commands change the kernels' determinism; the index, feature and loss commands run as before."""
    torch = _restore_determinism

    class Loaded(Exception):
        pass

    def load(req):
        raise Loaded
    monkeypatch.setattr(W, "_load", load)
    with pytest.raises(Loaded):
        W.run({"command": "index"})
    assert not torch.are_deterministic_algorithms_enabled() and W._DETERMINISM == {}
