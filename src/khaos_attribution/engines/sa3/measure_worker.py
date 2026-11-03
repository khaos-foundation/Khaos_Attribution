"""Gradient features on Stable Audio 3, inside the engine's environment: the request as JSON on stdin, the reply as
one JSON line last on stdout. ``index`` writes one projected adapter gradient per clip (the D-TRAK feature, the
contract's count-sketch), ``feature`` the mean feature of an output's clips, ``loss`` the output's rectified-flow
loss with the adapter at full strength and with it off, ``latent_stats`` a research latent transform measured
over the training clips; ``tid_factors``, ``tid_invert`` and ``tid_kernels`` build the TID research reading. The
forward is the trainer's with the randomness fixed, as the held-out worker takes it; the adapter's parameters are
the ones Stability's trainer trains."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(req: dict):
    import torch  # noqa: PLC0415
    from khaos_attribution.engines.sa3.loading import load_measured  # noqa: PLC0415
    from stable_audio_3.models.lora.utils import get_lora_params  # noqa: PLC0415

    from khaos_attribution import gradient as G  # noqa: PLC0415
    from stable_audio_3.models.lora.utils import load_lora_checkpoint, name_is_lora  # noqa: PLC0415

    wrapped = load_measured(req["model"])
    if not req.get("adapter"):
        raise RuntimeError("a feature needs the adapter; none was named")
    wrapped.load_lora([str(req["adapter"])])
    wrapped.set_lora_strength(1.0)
    model = wrapped.model
    model.eval()
    named = [(n, p) for n, p in model.named_parameters() if name_is_lora(n)]
    if not named or len(named) != len(list(get_lora_params(model))):
        raise RuntimeError("the adapter's parameters could not be named")
    state, _cfg = load_lora_checkpoint(str(req["adapter"]))
    # The adapter file's key order, shared with the MLX worker, so one adapter projects alike on either runtime.
    order = G.checkpoint_order([n for n, _p in named], list(state))
    params = [named[i][1] for i in order]
    if any(p.dtype != torch.float32 for p in params):
        raise RuntimeError("the adapter's parameters are not float32, the precision they trained at")
    for p in params:
        p.requires_grad_(True)
    names = [named[i][0] for i in order]
    # Any magnitude tensor means weight decomposition: at strength 0 such an adapter is still not the base model.
    decomposed = any(str(k).endswith(".magnitude") for k in state)
    return wrapped, model, params, torch, sorted(state), names, decomposed


def _encode(model, torch, wav: Path, device):
    """(latents, sample count, valid frame count) of one clip through the engine's autoencoder."""
    import soundfile as sf  # noqa: PLC0415
    sr = int(model.sample_rate)
    ratio = int(model.pretransform.downsampling_ratio)
    audio, file_sr = sf.read(str(wav), dtype="float32", always_2d=True)
    if file_sr != sr:
        raise RuntimeError(f"{wav.name} is at {file_sr} Hz, the model wants {sr}")
    x = torch.from_numpy(audio.T).unsqueeze(0).to(device)
    n = x.shape[-1]
    padded = ((n + ratio - 1) // ratio) * ratio
    if padded != n:
        x = torch.nn.functional.pad(x, (0, padded - n))
    with torch.no_grad():
        with torch.autocast(device_type=device if isinstance(device, str) else device.type, enabled=str(device).startswith("cuda")):
            latents = model.pretransform.encode(x)
    latents = latents.float().detach()
    return latents, n, max(1, min(int(round(n / ratio)), latents.shape[-1]))


def _clip(model, torch, wav: Path, device, pair_seed: int):
    """One clip as the trainer sees it: latents, conditioning with an empty inpaint mask, the padding mask, the
    seeded noise."""
    sr = int(model.sample_rate)
    latents, n, valid = _encode(model, torch, wav, device)
    caption = wav.with_suffix(".txt").read_text(encoding="utf-8").strip() if wav.with_suffix(".txt").exists() else ""
    # The conditioning is taken once and detached: the feature is the DiT adapter's gradient, and a conditioner
    # graph reused across timesteps would be walked twice.
    from khaos_attribution.engines.sa3.loading import measure_autocast  # noqa: PLC0415
    with torch.no_grad(), measure_autocast(device):
        cond = model.conditioner([{"prompt": caption, "seconds_total": n / sr}], device)
    cond = _detached(cond, torch)
    cond["inpaint_mask"] = [torch.zeros(1, 1, latents.shape[-1], dtype=torch.bool, device=latents.device)]
    cond["inpaint_masked_input"] = [torch.zeros_like(latents)]
    mask = torch.zeros(1, latents.shape[-1], dtype=torch.bool, device=latents.device)
    mask[:, :valid] = True
    from khaos_attribution import gradient as G  # noqa: PLC0415
    noise = torch.from_numpy(G.pair_noise(pair_seed, latents.shape)).to(latents.device)
    return latents, cond, mask, noise


def _detached(value, torch):
    if isinstance(value, dict):
        return {k: _detached(v, torch) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detached(v, torch) for v in value)
    if hasattr(value, "detach"):
        return value.detach()
    return value


def _forward(model, torch, latents, cond, mask, noise, t: float, *, checkpointing: bool = True):
    from khaos_attribution.engines.sa3.loading import measure_autocast  # noqa: PLC0415
    tt = torch.full((1,), t, device=latents.device, dtype=torch.float32)
    noised = latents * (1.0 - t) + noise * t
    # Checkpointed blocks run their forward twice, so a forward hook would see each layer twice.
    extra = {} if checkpointing else {"use_checkpointing": False}
    with measure_autocast(latents.device):
        return model(noised, tt, cond=cond, cfg_dropout_prob=0.0, padding_mask=mask, **extra).float()


# vmean and dmean take the gradient of the signed mean of the prediction (or of the adapter's delta): the scalar
# output function a leave-one-out kernel differentiates, where the norm targets differentiate a magnitude.
TARGETS = ("vnorm", "loss", "vdelta", "vmean", "dmean")


def _squared(torch, diff, where, lmap=None, pool: int = 1):
    """Mean square of ``diff`` over the valid frames, after the latent map over channels and the mean over blocks
    of ``pool`` frames when either is given."""
    if lmap is not None:
        if lmap.shape[1] != diff.shape[1]:
            raise RuntimeError(f"the latent transform was measured on {lmap.shape[1]} channels; this latent has {diff.shape[1]}")
        diff = torch.einsum("kc,bct->bkt", lmap, diff)
    if pool > 1:
        # The valid frames are a prefix; a tail shorter than one block is dropped unless it is all there is.
        valid = int(where[0, 0].sum())
        blocks = max(1, valid // pool)
        size = pool if valid >= pool else valid
        return diff[:, :, :blocks * size].reshape(diff.shape[0], diff.shape[1], blocks, size).mean(-1).pow(2).mean()
    return diff[where.expand_as(diff)].pow(2).mean()


def _signed_mean(torch, diff, where, lmap=None, pool: int = 1):
    """Mean of ``diff`` over the valid frames, after the same latent map and pooling as the square."""
    if lmap is not None:
        diff = torch.einsum("kc,bct->bkt", lmap, diff)
    if pool > 1:
        valid = int(where[0, 0].sum())
        blocks = max(1, valid // pool)
        size = pool if valid >= pool else valid
        return diff[:, :, :blocks * size].reshape(diff.shape[0], diff.shape[1], blocks, size).mean(-1).mean()
    return diff[where.expand_as(diff)].mean()


def _gradient(model, torch, params, clip, timesteps, target: str, *, wrapped=None, timestep_norm: bool = False,
              keep=None, lmap=None, pool: int = 1, noises=None):
    """(flat gradient, residual) of one clip: the mean adapter gradient over the timesteps, flat, float32, on the
    CPU, and the clip's training residual over the same timesteps. ``noises`` gives each timestep its own noise in
    place of the clip's; ``keep`` zeroes the tensors a filter drops."""
    from khaos_attribution import gradient as G  # noqa: PLC0415
    if target not in TARGETS:
        raise RuntimeError(f"target must be one of {TARGETS}, not {target!r}")
    latents, cond, mask, clip_noise = clip
    where = mask.unsqueeze(1)
    flats, residuals = [], []
    for i, t in enumerate(timesteps):
        noise = clip_noise if noises is None else noises[i]
        base = None
        if target in ("vdelta", "dmean"):
            wrapped.set_lora_strength(0.0)
            with torch.no_grad():
                base = _forward(model, torch, latents, cond, mask, noise, float(t))
            wrapped.set_lora_strength(1.0)
        with torch.enable_grad():
            out = _forward(model, torch, latents, cond, mask, noise, float(t))
            if target == "vnorm":
                loss = _squared(torch, out, where, lmap, pool)
            elif target == "vdelta":
                loss = _squared(torch, out - base, where, lmap, pool)
            elif target == "vmean":
                loss = _signed_mean(torch, out, where, lmap, pool)
            elif target == "dmean":
                loss = _signed_mean(torch, out - base, where, lmap, pool)
            else:
                loss = _squared(torch, out - (noise - latents), where, lmap, pool)
        # The residual is the trainer's, prediction minus the flow target, whatever the output function.
        with torch.no_grad():
            residuals.append(float(_signed_mean(torch, out.detach() - (noise - latents), where)))
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        parts = [(g if g is not None else torch.zeros_like(p)).reshape(-1).detach().to("cpu", torch.float32)
                 for g, p in zip(grads, params)]
        if keep is not None:
            parts = [part if ok else torch.zeros_like(part) for part, ok in zip(parts, keep)]
        flats.append(torch.cat(parts).numpy())
    return G.combine_timesteps(flats, timestep_norm=timestep_norm), G.residual_over_timesteps(residuals)


def _shared_noises(torch, noise_seed: int, draw: int, timesteps, shape, device):
    """One noise per timestep, the same for every clip and output of the request, cut to this clip's frames."""
    from khaos_attribution import gradient as G  # noqa: PLC0415
    return [torch.from_numpy(G.shared_noise(noise_seed + draw, i, shape)).to(device) for i in range(len(timesteps))]


def _latent_stats(model, torch, files, device, kind: str, variance: float) -> dict:
    """Channel moments of the valid latent frames over ``files``, in float64, made into the transform."""
    import numpy as np  # noqa: PLC0415
    from khaos_attribution import gradient as G  # noqa: PLC0415
    n, total, outer = 0, None, None
    for wav in files:
        latents, _n, valid = _encode(model, torch, wav, device)
        x = latents[0, :, :valid].to("cpu", torch.float64).numpy()
        total = x.sum(axis=1) if total is None else total + x.sum(axis=1)
        outer = x @ x.T if outer is None else outer + x @ x.T
        n += x.shape[1]
    if total is None:
        raise RuntimeError("the latent transform needs at least one clip")
    return G.latent_transform_from_moments(kind, n, total, outer, variance=variance)


# The truth's loss is the trainer's, so a research latent transform never reaches it.
def _loss(model, torch, clip, timesteps) -> float:
    latents, cond, mask, noise = clip
    total = 0.0
    with torch.no_grad():
        for t in timesteps:
            out = _forward(model, torch, latents, cond, mask, noise, float(t))
            total += float(((out - (noise - latents)) ** 2)[mask.unsqueeze(1).expand_as(out)].mean())
    return total / float(len(timesteps))


def _base_dtype(model) -> str:
    """The precision the base model's weights are held at, as a result's name records it."""
    from khaos_attribution.gradient import dtype_tag  # noqa: PLC0415
    from stable_audio_3.models.lora.utils import name_is_lora  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415
    lora = {id(p) for n, p in model.named_parameters() if name_is_lora(n)}
    counts = Counter(str(p.dtype) for p in model.parameters() if id(p) not in lora)
    # The dtype most of the base's weights are held at; a few kept at full precision do not change the label.
    return dtype_tag(counts.most_common(1)[0][0]) if counts else "unknown"


def _peak_gb() -> float | None:
    """The worker's peak device memory, so the room can plan for it."""
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    except Exception:  # noqa: BLE001
        return None
    return None


# ── TID: Kronecker-factored score-discrepancy influence; the algebra is khaos_attribution.tid ──

def _lora_modules(model, names, pattern=None) -> list:
    """(key, module, parametrization, kind) of each LoRA layer in the adapter file's order, limited by ``pattern``;
    a stacked, weight-decomposed or other-shaped adapter is refused, the factor algebra being plain LoRA's."""
    import re  # noqa: PLC0415
    import torch  # noqa: PLC0415
    found = {}
    for key, module in model.named_modules():
        plist = getattr(getattr(module, "parametrizations", None), "weight", None)
        if plist is None:
            continue
        loras = [p for p in plist if hasattr(p, "lora_A") and hasattr(p, "lora_B")]
        if not loras:
            continue
        if len(plist) != 1:
            raise RuntimeError(f"{key} carries {len(plist)} weight parametrizations; TID reads one plain LoRA")
        lora = loras[0]
        if getattr(lora, "adapter_type", "lora") != "lora":
            raise RuntimeError(f"{key} is a {lora.adapter_type} adapter; TID's factors are derived for plain LoRA")
        if isinstance(module, torch.nn.Conv1d):
            if tuple(module.kernel_size) != (1,) or module.groups != 1:
                raise RuntimeError(f"{key} is a Conv1d of kernel {tuple(module.kernel_size)}; TID reads 1-wide convolutions only")
            kind = "conv1d"
        elif isinstance(module, torch.nn.Linear):
            kind = "linear"
        else:
            raise RuntimeError(f"{key} is a {type(module).__name__}; TID reads Linear and 1-wide Conv1d layers")
        found[key] = (module, lora, kind)
    rx = re.compile(pattern) if pattern else None
    order = []
    for name in names:
        key = name.rpartition(".parametrizations.weight.0.")[0]
        if key in found and key not in order and (rx is None or rx.search(key)):
            order.append(key)
    if not order:
        raise RuntimeError("no LoRA layer of this adapter matches" + (f" {pattern!r}" if pattern else ""))
    return [(key, *found[key]) for key in order]


def _tid_guard(entries) -> None:
    """Refuses a run whose hooks or gradients would not be TID's: compiled blocks, or a LoRA layer with live dropout."""
    import os  # noqa: PLC0415
    if os.environ.get("ENABLE_TORCH_COMPILE") == "1":
        raise RuntimeError("ENABLE_TORCH_COMPILE=1 compiles the transformer blocks past the hooks TID reads; unset it")
    training = [key for key, _module, lora, _kind in entries if lora.training]
    if training:
        raise RuntimeError(f"{len(training)} LoRA layer(s) are in training mode, so their dropout is live, e.g. {training[0]}")


def _tokens(tensor, kind: str, dtype):
    """A layer's activations or output gradients as (features × tokens)."""
    t = tensor.detach().to(dtype)
    if kind == "conv1d":
        return t.transpose(0, 1).reshape(t.shape[1], -1)
    return t.reshape(-1, t.shape[-1]).transpose(0, 1)


def _lora_scale(lora) -> float:
    return float(lora.scaling) * float(lora.lora_strength)


class _TidHooks:
    """Forward hooks on the LoRA layers that hand ``sink(key, lora, X, G)`` each layer's input tokens (in × T) and
    output gradients (out × T) when the backward reaches it; a layer run twice in one forward is refused."""

    def __init__(self, entries, sink):
        self.entries = list(entries)
        self.sink = sink
        self.handles: list = []
        self.fired: set = set()

    def __enter__(self) -> "_TidHooks":
        for key, module, lora, kind in self.entries:
            self.handles.append(module.register_forward_hook(self._hook(key, lora, kind)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def reset(self) -> None:
        self.fired = set()

    def _hook(self, key, lora, kind):
        def forward_hook(_module, args, out):
            import torch  # noqa: PLC0415
            if not torch.is_grad_enabled() or not getattr(out, "requires_grad", False):
                return None
            if key in self.fired:
                raise RuntimeError(f"{key} ran twice in one forward; TID needs each layer once, with checkpointing off")
            self.fired.add(key)
            dtype = lora.lora_A.dtype
            x = _tokens(args[0], kind, dtype)

            def on_grad(grad):
                self.sink(key, lora, x, _tokens(grad, kind, dtype))

            out.register_hook(on_grad)
            return None
        return forward_hook


def _tid_backward(model, torch, clip, t: float, noise, target: str, hooks, *, wrapped=None, params=()):
    """One forward and backward of TID's target at (t, noise) under ``hooks``: autograd's gradients of ``params``.
    The adapter-delta target's base forward runs without gradient, so no hook sees it."""
    from khaos_attribution import tid as T  # noqa: PLC0415
    if target not in T.TARGETS:
        raise RuntimeError(f"TID's target is one of {T.TARGETS}, not {target!r}")
    latents, cond, mask, _ = clip
    where = mask.unsqueeze(1)
    base = None
    if target == "vdelta":
        wrapped.set_lora_strength(0.0)
        try:
            with torch.no_grad():
                base = _forward(model, torch, latents, cond, mask, noise, float(t), checkpointing=False)
        finally:
            wrapped.set_lora_strength(1.0)
    hooks.reset()
    with torch.enable_grad():
        out = _forward(model, torch, latents, cond, mask, noise, float(t), checkpointing=False)
        loss = _signed_mean(torch, out, where) if target == "vmean" else _squared(torch, out - base, where)
    return torch.autograd.grad(loss, list(params), allow_unused=True)


def _tid_noise(torch, noise_seed: int, m: int, latents):
    """Draw m's shared noise for one clip, on its device and in its dtype."""
    from khaos_attribution import tid as T  # noqa: PLC0415
    return torch.from_numpy(T.draw_noise(noise_seed, m, tuple(latents.shape))).to(latents.device, latents.dtype)


def _factor_groups(keys, groups: int) -> list:
    """The layers each factor pass holds: all at once, or the transformer's first half of layers and then the rest."""
    import re  # noqa: PLC0415
    if groups == 1:
        return [list(keys)]
    if groups != 2:
        raise RuntimeError(f"factor_groups is 1 or 2, not {groups!r}")
    depth = {k: int(m.group(1)) for k in keys if (m := re.search(r"\.layers\.(\d+)\.", k))}
    if not depth:
        raise RuntimeError("factor_groups=2 splits the transformer's layers, and these layers name none")
    half = (max(depth.values()) + 1) // 2
    first = [k for k in keys if depth.get(k, half) < half]
    return [first, [k for k in keys if k not in set(first)]]


def _tid_factor_pass(model, torch, clips, grid, noise_seed: int, target: str, entries, acc, *, wrapped=None,
                     params=(), on_draw=None) -> None:
    """Adds every (clip, draw) observation of the hooked layers to ``acc``: TID's curvature over the training clips."""
    def sink(key, lora, X, G_out):
        acc.add(key, X, G_out, lora.lora_A.detach().to(X.dtype), lora.lora_B.detach().to(X.dtype), _lora_scale(lora))

    with _TidHooks(entries, sink) as hooks:
        for m, t in enumerate(grid):
            for clip in clips:
                _tid_backward(model, torch, clip, t, _tid_noise(torch, noise_seed, m, clip[0]), target, hooks,
                              wrapped=wrapped, params=params)
            if on_draw is not None:
                on_draw(m)


def _tid_live(names, modules) -> tuple:
    """(live, dropped) tensor names in the adapter file's order: live where the factor pass saw a nonzero gradient."""
    seen = {m["key"]: m for m in modules}
    live, dropped = [], []
    for name in names:
        key, _, leaf = name.rpartition(".parametrizations.weight.0.")
        entry = seen.get(key)
        ok = entry is not None and entry["nonzero_A" if leaf == "lora_A" else "nonzero_B"] > 0
        (live if ok else dropped).append(name)
    return live, dropped


def _tid_layout(names, params, live) -> tuple:
    """Where each live tensor sits in a TID row, {layer: {"A"|"B": (offset, numel, shape, parameter index)}}, with
    every live tensor's (offset, numel) and name in the adapter file's order, and the row's length."""
    wanted = set(live)
    layout, bounds, order, off = {}, [], [], 0
    for i, (name, p) in enumerate(zip(names, params)):
        if name not in wanted:
            continue
        key, sep, leaf = name.rpartition(".parametrizations.weight.0.")
        if not sep or leaf not in ("lora_A", "lora_B"):
            raise RuntimeError(f"{name} is not a plain LoRA tensor")
        n = int(p.numel())
        layout.setdefault(key, {})["A" if leaf == "lora_A" else "B"] = (off, n, tuple(p.shape), i)
        bounds.append((off, n))
        order.append(name)
        off += n
    missing = wanted - set(order)
    if missing:
        raise RuntimeError(f"{len(missing)} live tensor(s) of the factors are not this adapter's, e.g. {sorted(missing)[0]}")
    return layout, bounds, order, off


class _RowSink:
    """Writes each hooked layer's normalised LoRA gradients into the current row at their places, counts the zero
    normalisers, and checks the row against autograd's gradients."""

    def __init__(self, layout):
        self.layout = layout
        self.row = None
        self.scales: dict = {}
        self.zero = 0
        self.live_index = {part[3] for parts in layout.values() for part in parts.values()}
        self.check = {"samples": 0, "min_cosine": None, "max_norm_ratio_dev": None, "dropped_max_abs": 0.0, "missing": 0}

    def begin(self, row) -> None:
        row.zero_()
        self.row = row
        self.scales = {}

    def __call__(self, key, lora, X, G_out) -> None:
        parts = self.layout.get(key)
        if parts is None:
            return
        from khaos_attribution import tid as T  # noqa: PLC0415
        gA, gB, norms = T.normalised_gradients(X, G_out, lora.lora_A.detach().to(X.dtype),
                                               lora.lora_B.detach().to(X.dtype), _lora_scale(lora))
        for which, grad, scale in (("A", gA, norms[0] * norms[1]), ("B", gB, norms[2] * norms[3])):
            if which not in parts:
                continue
            off, n, _shape, _i = parts[which]
            self.row[off:off + n] = grad.reshape(-1).to(self.row.dtype)
            self.scales[(key, which)] = scale
            self.zero += int(scale == 0.0)

    def compare(self, grads) -> None:
        """Autograd's gradients against the row just written: each live tensor's cosine and length ratio, and the
        largest gradient on a tensor left out, which must be zero."""
        import torch  # noqa: PLC0415
        cosines, devs = [], []
        for key, parts in self.layout.items():
            for which, (off, n, _shape, i) in parts.items():
                g = grads[i]
                scale = self.scales.get((key, which), 0.0)
                if g is None:
                    self.check["missing"] += 1
                    continue
                if scale == 0.0:
                    continue
                h = self.row[off:off + n]
                gv = g.detach().reshape(-1).to(h.dtype)
                gn, hn = gv.norm(), h.norm()
                cosines.append((gv @ h) / (gn * hn))
                devs.append((gn / (hn * scale) - 1.0).abs())
        dropped = [g.detach().abs().max() for i, g in enumerate(grads) if i not in self.live_index and g is not None and g.numel()]
        c = self.check
        c["samples"] += 1
        if cosines:
            low, high = float(torch.stack(cosines).min()), float(torch.stack(devs).max())
            c["min_cosine"] = low if c["min_cosine"] is None else min(c["min_cosine"], low)
            c["max_norm_ratio_dev"] = high if c["max_norm_ratio_dev"] is None else max(c["max_norm_ratio_dev"], high)
        if dropped:
            c["dropped_max_abs"] = max(c["dropped_max_abs"], float(torch.stack(dropped).max()))


def _precondition_rows(P, layout, inverses) -> None:
    """Replaces each live tensor's block of every row of P by Ṽ⁻¹·Γ̂·Ũ⁻¹, one layer's inverses on the device at a time."""
    from khaos_attribution import tid as T  # noqa: PLC0415
    rows = P.shape[0]
    for key, parts in layout.items():
        for which, (off, n, shape, _i) in parts.items():
            inv_V = inverses[key][f"V_{which}"].to(P.device, P.dtype, non_blocking=True)
            inv_U = inverses[key][f"U_{which}"].to(P.device, P.dtype, non_blocking=True)
            block = P[:, off:off + n].reshape(rows, *shape)
            P[:, off:off + n] = T.precondition(block, inv_V, inv_U).reshape(rows, n)


def _tid_draw(model, torch, train, queries, m: int, t: float, noise_seed: int, target: str, entries, layout, inverses,
              sink, *, n_live: int, wrapped=None, params=(), train_chunk: int = 1, bounds=None) -> tuple:
    """Draw m's kernels as (queries × rows) float64 on the CPU: the training rows' normalised gradients preconditioned,
    then each query clip's against them; with ``bounds``, each tensor's share as (queries × tensors × rows)."""
    from khaos_attribution import tid as T  # noqa: PLC0415
    dtype = entries[0][2].lora_A.dtype
    device = train[0][0].device
    K = torch.zeros(len(queries), len(train), dtype=torch.float64)
    layers = torch.zeros(len(queries), len(bounds), len(train), dtype=torch.float64) if bounds else None
    size = -(-len(train) // int(train_chunk))
    with _TidHooks(entries, sink) as hooks:
        for start in range(0, len(train), size):
            stop = min(start + size, len(train))
            P = torch.zeros(stop - start, n_live, dtype=dtype, device=device)
            for j, clip in enumerate(train[start:stop]):
                sink.begin(P[j])
                sink.compare(_tid_backward(model, torch, clip, t, _tid_noise(torch, noise_seed, m, clip[0]), target,
                                           hooks, wrapped=wrapped, params=params))
            _precondition_rows(P, layout, inverses)
            q = torch.zeros(n_live, dtype=dtype, device=device)
            for c, clip in enumerate(queries):
                sink.begin(q)
                sink.compare(_tid_backward(model, torch, clip, t, _tid_noise(torch, noise_seed, m, clip[0]), target,
                                           hooks, wrapped=wrapped, params=params))
                K[c, start:stop] = (P @ q).to("cpu", torch.float64)
                if layers is not None:
                    layers[c, :, start:stop] = T.layer_kernels(P, q, bounds).to("cpu", torch.float64)
            del P
    return K, layers


def _write_json(path: Path, record: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def _progress(row: dict) -> None:
    from khaos_attribution.engines.sa3.measure import ROW_PREFIX  # noqa: PLC0415
    sys.stdout.write(ROW_PREFIX + json.dumps(row) + "\n")
    sys.stdout.flush()


def _tid_done_path(out: Path) -> Path:
    return out.with_suffix(".done.json")


def _tid_store(out: Path, shape, config: dict, resume: bool) -> tuple:
    """(kernels, draws done): the file reopened for a resumed run whose settings match, else a new one."""
    import numpy as np  # noqa: PLC0415
    done_path = _tid_done_path(out)
    if resume and out.is_file() and done_path.is_file():
        record = json.loads(done_path.read_text(encoding="utf-8"))
        if record.get("config") != config:
            raise RuntimeError(f"the kernels at {out} were written under other settings; move them aside or run without resume")
        K = np.lib.format.open_memmap(str(out), mode="r+")
        if K.shape != tuple(shape) or K.dtype != np.float32:
            raise RuntimeError(f"the kernels at {out} are {K.shape} {K.dtype}, not {tuple(shape)} float32")
        return K, [int(m) for m in record["draws_done"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    K = np.lib.format.open_memmap(str(out), mode="w+", dtype=np.float32, shape=tuple(shape))
    _tid_mark(out, config, [])
    return K, []


def _tid_mark(out: Path, config: dict, done) -> None:
    _write_json(_tid_done_path(out), {"config": config, "draws_done": sorted(int(m) for m in done)})


def _tid_settings(req: dict) -> tuple:
    """(grid, noise seed, target) of a TID request, refused unless every time lies strictly inside (0, 1)."""
    from khaos_attribution import tid as T  # noqa: PLC0415
    grid = [float(t) for t in req["grid"]]
    if not grid or any(not 0.0 < t < 1.0 for t in grid):
        raise RuntimeError("TID's grid is a non-empty list of times strictly inside (0, 1)")
    target = str(req.get("target") or "vmean")
    if target not in T.TARGETS:
        raise RuntimeError(f"TID's target is one of {T.TARGETS}, not {target!r}")
    return grid, int(req["noise_seed"]), target


FLASH_ATTENTION = ("flash_attn_func", "flash_attn_kvpacked_func", "flash_attn_varlen_func")
CUBLAS_WORKSPACE = ":4096:8"
_DETERMINISM: dict = {}


def _tid_deterministic() -> dict:
    """Torch's deterministic kernels and cuBLAS's fixed workspace for a TID pass, set before any CUDA work; an op with
    no deterministic form is warned about and named in ``warned`` rather than refused."""
    import os  # noqa: PLC0415
    import warnings  # noqa: PLC0415
    import torch  # noqa: PLC0415
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    warned = _DETERMINISM.setdefault("warned", [])
    shown = warnings.showwarning
    if not getattr(shown, "_khaos_collects", False):
        def collect(message, category, filename, lineno, file=None, line=None):
            text = str(message)[:300]
            if "determinis" in text and text not in warned:
                warned.append(text)
            shown(message, category, filename, lineno, file, line)
        collect._khaos_collects = True
        warnings.showwarning = collect
    _DETERMINISM.update({"torch": True, "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"]})
    return _DETERMINISM


def _deterministic_attention(module) -> list:
    """Flash attention's deterministic backward in the engine's transformer, whose default accumulates in no fixed
    order; returns the functions wrapped, once each however often it is called."""
    import functools  # noqa: PLC0415
    wrapped = []
    for name in FLASH_ATTENTION:
        fn = getattr(module, name, None) if module is not None else None
        if fn is None:
            continue
        if not getattr(fn, "_khaos_deterministic", False):
            fn = functools.partial(fn, deterministic=True)
            fn._khaos_deterministic = True
            setattr(module, name, fn)
        wrapped.append(name)
    return wrapped


def _tid_setup(req: dict, pattern=None) -> tuple:
    """The model with its adapter, the LoRA layers TID reads, and the guards every TID command needs."""
    _tid_deterministic()
    wrapped, model, params, torch, _keys, names, decomposed = _load(req)
    _DETERMINISM["flash_attention"] = _deterministic_attention(sys.modules.get("stable_audio_3.models.transformer"))
    if decomposed:
        raise RuntimeError("a weight-decomposed adapter is refused: TID's factors are derived for plain LoRA")
    entries = _lora_modules(model, names, pattern)
    _tid_guard(entries)
    # The factor and preconditioning products are float32 sums; TF32 would round their inputs to 10 bits.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return wrapped, model, params, torch, names, entries


def _tid_factors(req: dict) -> dict:
    """TID's curvature: every LoRA layer's normalised Kronecker factors over the training clips at every draw of the
    grid, written per layer under ``out_dir`` with the record ``factors.json``, which is the reply."""
    import time  # noqa: PLC0415
    from khaos_attribution import gradient as G  # noqa: PLC0415
    from khaos_attribution import tid as T  # noqa: PLC0415
    started = time.time()
    grid, noise_seed, target = _tid_settings(req)
    wrapped, model, params, torch, names, entries = _tid_setup(req, req.get("modules") or None)
    device = wrapped.device
    files = [Path(f) for f in req["files"]]
    out_dir = Path(req["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = [_clip(model, torch, wav, device, 0) for wav in files]
    groups = _factor_groups([e[0] for e in entries], int(req.get("factor_groups") or 1))
    modules = []
    for g, keys in enumerate(groups):
        acc = T.FactorAccumulator(keys, device=device)
        part = [e for e in entries if e[0] in set(keys)]
        _tid_factor_pass(model, torch, clips, grid, noise_seed, target, part, acc, wrapped=wrapped, params=params,
                         on_draw=lambda m, g=g: _progress({"phase": "factors", "group": g, "draw": m, "of": len(grid)}))
        modules += acc.save(out_dir)
        del acc
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    live, dropped = _tid_live(names, modules)
    record = {"tid_version": T.TID_VERSION, "target": target, "grid": grid, "noise_seed": noise_seed,
              "files": [str(f) for f in files], "observations": len(files) * len(grid), "factor_groups": len(groups),
              "modules_pattern": req.get("modules") or None, "normalisation": "factor matrices at unit Frobenius norm",
              "modules": modules, "live_tensors": len(live), "live_tensor_names": live, "dropped_tensors": dropped,
              "signature": G.parameter_signature(params), "compute_dtype": _base_dtype(model),
              "device": str(device).split(":")[0], "deterministic": _DETERMINISM,
              "seconds": round(time.time() - started, 1)}
    _write_json(out_dir / T.FACTORS_RECORD, record)
    return record


def _tid_invert(req: dict) -> dict:
    """Each live factor's damped inverse from a float64 eigendecomposition, written under ``inverses/`` with the
    spectra in ``spectra.npz`` and their summary in ``inverses.json``, which is the reply. No model is loaded."""
    import time  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from khaos_attribution import tid as T  # noqa: PLC0415
    started = time.time()
    factors_dir = Path(req["factors_dir"])
    alpha = float(req.get("alpha", T.RELATIVE_DAMPING))
    record = json.loads((factors_dir / T.FACTORS_RECORD).read_text(encoding="utf-8"))
    if record.get("tid_version") != T.TID_VERSION:
        raise RuntimeError(f"the factors under {factors_dir} are {record.get('tid_version')}, not {T.TID_VERSION}")
    (factors_dir / "inverses").mkdir(exist_ok=True)
    spectra, modules = {}, []
    for entry in record["modules"]:
        wanted = [n for n, count in (("V_A", entry["nonzero_A"]), ("U_A", entry["nonzero_A"]),
                                     ("U_B", entry["nonzero_B"]), ("V_B", entry["nonzero_B"])) if count]
        if not wanted:
            continue
        data = torch.load(str(factors_dir / entry["file"]), map_location="cpu")
        inverses, stats = {}, {}
        for name in wanted:
            F = data[name] / int(data["n"])
            inverse, lam = T.damped_inverse(F, alpha)
            inverses[name] = inverse
            spectra[f"{entry['key']}:{name}"] = lam.numpy()
            trace = float(lam.sum())
            stats[name] = {"dim": int(F.shape[0]), "trace": trace, "delta": alpha * trace / F.shape[0],
                           "min": float(lam.min()), "max": float(lam.max())}
        torch.save(inverses, str(factors_dir / "inverses" / entry["file"]))
        modules.append({"key": entry["key"], "file": entry["file"], "factors": stats})
    np.savez(str(factors_dir / "spectra.npz"), **spectra)
    stats = [s for m in modules for s in m["factors"].values()]
    summary = {"tid_version": T.TID_VERSION, "alpha": alpha, "inverses": str(factors_dir / "inverses"),
               "spectra": str(factors_dir / "spectra.npz"), "factors": len(stats),
               "min_eigen_ratio": min((s["min"] / s["max"] for s in stats if s["max"] > 0), default=None),
               "max_trace_deviation": max((abs(s["trace"] - 1.0) for s in stats), default=None),
               "modules": modules, "seconds": round(time.time() - started, 1)}
    _write_json(factors_dir / "inverses.json", summary)
    return summary


def _tid_inverses(factors_dir: Path, keys, pin: bool) -> dict:
    """Each layer's damped inverse factors from ``tid_invert``, on the CPU, pinned when a GPU will read them."""
    import torch  # noqa: PLC0415
    record = json.loads((factors_dir / "inverses.json").read_text(encoding="utf-8"))
    files = {m["key"]: m["file"] for m in record["modules"]}
    out = {}
    for key in keys:
        if key not in files:
            raise RuntimeError(f"no inverse factors for {key} under {factors_dir}; run tid_invert")
        data = torch.load(str(factors_dir / "inverses" / files[key]), map_location="cpu")
        out[key] = {n: (v.pin_memory() if pin else v) for n, v in data.items()}
    return out


def _tid_kernels(req: dict) -> dict:
    """TID's per-draw kernels between every query clip and every training clip, written to ``out`` as float32
    (draws × queries × rows) in grid, query and training order; a resumed run skips the draws already done."""
    import time  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from khaos_attribution import gradient as G  # noqa: PLC0415
    from khaos_attribution import tid as T  # noqa: PLC0415
    started = time.time()
    factors_dir = Path(req["factors_dir"])
    record = json.loads((factors_dir / T.FACTORS_RECORD).read_text(encoding="utf-8"))
    if record.get("tid_version") != T.TID_VERSION:
        raise RuntimeError(f"the factors under {factors_dir} are {record.get('tid_version')}, not {T.TID_VERSION}")
    grid, noise_seed, target = _tid_settings(req)
    if grid != [float(t) for t in record["grid"]] or noise_seed != int(record["noise_seed"]):
        raise RuntimeError("the factors were taken at another grid or noise seed; kernels are read at the curvature's draws")
    train_files = [Path(f) for f in req["train_files"]]
    query_files = [Path(f) for f in req["query_files"]]
    groups = list(req["query_groups"])
    if len(groups) != len(query_files):
        raise RuntimeError(f"{len(groups)} query groups for {len(query_files)} query files")
    train_chunk = int(req.get("train_chunk") or 1)
    if not 1 <= train_chunk <= max(1, len(train_files)):
        raise RuntimeError(f"train_chunk must lie between 1 and the {len(train_files)} training clips")
    wrapped, model, params, torch, names, entries = _tid_setup(req)
    if record.get("signature") and record["signature"] != G.parameter_signature(params):
        raise RuntimeError("the factors were taken on an adapter laid out differently; take them again")
    layout, bounds, live, n_live = _tid_layout(names, params, record["live_tensor_names"])
    entries = [e for e in entries if e[0] in layout]
    device = wrapped.device
    inverses = _tid_inverses(factors_dir, list(layout), pin=str(device).startswith("cuda"))
    keep_layers = bool(req.get("keep_layer_kernels", False))
    config = {"tid_version": T.TID_VERSION, "adapter": str(req["adapter"]), "factors_dir": str(factors_dir),
              "grid": grid, "noise_seed": noise_seed, "target": target, "train_files": [str(f) for f in train_files],
              "query_files": [str(f) for f in query_files], "keep_layer_kernels": keep_layers}
    out = Path(req["out"])
    K, done = _tid_store(out, (len(grid), len(query_files), len(train_files)), config, bool(req.get("resume", True)))
    layer_path = out.with_suffix(".layers.npy") if keep_layers else None
    L = None
    if layer_path is not None:
        shape = (len(grid), len(query_files), len(bounds), len(train_files))
        L = np.lib.format.open_memmap(str(layer_path), mode="r+" if done else "w+",
                                      **({} if done else {"dtype": np.float32, "shape": shape}))
    train = [_clip(model, torch, wav, device, 0) for wav in train_files]
    queries = [_clip(model, torch, wav, device, 0) for wav in query_files]
    sink = _RowSink(layout)
    for m, t in enumerate(grid):
        if m in done:
            continue
        k_m, l_m = _tid_draw(model, torch, train, queries, m, t, noise_seed, target, entries, layout, inverses, sink,
                             n_live=n_live, wrapped=wrapped, params=params, train_chunk=train_chunk,
                             bounds=bounds if L is not None else None)
        K[m] = k_m.numpy().astype(np.float32)
        K.flush()
        if L is not None:
            L[m] = l_m.numpy().astype(np.float32)
            L.flush()
        done.append(m)
        _tid_mark(out, config, done)
        _progress({"phase": "kernels", "draw": m, "t": t, "done": len(done), "of": len(grid)})
    del K, L
    return {"tid_version": T.TID_VERSION, "kernels": str(out), "shape": [len(grid), len(query_files), len(train_files)],
            "layer_kernels": str(layer_path) if layer_path else None, "layer_names": live if keep_layers else None,
            "draws_done": sorted(done), "grid": grid, "noise_seed": noise_seed, "target": target,
            "rows": [{"row": i, "pair_id": f.stem, "track_id": G.track_of_pair(f.stem)} for i, f in enumerate(train_files)],
            "clips": [{"clip": c, "file": str(f), "group": g} for c, (f, g) in enumerate(zip(query_files, groups))],
            "live_tensors": len(live), "live_parameters": n_live, "dropped_tensors": [n for n in names if n not in set(live)],
            "normaliser_zero": sink.zero, "autograd_check": sink.check, "factors_dir": str(factors_dir),
            "train_chunk": train_chunk, "signature": G.parameter_signature(params), "compute_dtype": _base_dtype(model),
            "device": str(device).split(":")[0], "deterministic": _DETERMINISM,
            "seconds": round(time.time() - started, 1)}


def main() -> int:
    req = json.loads(sys.stdin.read())
    try:
        reply = run(req)
        reply["peak_gb"] = _peak_gb()
    except Exception as exc:  # noqa: BLE001 — the reply carries it
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("\n" + json.dumps(reply) + "\n")
    sys.stdout.flush()
    return 0 if not reply.get("error") else 1


TID_COMMANDS = {"tid_factors": _tid_factors, "tid_invert": _tid_invert, "tid_kernels": _tid_kernels}


def run(req: dict) -> dict:
    import numpy as np  # noqa: PLC0415
    from khaos_attribution.engines.sa3.measure import ROW_PREFIX  # noqa: PLC0415
    from khaos_attribution import gradient as G  # noqa: PLC0415

    # tid_invert loads no model, so the TID commands are routed before the adapter is.
    if req.get("command") in TID_COMMANDS:
        return TID_COMMANDS[req["command"]](req)
    wrapped, model, params, torch, keys, names, decomposed = _load(req)
    device = wrapped.device
    timesteps = [float(t) for t in req["timesteps"]]
    noise_seed = int(req["noise_seed"])
    target = str(req.get("target") or "vnorm")
    if target == "vdelta" and decomposed:
        raise RuntimeError("a weight-decomposed adapter cannot be switched off for the adapter-delta target")
    timestep_norm = bool(req.get("timestep_norm", False))
    param_filter = req.get("param_filter") or None
    keep = G.param_filter_mask(names, param_filter) if param_filter else None
    transform = req.get("latent_transform") or None
    lmap = torch.from_numpy(G.latent_map(transform)).to(device) if transform else None
    pool = G.research_int(req.get("latent_pool"), "latent_pool")
    draws = G.research_int(req.get("noise_draws"), "noise_draws")
    noise_shared = bool(req.get("noise_shared", False))
    groups_pattern = req.get("param_groups") or None
    groups_out = req.get("groups_out") or None
    if groups_out and not groups_pattern:
        raise RuntimeError("groups_out needs param_groups, the pattern that names the blocks")
    labels = G.param_groups(names, groups_pattern) if groups_pattern else None
    files = [Path(f) for f in req["files"]]
    n_params = int(sum(int(p.numel()) for p in params))
    signature = G.parameter_signature(params)
    command = req["command"]
    filtered = {"latent_transform": (transform or {}).get("kind"), "param_filter": param_filter, "latent_pool": pool,
                "noise_draws": draws, "noise_shared": noise_shared, "param_groups": groups_pattern, "tensors_kept": int(sum(keep)) if keep else len(params),
                "parameters_kept": int(sum(int(p.numel()) for p, ok in zip(params, keep) if ok)) if keep else n_params}
    if command == "latent_stats":
        return {"transform": _latent_stats(model, torch, files, device, str(req["latent_kind"]),
                                           float(req.get("latent_variance", 0.9)))}
    if command == "parameters":
        return {"n_params": n_params, "n_tensors": len(params), "signature": signature, "device": str(device).split(":")[0],
                "compute_dtype": _base_dtype(model), "names": names}
    if command in ("index", "feature"):
        projection = (G.make_projection(n_params, int(req["projection"]["dim"]), int(req["projection"]["seed"]))
                      if command == "index" else G.projection_from_meta(req["index_meta"]))
        if projection.n_params != n_params:
            raise RuntimeError(f"the index was built on {projection.n_params} adapter parameters; this adapter has {n_params}")
        if command == "index" and labels and not groups_out:
            raise RuntimeError("an index with param_groups writes its block sketches; name groups_out")
        sketch = G.group_sketch(projection, [int(p.numel()) for p in params], labels) if labels and groups_out else None
        # The index's blocks are written row by row, so a long index never holds them all in memory.
        group_rows = (np.lib.format.open_memmap(str(groups_out), mode="w+", dtype=np.float32,
                                                shape=(len(files), len(sketch.labels), projection.dim))
                      if sketch is not None and command == "index" else None)
        group_feats = []
        rows, vectors, raw = [], [], []
        for i, wav in enumerate(files):
            # Draw 0 is the index's own noise, so one draw reads exactly as an index built without the option.
            flats, residuals = [], []
            for d in range(draws):
                clip = _clip(model, torch, wav, device, G.pair_noise_seed(noise_seed + d, wav.name))
                noises = _shared_noises(torch, noise_seed, d, timesteps, clip[0].shape, device) if noise_shared else None
                flat_d, residual_d = _gradient(model, torch, params, clip, timesteps, target, wrapped=wrapped,
                                               timestep_norm=timestep_norm, keep=keep, lmap=lmap, pool=pool, noises=noises)
                flats.append(flat_d)
                residuals.append(residual_d)
            flat = flats[0] if draws == 1 else np.mean(np.stack(flats), axis=0).astype(np.float32)
            residual = float(np.mean(residuals))
            if req.get("keep_flat"):
                raw.append(flat)
            if sketch is not None:
                parts = G.project_groups(flat, sketch)
                if group_rows is not None:
                    group_rows[i] = parts
                else:
                    group_feats.append(parts)
            projected = G.project(flat, projection)
            if command == "index":
                norm = float(np.linalg.norm(projected))
                vectors.append(G.l2_normalise(projected))
                rows.append({"row": i, "pair_id": wav.stem, "track_id": G.track_of_pair(wav.stem), "gradient_norm": round(norm, 8),
                             "residual": round(residual, 8)})
                sys.stdout.write(ROW_PREFIX + json.dumps(rows[-1]) + "\n")
                sys.stdout.flush()
            else:
                vectors.append(projected)
            del clip
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        extra = {}
        if raw:
            np.save(str(req["keep_flat"]), np.stack(raw).astype(np.float32))
            extra = {"flat": str(req["keep_flat"]), "keys": keys}
        if group_rows is not None:
            group_rows.flush()
            del group_rows
        if sketch is not None and command == "feature":
            Path(groups_out).parent.mkdir(parents=True, exist_ok=True)
            np.save(str(groups_out), np.mean(np.stack(group_feats), axis=0).astype(np.float32))
        if sketch is not None:
            extra["group_labels"] = sketch.labels
            extra["groups"] = str(groups_out)
        if command == "index":
            out = Path(req["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out), np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, projection.dim), np.float32))
            return {**extra, **filtered, "rows": rows, "matrix": str(out), "n_params": n_params, "n_tensors": len(params),
                    "signature": signature, "device": str(device).split(":")[0], "compute_dtype": _base_dtype(model)}
        return {**extra, **filtered, "feature": np.mean(np.stack(vectors), axis=0).astype(float).tolist(), "n": len(vectors),
                "signature": signature}
    if command == "loss":
        # A weight-decomposed adapter at strength 0 is still not the base model: the reading is refused, never faked.
        if decomposed:
            raise RuntimeError("a weight-decomposed adapter cannot be switched off for a measurement")
        clips = [_clip(model, torch, wav, device, G.pair_noise_seed(noise_seed, wav.name)) for wav in files]
        wrapped.set_lora_strength(1.0)
        with_adapter = float(np.mean([_loss(model, torch, c, timesteps) for c in clips]))
        wrapped.set_lora_strength(0.0)
        without = float(np.mean([_loss(model, torch, c, timesteps) for c in clips]))
        wrapped.set_lora_strength(1.0)
        return {"with": with_adapter, "without": without, "n": len(clips)}
    raise RuntimeError(f"unknown command {command!r}")


if __name__ == "__main__":
    sys.exit(main())
