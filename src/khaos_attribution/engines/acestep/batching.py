"""Training seams installed from our code onto ACE-Step's classes.

``masked_loss`` normalises the padded-batch MSE over valid frames and
``length_buckets`` batches by latent length (both off by default; turn both
on before any batch_size > 1 run). ``latent_crop_sec``, ``stem_loss_weight``
and ``timestep_high_noise_bias`` add variety per step and are recorded as
what actually ran. Every seam is undone by the handle it returns, so a class
patch never outlives the run that asked for it. Single machine only: DDP
would replace the batch sampler and break the per-process masked mean.
"""

from __future__ import annotations

import contextvars
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import logging

from khaos_attribution.engines.acestep import DESCRIPTOR

log = logging.getLogger(__name__)

# The attention mask of the batch currently inside training_step (B, T),
# or None outside it. A context variable, so a stray F.mse_loss elsewhere
# in the same process never sees a stale mask.
_CURRENT_MASK: contextvars.ContextVar[Any] = contextvars.ContextVar("khaos_attention_mask", default=None)


# ── masked MSE ───────────────────────────────────────────────────────────────

def masked_mse(pred, target, mask):
    """Mean squared error over valid frames only: ``pred``/``target`` (B, T, D),
    ``mask`` (B, T) nonzero = valid; the masked squared error divided by
    (valid frames × D), so each sample weighs by its frame count. fp32 accumulation."""
    import torch  # noqa: PLC0415

    if pred.dim() != 3 or mask.dim() != 2 or mask.shape[0] != pred.shape[0] or mask.shape[1] != pred.shape[1]:
        raise ValueError(f"masked_mse expects pred (B,T,D) and mask (B,T); got {tuple(pred.shape)} / {tuple(mask.shape)}")
    m = (mask.to(device=pred.device) != 0).to(torch.float32).unsqueeze(-1)
    se = (pred.to(torch.float32) - target.to(torch.float32)) ** 2 * m
    denom = m.sum() * pred.shape[-1]
    return se.sum() / denom.clamp_min(1.0)


class _FunctionalProxy:
    """``torch.nn.functional`` with ``mse_loss`` mask-aware while a batch is
    in scope. Attribute access otherwise delegates to the real module."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def mse_loss(self, input, target, *args, **kwargs):  # noqa: A002 — torch's own names
        # Only the 2-arg mean form ACE-Step's training_step makes is intercepted;
        # a positional or non-mean reduction is a different contract and passes through.
        mask = _CURRENT_MASK.get()
        if (mask is not None and getattr(input, "dim", lambda: 0)() == 3 and mask.dim() == 2
                and tuple(mask.shape) == tuple(input.shape[:2]) and not args
                and kwargs.get("reduction", "mean") == "mean"):
            return masked_mse(input, target, mask)
        return self._real.mse_loss(input, target, *args, **kwargs)


def install_masked_loss(module_cls) -> bool:
    """Patch ``module_cls`` so ``training_step``'s module-global ``F.mse_loss``
    is normalised by the batch's ``attention_mask``. Idempotent. False when the
    class's module has no ``F`` to swap (the caller must not claim masked_loss)."""
    if getattr(module_cls, "_khaos_masked_loss", False):
        return True
    mod = sys.modules.get(module_cls.__module__)
    real_f = getattr(mod, "F", None)
    if mod is None or real_f is None or not hasattr(real_f, "mse_loss"):
        log.error("masked_loss: %s has no module-level F to patch — leaving the loss unmasked",
                  module_cls.__name__)
        return False
    # Claimed only where the step computes the loss it masks; a step whose code cannot be read is not claimed.
    names = getattr(getattr(module_cls.training_step, "__code__", None), "co_names", ())
    if "mse_loss" not in names:
        log.error("masked_loss: %s.training_step computes no mse_loss that can be seen — leaving the loss unmasked",
                  module_cls.__name__)
        return False
    setattr(mod, "F", _FunctionalProxy(real_f))
    orig = module_cls.training_step
    module_cls._khaos_masked_loss_undo = (mod, real_f, orig)

    def training_step(self, batch, *args, **kwargs):
        mask = batch.get("attention_mask") if isinstance(batch, dict) else None
        token = _CURRENT_MASK.set(mask)
        try:
            return orig(self, batch, *args, **kwargs)
        finally:
            _CURRENT_MASK.reset(token)

    training_step.__wrapped__ = orig  # type: ignore[attr-defined]
    module_cls.training_step = training_step
    module_cls._khaos_masked_loss = True
    log.info("masked_loss ON: %s.training_step now normalises the MSE over valid frames", module_cls.__name__)
    return True


def uninstall_masked_loss(module_cls) -> bool:
    """Undo :func:`install_masked_loss`: restore the module's real ``F`` and
    the original ``training_step``. Idempotent; False when nothing was on."""
    undo = getattr(module_cls, "_khaos_masked_loss_undo", None)
    if not getattr(module_cls, "_khaos_masked_loss", False) or undo is None:
        return False
    mod, real_f, orig = undo
    setattr(mod, "F", real_f)
    module_cls.training_step = orig
    module_cls._khaos_masked_loss = False
    module_cls._khaos_masked_loss_undo = None
    log.info("masked_loss OFF: %s restored", module_cls.__name__)
    return True


# ── length buckets ───────────────────────────────────────────────────────────

_EPOCH_STRIDE = 1_000_003   # Random(seed * K + epoch): one stream per (seed, epoch)


def length_bucket_batches(lengths: Sequence[int], batch_size: int, seed: int, epoch: int = 0) -> list[list[int]]:
    """Indices grouped by ascending length into batches of ``batch_size``
    (the last may be short); every index appears once. Deterministic per
    ``(seed, epoch)``: shuffled before the stable sort so equal lengths change
    partners each epoch, then the batch order is shuffled."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    rng = random.Random(int(seed) * _EPOCH_STRIDE + int(epoch))
    order = list(range(len(lengths)))
    rng.shuffle(order)
    order.sort(key=lambda i: int(lengths[i]))   # stable: ties keep the shuffled order
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


def padding_fraction(lengths: Sequence[int], batches: Sequence[Sequence[int]]) -> float:
    """Share of collated frames that are padding — what bucketing minimises."""
    total = padded = 0
    for b in batches:
        longest = max(int(lengths[i]) for i in b)
        for i in b:
            total += longest
            padded += longest - int(lengths[i])
    return padded / total if total else 0.0


class LengthBucketBatchSampler:
    """A ``batch_sampler`` for ``torch.utils.data.DataLoader`` over
    :func:`length_bucket_batches`. ``__iter__`` is the ONLY thing that advances
    the epoch (no ``set_epoch``), so nothing can move the counter behind the loader's back."""

    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int, start_epoch: int = 0):
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        # A resume passes the epochs already trained so the pairing continues.
        self.epoch = int(start_epoch)

    def __iter__(self) -> Iterator[list[int]]:
        batches = length_bucket_batches(self.lengths, self.batch_size, self.seed, self.epoch)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size) if self.lengths else 0


def dataset_lengths(dataset, cache_name: str = ".khaos_lengths.json") -> list[int]:
    """Latent length (frames) of every sample of an ACE-Step ``PreprocessedTensorDataset`` (or a ``Subset`` of
    one): :func:`dataset_facts`' frames, so both read the tensors once into one cache."""
    return [f["frames"] for f in dataset_facts(dataset, cache_name)]


def install_length_buckets(data_module_cls, seed: int, start_epoch: int = 0) -> bool:
    """Replace ``data_module_cls.train_dataloader`` with a length-bucketed
    loader: same dataset, collate and worker knobs, ``batch_sampler`` instead
    of shuffle. ``start_epoch`` continues the sampler's counter on a resume.
    Idempotent (re-installing updates the seed / start_epoch)."""
    if getattr(data_module_cls, "_khaos_length_buckets", False):
        data_module_cls._khaos_bucket_seed = int(seed)
        data_module_cls._khaos_bucket_start_epoch = int(start_epoch)
        return True
    mod = sys.modules.get(data_module_cls.__module__)
    collate = getattr(mod, "collate_preprocessed_batch", None)
    if collate is None:
        log.error("length_buckets: %s's module has no collate_preprocessed_batch — leaving the loader alone",
                  data_module_cls.__name__)
        return False
    data_module_cls._khaos_bucket_seed = int(seed)
    data_module_cls._khaos_bucket_start_epoch = int(start_epoch)
    orig_loader = data_module_cls.__dict__.get("train_dataloader")

    def train_dataloader(self):
        from torch.utils.data import DataLoader  # noqa: PLC0415
        lengths = dataset_lengths(self.train_dataset)
        sampler = LengthBucketBatchSampler(lengths, self.batch_size, data_module_cls._khaos_bucket_seed,
                                           start_epoch=data_module_cls._khaos_bucket_start_epoch)
        prefetch = None if self.num_workers == 0 else self.prefetch_factor
        persistent = False if self.num_workers == 0 else self.persistent_workers
        kwargs = dict(dataset=self.train_dataset, batch_sampler=sampler,
                      num_workers=self.num_workers, pin_memory=self.pin_memory,
                      collate_fn=collate, prefetch_factor=prefetch, persistent_workers=persistent)
        if getattr(self, "pin_memory_device", ""):
            kwargs["pin_memory_device"] = self.pin_memory_device
        first = length_bucket_batches(lengths, self.batch_size, data_module_cls._khaos_bucket_seed,
                                      data_module_cls._khaos_bucket_start_epoch)
        log.info("length_buckets ON: %d samples, batch %d, padding %.1f%% of frames (vs the longest-song worst case)",
                 len(lengths), self.batch_size, 100.0 * padding_fraction(lengths, first))
        return DataLoader(**kwargs)

    train_dataloader.__wrapped__ = orig_loader  # type: ignore[attr-defined]
    data_module_cls.train_dataloader = train_dataloader
    data_module_cls._khaos_length_buckets = True
    data_module_cls._khaos_length_buckets_orig = orig_loader
    return True


def uninstall_length_buckets(data_module_cls) -> bool:
    """Undo :func:`install_length_buckets`: put the class's own
    ``train_dataloader`` back (or drop ours if it inherited one).
    Idempotent; False when nothing was on."""
    if not getattr(data_module_cls, "_khaos_length_buckets", False):
        return False
    orig = getattr(data_module_cls, "_khaos_length_buckets_orig", None)
    if orig is not None:
        data_module_cls.train_dataloader = orig
    else:
        try:
            del data_module_cls.train_dataloader
        except AttributeError:
            pass
    data_module_cls._khaos_length_buckets = False
    data_module_cls._khaos_length_buckets_orig = None
    log.info("length_buckets OFF: %s.train_dataloader restored", data_module_cls.__name__)
    return True


# ── latent crops (variety per step) ─────────────────────────────────────────
# A run at batch 1 shows the model the same latent, whole, every epoch. This
# seam takes a FRESH random window from every INSTRUMENTAL sample each time it
# is served (sung pairs are never cropped: lyric conditioning is per file, not
# per frame). The crop is cut in the step hook on the main process, so loader
# workers hold no state and the manifest counts what the model actually saw.

# ACE-Step's VAE (AutoencoderOobleck, 48 kHz, 1920 samples a frame) puts target_latents at the descriptor's latent
# frame rate; every per-frame companion shares the axis.
LATENT_FPS = DESCRIPTOR.latent_fps
INSTRUMENTAL_LYRICS = "[Instrumental]"
# The keys that share target_latents' time axis; named explicitly rather than
# "every tensor whose first dim is T", or a caption of T tokens would be sliced.
PER_FRAME_KEYS = ("target_latents", "attention_mask", "context_latents")
# A sample is cropped only when it is at least this much LONGER than the
# window (a window that can move by two frames is not variety); the Training tab's plan applies the same margin.
CROP_MIN_EXTRA_SEC = 1.0


def crop_min_frames(window_frames: int) -> int:
    """The shortest valid length that gets cropped: window + the margin."""
    return int(window_frames) + int(round(CROP_MIN_EXTRA_SEC * LATENT_FPS))
CROP_META_KEY = "khaos_latent_crop"


def latent_crop_frames(window_sec: float) -> int:
    """The window in latent frames; 0 when cropping is off."""
    sec = float(window_sec or 0.0)
    return int(round(sec * LATENT_FPS)) if sec > 0 else 0


def is_instrumental_sample(metadata: Any) -> bool:
    """A pair with no lyrics — the preprocessor's ``is_instrumental`` or the
    ``[Instrumental]`` lyric string the pair builder writes (either alone
    is the same statement; a sample missing both is treated as sung)."""
    if not isinstance(metadata, dict):
        return False
    return bool(metadata.get("is_instrumental")) or metadata.get("lyrics") == INSTRUMENTAL_LYRICS


def crop_sample(sample: dict, window_frames: int, rng: random.Random) -> dict:
    """``sample`` with a random ``window_frames`` window cut from its latent
    and every per-frame companion, when it is instrumental and longer than
    the window; otherwise the sample itself, untouched. The window is drawn
    inside the VALID frames (attention_mask nonzero) so a padded tail can
    never be all a crop sees. The cropped sample's metadata carries
    :data:`CROP_META_KEY` = {start, frames, full_frames}."""
    window_frames = int(window_frames)
    meta = sample.get("metadata")
    latents = sample.get("target_latents")
    if window_frames <= 0 or latents is None or not is_instrumental_sample(meta):
        return sample
    full = int(latents.shape[0])
    mask = sample.get("attention_mask")
    valid = int((mask != 0).sum().item()) if mask is not None else full
    if valid < crop_min_frames(window_frames):
        return sample
    start = rng.randint(0, valid - window_frames)
    out = dict(sample)
    for key in PER_FRAME_KEYS:
        t = sample.get(key)
        if t is None:
            continue
        if int(t.shape[0]) != full:
            raise ValueError(f"latent_crop: {key} has {int(t.shape[0])} frames but target_latents has {full}"
                             " — the per-frame tensors disagree and a crop cannot be consistent")
        out[key] = t[start:start + window_frames]
    out["metadata"] = {**meta, CROP_META_KEY: {"start": start, "frames": window_frames, "full_frames": full}}
    return out


def dataset_facts(dataset, cache_name: str = ".khaos_lengths.json") -> list[dict]:
    """``{"frames", "instrumental"}`` for every sample of an ACE-Step
    ``PreprocessedTensorDataset`` (or a ``Subset`` of one), read from the
    tensor files once and cached beside them keyed by name + mtime. A cache
    row from before ``instrumental`` was stored is re-read, not trusted."""
    import torch  # noqa: PLC0415

    base = dataset
    indices = None
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):   # torch Subset
        base, indices = dataset.dataset, list(dataset.indices)
    paths = [Path(p) for p in getattr(base, "valid_paths", [])]
    if not paths:
        raise ValueError("the dataset exposes no valid_paths (not a PreprocessedTensorDataset)")
    cache_path = paths[0].parent / cache_name
    cache: dict[str, list] = {}
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    facts: list[dict] = []
    dirty = False
    for p in paths:
        key, mtime = p.name, p.stat().st_mtime
        hit = cache.get(key)
        if hit and len(hit) >= 3 and abs(float(hit[0]) - mtime) < 1e-6:
            facts.append({"frames": int(hit[1]), "instrumental": bool(hit[2])})
            continue
        data = torch.load(str(p), map_location="cpu", weights_only=True)
        row = {"frames": int(data["target_latents"].shape[0]),
               "instrumental": is_instrumental_sample(data.get("metadata"))}
        cache[key] = [mtime, row["frames"], row["instrumental"]]
        facts.append(row)
        dirty = True
    if dirty:
        try:
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass
    if indices is not None:
        facts = [facts[i] for i in indices]
    return facts


def crop_eligible_count(facts: Sequence[dict], window_frames: int) -> int:
    """How many samples :func:`crop_sample` would actually crop."""
    if window_frames <= 0:
        return 0
    return sum(1 for f in facts if f["instrumental"] and int(f["frames"]) >= crop_min_frames(window_frames))


class LatentCropState:
    """What the crop seam is doing this run (the manifest's ``latent_crops``
    record), kept on the main process. ``eligible`` is set when the loader is
    built; ``served`` is advanced by the step hook, where the crop is cut (a
    counter in the dataset would be copied into every loader worker)."""

    def __init__(self, window_sec: float, seed: int):
        self.window_sec = float(window_sec)
        self.window_frames = latent_crop_frames(window_sec)
        self.seed = int(seed)
        self.eligible: int | None = None
        self.served = 0
        self.serves: dict[str, int] = {}     # per sample: how many times it has been served

    def rng_for(self, identity: str) -> random.Random:
        """A fresh stream per (run seed, sample, serve): the n-th time a
        sample is served it gets its n-th window, whatever process built
        the batch — the counter lives here, on the main process."""
        serve = self.serves.get(identity, 0)
        self.serves[identity] = serve + 1
        return random.Random(f"{self.seed}:{identity}:{serve}")

    def record(self) -> dict:
        # The text the model is conditioned on is the pair's, encoded before training;
        # a crop cannot re-encode it, so its duration line still states the whole pair's.
        return {"window_sec": self.window_sec, "applied_to": "instrumental_pairs",
                "eligible_samples": self.eligible, "crops_served": int(self.served),
                "conditioning": "the pair's own text, duration line included; a crop is not re-captioned"}


def install_latent_crops(data_module_cls, window_sec: float, seed: int) -> LatentCropState | None:
    """Wrap ``data_module_cls.train_dataloader`` so the run's
    :class:`LatentCropState` learns how many samples are eligible once the
    dataset is known; the crop itself is cut by the step hook on the main
    process. Whatever ``train_dataloader`` the class has AT INSTALL TIME is
    wrapped. Returns the state, or None when the window is 0 or there is no
    loader to wrap. Idempotent (re-install updates the window / seed)."""
    frames = latent_crop_frames(window_sec)
    if frames <= 0:
        return None
    if getattr(data_module_cls, "_khaos_latent_crops", None) is not None:
        state = data_module_cls._khaos_latent_crops
        state.window_sec, state.window_frames, state.seed = float(window_sec), frames, int(seed)
        return state
    inner = getattr(data_module_cls, "train_dataloader", None)
    if inner is None:
        log.error("latent_crops: %s has no train_dataloader to wrap — serving whole latents",
                  data_module_cls.__name__)
        return None
    state = LatentCropState(window_sec, seed)

    def train_dataloader(self):
        original = self.train_dataset
        try:
            state.eligible = crop_eligible_count(dataset_facts(original), state.window_frames)
        except Exception as exc:  # noqa: BLE001 — a count, not the seam
            log.warning("latent_crops: could not count eligible samples (%s)", exc)
            state.eligible = None
        loader = inner(self)
        log.info("latent_crops ON: %.1f s (%d frames) windows cut per step, %s of %d samples eligible "
                 "(instrumental and longer)", state.window_sec, state.window_frames, state.eligible, len(original))
        return loader

    train_dataloader.__wrapped__ = inner  # type: ignore[attr-defined]
    # Whether the loader we wrap was the class's own or inherited decides
    # what uninstall puts back: the function, or nothing.
    data_module_cls._khaos_latent_crops_inherited = "train_dataloader" not in data_module_cls.__dict__
    data_module_cls.train_dataloader = train_dataloader
    data_module_cls._khaos_latent_crops = state
    return state


def uninstall_latent_crops(data_module_cls) -> bool:
    """Undo :func:`install_latent_crops`: put back the loader it wrapped
    (or drop ours if that loader was inherited). Idempotent."""
    state = getattr(data_module_cls, "_khaos_latent_crops", None)
    if state is None:
        return False
    inner = getattr(data_module_cls.train_dataloader, "__wrapped__", None)
    if getattr(data_module_cls, "_khaos_latent_crops_inherited", False) or inner is None:
        try:
            del data_module_cls.train_dataloader
        except AttributeError:
            pass
    else:
        data_module_cls.train_dataloader = inner
    data_module_cls._khaos_latent_crops = None
    log.info("latent_crops OFF: %s.train_dataloader restored", data_module_cls.__name__)
    return True


# ── step hooks: crop count + stem loss weight ───────────────────────────────

def stem_sample_ids(pairs) -> frozenset:
    """The segment ids of the STEM pairs among ``pairs``. ``_write_dataset``
    names a pair's WAV ``{segment_id}.wav`` and the preprocessor keeps that
    name in ``metadata.filename``, which is how a served batch says which pair it is."""
    return frozenset(str(p.segment_id) for p in pairs if getattr(p, "stem_name", None))


def sample_identity(metadata: Any) -> str | None:
    """The pair's segment id from a served sample's metadata, or None when
    the tensor carries no filename (a batch we cannot name is never
    weighted — a guess would weight the wrong pair)."""
    if not isinstance(metadata, dict):
        return None
    name = metadata.get("filename")
    return Path(str(name)).stem if name else None


class StepHookState:
    """The step hook's own record: how many steps the stem weight
    changed, how many samples could not be named, plus the crop count it
    advances on behalf of :class:`LatentCropState`."""

    def __init__(self, stem_ids: frozenset, weight: float, crop_state: LatentCropState | None):
        self.stem_ids = frozenset(stem_ids)
        self.weight = float(weight)
        self.crop_state = crop_state
        self.weighted_steps = 0
        self.unnamed_samples = 0
        self._warned = False

    def batch_weight(self, metas: list) -> float:
        """The factor for a batch: the mean over its samples of ``weight``
        (stem pair) or 1 (anything else). Exact at batch 1 — the only batch
        size any profile runs; at batch > 1 a mixed batch gets the mean,
        since the upstream loss is already one scalar by then."""
        if not metas:
            return 1.0
        weights = []
        for m in metas:
            ident = sample_identity(m)
            if ident is None:
                self.unnamed_samples += 1
                if not self._warned:
                    self._warned = True
                    log.warning("stem_loss_weight: a served sample carries no metadata.filename — "
                                "such samples train at weight 1")
            weights.append(self.weight if ident in self.stem_ids else 1.0)
        return sum(weights) / len(weights)


def scale_grad(loss, weight: float):
    """``loss`` with its GRADIENT scaled by ``weight`` and its VALUE untouched:
    ACE-Step's loop records the returned value into every loss curve this repo
    reads, so scaling the scalar would rewrite the curve by fiat. A tensor
    with no grad (the stub, a fake) is returned as it is."""
    if float(weight) == 1.0 or not hasattr(loss, "requires_grad") or not loss.requires_grad:
        return loss
    import torch  # noqa: PLC0415 — only where a real loss exists

    class _ScaleGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.view_as(value)

        @staticmethod
        def backward(ctx, grad):
            return grad * float(weight)

    return _ScaleGrad.apply(loss)


def crop_batch(batch, crop_state: LatentCropState):
    """``batch`` with its ONE sample cropped by :func:`crop_sample`, or the
    batch itself. Batch size 1 is every profile's; a larger batch is
    served whole (the per-sample windows would need re-padding) and said
    once. Per-frame tensors arrive as ``[1, T, ...]``; the sample is
    cropped on its ``[T, ...]`` view and put back with the batch axis."""
    if not isinstance(batch, dict) or batch.get("target_latents") is None:
        return batch
    metas = batch.get("metadata")
    metas = list(metas) if isinstance(metas, (list, tuple)) else []
    if int(batch["target_latents"].shape[0]) != 1 or len(metas) != 1:
        if not crop_state.serves.get("__batch_gt_1__"):
            crop_state.serves["__batch_gt_1__"] = 1
            log.warning("latent_crops: batch size > 1 — samples served whole (crops cut at batch 1 only)")
        return batch
    meta = metas[0] if isinstance(metas[0], dict) else {}
    sample = {k: batch[k][0] for k in PER_FRAME_KEYS if batch.get(k) is not None}
    sample["metadata"] = meta
    identity = sample_identity(meta) or "unnamed"
    out = crop_sample(sample, crop_state.window_frames, crop_state.rng_for(identity))
    if CROP_META_KEY not in (out.get("metadata") or {}):
        return batch
    cropped = dict(batch)
    for k in PER_FRAME_KEYS:
        if out.get(k) is not None:
            cropped[k] = out[k].unsqueeze(0)
    cropped["metadata"] = [out["metadata"]]
    crop_state.served += 1
    return cropped


def install_step_hooks(module_cls, stem_ids, weight: float,
                       crop_state: LatentCropState | None) -> StepHookState | None:
    """Wrap ``module_cls.training_step`` so that (a) the batch's sample is
    cropped by :func:`crop_batch` when ``crop_state`` is set, on the main
    process, and (b) a stem pair's loss is scaled by the stem weight on the
    GRADIENT only (:func:`scale_grad`), so the recorded curve stays raw.
    Returns None when there is nothing to do: weight 1 and no crop state."""
    if float(weight) == 1.0 and crop_state is None:
        return None
    existing = getattr(module_cls, "_khaos_step_hooks", None)
    if existing is not None:
        existing.stem_ids, existing.weight, existing.crop_state = frozenset(stem_ids), float(weight), crop_state
        return existing
    state = StepHookState(frozenset(stem_ids), weight, crop_state)
    orig = module_cls.training_step

    def training_step(self, batch, *args, **kwargs):
        if state.crop_state is not None:
            batch = crop_batch(batch, state.crop_state)
        loss = orig(self, batch, *args, **kwargs)
        metas = batch.get("metadata") if isinstance(batch, dict) else None
        metas = list(metas) if isinstance(metas, (list, tuple)) else []
        if state.weight != 1.0:
            w = state.batch_weight(metas)
            if w != 1.0:
                loss = scale_grad(loss, w)
                state.weighted_steps += 1
        return loss

    training_step.__wrapped__ = orig  # type: ignore[attr-defined]
    module_cls.training_step = training_step
    module_cls._khaos_step_hooks = state
    module_cls._khaos_step_hooks_orig = orig
    log.info("step hooks ON: stem_loss_weight %.3g over %d stem pairs%s", state.weight, len(state.stem_ids),
             "; counting latent crops" if crop_state is not None else "")
    return state


def uninstall_step_hooks(module_cls) -> bool:
    """Undo :func:`install_step_hooks`. Idempotent; False when nothing was on."""
    if getattr(module_cls, "_khaos_step_hooks", None) is None:
        return False
    module_cls.training_step = module_cls._khaos_step_hooks_orig
    module_cls._khaos_step_hooks = None
    module_cls._khaos_step_hooks_orig = None
    log.info("step hooks OFF: %s.training_step restored", module_cls.__name__)
    return True


# ── timestep bias (an experiment) ───────────────────────────────────────────
# ACE-Step draws t = max(a, b) with a, b ~ sigmoid(N(mu, sigma)); mu / sigma
# are plain TrainingConfigV2 fields read when the module is built, so the seam
# is a config field, nothing patched. t → 1 is all noise: at the defaults ~7 %
# of steps land in the high-noise band, +1 on mu makes it ~38 %. Unmeasured; off by default.

def high_noise_share(mu: float, sigma: float, threshold: float | None = None) -> float:
    """P(t ≥ threshold) for ACE-Step's max-of-two logit-normal draw:
    1 − Φ((logit(threshold) − mu) / sigma)²."""
    from khaos_attribution.engines.acestep.heldout import HIGH_NOISE_T  # noqa: PLC0415 — the evaluator's band
    th = HIGH_NOISE_T if threshold is None else float(threshold)
    z = (math.log(th / (1.0 - th)) - float(mu)) / float(sigma)
    single_below = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return 1.0 - single_below ** 2


def apply_timestep_bias(training_config, bias: float) -> dict | None:
    """Shift ``training_config.timestep_mu`` by ``bias`` (0 = as ACE-Step
    ships it) and return the manifest's ``timestep_sampling`` record of the
    resolved parameters; None when the config has no such fields (an older
    ACE-Step), in which case nothing was changed. The original mu is kept
    on the config as ``_khaos_timestep_mu_orig`` for
    :func:`restore_timestep_bias`."""
    mu0 = getattr(training_config, "timestep_mu", None)
    sigma = getattr(training_config, "timestep_sigma", None)
    if mu0 is None or sigma is None:
        log.error("timestep_high_noise_bias: this ACE-Step's TrainingConfigV2 has no timestep_mu / "
                  "timestep_sigma — sampling as upstream, bias NOT applied")
        return None
    if getattr(training_config, "_khaos_timestep_mu_orig", None) is None:
        training_config._khaos_timestep_mu_orig = float(mu0)
    base = float(training_config._khaos_timestep_mu_orig)
    training_config.timestep_mu = base + float(bias)
    rec = timestep_record(float(bias), training_config.timestep_mu, float(sigma))
    if float(bias) != 0.0:
        log.info("timestep_high_noise_bias %+.3g: mu %.3g -> %.3g, P(t >= high-noise) %.1f%% -> %.1f%%",
                 float(bias), base, training_config.timestep_mu,
                 100.0 * high_noise_share(base, float(sigma)), 100.0 * rec["p_high_noise"])
    return rec


def restore_timestep_bias(training_config) -> bool:
    """Put ``timestep_mu`` back to what the config had before
    :func:`apply_timestep_bias`. Idempotent."""
    orig = getattr(training_config, "_khaos_timestep_mu_orig", None)
    if orig is None:
        return False
    training_config.timestep_mu = float(orig)
    training_config._khaos_timestep_mu_orig = None
    return True


def timestep_record(bias: float, mu: float | None, sigma: float | None) -> dict:
    """The manifest's ``timestep_sampling`` record."""
    p = high_noise_share(mu, sigma) if mu is not None and sigma is not None else None
    return {"high_noise_bias": float(bias), "mu": mu, "sigma": sigma, "p_high_noise": p}


# ── the manifest record ─────────────────────────────────────────────────────

def idle_record(cfg: dict) -> dict:
    """The variety fields for a run on which none of these seams ran (the
    stub trainer, or a real run before they are installed): the window and
    weight the config asked for, zero counts, no resolved timestep
    parameters — the manifest says what happened, and here nothing did."""
    return {
        "latent_crops": {"window_sec": float(cfg.get("latent_crop_sec") or 0.0),
                         "applied_to": "instrumental_pairs", "eligible_samples": 0, "crops_served": 0,
                         "conditioning": "the pair's own text, duration line included; a crop is not re-captioned"},
        "stem_loss_weight": float(cfg.get("stem_loss_weight", 1.0)),
        "stem_weighted_steps": 0,
        "timestep_sampling": timestep_record(float(cfg.get("timestep_high_noise_bias") or 0.0), None, None),
    }


def variety_record(crop_state: LatentCropState | None, step_state: StepHookState | None,
                   timestep: dict | None, cfg: dict) -> dict:
    """The variety fields as ACTUALLY installed: a seam that did not go on
    records its off value (window 0, weight 1, no resolved mu), whatever
    the config asked for — the masked_loss rule."""
    rec = idle_record(cfg)
    if crop_state is not None:
        rec["latent_crops"] = crop_state.record()
    else:
        rec["latent_crops"]["window_sec"] = 0.0
    if step_state is not None:
        rec["stem_loss_weight"] = step_state.weight
        rec["stem_weighted_steps"] = int(step_state.weighted_steps)
    else:
        rec["stem_loss_weight"] = 1.0
    rec["timestep_sampling"] = timestep if timestep is not None else timestep_record(0.0, None, None)
    return rec
