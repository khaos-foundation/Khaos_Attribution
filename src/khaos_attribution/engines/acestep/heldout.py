"""The held-out loss on ACE-Step's live training module: the flow-matching velocity MSE on pairs the adapter never
trained on, at fixed timesteps with fixed noise, reproducing training_step with the randomness pinned. The pure
rules that read the history stay in the room; the forward lives with the engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from khaos_attribution.engines.heldout import (DEFAULT_BUCKETS, HIGH_NOISE_T, bucket_key, bucket_timesteps,  # noqa: F401
                                              pair_noise_seed, summary_of)
from khaos_attribution.identifiers import track_of_segment


def evaluate_heldout(
    forward: Callable[[Any, float, Any], Any],
    samples: Sequence[tuple[str, Any]],
    *,
    seed: int,
    timesteps: Sequence[float] | None = None,
    noise_for: Callable[[Any, int], Any] | None = None,
) -> dict:
    """Timestep-bucketed velocity MSE over ``(name, sample)`` pairs, comparable with the train loss:
    ``forward(sample, t, noise)`` predicts the velocity at ``x_t = t·noise + (1−t)·x0`` against
    ``noise − x0``; ``mean`` averages the buckets, ``high_noise`` those at t ≥ HIGH_NOISE_T, per track too.
    """
    import torch  # noqa: PLC0415

    ts = list(timesteps) if timesteps is not None else bucket_timesteps()
    if not samples:
        raise ValueError("evaluate_heldout needs at least one held-out sample")

    def _default_noise(sample, s):
        x0 = _target_of(sample)
        gen = torch.Generator(device="cpu").manual_seed(int(s))
        return torch.randn(x0.shape, generator=gen, dtype=torch.float32)

    make_noise = noise_for or _default_noise
    sums: dict[str, float] = {bucket_key(t): 0.0 for t in ts}
    # Per track as well as pooled: the pooled numbers are what best_epoch and
    # the turn rule read; by_track is the regressand of evaluation.gradient_calibration.
    per_track: dict[str, dict[str, float]] = {}
    for name, sample in samples:
        noise = make_noise(sample, pair_noise_seed(seed, name))
        x0 = _target_of(sample).to(torch.float32)
        target = noise.to(torch.float32) - x0
        track = track_of_segment(name)
        row = per_track.setdefault(track, {"sums": {bucket_key(t): 0.0 for t in ts},
                                           "n_pairs": 0})
        row["n_pairs"] += 1
        for t in ts:
            pred = forward(sample, float(t), noise)
            pred = pred.detach().to("cpu", torch.float32)
            mse = float(torch.nn.functional.mse_loss(pred, target))
            sums[bucket_key(t)] += mse
            row["sums"][bucket_key(t)] += mse

    return summary_of(sums, per_track, len(samples), ts)


def _target_of(sample: Any):
    """The data latents ``x0`` of a sample: a preprocessed ``.pt`` dict
    (``target_latents``) or a bare tensor (tests)."""
    if isinstance(sample, dict):
        return sample["target_latents"]
    return sample


class HeldoutEvaluator:
    """Held-out loss on a live ``FixedLoRAModule``, reproducing ``training_step`` with the randomness
    fixed: preprocessed ``.pt`` batches, no CFG dropout, ``t`` the bucket centre, noise seeded per
    pair, the trainer's unmasked ``F.mse_loss(decoder_out[0], x1 − x0)`` under no_grad in eval mode.
    """

    def __init__(self, module: Any, tensor_dir: Path | str, *, seed: int,
                 n_buckets: int = DEFAULT_BUCKETS) -> None:
        self.module = module
        self.tensor_dir = Path(tensor_dir)
        self.seed = int(seed)
        self.timesteps = bucket_timesteps(n_buckets)
        self.samples = self._load_samples()

    def _load_samples(self) -> list[tuple[str, dict]]:
        """Every ``tensors_val/*.pt`` in CPU RAM for the run's lifetime (the same
        pairs are scored every evaluation), so ``heldout_pairs_cap`` scales the
        trainer's resident memory and is raised only against a measurement."""
        import torch  # noqa: PLC0415

        out: list[tuple[str, dict]] = []
        for p in sorted(self.tensor_dir.glob("*.pt")):
            data = torch.load(str(p), map_location="cpu", weights_only=True)
            out.append((p.name, {
                "target_latents": data["target_latents"],
                "attention_mask": data["attention_mask"],
                "encoder_hidden_states": data["encoder_hidden_states"],
                "encoder_attention_mask": data["encoder_attention_mask"],
                "context_latents": data["context_latents"],
            }))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def _forward(self, sample: dict, t: float, noise) -> Any:
        import torch  # noqa: PLC0415
        from contextlib import nullcontext  # noqa: PLC0415

        m = self.module
        dev, dtype = m.device, m.dtype
        x0 = sample["target_latents"].unsqueeze(0).to(dev, dtype=dtype)
        x1 = noise.unsqueeze(0).to(dev, dtype=dtype)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(dev, dtype=dtype)
        ehs = sample["encoder_hidden_states"].unsqueeze(0).to(dev, dtype=dtype)
        eam = sample["encoder_attention_mask"].unsqueeze(0).to(dev, dtype=dtype)
        ctx = sample["context_latents"].unsqueeze(0).to(dev, dtype=dtype)
        tt = torch.full((1,), float(t), device=dev, dtype=dtype)
        xt = tt.unsqueeze(-1).unsqueeze(-1) * x1 + (1.0 - tt.unsqueeze(-1).unsqueeze(-1)) * x0
        if m.device_type in ("cuda", "xpu", "mps") and dtype != torch.float32:
            ctx_mgr = torch.autocast(device_type=m.device_type, dtype=dtype)
        else:
            ctx_mgr = nullcontext()
        with ctx_mgr:
            out = m.model.decoder(
                hidden_states=xt, timestep=tt, timestep_r=tt,
                attention_mask=attention_mask, encoder_hidden_states=ehs,
                encoder_attention_mask=eam, context_latents=ctx,
            )
        return out[0].squeeze(0)

    def evaluate(self) -> dict:
        import torch  # noqa: PLC0415

        decoder = self.module.model.decoder
        was_training = bool(getattr(decoder, "training", True))
        decoder.eval()
        try:
            with torch.no_grad():
                result = evaluate_heldout(self._forward, self.samples,
                                          seed=self.seed, timesteps=self.timesteps)
        finally:
            if was_training:
                decoder.train()
        return result

    def load_adapter_weights(self, weights_path: Path | str) -> bool:
        """Swap the module's LoKR weights for the file's (used to score the
        soup). LoKR only — the LyCORIS net exposes ``load_weights``; a LoRA
        module has no equivalent here. Returns False when it cannot."""
        net = getattr(self.module, "lycoris_net", None)
        if net is None or not hasattr(net, "load_weights"):
            return False
        net.load_weights(str(weights_path))
        return True
