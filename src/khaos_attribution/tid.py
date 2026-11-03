"""TID (Liu et al., arXiv 2609.38776) on a LoRA adapter. At each aligned draw (t, noise), every LoRA tensor's gradient
is divided by its two Kronecker factor matrices' norms and preconditioned by damped Kronecker-factored curvature.
The kernel summed over tensors is squared and averaged over draws. Pure functions: numpy at import, torch where used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from khaos_attribution import gradient as G

TID_VERSION = "sa3-tid/1"
RELATIVE_DAMPING = 0.1
CLIP_AGGREGATION = "mean_of_squares"
TARGETS = ("vmean", "vdelta")
FACTORS = ("U_A", "V_A", "U_B", "V_B")
FACTORS_RECORD = "factors.json"


def served_grid(steps: int = 50, anchor_logsnr: float = -6.2, logsnr_end: float = 2.0) -> list:
    """The times the served Euler sampler evaluates, without t = 1: Stability's LogSNR shift at rate 0, so
    t_k = sigmoid(−(end − u_k·(end − anchor))) with u_k = 1 − k/steps, for k = 1 … steps − 1."""
    out = []
    for k in range(1, int(steps)):
        u = 1.0 - k / float(steps)
        logsnr = logsnr_end - u * (logsnr_end - anchor_logsnr)
        out.append(1.0 / (1.0 + math.exp(logsnr)))
    return out


def fm_weight(t: float) -> float:
    """The flow-matching score weight ((1 − t)/t)², which the paper derives and then drops."""
    return ((1.0 - float(t)) / float(t)) ** 2


def draw_noise(seed: int, m: int, shape: Sequence[int]) -> np.ndarray:
    """Draw m's noise: the same for every training clip and query clip, cut to the clip's frames."""
    return G.shared_noise(int(seed), int(m), shape)


def lora_factor_matrices(X, G_out, A, B, s: float) -> tuple:
    """(a_A, b_A, a_B, b_B) of one LoRA layer from its input tokens X (in × T) and output gradients G (out × T):
    lora_A's gradient is b_A a_Aᵀ = s·BᵀG·Xᵀ and lora_B's is b_B a_Bᵀ = s·G·(AX)ᵀ."""
    return X, s * (B.transpose(0, 1) @ G_out), s * (A @ X), G_out


def _outer(b, a, scale: float):
    import torch  # noqa: PLC0415
    if scale > 0.0:
        return (b @ a.transpose(0, 1)) / scale
    return torch.zeros(b.shape[0], a.shape[0], dtype=b.dtype, device=b.device)


def normalised_gradients(X, G_out, A, B, s: float) -> tuple:
    """(Γ̂_A, Γ̂_B, norms): each tensor's gradient over the product of its factor matrices' Frobenius norms, zero
    where a norm is zero; norms are (‖a_A‖, ‖b_A‖, ‖a_B‖, ‖b_B‖)."""
    a_A, b_A, a_B, b_B = lora_factor_matrices(X, G_out, A, B, s)
    norms = tuple(float(m.norm()) for m in (a_A, b_A, a_B, b_B))
    return _outer(b_A, a_A, norms[0] * norms[1]), _outer(b_B, a_B, norms[2] * norms[3]), norms


class FactorAccumulator:
    """Per LoRA layer, the summed outer products of its four factor matrices, each scaled to unit Frobenius norm,
    over (clip, draw) observations; ``keys`` limits it to some layers."""

    def __init__(self, keys=None, device=None):
        self.keys = None if keys is None else set(keys)
        self.device = device
        self.sums: dict = {}
        self.counts: dict = {}

    def add(self, key: str, X, G_out, A, B, s: float) -> None:
        if self.keys is not None and key not in self.keys:
            return
        import torch  # noqa: PLC0415
        mats = dict(zip(FACTORS, lora_factor_matrices(X, G_out, A, B, s)))
        if key not in self.sums:
            device = self.device if self.device is not None else X.device
            self.sums[key] = {n: torch.zeros(m.shape[0], m.shape[0], dtype=m.dtype, device=device) for n, m in mats.items()}
            self.counts[key] = {"n": 0, "nonzero_A": 0, "nonzero_B": 0}
        norms = {}
        for n, m in mats.items():
            norms[n] = float(m.norm())
            if norms[n] > 0.0:
                S = self.sums[key][n]
                m = m.to(S.device)
                S.addmm_(m, m.transpose(0, 1), alpha=1.0 / (norms[n] * norms[n]))
        c = self.counts[key]
        c["n"] += 1
        c["nonzero_A"] += int(norms["U_A"] > 0.0 and norms["V_A"] > 0.0)
        c["nonzero_B"] += int(norms["U_B"] > 0.0 and norms["V_B"] > 0.0)

    def merge(self, other: "FactorAccumulator") -> "FactorAccumulator":
        for key, sums in other.sums.items():
            if key not in self.sums:
                self.sums[key] = {n: S.clone() for n, S in sums.items()}
                self.counts[key] = dict(other.counts[key])
                continue
            for n, S in sums.items():
                self.sums[key][n] += S.to(self.sums[key][n].device)
            for k, v in other.counts[key].items():
                self.counts[key][k] += v
        return self

    def means(self, key: str) -> dict:
        n = self.counts[key]["n"]
        return {name: S / n for name, S in self.sums[key].items()}

    def live(self) -> dict:
        """Which of each layer's two tensors had a nonzero gradient on at least one observation."""
        return {key: {"A": c["nonzero_A"] > 0, "B": c["nonzero_B"] > 0} for key, c in self.counts.items()}

    def save(self, out_dir) -> list:
        """Writes one file per layer (sums and counts) and returns each layer's entry for the factors record."""
        import torch  # noqa: PLC0415
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        entries = []
        for key, sums in self.sums.items():
            counts = self.counts[key]
            torch.save({**{n: S.detach().cpu() for n, S in sums.items()}, **counts}, str(out / f"{key}.pt"))
            n = max(1, counts["n"])
            entries.append({"key": key, "file": f"{key}.pt", "dims": {name: int(S.shape[0]) for name, S in sums.items()},
                            **counts, "trace": {name: float(torch.trace(S)) / n for name, S in sums.items()}})
        return entries

    @classmethod
    def load(cls, out_dir, keys=None, device="cpu") -> "FactorAccumulator":
        import torch  # noqa: PLC0415
        out = Path(out_dir)
        record = json.loads((out / FACTORS_RECORD).read_text(encoding="utf-8"))
        acc = cls(keys, device)
        for entry in record["modules"]:
            if keys is not None and entry["key"] not in set(keys):
                continue
            data = torch.load(str(out / entry["file"]), map_location=device)
            acc.sums[entry["key"]] = {n: data[n] for n in FACTORS}
            acc.counts[entry["key"]] = {k: int(data[k]) for k in ("n", "nonzero_A", "nonzero_B")}
        return acc


def damped_inverse(F, alpha: float = RELATIVE_DAMPING) -> tuple:
    """(inverse, eigenvalues) of F + alpha·(tr F / n)·I through a float64 eigendecomposition: the inverse in F's
    dtype, F's eigenvalues in float64. A factor with zero trace is refused, having nothing to precondition."""
    import torch  # noqa: PLC0415
    F64 = F.detach().to("cpu", torch.float64)
    F64 = (F64 + F64.transpose(0, 1)) / 2.0
    delta = float(alpha) * float(torch.trace(F64)) / F64.shape[0]
    if not delta > 0.0:
        raise ValueError("a factor with zero trace has nothing to precondition")
    lam, Q = torch.linalg.eigh(F64)
    # Eigenvalues below zero are rounding on a sum of outer products; the damping keeps every divisor positive.
    inverse = (Q / (lam.clamp(min=0.0) + delta)) @ Q.transpose(0, 1)
    return inverse.to(F.dtype), lam


def precondition(rows, inv_V, inv_U):
    """Ṽ⁻¹·Γ·Ũ⁻¹ for one gradient matrix or a batch of them (…, out, in)."""
    return inv_V @ rows @ inv_U


def layer_kernels(P, q, bounds: Sequence[tuple]):
    """Each tensor's share of the kernel, (tensors × rows), from preconditioned rows P and a query row q."""
    import torch  # noqa: PLC0415
    return torch.stack([P[:, off:off + n] @ q[off:off + n] for off, n in bounds])


def tau_from_kernels(K, weights: Sequence[float] | None = None) -> np.ndarray:
    """TID's score per (query clip, training row): the mean over draws of the squared kernel, weighted when asked."""
    K = np.asarray(K, dtype=np.float64)
    w = np.ones(K.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (K.shape[0],) or not w.sum() > 0.0:
        raise ValueError(f"{w.shape[0] if w.ndim else 0} weights for {K.shape[0]} draws")
    return np.tensordot(w, K * K, axes=1) / w.sum()


def output_scores(tau, clip_groups: Sequence) -> dict:
    """Each output's score per training row: the mean of its clips' scores, outputs in first-appearance order."""
    tau = np.asarray(tau, dtype=np.float64)
    if len(clip_groups) != tau.shape[0]:
        raise ValueError(f"{len(clip_groups)} groups for {tau.shape[0]} clips")
    out: dict = {}
    for row, group in zip(tau, clip_groups):
        out.setdefault(group, []).append(row)
    return {g: np.mean(np.stack(rows), axis=0).astype(np.float32) for g, rows in out.items()}
