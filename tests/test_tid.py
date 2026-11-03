"""TID's algebra on a LoRA adapter: the damped Kronecker preconditioner is the explicit inverse, the token form equals
the matrix form, the normalisation removes every scale, the served grid is Stability's, and the score is the mean of
squares over draws."""
import json
import math

import numpy as np
import pytest

from khaos_attribution import gradient as G
from khaos_attribution import tid as T


def _torch():
    return pytest.importorskip("torch")


def _psd(torch, n, gen):
    M = torch.randn(n, n + 2, generator=gen, dtype=torch.float64)
    S = M @ M.T
    return S / torch.trace(S)


def _vec(X):
    return X.T.reshape(-1)


def test_the_constants_the_workshop_reads():
    assert T.TID_VERSION == "sa3-tid/1"
    assert T.RELATIVE_DAMPING == 0.1
    assert T.CLIP_AGGREGATION == "mean_of_squares"
    assert T.TARGETS == ("vmean", "vdelta")


def test_the_served_grid_is_stabilitys_euler_schedule_without_t_one():
    grid = T.served_grid()
    assert len(grid) == 49 and all(0.0 < t < 1.0 for t in grid)
    assert all(a > b for a, b in zip(grid, grid[1:])), "the sampler runs from noise to data"
    for seq_len in (256, 4096):
        # Stability's build_schedule with LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0), in its float32.
        t = np.linspace(1.0, 0.0, 51, dtype=np.float32)
        start = np.float32(-6.2 - 0.0 * math.log2(seq_len / 2000))
        logsnr = np.float32(2.0) - t * (np.float32(2.0) - start)
        shifted = (1.0 / (1.0 + np.exp(logsnr))).astype(np.float32)
        shifted[0], shifted[-1] = 1.0, 0.0
        evaluated = shifted[:-1]
        assert evaluated[0] == 1.0
        np.testing.assert_allclose(grid, evaluated[1:], atol=1e-6)
    golden = {0: 0.997615, 24: 0.890903, 44: 0.235052, 48: 0.137525}
    for k, v in golden.items():
        assert grid[k] == pytest.approx(v, abs=1e-6)


def test_the_flow_matching_weight_is_the_papers():
    assert T.fm_weight(0.5) == pytest.approx(1.0)
    assert T.fm_weight(0.2) == pytest.approx(16.0)
    assert T.fm_weight(0.9) == pytest.approx((0.1 / 0.9) ** 2)


def test_a_draws_noise_is_shared_by_every_clip_on_their_common_frames():
    a = T.draw_noise(1000, 3, (1, 4, 50))
    b = T.draw_noise(1000, 3, (1, 4, 20))
    assert np.array_equal(a[:, :, :20], b), "a query clip and a training clip share draw 3's noise"
    assert np.array_equal(a, G.shared_noise(1000, 3, (1, 4, 50)))
    assert not np.array_equal(T.draw_noise(1000, 4, (1, 4, 20)), b), "another draw is another noise"
    assert not np.array_equal(T.draw_noise(1001, 3, (1, 4, 20)), b), "another seed is another noise"


def test_the_damped_kronecker_preconditioner_is_the_explicit_inverse():
    torch = _torch()
    gen = torch.Generator().manual_seed(3)
    o, i = 4, 5
    U, V = _psd(torch, i, gen), _psd(torch, o, gen)
    Gc = torch.randn(o, i, generator=gen, dtype=torch.float64)
    Gi = torch.randn(o, i, generator=gen, dtype=torch.float64)
    inv_U, lam_U = T.damped_inverse(U)
    inv_V, _lam_V = T.damped_inverse(V)
    got = float((Gc * T.precondition(Gi, inv_V, inv_U)).sum())
    dU, dV = 0.1 * float(torch.trace(U)) / i, 0.1 * float(torch.trace(V)) / o
    full = torch.kron(U + dU * torch.eye(i, dtype=torch.float64), V + dV * torch.eye(o, dtype=torch.float64))
    want = float(_vec(Gc) @ torch.linalg.solve(full, _vec(Gi)))
    assert got == pytest.approx(want, rel=1e-10)
    batch = T.precondition(torch.stack([Gc, Gi]), inv_V, inv_U)
    assert torch.allclose(batch[1], T.precondition(Gi, inv_V, inv_U), rtol=0, atol=1e-13)
    assert torch.allclose(lam_U, torch.linalg.eigvalsh(U), atol=1e-13)


def test_the_token_form_of_the_kernel_equals_the_matrix_form_and_reduces_to_eq_6():
    torch = _torch()
    gen = torch.Generator().manual_seed(4)
    o, i, tc, ti = 3, 4, 5, 6
    U = _psd(torch, i, gen) + 0.01 * torch.eye(i, dtype=torch.float64)
    V = _psd(torch, o, gen) + 0.01 * torch.eye(o, dtype=torch.float64)
    Ui, Vi = torch.linalg.inv(U), torch.linalg.inv(V)
    Ac, Ai = (torch.randn(i, n, generator=gen, dtype=torch.float64) for n in (tc, ti))
    Bc, Bi = (torch.randn(o, n, generator=gen, dtype=torch.float64) for n in (tc, ti))
    ghost = sum(float(Ac[:, t] @ Ui @ Ai[:, s]) * float(Bc[:, t] @ Vi @ Bi[:, s]) for t in range(tc) for s in range(ti))
    matrix = float((Bc @ Ac.T * T.precondition(Bi @ Ai.T, Vi, Ui)).sum())
    assert matrix == pytest.approx(ghost, rel=1e-11)
    a, b, a2, b2 = Ac[:, :1], Bc[:, :1], Ai[:, :1], Bi[:, :1]
    eq6 = float(a.T @ Ui @ a2) * float(b.T @ Vi @ b2)
    assert float((b @ a.T * T.precondition(b2 @ a2.T, Vi, Ui)).sum()) == pytest.approx(eq6, rel=1e-11)


def test_normalised_gradients_remove_every_scale_and_read_zero_where_a_factor_is_zero():
    torch = _torch()
    gen = torch.Generator().manual_seed(5)
    X = torch.randn(7, 9, generator=gen, dtype=torch.float64)
    G_out = torch.randn(5, 9, generator=gen, dtype=torch.float64)
    A = torch.randn(3, 7, generator=gen, dtype=torch.float64)
    B = torch.randn(5, 3, generator=gen, dtype=torch.float64)
    gA, gB, norms = T.normalised_gradients(X, G_out, A, B, 2.0)
    for args in ((3.7 * X, G_out, A, B, 2.0), (X, 0.01 * G_out, A, B, 2.0), (X, G_out, A, B, 11.0)):
        hA, hB, _ = T.normalised_gradients(*args)
        assert torch.allclose(hA, gA, atol=1e-13) and torch.allclose(hB, gB, atol=1e-13)
    a_A, b_A, a_B, b_B = T.lora_factor_matrices(X, G_out, A, B, 2.0)
    assert torch.allclose(gA, (b_A @ a_A.T) / (a_A.norm() * b_A.norm()), atol=1e-13)
    assert norms == pytest.approx((float(a_A.norm()), float(b_A.norm()), float(a_B.norm()), float(b_B.norm())))
    zA, zB, znorms = T.normalised_gradients(torch.zeros_like(X), G_out, A, B, 2.0)
    assert not zA.any() and not zB.any() and znorms[0] == 0.0 and znorms[2] == 0.0


def test_the_accumulated_factors_have_unit_trace_and_survive_a_save_and_load(tmp_path):
    torch = _torch()
    gen = torch.Generator().manual_seed(6)
    A = torch.randn(2, 4, generator=gen, dtype=torch.float64)
    B = torch.randn(3, 2, generator=gen, dtype=torch.float64)
    acc = T.FactorAccumulator()
    for _ in range(5):
        acc.add("layer", torch.randn(4, 6, generator=gen, dtype=torch.float64),
                torch.randn(3, 6, generator=gen, dtype=torch.float64), A, B, 1.0)
    acc.add("dead", torch.zeros(4, 6, dtype=torch.float64), torch.randn(3, 6, generator=gen, dtype=torch.float64), A, B, 1.0)
    means = acc.means("layer")
    for name in T.FACTORS:
        assert float(torch.trace(means[name])) == pytest.approx(1.0, abs=1e-12)
        assert torch.allclose(means[name], means[name].T, atol=1e-14)
    assert acc.live() == {"layer": {"A": True, "B": True}, "dead": {"A": False, "B": False}}
    entries = acc.save(tmp_path)
    (tmp_path / T.FACTORS_RECORD).write_text(json.dumps({"modules": entries}))
    again = T.FactorAccumulator.load(tmp_path)
    for name in T.FACTORS:
        assert torch.allclose(again.means("layer")[name], means[name], atol=0)
    assert again.counts == acc.counts
    halves = T.FactorAccumulator.load(tmp_path, keys=["layer"]).merge(T.FactorAccumulator.load(tmp_path, keys=["layer"]))
    assert torch.allclose(halves.means("layer")["V_B"], means["V_B"], atol=1e-14), "merging two passes keeps the mean"


def test_the_damping_is_a_tenth_of_the_mean_eigenvalue_and_scales_with_the_factor():
    torch = _torch()
    gen = torch.Generator().manual_seed(7)
    F = _psd(torch, 6, gen) * 3.0
    inverse, lam = T.damped_inverse(F)
    delta = 0.1 * float(torch.trace(F)) / 6
    assert torch.allclose(inverse, torch.linalg.inv(F + delta * torch.eye(6, dtype=torch.float64)), atol=1e-10)
    assert float(lam.mean()) * 0.1 == pytest.approx(delta)
    scaled, _ = T.damped_inverse(F * 40.0)
    assert torch.allclose(scaled, inverse / 40.0, atol=1e-12), "relative damping is scale-equivariant"
    with pytest.raises(ValueError, match="zero trace"):
        T.damped_inverse(torch.zeros(3, 3, dtype=torch.float64))


def test_tau_is_the_mean_over_draws_of_the_squared_kernel_and_an_output_averages_its_clips():
    K = np.array([[[1.0, -2.0], [0.5, 0.0], [3.0, 1.0]],
                  [[-1.0, 1.0], [0.5, 2.0], [0.0, 1.0]]])
    tau = T.tau_from_kernels(K)
    assert np.allclose(tau, [[1.0, 2.5], [0.25, 2.0], [4.5, 1.0]])
    weighted = T.tau_from_kernels(K, weights=[3.0, 1.0])
    assert np.allclose(weighted[0], [(3 * 1 + 1 * 1) / 4, (3 * 4 + 1 * 1) / 4])
    scores = T.output_scores(tau, ["o2", "o1", "o2"])
    assert list(scores) == ["o2", "o1"]
    assert np.allclose(scores["o2"], [(1.0 + 4.5) / 2, (2.5 + 1.0) / 2]) and np.allclose(scores["o1"], [0.25, 2.0])
    with pytest.raises(ValueError):
        T.tau_from_kernels(K, weights=[1.0])
    with pytest.raises(ValueError):
        T.output_scores(tau, ["o1"])
