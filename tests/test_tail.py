"""
Tests for tail.py — Student-t warping tail transform and Hill estimator.

TailTransform tests require torch and scipy.
Hill estimator tests require only numpy.
"""

import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from insurance_nflow.tail import (
    TailTransform,
    hill_estimator,
    hill_double_bootstrap,
    estimate_tail_params,
    _erfc,
    _erfcinv,
)


# ---------------------------------------------------------------------------
# TailTransform: forward / inverse consistency
# ---------------------------------------------------------------------------


class TestTailTransformRoundTrip:
    """Forward and inverse must be exact inverses."""

    def test_round_trip_z_to_u_to_z_identity(self):
        """z -> u -> z should recover z (small lambda = near Gaussian)."""
        tail = TailTransform(lambda_pos=0.01, lambda_neg=0.01)
        z = torch.linspace(-2.0, 2.0, 20)
        u, _ = tail.forward(z)
        z_rec, _ = tail.inverse(u)
        assert torch.allclose(z, z_rec, atol=1e-3), (
            f"Max round-trip error: {(z - z_rec).abs().max():.4e}"
        )

    def test_round_trip_z_to_u_to_z_standard(self):
        """Round-trip with standard heavy tail parameters."""
        tail = TailTransform(lambda_pos=0.25, lambda_neg=0.1)
        z = torch.linspace(-2.0, 2.0, 20)
        u, _ = tail.forward(z)
        z_rec, _ = tail.inverse(u)
        assert torch.allclose(z, z_rec, atol=1e-3), (
            f"Max round-trip error: {(z - z_rec).abs().max():.4e}"
        )

    def test_round_trip_u_to_z_to_u(self):
        """Inverse then forward should recover u."""
        tail = TailTransform(lambda_pos=0.2, lambda_neg=0.15)
        u = torch.linspace(-2.0, 2.0, 20)
        z, _ = tail.inverse(u)
        u_rec, _ = tail.forward(z)
        assert torch.allclose(u, u_rec, atol=1e-3), (
            f"Max round-trip error: {(u - u_rec).abs().max():.4e}"
        )

    def test_jacobian_sum_forward_inverse_zero(self):
        """Sum of forward and inverse log Jacobians should be 0."""
        tail = TailTransform(lambda_pos=0.25, lambda_neg=0.1)
        z = torch.tensor([0.5, 1.0, 2.0, -0.5, -1.0])
        u, ladj_fwd = tail.forward(z)
        _, ladj_inv = tail.inverse(u)
        total = ladj_fwd + ladj_inv
        assert torch.allclose(total, torch.zeros_like(total), atol=1e-4), (
            f"Max Jacobian error: {total.abs().max():.4e}"
        )

    def test_jacobian_consistency_forward(self):
        """Forward Jacobian should match numerical Jacobian."""
        tail = TailTransform(lambda_pos=0.25, lambda_neg=0.25)
        z = torch.tensor([0.5, 1.0, -0.5, -1.0])
        u, ladj_analytic = tail.forward(z)

        eps = 1e-4
        ladj_numerical = []
        for i in range(len(z)):
            z_plus = z.clone()
            z_minus = z.clone()
            z_plus[i] = z[i] + eps
            z_minus[i] = z[i] - eps
            u_plus, _ = tail.forward(z_plus)
            u_minus, _ = tail.forward(z_minus)
            dui_dzi = float(u_plus[i] - u_minus[i]) / (2 * eps)
            ladj_numerical.append(math.log(abs(dui_dzi) + 1e-30))

        ladj_numerical_t = torch.tensor(ladj_numerical, dtype=torch.float32)
        assert torch.allclose(ladj_analytic, ladj_numerical_t, atol=5e-3), (
            f"Analytic: {ladj_analytic.tolist()}\nNumerical: {ladj_numerical_t.tolist()}"
        )


class TestTailTransformProperties:
    """Mathematical properties of the Student-t warping."""

    def test_z_zero_maps_to_u_zero(self):
        """T^{-1}(0) = 0 (F_t(0) = 0.5 = Phi(0))."""
        tail = TailTransform(lambda_pos=0.25, lambda_neg=0.1)
        z_zero = torch.tensor([0.0])
        u, _ = tail.forward(z_zero)
        assert abs(float(u[0])) < 1e-4, f"T^{{-1}}(0) = {float(u[0]):.4e}, expected 0"

    def test_u_zero_maps_to_z_zero(self):
        """T(0) = 0."""
        tail = TailTransform(lambda_pos=0.25, lambda_neg=0.1)
        u_zero = torch.tensor([0.0])
        z, _ = tail.inverse(u_zero)
        assert abs(float(z[0])) < 1e-4, f"T(0) = {float(z[0]):.4e}, expected 0"

    def test_sign_preserving(self):
        """Both T and T^{-1} should preserve sign."""
        tail = TailTransform(lambda_pos=0.25, lambda_neg=0.25)
        z_pos = torch.linspace(0.1, 2.0, 10)
        z_neg = torch.linspace(-2.0, -0.1, 10)
        u_pos, _ = tail.forward(z_pos)
        u_neg, _ = tail.forward(z_neg)
        assert (u_pos > 0).all()
        assert (u_neg < 0).all()

    def test_monotone_forward(self):
        """T^{-1} should be non-decreasing."""
        tail = TailTransform(lambda_pos=0.25, lambda_neg=0.25)
        z = torch.linspace(-1.5, 1.5, 30)
        u, _ = tail.forward(z)
        diffs = u[1:] - u[:-1]
        assert (diffs > -1e-4).all(), f"Min diff: {diffs.min():.4e}"

    def test_monotone_inverse(self):
        """T should be non-decreasing."""
        tail = TailTransform(lambda_pos=0.25, lambda_neg=0.25)
        u = torch.linspace(-1.5, 1.5, 30)
        z, _ = tail.inverse(u)
        diffs = z[1:] - z[:-1]
        assert (diffs > -1e-4).all(), f"Min diff: {diffs.min():.4e}"

    def test_lambda_property(self):
        tail = TailTransform(lambda_pos=0.3, lambda_neg=0.1)
        assert float(tail.lambda_pos) > 0
        assert float(tail.lambda_neg) > 0
        assert abs(float(tail.lambda_pos) - 0.3) < 0.001

    def test_nu_pos_from_lambda(self):
        tail = TailTransform(lambda_pos=0.25, min_nu=2.0)
        expected_nu = max(2.0, 1.0 / 0.25)  # = 4.0
        assert abs(tail.nu_pos - expected_nu) < 0.01

    def test_trainable_false_no_grad(self):
        tail = TailTransform(trainable=False)
        assert not tail._log_lambda_pos.requires_grad
        assert not tail._log_lambda_neg.requires_grad

    def test_trainable_true_has_grad(self):
        tail = TailTransform(trainable=True)
        assert tail._log_lambda_pos.requires_grad
        assert tail._log_lambda_neg.requires_grad

    def test_heavier_tail_larger_extreme_values(self):
        """
        Heavier tail (larger lambda = smaller nu) should map same Gaussian u
        to larger extreme values |z|.
        """
        tail_light = TailTransform(lambda_pos=0.1, lambda_neg=0.1)   # nu=10
        tail_heavy = TailTransform(lambda_pos=0.4, lambda_neg=0.4)   # nu=2.5

        u = torch.tensor([2.0])
        z_light, _ = tail_light.inverse(u)
        z_heavy, _ = tail_heavy.inverse(u)
        assert float(z_heavy[0]) > float(z_light[0]), (
            f"Heavy z={float(z_heavy[0]):.3f} should exceed light z={float(z_light[0]):.3f}"
        )

    def test_batch_forward_matches_elementwise(self):
        """Batch and element-wise should agree."""
        tail = TailTransform(lambda_pos=0.25, lambda_neg=0.1)
        z_batch = torch.tensor([0.5, 1.0, -0.5, 2.0, -1.5])
        u_batch, ladj_batch = tail.forward(z_batch)
        for i, zi in enumerate(z_batch):
            u_i, ladj_i = tail.forward(zi.unsqueeze(0))
            assert abs(float(u_batch[i]) - float(u_i[0])) < 1e-5
            assert abs(float(ladj_batch[i]) - float(ladj_i[0])) < 1e-5


class TestErfc:
    def test_erfc_zero_is_one(self):
        z = torch.tensor([0.0])
        assert abs(float(_erfc(z)[0]) - 1.0) < 1e-6

    def test_erfc_positive_decreasing(self):
        z = torch.linspace(0.0, 3.0, 50)
        erfc_vals = _erfc(z)
        diffs = erfc_vals[1:] - erfc_vals[:-1]
        assert (diffs <= 0).all()

    def test_erfcinv_round_trip(self):
        z = torch.linspace(0.1, 2.5, 30)
        erfc_z = _erfc(z)
        z_rec = _erfcinv(erfc_z)
        assert torch.allclose(z, z_rec, atol=1e-5)


# ---------------------------------------------------------------------------
# Hill estimator
# ---------------------------------------------------------------------------


class TestHillEstimator:
    def test_pareto_recovery(self):
        rng = np.random.default_rng(42)
        alpha = 2.0
        n = 5000
        x = (rng.uniform(0, 1, n) ** (-1 / alpha))
        k = int(np.sqrt(n))
        xi_hat = hill_estimator(x, k)
        assert abs(xi_hat - 1.0 / alpha) < 0.15, (
            f"Hill estimate {xi_hat:.3f}, expected {1/alpha:.3f}"
        )

    def test_heavier_tail_larger_estimate(self):
        rng = np.random.default_rng(42)
        n = 2000
        k = int(np.sqrt(n))
        x_heavy = (rng.uniform(0, 1, n) ** (-1 / 1.5))
        x_light = (rng.uniform(0, 1, n) ** (-1 / 3.0))
        assert hill_estimator(x_heavy, k) > hill_estimator(x_light, k)

    def test_minimum_k(self):
        x = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
        assert np.isfinite(hill_estimator(x, 2))

    def test_returns_float(self):
        x = np.array([10.0, 5.0, 3.0, 2.0, 1.0])
        assert isinstance(hill_estimator(x, 3), float)


class TestHillDoubleBootstrap:
    def test_returns_positive_float(self):
        rng = np.random.default_rng(0)
        x = rng.pareto(1.5, size=500) + 1.0
        result = hill_double_bootstrap(x)
        assert isinstance(result, float)
        assert result > 0

    def test_consistent_with_hill(self):
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.pareto(2.0, size=n) + 1.0
        k = int(np.sqrt(n))
        hill_direct = hill_estimator(x, k)
        hill_boot = hill_double_bootstrap(x)
        assert abs(hill_boot - hill_direct) < 0.5


class TestEstimateTailParams:
    def test_returns_positive_params(self):
        rng = np.random.default_rng(42)
        log_claims = np.log(rng.lognormal(8.5, 1.0, 500))
        lp, ln = estimate_tail_params(log_claims)
        assert lp > 0
        assert ln > 0

    def test_clamped_to_valid_range(self):
        rng = np.random.default_rng(42)
        log_claims = np.log(rng.exponential(scale=5000, size=200))
        lp, ln = estimate_tail_params(log_claims)
        assert 0.05 <= lp <= 0.5
        assert 0.05 <= ln <= 0.5

    def test_small_sample(self):
        log_claims = np.array([8.0, 8.5, 9.0, 10.0, 12.0])
        lp, ln = estimate_tail_params(log_claims)
        assert lp > 0
        assert ln > 0

    def test_heavy_tail_gives_larger_lambda(self):
        rng = np.random.default_rng(42)
        n = 1000
        light_log = np.log(rng.lognormal(8.5, 0.5, n))
        heavy_log = np.log(rng.lognormal(8.5, 2.0, n))
        lp_light, _ = estimate_tail_params(light_log)
        lp_heavy, _ = estimate_tail_params(heavy_log)
        assert lp_heavy >= lp_light - 0.1
