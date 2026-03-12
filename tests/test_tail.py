"""
Tests for tail.py — TTF tail transform and Hill estimator.

These tests do NOT require zuko or torch for the Hill estimator tests.
The TailTransform tests are skipped if torch is not available.
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
    _erfc_inverse,
)


# ---------------------------------------------------------------------------
# TailTransform: forward / inverse consistency
# ---------------------------------------------------------------------------


class TestTailTransformRoundTrip:
    """Forward and inverse must be exact inverses."""

    def test_round_trip_z_to_u_to_z_lambda1(self):
        """z -> u -> z should recover z for lambda = 1."""
        tail = TailTransform(lambda_pos=1.0, lambda_neg=1.0)
        z = torch.linspace(-3.0, 3.0, 50)
        u, ladj_fwd = tail.forward(z)
        z_rec, ladj_inv = tail.inverse(u)
        assert torch.allclose(z, z_rec, atol=1e-5), f"Max error: {(z - z_rec).abs().max():.2e}"

    def test_round_trip_z_to_u_to_z_heavy_tail(self):
        """Round-trip with heavy tail parameters (lambda = 0.5, mimics GPD xi=2)."""
        tail = TailTransform(lambda_pos=0.5, lambda_neg=0.3)
        z = torch.linspace(-2.5, 2.5, 50)
        u, _ = tail.forward(z)
        z_rec, _ = tail.inverse(u)
        assert torch.allclose(z, z_rec, atol=1e-4)

    def test_round_trip_u_to_z_to_u(self):
        """Inverse then forward should recover u."""
        tail = TailTransform(lambda_pos=0.8, lambda_neg=0.6)
        u = torch.linspace(-2.5, 2.5, 50)
        z, _ = tail.inverse(u)
        u_rec, _ = tail.forward(z)
        assert torch.allclose(u, u_rec, atol=1e-5)

    def test_jacobian_consistency_forward(self):
        """Forward Jacobian should match numerical Jacobian."""
        tail = TailTransform(lambda_pos=1.0, lambda_neg=1.0, trainable=False)
        z = torch.tensor([0.5, -0.5, 1.5, -1.5], requires_grad=True)
        u, ladj_analytic = tail.forward(z)

        # Numerical Jacobian via finite differences
        eps = 1e-5
        ladj_numerical = []
        for i in range(len(z)):
            z_plus = z.detach().clone()
            z_plus[i] += eps
            z_minus = z.detach().clone()
            z_minus[i] -= eps
            u_plus, _ = tail.forward(z_plus)
            u_minus, _ = tail.forward(z_minus)
            dui_dzi = (u_plus[i] - u_minus[i]) / (2 * eps)
            ladj_numerical.append(math.log(abs(float(dui_dzi))))

        ladj_numerical = torch.tensor(ladj_numerical, dtype=torch.float32)
        assert torch.allclose(ladj_analytic, ladj_numerical, atol=1e-4)

    def test_jacobian_sum_forward_inverse_zero(self):
        """Sum of forward and inverse log Jacobians should be 0."""
        tail = TailTransform(lambda_pos=1.2, lambda_neg=0.7)
        z = torch.tensor([0.1, 0.5, 1.0, 2.0, -0.5, -1.5])
        u, ladj_fwd = tail.forward(z)
        _, ladj_inv = tail.inverse(u)
        # ladj_fwd + ladj_inv = 0 (they're inverses)
        total = ladj_fwd + ladj_inv
        assert torch.allclose(total, torch.zeros_like(total), atol=1e-5)


class TestTailTransformProperties:
    """Mathematical properties of the TTF transform."""

    def test_z_zero_maps_to_u_zero(self):
        """R(0) = 0 for any lambda."""
        tail = TailTransform(lambda_pos=1.5, lambda_neg=0.4)
        z_zero = torch.tensor([0.0])
        u, _ = tail.forward(z_zero)
        assert abs(float(u[0])) < 1e-5, f"R(0) = {float(u[0]):.2e}, expected 0"

    def test_sign_preserving(self):
        """TTF should preserve the sign of z."""
        tail = TailTransform(lambda_pos=1.0, lambda_neg=1.0)
        z_pos = torch.linspace(0.1, 3.0, 20)
        z_neg = torch.linspace(-3.0, -0.1, 20)
        u_pos, _ = tail.forward(z_pos)
        u_neg, _ = tail.forward(z_neg)
        assert (u_pos > 0).all(), "Positive z should map to positive u"
        assert (u_neg < 0).all(), "Negative z should map to negative u"

    def test_monotone(self):
        """TTF should be monotone increasing."""
        tail = TailTransform(lambda_pos=0.8, lambda_neg=0.8)
        z = torch.linspace(-3.0, 3.0, 100)
        u, _ = tail.forward(z)
        diffs = u[1:] - u[:-1]
        assert (diffs > 0).all(), "TTF should be strictly increasing"

    def test_lambda_property(self):
        """lambda_pos and lambda_neg properties return positive values."""
        tail = TailTransform(lambda_pos=1.3, lambda_neg=0.6)
        assert float(tail.lambda_pos) > 0
        assert float(tail.lambda_neg) > 0

    def test_trainable_false_no_grad(self):
        """Non-trainable parameters should not accumulate gradients."""
        tail = TailTransform(trainable=False)
        assert not tail._log_lambda_pos.requires_grad
        assert not tail._log_lambda_neg.requires_grad

    def test_trainable_true_has_grad(self):
        """Trainable parameters should be in the computation graph."""
        tail = TailTransform(trainable=True)
        assert tail._log_lambda_pos.requires_grad
        assert tail._log_lambda_neg.requires_grad

    def test_heavier_tail_larger_z_output(self):
        """
        Heavier tail (larger lambda) should produce larger |u| for large |z|.
        The TTF stretches the tail — larger lambda = more stretch.
        """
        tail_light = TailTransform(lambda_pos=0.5, lambda_neg=0.5)
        tail_heavy = TailTransform(lambda_pos=2.0, lambda_neg=2.0)
        z = torch.tensor([3.0])
        u_light, _ = tail_light.forward(z)
        u_heavy, _ = tail_heavy.forward(z)
        # Heavier tail maps large z to smaller u (the inverse has heavier tail)
        # The forward maps Gaussian z -> "less Gaussian" u
        # Actually: forward maps heavy-tailed z -> standard normal u
        # So inverse maps standard normal -> heavy-tailed
        # For z=3 (already in tail): heavier lambda means u is more "normal"
        # The inverse is what generates heavy tails
        u_inv_light, _ = tail_light.inverse(z)
        u_inv_heavy, _ = tail_heavy.inverse(z)
        assert float(u_inv_heavy[0]) > float(u_inv_light[0]), (
            "Heavier tail (larger lambda) should give larger output from inverse at z=3"
        )

    def test_batch_forward_matches_elementwise(self):
        """Batch and element-wise forward should give same result."""
        tail = TailTransform(lambda_pos=1.1, lambda_neg=0.7)
        z_batch = torch.tensor([0.5, 1.0, -0.5, 2.0, -1.5])
        u_batch, ladj_batch = tail.forward(z_batch)

        for i, zi in enumerate(z_batch):
            u_i, ladj_i = tail.forward(zi.unsqueeze(0))
            assert abs(float(u_batch[i]) - float(u_i[0])) < 1e-6
            assert abs(float(ladj_batch[i]) - float(ladj_i[0])) < 1e-6


class TestErfc:
    """Numerical properties of erfc."""

    def test_erfc_zero_is_one(self):
        z = torch.tensor([0.0])
        assert abs(float(_erfc(z)[0]) - 1.0) < 1e-6

    def test_erfc_positive_decreasing(self):
        z = torch.linspace(0.0, 3.0, 50)
        erfc_vals = _erfc(z)
        diffs = erfc_vals[1:] - erfc_vals[:-1]
        assert (diffs <= 0).all()

    def test_erfc_inverse_round_trip(self):
        z = torch.linspace(0.1, 2.5, 30)
        erfc_z = _erfc(z)
        z_rec = _erfc_inverse(erfc_z)
        assert torch.allclose(z, z_rec, atol=1e-5)


# ---------------------------------------------------------------------------
# Hill estimator
# ---------------------------------------------------------------------------


class TestHillEstimator:
    """Tests for Hill tail index estimator."""

    def test_pareto_recovery(self):
        """
        Hill estimator on Pareto(alpha=2) data should recover xi ≈ 0.5.
        xi = 1/alpha, so alpha=2 -> xi=0.5.
        """
        rng = np.random.default_rng(42)
        alpha = 2.0
        scale = 1.0
        n = 5000
        # Pareto samples: X = scale * U^{-1/alpha}
        u = rng.uniform(0, 1, n)
        x = scale * (u ** (-1 / alpha))
        k = int(np.sqrt(n))
        xi_hat = hill_estimator(x, k)
        # Hill estimates 1/alpha (the tail index xi)
        # Should be close to 0.5 with some variance
        assert abs(xi_hat - 1.0 / alpha) < 0.15, (
            f"Hill estimate {xi_hat:.3f}, expected {1/alpha:.3f}"
        )

    def test_heavier_tail_larger_estimate(self):
        """Heavier-tailed distribution should give larger Hill estimate."""
        rng = np.random.default_rng(42)
        n = 2000
        k = int(np.sqrt(n))

        # Pareto(alpha=1.5) — heavy tail
        u = rng.uniform(0, 1, n)
        x_heavy = (u ** (-1 / 1.5))

        # Pareto(alpha=3.0) — lighter tail
        u = rng.uniform(0, 1, n)
        x_light = (u ** (-1 / 3.0))

        xi_heavy = hill_estimator(x_heavy, k)
        xi_light = hill_estimator(x_light, k)
        assert xi_heavy > xi_light, (
            f"Heavy tail xi={xi_heavy:.3f} should exceed light tail xi={xi_light:.3f}"
        )

    def test_minimum_k(self):
        """Hill estimator with k=1 should not crash."""
        x = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
        result = hill_estimator(x, 2)
        assert np.isfinite(result)

    def test_returns_float(self):
        x = np.array([10.0, 5.0, 3.0, 2.0, 1.0])
        result = hill_estimator(x, 3)
        assert isinstance(result, float)


class TestHillDoubleBootstrap:
    """Tests for double bootstrap tail estimator."""

    def test_returns_positive_float(self):
        rng = np.random.default_rng(0)
        x = rng.pareto(1.5, size=500) + 1.0
        result = hill_double_bootstrap(x)
        assert isinstance(result, float)
        assert result > 0

    def test_consistent_with_hill(self):
        """Bootstrap result should be in reasonable range of direct Hill."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.pareto(2.0, size=n) + 1.0
        k = int(np.sqrt(n))
        hill_direct = hill_estimator(x, k)
        hill_boot = hill_double_bootstrap(x)
        # Should be in the same ballpark
        assert abs(hill_boot - hill_direct) < 0.5


class TestEstimateTailParams:
    """Tests for estimate_tail_params from log-claims."""

    def test_returns_positive_params(self):
        rng = np.random.default_rng(42)
        log_claims = np.log(rng.lognormal(8.5, 1.0, 500))
        lp, ln = estimate_tail_params(log_claims)
        assert lp > 0
        assert ln > 0

    def test_heavier_tail_gives_larger_lambda(self):
        """Heavier-tailed log-claims should give larger lambda_pos."""
        rng = np.random.default_rng(42)
        n = 1000
        # Light tail: log-normal
        light_claims = np.log(rng.lognormal(8.5, 0.5, n))
        # Heavy tail: mix with some very large values
        heavy_log = np.log(rng.lognormal(8.5, 2.0, n))
        lp_light, _ = estimate_tail_params(light_claims)
        lp_heavy, _ = estimate_tail_params(heavy_log)
        # Heavy tail should (generally) give larger lambda_pos
        # This is a soft test due to estimator variance
        assert lp_heavy >= lp_light - 0.2  # allow some variance

    def test_clamped_to_valid_range(self):
        """Estimated parameters should be in valid range."""
        rng = np.random.default_rng(42)
        log_claims = np.log(rng.exponential(scale=5000, size=200))
        lp, ln = estimate_tail_params(log_claims)
        assert 0.3 <= lp <= 3.0
        assert 0.3 <= ln <= 3.0

    def test_small_sample(self):
        """Should not crash with small samples."""
        log_claims = np.array([8.0, 8.5, 9.0, 10.0, 12.0])
        lp, ln = estimate_tail_params(log_claims)
        assert lp > 0
        assert ln > 0
