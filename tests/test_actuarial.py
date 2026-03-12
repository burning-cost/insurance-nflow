"""
Tests for actuarial.py — TVaR, ILF, LEV, reinsurance layer functions.

No torch dependency. Pure numpy computation.
"""

import numpy as np
import pytest

from insurance_nflow.actuarial import (
    quantile,
    tvar,
    limited_expected_value,
    excess_expected_value,
    ilf,
    reinsurance_layer_cost,
    layer_loss_ratio,
    burning_cost_summary,
    _skewness,
)


@pytest.fixture
def pareto_samples():
    """Large Pareto(alpha=2, scale=10_000) sample for verification."""
    rng = np.random.default_rng(42)
    n = 500_000
    u = rng.uniform(0, 1, n)
    return 10_000.0 * (u ** (-1 / 2.0))  # Pareto scale=10k, alpha=2


@pytest.fixture
def lognormal_samples():
    """Log-normal samples for exact formula checks."""
    rng = np.random.default_rng(42)
    return rng.lognormal(mean=8.5, sigma=0.8, size=200_000)


# ---------------------------------------------------------------------------
# quantile
# ---------------------------------------------------------------------------


class TestQuantile:
    def test_median_matches_numpy(self, lognormal_samples):
        q = quantile(lognormal_samples, 0.5)
        expected = float(np.median(lognormal_samples))
        assert abs(q - expected) / expected < 0.001

    def test_quantile_0_returns_min(self, lognormal_samples):
        q = quantile(lognormal_samples, 0.0)
        assert q == pytest.approx(float(np.min(lognormal_samples)))

    def test_quantile_1_returns_max(self, lognormal_samples):
        q = quantile(lognormal_samples, 1.0)
        assert q == pytest.approx(float(np.max(lognormal_samples)))

    def test_quantile_monotone(self, lognormal_samples):
        """q(p1) <= q(p2) when p1 < p2."""
        levels = [0.5, 0.75, 0.9, 0.95, 0.99]
        qs = [quantile(lognormal_samples, p) for p in levels]
        for i in range(len(qs) - 1):
            assert qs[i] <= qs[i + 1]

    def test_returns_float(self, lognormal_samples):
        assert isinstance(quantile(lognormal_samples, 0.9), float)


# ---------------------------------------------------------------------------
# tvar
# ---------------------------------------------------------------------------


class TestTVaR:
    def test_tvar_exceeds_quantile(self, lognormal_samples):
        """TVaR(p) > VaR(p) for continuous distributions."""
        p = 0.95
        q = quantile(lognormal_samples, p)
        tv = tvar(lognormal_samples, p)
        assert tv > q, f"TVaR={tv:.2f} should exceed VaR={q:.2f}"

    def test_tvar_pareto_formula(self, pareto_samples):
        """
        For Pareto(alpha=2, scale=x_m), TVaR(p) = x_m / (1-p)^{1/alpha} / (1 - 1/alpha).
        Actually: TVaR(p) = VaR(p) * alpha / (alpha - 1) for alpha > 1.
        With alpha=2: TVaR(p) = 2 * VaR(p).
        """
        p = 0.90
        alpha = 2.0
        scale = 10_000.0
        tv = tvar(pareto_samples, p)
        # VaR(p) for Pareto = scale / (1-p)^{1/alpha}
        var_p = scale * ((1 - p) ** (-1 / alpha))
        # TVaR = VaR * alpha/(alpha-1) = VaR * 2
        expected_tvar = var_p * alpha / (alpha - 1)
        # Allow 3% tolerance from MC sampling
        assert abs(tv - expected_tvar) / expected_tvar < 0.03, (
            f"TVaR={tv:.2f}, expected~{expected_tvar:.2f}"
        )

    def test_tvar_monotone_in_p(self, lognormal_samples):
        """TVaR should be increasing in p."""
        levels = [0.5, 0.75, 0.9, 0.95, 0.99]
        tvars = [tvar(lognormal_samples, p) for p in levels]
        for i in range(len(tvars) - 1):
            assert tvars[i] <= tvars[i + 1]

    def test_tvar_equals_mean_at_p_zero(self, lognormal_samples):
        """TVaR(0) = E[X] (the full mean)."""
        tv = tvar(lognormal_samples, 0.0)
        mean = float(np.mean(lognormal_samples))
        # At p=0, all samples are in the tail
        assert abs(tv - mean) / mean < 0.01

    def test_tvar_high_p(self, lognormal_samples):
        """TVaR(0.999) should be finite and exceed TVaR(0.99)."""
        tv99 = tvar(lognormal_samples, 0.99)
        tv999 = tvar(lognormal_samples, 0.999)
        assert tv999 > tv99
        assert np.isfinite(tv999)


# ---------------------------------------------------------------------------
# limited_expected_value
# ---------------------------------------------------------------------------


class TestLimitedExpectedValue:
    def test_lev_at_infinity_equals_mean(self, lognormal_samples):
        """LEV(large limit) ≈ E[X]."""
        lev = limited_expected_value(lognormal_samples, 1e10)
        mean = float(np.mean(lognormal_samples))
        assert abs(lev - mean) / mean < 0.001

    def test_lev_at_zero_is_zero(self, lognormal_samples):
        """LEV(0) = 0."""
        lev = limited_expected_value(lognormal_samples, 0.0)
        assert lev == pytest.approx(0.0, abs=1e-6)

    def test_lev_increasing_in_limit(self, lognormal_samples):
        """LEV is non-decreasing in limit."""
        limits = [10_000, 50_000, 100_000, 250_000, 1_000_000]
        levs = [limited_expected_value(lognormal_samples, L) for L in limits]
        for i in range(len(levs) - 1):
            assert levs[i] <= levs[i + 1]

    def test_lev_bounded_by_limit(self, lognormal_samples):
        """LEV(u) <= u for any u."""
        for u in [5_000, 50_000, 500_000]:
            lev = limited_expected_value(lognormal_samples, u)
            assert lev <= u + 1e-6, f"LEV({u}) = {lev:.2f} exceeds limit"

    def test_excess_plus_lev_equals_mean(self, lognormal_samples):
        """E[X] = LEV(d) + E[(X - d)_+]."""
        d = 50_000
        mean = float(np.mean(lognormal_samples))
        lev = limited_expected_value(lognormal_samples, d)
        excess = excess_expected_value(lognormal_samples, d)
        assert abs(lev + excess - mean) / mean < 0.001

    def test_lognormal_lev_formula(self):
        """
        For Lognormal(mu, sigma), LEV has analytic formula:
        LEV(u) = exp(mu + sigma^2/2) * Phi((log(u) - mu - sigma^2) / sigma)
               + u * (1 - Phi((log(u) - mu) / sigma))
        """
        from scipy.stats import norm
        mu, sigma, u = 8.5, 0.8, 50_000

        rng = np.random.default_rng(42)
        samples = rng.lognormal(mean=mu, sigma=sigma, size=1_000_000)
        lev_mc = limited_expected_value(samples, u)

        # Analytic formula
        lev_analytic = (
            np.exp(mu + 0.5 * sigma ** 2) * norm.cdf((np.log(u) - mu - sigma ** 2) / sigma)
            + u * (1 - norm.cdf((np.log(u) - mu) / sigma))
        )
        assert abs(lev_mc - lev_analytic) / lev_analytic < 0.01, (
            f"MC LEV={lev_mc:.2f}, analytic LEV={lev_analytic:.2f}"
        )


# ---------------------------------------------------------------------------
# ilf
# ---------------------------------------------------------------------------


class TestILF:
    def test_ilf_at_basic_limit_is_one(self, lognormal_samples):
        """ILF(basic_limit) == 1.0 by definition."""
        basic = 50_000
        result = ilf(lognormal_samples, limits=[basic], basic_limit=basic)
        assert result[float(basic)] == pytest.approx(1.0, abs=1e-6)

    def test_ilf_increasing_in_limit(self, lognormal_samples):
        """ILF should be increasing in policy limit."""
        limits = [25_000, 50_000, 100_000, 250_000, 500_000]
        basic = 25_000
        result = ilf(lognormal_samples, limits=limits, basic_limit=basic)
        ilf_values = [result[float(L)] for L in limits]
        for i in range(len(ilf_values) - 1):
            assert ilf_values[i] <= ilf_values[i + 1]

    def test_ilf_converges_to_mean_ratio(self, lognormal_samples):
        """
        ILF(large limit) / ILF(small limit) ≈ mean / LEV(small limit).
        """
        basic = 50_000
        large = 10_000_000
        result = ilf(lognormal_samples, limits=[basic, large], basic_limit=basic)
        # LEV(large) ≈ mean; LEV(basic) is known
        mean = float(np.mean(lognormal_samples))
        lev_basic = limited_expected_value(lognormal_samples, basic)
        expected_ratio = mean / lev_basic
        ilf_large = result[float(large)]
        assert abs(ilf_large - expected_ratio) / expected_ratio < 0.01

    def test_ilf_raises_on_zero_lev(self):
        """ILF should raise if basic_limit LEV is zero."""
        samples = np.array([1000.0, 2000.0, 3000.0])
        with pytest.raises(ValueError, match="basic_limit"):
            ilf(samples, limits=[10_000], basic_limit=0.0)

    def test_ilf_returns_dict(self, lognormal_samples):
        result = ilf(lognormal_samples, [50_000, 100_000], basic_limit=50_000)
        assert isinstance(result, dict)
        assert 50_000.0 in result
        assert 100_000.0 in result


# ---------------------------------------------------------------------------
# reinsurance_layer_cost
# ---------------------------------------------------------------------------


class TestReinsuranceLayerCost:
    def test_layer_below_attachment_is_zero(self, lognormal_samples):
        """
        If all samples are below attachment, layer cost is 0.
        Use a very high attachment.
        """
        cost = reinsurance_layer_cost(lognormal_samples, attachment=1e9, limit=1e9)
        assert cost == pytest.approx(0.0, abs=1.0)

    def test_layer_above_max_is_limit(self, lognormal_samples):
        """
        If all samples exceed attachment and no limit binding,
        layer cost ≈ E[(X - attachment)_+].
        Use attachment = 0 so all samples contribute.
        """
        cost = reinsurance_layer_cost(lognormal_samples, attachment=0.0, limit=1e9)
        expected = excess_expected_value(lognormal_samples, 0.0)
        assert abs(cost - expected) / expected < 0.001

    def test_layer_cost_increases_with_limit(self, lognormal_samples):
        """Larger layer limit -> higher or equal layer cost."""
        att = 50_000
        cost1 = reinsurance_layer_cost(lognormal_samples, att, 50_000)
        cost2 = reinsurance_layer_cost(lognormal_samples, att, 200_000)
        assert cost2 >= cost1

    def test_lev_decomposition(self, lognormal_samples):
        """
        Layer(att, lim) = LEV(att + lim) - LEV(att).
        This is the layer cost identity for continuous distributions.
        """
        att, lim = 50_000, 100_000
        lev_top = limited_expected_value(lognormal_samples, att + lim)
        lev_att = limited_expected_value(lognormal_samples, att)
        expected = lev_top - lev_att
        actual = reinsurance_layer_cost(lognormal_samples, att, lim)
        assert abs(actual - expected) / (expected + 1) < 0.001

    def test_layer_loss_ratio(self, lognormal_samples):
        """layer_loss_ratio = layer_cost / premium."""
        att, lim = 50_000, 200_000
        premium = 5_000.0
        cost = reinsurance_layer_cost(lognormal_samples, att, lim)
        ratio = layer_loss_ratio(lognormal_samples, att, lim, premium)
        assert abs(ratio - cost / premium) < 1e-6


# ---------------------------------------------------------------------------
# burning_cost_summary
# ---------------------------------------------------------------------------


class TestBurningCostSummary:
    def test_summary_keys(self, lognormal_samples):
        result = burning_cost_summary(lognormal_samples)
        assert "mean" in result
        assert "median" in result
        assert "std" in result
        assert "skewness" in result
        assert "quantiles" in result
        assert "tvar" in result
        assert "ilf" in result
        assert "n_samples" in result

    def test_summary_values_positive(self, lognormal_samples):
        result = burning_cost_summary(lognormal_samples)
        assert result["mean"] > 0
        assert result["std"] > 0
        assert result["n_samples"] == len(lognormal_samples)

    def test_summary_quantiles_ordered(self, lognormal_samples):
        result = burning_cost_summary(lognormal_samples)
        qs = result["quantiles"]
        vals = sorted(qs.items())
        for i in range(len(vals) - 1):
            assert vals[i][1] <= vals[i + 1][1]

    def test_summary_tvar_exceeds_quantile(self, lognormal_samples):
        result = burning_cost_summary(lognormal_samples)
        for p in [0.90, 0.95, 0.99]:
            if p in result["quantiles"]:
                tv = result["tvar"][p]
                q = result["quantiles"][p]
                assert tv >= q

    def test_summary_ilf_at_basic_limit(self, lognormal_samples):
        result = burning_cost_summary(lognormal_samples, basic_limit=50_000)
        assert result["ilf"][50_000.0] == pytest.approx(1.0, abs=1e-5)

    def test_skewness_lognormal_positive(self, lognormal_samples):
        """Lognormal has positive skewness."""
        result = burning_cost_summary(lognormal_samples)
        assert result["skewness"] > 0


class TestSkewness:
    def test_symmetric_distribution_near_zero(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100_000)
        sk = _skewness(x)
        assert abs(sk) < 0.05

    def test_lognormal_positive_skewness(self):
        rng = np.random.default_rng(42)
        x = rng.lognormal(0, 1, 100_000)
        sk = _skewness(x)
        assert sk > 1.0

    def test_small_sample(self):
        x = np.array([1.0, 2.0])
        sk = _skewness(x)
        assert np.isnan(sk) or isinstance(sk, float)
