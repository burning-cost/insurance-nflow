"""
Tests for diagnostics.py — PIT, QQ plots, tail index comparison, AIC/BIC table.

No torch dependency. Some tests use matplotlib; those are skipped if not available.
"""

import numpy as np
import pytest
from scipy import stats

from insurance_nflow.diagnostics import (
    ks_test,
    tail_index_comparison,
    model_comparison_table,
    fit_parametric_benchmarks,
)


# ---------------------------------------------------------------------------
# ks_test
# ---------------------------------------------------------------------------


class TestKSTest:
    def test_uniform_large_pvalue(self):
        """Truly uniform PIT should not be rejected."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 1000)
        stat, pval = ks_test(pit)
        assert pval > 0.05, f"Should not reject uniform: p={pval:.3f}"

    def test_non_uniform_low_pvalue(self):
        """Biased PIT should be rejected."""
        rng = np.random.default_rng(42)
        pit = rng.beta(0.5, 0.5, 1000)  # U-shaped, not uniform
        stat, pval = ks_test(pit)
        assert pval < 0.05, f"Should reject biased PIT: p={pval:.3f}"

    def test_returns_floats(self):
        pit = np.linspace(0, 1, 100)
        stat, pval = ks_test(pit)
        assert isinstance(stat, float)
        assert isinstance(pval, float)

    def test_stat_nonnegative(self):
        pit = np.random.default_rng(0).uniform(0, 1, 200)
        stat, pval = ks_test(pit)
        assert stat >= 0
        assert 0 <= pval <= 1


# ---------------------------------------------------------------------------
# tail_index_comparison
# ---------------------------------------------------------------------------


class TestTailIndexComparison:
    def test_keys_present(self):
        rng = np.random.default_rng(42)
        x = rng.pareto(2.0, 1000) * 10000 + 10000
        result = tail_index_comparison(x)
        assert "hill_median" in result
        assert "hill_std" in result
        assert "pareto_alpha_upper10pct" in result
        assert "lognormal_tail_index" in result

    def test_lognormal_tail_index_is_infinite(self):
        result = tail_index_comparison(np.array([100.0, 200.0, 500.0, 1000.0, 5000.0]))
        assert result["lognormal_tail_index"] == float("inf")

    def test_flow_lambda_included(self):
        rng = np.random.default_rng(42)
        x = rng.pareto(2.0, 500) * 10000
        result = tail_index_comparison(x, flow_lambda_pos=0.5)
        assert "flow_lambda_pos" in result
        assert "flow_equivalent_alpha" in result
        assert result["flow_equivalent_alpha"] == pytest.approx(2.0, rel=0.001)

    def test_hill_positive(self):
        rng = np.random.default_rng(42)
        x = rng.pareto(1.5, 500) * 5000 + 1000
        result = tail_index_comparison(x)
        assert result["hill_median"] > 0

    def test_pareto_alpha_positive(self):
        rng = np.random.default_rng(42)
        x = rng.pareto(2.0, 500) * 10000 + 5000
        result = tail_index_comparison(x)
        assert result["pareto_alpha_upper10pct"] > 0

    def test_k_range_respected(self):
        rng = np.random.default_rng(42)
        x = rng.pareto(2.0, 500) * 1000
        result = tail_index_comparison(x, k_range=(5, 20))
        assert result["hill_k_range"] == (5, 20)


# ---------------------------------------------------------------------------
# model_comparison_table
# ---------------------------------------------------------------------------


class TestModelComparisonTable:
    @pytest.fixture
    def comparison_data(self):
        rng = np.random.default_rng(42)
        x = rng.lognormal(8.5, 0.8, 1000)
        log_likelihoods = {
            "lognormal": -5000.0,
            "gamma": -5100.0,
            "flow": -4900.0,
        }
        n_params = {
            "lognormal": 2,
            "gamma": 2,
            "flow": 100_000,
        }
        return x, log_likelihoods, n_params

    def test_returns_list_of_dicts(self, comparison_data):
        x, ll, k = comparison_data
        table = model_comparison_table(x, ll, k)
        assert isinstance(table, list)
        assert all(isinstance(row, dict) for row in table)

    def test_sorted_by_aic(self, comparison_data):
        x, ll, k = comparison_data
        table = model_comparison_table(x, ll, k)
        aics = [row["aic"] for row in table]
        assert aics == sorted(aics)

    def test_required_columns(self, comparison_data):
        x, ll, k = comparison_data
        table = model_comparison_table(x, ll, k)
        for row in table:
            assert "model" in row
            assert "n_params" in row
            assert "log_lik" in row
            assert "log_lik_per_obs" in row
            assert "aic" in row
            assert "bic" in row

    def test_aic_formula(self, comparison_data):
        x, ll, k = comparison_data
        table = model_comparison_table(x, ll, k)
        for row in table:
            expected_aic = 2 * row["n_params"] - 2 * row["log_lik"]
            assert abs(row["aic"] - expected_aic) < 1e-6

    def test_bic_formula(self, comparison_data):
        x, ll, k = comparison_data
        table = model_comparison_table(x, ll, k)
        n = len(x)
        for row in table:
            expected_bic = row["n_params"] * np.log(n) - 2 * row["log_lik"]
            assert abs(row["bic"] - expected_bic) < 1e-6

    def test_flow_aic_penalised(self, comparison_data):
        """Flow has much higher AIC despite better log-lik, due to parameter count."""
        x, ll, k = comparison_data
        table = model_comparison_table(x, ll, k)
        table_dict = {row["model"]: row for row in table}
        # Flow has best log-lik but highest AIC due to 100k params
        assert table_dict["flow"]["log_lik"] > table_dict["lognormal"]["log_lik"]
        assert table_dict["flow"]["aic"] > table_dict["lognormal"]["aic"]

    def test_log_lik_per_obs(self, comparison_data):
        x, ll, k = comparison_data
        n = len(x)
        table = model_comparison_table(x, ll, k)
        for row in table:
            assert abs(row["log_lik_per_obs"] - row["log_lik"] / n) < 1e-10


# ---------------------------------------------------------------------------
# fit_parametric_benchmarks
# ---------------------------------------------------------------------------


class TestFitParametricBenchmarks:
    @pytest.fixture
    def lognormal_data(self):
        rng = np.random.default_rng(42)
        return rng.lognormal(mean=8.5, sigma=0.8, size=500)

    def test_returns_two_dicts(self, lognormal_data):
        ll, k = fit_parametric_benchmarks(lognormal_data)
        assert isinstance(ll, dict)
        assert isinstance(k, dict)

    def test_models_present(self, lognormal_data):
        ll, k = fit_parametric_benchmarks(lognormal_data)
        assert "lognormal" in ll
        assert "gamma" in ll
        assert "pareto" in ll

    def test_n_params(self, lognormal_data):
        ll, k = fit_parametric_benchmarks(lognormal_data)
        assert k["lognormal"] == 2
        assert k["gamma"] == 2
        assert k["pareto"] == 2

    def test_log_likelihoods_finite(self, lognormal_data):
        ll, k = fit_parametric_benchmarks(lognormal_data)
        for model, loglik in ll.items():
            assert np.isfinite(loglik), f"{model} log-lik is non-finite"

    def test_lognormal_data_best_lognormal_fit(self):
        """On lognormal data, lognormal should have highest log-likelihood."""
        rng = np.random.default_rng(42)
        x = rng.lognormal(mean=8.5, sigma=0.8, size=5000)
        ll, _ = fit_parametric_benchmarks(x)
        # Lognormal should beat Pareto on lognormal data
        assert ll["lognormal"] > ll["pareto"]


# ---------------------------------------------------------------------------
# matplotlib tests (skipped if not installed)
# ---------------------------------------------------------------------------


try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

pytestmark_mpl = pytest.mark.skipif(
    not MATPLOTLIB_AVAILABLE,
    reason="matplotlib not available"
)


@pytestmark_mpl
class TestPITHistogram:
    def test_returns_figure(self):
        from insurance_nflow.diagnostics import pit_histogram
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 200)
        fig = pit_histogram(pit)
        assert fig is not None
        plt.close("all")


@pytestmark_mpl
class TestQQPlotLognormal:
    def test_returns_figure(self):
        from insurance_nflow.diagnostics import qq_plot_lognormal
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(42)
        x = rng.lognormal(8.5, 0.8, 200)
        fig = qq_plot_lognormal(x)
        assert fig is not None
        plt.close("all")


@pytestmark_mpl
class TestQQPlotFlow:
    def test_returns_figure(self):
        from insurance_nflow.diagnostics import qq_plot_flow
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(42)
        x_obs = rng.lognormal(8.5, 0.8, 200)
        x_sim = rng.lognormal(8.5, 0.8, 2000)
        fig = qq_plot_flow(x_obs, x_sim)
        assert fig is not None
        plt.close("all")


@pytestmark_mpl
class TestTailPlot:
    def test_returns_figure(self):
        from insurance_nflow.diagnostics import tail_plot
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(42)
        x = rng.lognormal(8.5, 0.8, 200)
        fig = tail_plot(x)
        assert fig is not None
        plt.close("all")

    def test_with_simulated(self):
        from insurance_nflow.diagnostics import tail_plot
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(42)
        x = rng.lognormal(8.5, 0.8, 200)
        x_sim = rng.lognormal(8.5, 0.8, 2000)
        fig = tail_plot(x, x_simulated=x_sim)
        assert fig is not None
        plt.close("all")
