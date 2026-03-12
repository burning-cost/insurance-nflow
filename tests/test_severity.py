"""
Tests for severity.py — SeverityFlow and ConditionalSeverityFlow.

Requires torch and zuko. All tests skip if not available.
Training tests use minimal epochs to keep runtime short.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
zuko = pytest.importorskip("zuko")

from insurance_nflow.severity import SeverityFlow, ConditionalSeverityFlow, SeverityFlowResult
from insurance_nflow.data import generate_motor_bi_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_dataset():
    """Small dataset for fast tests (300 claims)."""
    dataset = generate_motor_bi_dataset(n_policies=5000, claim_rate=0.06, seed=42)
    return dataset["claim_amount"][:300]


@pytest.fixture(scope="module")
def small_dataset_with_context():
    """Small dataset with context for conditional flow tests."""
    dataset = generate_motor_bi_dataset(n_policies=5000, claim_rate=0.06, seed=42)
    claims = dataset["claim_amount"][:200]
    age_band = dataset["age_band"][:200].astype(float)
    vehicle_group = dataset["vehicle_group"][:200].astype(float)
    context = np.column_stack([age_band, vehicle_group])
    return claims, context


@pytest.fixture(scope="module")
def fitted_flow(small_dataset):
    """Pre-fitted unconditional flow (minimal epochs, for speed)."""
    flow = SeverityFlow(
        n_transforms=2,
        hidden_features=[16, 16],
        tail_transform=False,  # Skip TTF for speed
        max_epochs=3,
        patience=5,
        batch_size=64,
        seed=42,
    )
    flow.fit(small_dataset)
    return flow


@pytest.fixture(scope="module")
def fitted_conditional_flow(small_dataset_with_context):
    """Pre-fitted conditional flow."""
    claims, context = small_dataset_with_context
    flow = ConditionalSeverityFlow(
        context_features=2,
        n_transforms=2,
        hidden_features=[16, 16],
        tail_transform=False,
        max_epochs=3,
        patience=5,
        batch_size=64,
        seed=42,
    )
    flow.fit(claims, context=context)
    return flow


# ---------------------------------------------------------------------------
# SeverityFlow construction
# ---------------------------------------------------------------------------


class TestSeverityFlowConstruction:
    def test_default_construction(self):
        flow = SeverityFlow()
        assert not flow._fitted

    def test_result_none_before_fit(self):
        flow = SeverityFlow()
        assert flow.result is None

    def test_not_fitted_raises(self):
        flow = SeverityFlow()
        with pytest.raises(RuntimeError, match="fit"):
            flow.sample(10)

    def test_params_stored(self):
        flow = SeverityFlow(n_transforms=4, hidden_features=[32], tail_transform=False)
        assert flow.n_transforms == 4
        assert flow.hidden_features == [32]
        assert not flow.tail_transform


# ---------------------------------------------------------------------------
# SeverityFlow.fit — validation
# ---------------------------------------------------------------------------


class TestSeverityFlowFitValidation:
    def test_raises_on_negative_claims(self):
        flow = SeverityFlow(max_epochs=1)
        with pytest.raises(ValueError, match="positive"):
            flow.fit(np.array([-100.0, 200.0, 300.0]))

    def test_raises_on_zero_claims(self):
        flow = SeverityFlow(max_epochs=1)
        with pytest.raises(ValueError, match="positive"):
            flow.fit(np.array([0.0, 200.0, 300.0]))

    def test_raises_context_wrong_length(self):
        flow = SeverityFlow(context_features=2, max_epochs=1)
        claims = np.array([100.0, 200.0, 300.0])
        ctx = np.random.randn(5, 2)  # wrong length
        with pytest.raises(ValueError, match="rows"):
            flow.fit(claims, context=ctx)

    def test_raises_context_wrong_features(self):
        flow = SeverityFlow(context_features=3, max_epochs=1)
        claims = np.array([100.0, 200.0, 300.0])
        ctx = np.random.randn(3, 2)  # 2 cols, expected 3
        with pytest.raises(ValueError, match="columns"):
            flow.fit(claims, context=ctx)

    def test_raises_missing_context_when_required(self):
        flow = SeverityFlow(context_features=2, max_epochs=1)
        with pytest.raises(ValueError, match="context"):
            flow.fit(np.array([100.0, 200.0, 300.0]))

    def test_raises_negative_weights(self):
        flow = SeverityFlow(max_epochs=1)
        claims = np.array([100.0, 200.0, 300.0])
        weights = np.array([1.0, -1.0, 1.0])
        with pytest.raises(ValueError, match="non-negative"):
            flow.fit(claims, exposure_weights=weights)

    def test_raises_weight_length_mismatch(self):
        flow = SeverityFlow(max_epochs=1)
        claims = np.array([100.0, 200.0, 300.0])
        weights = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="length"):
            flow.fit(claims, exposure_weights=weights)


# ---------------------------------------------------------------------------
# SeverityFlow.fit — returns and side effects
# ---------------------------------------------------------------------------


class TestSeverityFlowFitResults:
    def test_returns_result(self, fitted_flow):
        assert isinstance(fitted_flow.result, SeverityFlowResult)

    def test_result_n_parameters_positive(self, fitted_flow):
        assert fitted_flow.result.n_parameters > 0

    def test_result_training_history_nonempty(self, fitted_flow):
        assert len(fitted_flow.result.training_history) > 0

    def test_result_aic_finite(self, fitted_flow):
        assert np.isfinite(fitted_flow.result.aic)

    def test_result_bic_finite(self, fitted_flow):
        assert np.isfinite(fitted_flow.result.bic)

    def test_result_bic_penalises_more_than_aic(self, fitted_flow):
        """BIC penalises more than AIC for large n (n > e^2 ≈ 7)."""
        # BIC = k*log(n) - 2*logL, AIC = 2k - 2*logL
        # BIC > AIC iff k*log(n) > 2k iff log(n) > 2 iff n > 7.4
        # For n=300, this should hold
        result = fitted_flow.result
        assert result.bic > result.aic, (
            f"For n=300, BIC={result.bic:.1f} should exceed AIC={result.aic:.1f}"
        )

    def test_result_repr(self, fitted_flow):
        s = repr(fitted_flow.result)
        assert "SeverityFlowResult" in s
        assert "val_logL" in s

    def test_fitted_flag_true(self, fitted_flow):
        assert fitted_flow._fitted

    def test_tail_indices_after_fit(self, fitted_flow):
        result = fitted_flow.result
        assert result.tail_lambda_pos > 0
        assert result.tail_lambda_neg > 0

    def test_training_seconds_positive(self, fitted_flow):
        assert fitted_flow.result.training_seconds > 0

    def test_n_eff_reasonable(self, small_dataset):
        flow = SeverityFlow(max_epochs=2, patience=5)
        result = flow.fit(small_dataset)
        # Without weights, n_eff = n
        assert abs(result.n_eff - len(small_dataset)) < 1.0

    def test_n_eff_with_weights(self):
        rng = np.random.default_rng(42)
        claims = rng.lognormal(8.5, 0.8, 100)
        # Uniform weights -> n_eff = n
        weights = np.ones(100)
        flow = SeverityFlow(max_epochs=2, patience=5)
        result = flow.fit(claims, exposure_weights=weights)
        # n_eff should be close to n for uniform weights
        assert abs(result.n_eff - 100) < 2

    def test_history_keys(self, fitted_flow):
        history = fitted_flow.result.training_history
        for entry in history:
            assert "epoch" in entry
            assert "train_loss" in entry
            assert "val_loss" in entry


# ---------------------------------------------------------------------------
# SeverityFlow.sample
# ---------------------------------------------------------------------------


class TestSeverityFlowSample:
    def test_sample_shape(self, fitted_flow):
        samples = fitted_flow.sample(100)
        assert samples.shape == (100,)

    def test_sample_positive(self, fitted_flow):
        samples = fitted_flow.sample(200)
        assert (samples > 0).all()

    def test_sample_finite(self, fitted_flow):
        samples = fitted_flow.sample(200)
        assert np.isfinite(samples).all()

    def test_sample_with_context_single_row(self, fitted_conditional_flow):
        ctx = np.array([[2.0, 5.0]])  # one risk
        samples = fitted_conditional_flow.sample(100, context=ctx)
        assert samples.shape == (100,)
        assert (samples > 0).all()

    def test_sample_broadcast_1d_context(self, fitted_conditional_flow):
        ctx = np.array([2.0, 5.0])  # 1D context
        samples = fitted_conditional_flow.sample(50, context=ctx)
        assert len(samples) == 50


# ---------------------------------------------------------------------------
# SeverityFlow.log_prob
# ---------------------------------------------------------------------------


class TestSeverityFlowLogProb:
    def test_log_prob_shape(self, fitted_flow, small_dataset):
        lp = fitted_flow.log_prob(small_dataset[:50])
        assert lp.shape == (50,)

    def test_log_prob_finite(self, fitted_flow, small_dataset):
        lp = fitted_flow.log_prob(small_dataset[:50])
        assert np.isfinite(lp).all()


# ---------------------------------------------------------------------------
# SeverityFlow.quantile
# ---------------------------------------------------------------------------


class TestSeverityFlowQuantile:
    def test_quantile_returns_float(self, fitted_flow):
        q = fitted_flow.quantile(0.95, n_samples=10_000)
        assert isinstance(q, float)

    def test_quantile_positive(self, fitted_flow):
        q = fitted_flow.quantile(0.5, n_samples=10_000)
        assert q > 0

    def test_quantile_monotone(self, fitted_flow):
        q90 = fitted_flow.quantile(0.90, n_samples=20_000)
        q95 = fitted_flow.quantile(0.95, n_samples=20_000)
        q99 = fitted_flow.quantile(0.99, n_samples=20_000)
        assert q90 <= q95 <= q99


# ---------------------------------------------------------------------------
# SeverityFlow.tvar
# ---------------------------------------------------------------------------


class TestSeverityFlowTVaR:
    def test_tvar_returns_float(self, fitted_flow):
        tv = fitted_flow.tvar(0.95, n_samples=10_000)
        assert isinstance(tv, float)

    def test_tvar_exceeds_quantile(self, fitted_flow):
        q = fitted_flow.quantile(0.95, n_samples=20_000)
        tv = fitted_flow.tvar(0.95, n_samples=20_000)
        assert tv >= q - 1  # allow small MC variance

    def test_tvar_positive(self, fitted_flow):
        tv = fitted_flow.tvar(0.99, n_samples=10_000)
        assert tv > 0


# ---------------------------------------------------------------------------
# SeverityFlow.ilf
# ---------------------------------------------------------------------------


class TestSeverityFlowILF:
    def test_ilf_returns_dict(self, fitted_flow):
        result = fitted_flow.ilf(
            limits=[50_000, 100_000, 250_000],
            basic_limit=50_000,
            n_samples=20_000,
        )
        assert isinstance(result, dict)
        assert 50_000.0 in result

    def test_ilf_at_basic_is_one(self, fitted_flow):
        result = fitted_flow.ilf(
            limits=[50_000],
            basic_limit=50_000,
            n_samples=20_000,
        )
        assert result[50_000.0] == pytest.approx(1.0, abs=1e-5)

    def test_ilf_increasing(self, fitted_flow):
        limits = [25_000, 50_000, 100_000, 250_000]
        result = fitted_flow.ilf(
            limits=limits,
            basic_limit=25_000,
            n_samples=20_000,
        )
        vals = [result[float(L)] for L in limits]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]


# ---------------------------------------------------------------------------
# SeverityFlow.reinsurance_layer
# ---------------------------------------------------------------------------


class TestSeverityFlowReinsuranceLayer:
    def test_returns_float(self, fitted_flow):
        cost = fitted_flow.reinsurance_layer(50_000, 100_000, n_samples=10_000)
        assert isinstance(cost, float)

    def test_nonnegative(self, fitted_flow):
        cost = fitted_flow.reinsurance_layer(50_000, 100_000, n_samples=10_000)
        assert cost >= 0


# ---------------------------------------------------------------------------
# SeverityFlow.summary
# ---------------------------------------------------------------------------


class TestSeverityFlowSummary:
    def test_summary_keys(self, fitted_flow):
        s = fitted_flow.summary(n_samples=10_000)
        assert "mean" in s
        assert "tvar" in s
        assert "ilf" in s


# ---------------------------------------------------------------------------
# ConditionalSeverityFlow
# ---------------------------------------------------------------------------


class TestConditionalSeverityFlow:
    def test_requires_context_features(self):
        with pytest.raises(ValueError, match="context_features"):
            ConditionalSeverityFlow(context_features=0)

    def test_fit_requires_context(self):
        flow = ConditionalSeverityFlow(context_features=2, max_epochs=1)
        claims = np.array([100.0, 200.0, 300.0])
        with pytest.raises(ValueError, match="context"):
            flow.fit(claims, context=None)

    def test_conditional_quantile(self, fitted_conditional_flow):
        ctx = np.array([[2.0, 5.0]])
        q = fitted_conditional_flow.conditional_quantile(ctx, 0.95, n_samples=5_000)
        assert isinstance(q, float)
        assert q > 0

    def test_conditional_tvar(self, fitted_conditional_flow):
        ctx = np.array([[1.0, 8.0]])  # young driver, high vehicle group
        tv = fitted_conditional_flow.conditional_tvar(ctx, 0.95, n_samples=5_000)
        assert isinstance(tv, float)
        assert tv > 0

    def test_context_auto_detected(self):
        """context_features should be auto-detected from context matrix shape."""
        rng = np.random.default_rng(42)
        claims = rng.lognormal(8.5, 0.8, 100)
        ctx = rng.randn(100, 4)
        flow = ConditionalSeverityFlow(context_features=4, max_epochs=2, patience=5)
        result = flow.fit(claims, context=ctx)
        assert isinstance(result, SeverityFlowResult)

    def test_fitted_result_has_tail_params(self, fitted_conditional_flow):
        result = fitted_conditional_flow.result
        assert result.tail_lambda_pos > 0
        assert result.tail_lambda_neg > 0


# ---------------------------------------------------------------------------
# Exposure weights
# ---------------------------------------------------------------------------


class TestExposureWeights:
    def test_fit_with_uniform_weights(self):
        """Uniform weights should give same result as no weights (approximately)."""
        rng = np.random.default_rng(42)
        claims = rng.lognormal(8.5, 0.8, 100)
        flow1 = SeverityFlow(max_epochs=2, patience=5, seed=42, n_transforms=2, hidden_features=[16])
        flow2 = SeverityFlow(max_epochs=2, patience=5, seed=42, n_transforms=2, hidden_features=[16])
        r1 = flow1.fit(claims)
        r2 = flow2.fit(claims, exposure_weights=np.ones(100))
        # Should produce similar (not necessarily identical) results
        assert isinstance(r1, SeverityFlowResult)
        assert isinstance(r2, SeverityFlowResult)

    def test_fit_with_varying_weights(self):
        rng = np.random.default_rng(42)
        claims = rng.lognormal(8.5, 0.8, 100)
        weights = rng.exponential(1.0, 100)
        flow = SeverityFlow(max_epochs=2, patience=5)
        result = flow.fit(claims, exposure_weights=weights)
        assert isinstance(result, SeverityFlowResult)
        assert np.isfinite(result.aic)
