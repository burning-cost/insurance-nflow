"""
Tests for data.py — synthetic DGP and data generator.

No torch dependency. Pure numpy.
"""

import numpy as np
import pytest

from insurance_nflow.data import (
    DGPParams,
    generate_motor_bi_dataset,
    sample_severity,
    theoretical_tvar,
    theoretical_ilf,
    REGIONS,
    DEFAULT_DGP,
    _logistic,
    _p_cat,
    _mu_soft,
)


# ---------------------------------------------------------------------------
# DGPParams
# ---------------------------------------------------------------------------


class TestDGPParams:
    def test_default_params_valid(self):
        dgp = DGPParams()
        assert dgp.mu_soft_base > 0
        assert dgp.sigma_soft > 0
        assert dgp.alpha_cat > 1.0  # Must be > 1 for finite mean
        assert 0 < dgp.p_cat_base < 1

    def test_age_band_effects_all_bands(self):
        dgp = DGPParams()
        for band in [1, 2, 3, 4, 5]:
            assert band in dgp.age_band_effects

    def test_vehicle_group_effects_all_groups(self):
        dgp = DGPParams()
        for group in range(1, 11):
            assert group in dgp.vehicle_group_effects

    def test_region_effects_all_regions(self):
        dgp = DGPParams()
        for region in REGIONS:
            assert region in dgp.region_effects

    def test_custom_dgp(self):
        dgp = DGPParams(mu_soft_base=9.0, sigma_soft=0.5, alpha_cat=2.0)
        assert dgp.mu_soft_base == 9.0
        assert dgp.sigma_soft == 0.5
        assert dgp.alpha_cat == 2.0


# ---------------------------------------------------------------------------
# _logistic
# ---------------------------------------------------------------------------


class TestLogistic:
    def test_logistic_zero_is_half(self):
        assert abs(_logistic(0.0) - 0.5) < 1e-10

    def test_logistic_range(self):
        for x in [-10, -1, 0, 1, 10]:
            v = _logistic(x)
            assert 0 < v < 1

    def test_logistic_large_positive(self):
        assert _logistic(100.0) > 0.9999

    def test_logistic_large_negative(self):
        assert _logistic(-100.0) < 0.0001


# ---------------------------------------------------------------------------
# _p_cat
# ---------------------------------------------------------------------------


class TestPCat:
    def test_in_range(self):
        for ab in [1, 2, 3, 4, 5]:
            for vg in [1, 5, 10]:
                for r in REGIONS:
                    p = _p_cat(ab, vg, r)
                    assert 0 < p < 1, f"p_cat={p} out of range for {ab},{vg},{r}"

    def test_young_driver_higher_p_cat(self):
        """Young drivers (band 1) should have higher p_cat than baseline (band 3)."""
        p_young = _p_cat(1, 5, "midlands")
        p_base = _p_cat(3, 5, "midlands")
        assert p_young > p_base

    def test_london_higher_than_north(self):
        """London should have higher p_cat than north."""
        p_london = _p_cat(3, 5, "london")
        p_north = _p_cat(3, 5, "north")
        assert p_london > p_north


# ---------------------------------------------------------------------------
# sample_severity
# ---------------------------------------------------------------------------


class TestSampleSeverity:
    def test_all_positive(self):
        rng = np.random.default_rng(42)
        x = sample_severity(3, 5, "midlands", 1000, rng=rng)
        assert (x > 0).all()

    def test_shape(self):
        rng = np.random.default_rng(42)
        x = sample_severity(1, 3, "london", 500, rng=rng)
        assert x.shape == (500,)

    def test_reproducible(self):
        x1 = sample_severity(3, 5, "midlands", 100, rng=np.random.default_rng(42))
        x2 = sample_severity(3, 5, "midlands", 100, rng=np.random.default_rng(42))
        assert np.allclose(x1, x2)

    def test_different_rng_different(self):
        x1 = sample_severity(3, 5, "midlands", 100, rng=np.random.default_rng(1))
        x2 = sample_severity(3, 5, "midlands", 100, rng=np.random.default_rng(2))
        assert not np.allclose(x1, x2)

    def test_finite_values(self):
        rng = np.random.default_rng(42)
        x = sample_severity(1, 10, "london", 500, rng=rng)
        assert np.isfinite(x).all()

    def test_young_london_higher_severity(self):
        """
        Young driver in London should have higher mean severity than
        middle-aged driver in north.
        """
        n = 50_000
        x_young_london = sample_severity(1, 5, "london", n, rng=np.random.default_rng(42))
        x_mid_north = sample_severity(3, 5, "north", n, rng=np.random.default_rng(42))
        mean_yl = float(np.mean(x_young_london))
        mean_mn = float(np.mean(x_mid_north))
        assert mean_yl > mean_mn, (
            f"Young London mean={mean_yl:.0f} should exceed mid North mean={mean_mn:.0f}"
        )

    def test_regions_consistent(self):
        """All regions should produce valid samples."""
        rng = np.random.default_rng(42)
        for region in REGIONS:
            x = sample_severity(3, 5, region, 100, rng=rng)
            assert (x > 0).all()
            assert np.isfinite(x).all()


# ---------------------------------------------------------------------------
# generate_motor_bi_dataset
# ---------------------------------------------------------------------------


class TestGenerateMotorBIDataset:
    def test_keys(self):
        dataset = generate_motor_bi_dataset(n_policies=1000, seed=42)
        assert "claim_amount" in dataset
        assert "age_band" in dataset
        assert "vehicle_group" in dataset
        assert "region" in dataset
        assert "exposure" in dataset

    def test_consistent_lengths(self):
        dataset = generate_motor_bi_dataset(n_policies=1000, seed=42)
        n = len(dataset["claim_amount"])
        for key in dataset:
            assert len(dataset[key]) == n, f"{key} has wrong length"

    def test_positive_claims(self):
        dataset = generate_motor_bi_dataset(n_policies=1000, seed=42)
        assert (dataset["claim_amount"] > 0).all()

    def test_valid_age_bands(self):
        dataset = generate_motor_bi_dataset(n_policies=2000, seed=42)
        assert set(dataset["age_band"]).issubset({1, 2, 3, 4, 5})

    def test_valid_vehicle_groups(self):
        dataset = generate_motor_bi_dataset(n_policies=2000, seed=42)
        assert set(dataset["vehicle_group"]).issubset(set(range(1, 11)))

    def test_valid_regions(self):
        dataset = generate_motor_bi_dataset(n_policies=2000, seed=42)
        assert set(dataset["region"]).issubset(set(REGIONS))

    def test_exposure_all_ones(self):
        dataset = generate_motor_bi_dataset(n_policies=500, seed=42)
        assert np.all(dataset["exposure"] == 1.0)

    def test_claim_count_roughly_expected(self):
        """With 10k policies and 5% claim rate, expect ~500 claims."""
        dataset = generate_motor_bi_dataset(n_policies=10_000, claim_rate=0.05, seed=42)
        n_claims = len(dataset["claim_amount"])
        assert 200 < n_claims < 800, f"Expected ~500 claims, got {n_claims}"

    def test_reproducible(self):
        d1 = generate_motor_bi_dataset(n_policies=500, seed=99)
        d2 = generate_motor_bi_dataset(n_policies=500, seed=99)
        assert np.allclose(d1["claim_amount"], d2["claim_amount"])

    def test_different_seeds_different(self):
        d1 = generate_motor_bi_dataset(n_policies=500, seed=1)
        d2 = generate_motor_bi_dataset(n_policies=500, seed=2)
        # Different seeds -> different number of claims or different values
        if len(d1["claim_amount"]) == len(d2["claim_amount"]):
            assert not np.allclose(d1["claim_amount"], d2["claim_amount"])

    def test_large_dataset(self):
        dataset = generate_motor_bi_dataset(n_policies=50_000, claim_rate=0.05, seed=42)
        assert len(dataset["claim_amount"]) > 0
        assert (dataset["claim_amount"] > 0).all()


# ---------------------------------------------------------------------------
# theoretical_tvar
# ---------------------------------------------------------------------------


class TestTheoreticalTVaR:
    def test_returns_positive(self):
        tv = theoretical_tvar(0.95, 3, 5, "midlands", n_mc=50_000)
        assert tv > 0

    def test_monotone_in_p(self):
        """TVaR(0.99) > TVaR(0.95) for motor BI."""
        tv95 = theoretical_tvar(0.95, 3, 5, "midlands", n_mc=50_000, seed=42)
        tv99 = theoretical_tvar(0.99, 3, 5, "midlands", n_mc=50_000, seed=42)
        assert tv99 > tv95

    def test_young_london_higher_tvar(self):
        """Young London driver should have higher TVaR than mid-aged north driver."""
        tv_yl = theoretical_tvar(0.95, 1, 5, "london", n_mc=50_000, seed=42)
        tv_mn = theoretical_tvar(0.95, 3, 5, "north", n_mc=50_000, seed=42)
        assert tv_yl > tv_mn


# ---------------------------------------------------------------------------
# theoretical_ilf
# ---------------------------------------------------------------------------


class TestTheoreticalILF:
    def test_ilf_at_basic_is_one(self):
        basic = 50_000
        result = theoretical_ilf(
            limits=[basic],
            basic_limit=basic,
            age_band=3,
            vehicle_group=5,
            region="midlands",
            n_mc=50_000,
        )
        assert result[float(basic)] == pytest.approx(1.0, abs=0.001)

    def test_ilf_increasing(self):
        limits = [50_000, 100_000, 250_000]
        result = theoretical_ilf(
            limits=limits,
            basic_limit=50_000,
            age_band=3,
            vehicle_group=5,
            region="midlands",
            n_mc=50_000,
        )
        vals = [result[float(L)] for L in limits]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]
