"""
Synthetic severity data generator for testing and validation.

The data generating process (DGP) mimics UK motor bodily injury severity:
- Bimodal: soft-tissue claims peak around GBP 5,000, catastrophic above GBP 100,000
- Heavy right tail: power-law behaviour for catastrophic claims
- Rating factor conditioning: age_band (younger = higher mean/tail),
  vehicle_group (higher group = higher severity), region (London premium)

The DGP is fully specified so test results are reproducible and tail indices
are known ground-truth for Hill estimator validation.

Design: implement as a mixture model rather than a normalizing flow. This is
deliberate — using a parametric DGP to validate a nonparametric flow avoids
the circular validation problem.

Motor BI bimodal DGP:
    P(catastrophic) = p_cat(X)  # function of rating factors
    If not catastrophic: severity ~ LogNormal(mu_soft(X), sigma_soft)
    If catastrophic: severity ~ Pareto(alpha=1.5, scale=mu_cat(X))

This gives a known TVaR, ILF, and tail index at each covariate value —
useful for checking flow outputs are in the right ballpark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DGPParams:
    """
    Parameters for the bimodal severity DGP.

    All monetary values in GBP. Calibrated to approximate UK motor BI
    severity distributions as of 2024.
    """

    # Soft-tissue component (lognormal)
    mu_soft_base: float = 8.5        # log(~5,000 GBP) = 8.52
    sigma_soft: float = 0.8          # log-scale SD

    # Catastrophic component (Pareto)
    alpha_cat: float = 1.5           # Pareto shape. Power-law exponent.
    scale_cat_base: float = 80_000.0  # Pareto location (min of catastrophic range)

    # Probability of catastrophic claim (baseline, no rating factors)
    p_cat_base: float = 0.03

    # Rating factor effects (additive on logit scale for p_cat, on log scale for mu_soft)
    age_band_effects: dict[int, float] = None  # age_band (1-5) -> effect
    vehicle_group_effects: dict[int, float] = None  # vehicle_group (1-10) -> effect
    region_effects: dict[str, float] = None  # region code -> effect

    def __post_init__(self) -> None:
        if self.age_band_effects is None:
            # Age band 1 (17-25) has highest severity; age band 5 (65+) also elevated
            self.age_band_effects = {
                1: 0.4,   # 17-25: higher catastrophic p and higher soft-tissue mean
                2: 0.1,   # 26-35
                3: 0.0,   # 36-50 (baseline)
                4: 0.05,  # 51-65
                5: 0.2,   # 65+: higher severity due to slower healing
            }
        if self.vehicle_group_effects is None:
            # Higher vehicle group = more expensive = slightly higher severity
            self.vehicle_group_effects = {
                k: (k - 5) * 0.05 for k in range(1, 11)
            }
        if self.region_effects is None:
            self.region_effects = {
                "london": 0.3,
                "south_east": 0.15,
                "midlands": 0.0,
                "north": -0.05,
                "scotland": -0.1,
            }


# Default DGP parameters
DEFAULT_DGP = DGPParams()

# Available regions for synthetic data
REGIONS = ["london", "south_east", "midlands", "north", "scotland"]


def _logistic(x: float) -> float:
    """Stable logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def _p_cat(
    age_band: int,
    vehicle_group: int,
    region: str,
    dgp: DGPParams = DEFAULT_DGP,
) -> float:
    """Probability of catastrophic claim for given rating factors."""
    logit_base = np.log(dgp.p_cat_base / (1 - dgp.p_cat_base))
    logit = (
        logit_base
        + dgp.age_band_effects.get(age_band, 0.0)
        + dgp.vehicle_group_effects.get(vehicle_group, 0.0)
        + dgp.region_effects.get(region, 0.0)
    )
    return _logistic(logit)


def _mu_soft(
    age_band: int,
    vehicle_group: int,
    region: str,
    dgp: DGPParams = DEFAULT_DGP,
) -> float:
    """Log-mean of soft-tissue component."""
    return (
        dgp.mu_soft_base
        + 0.3 * dgp.age_band_effects.get(age_band, 0.0)
        + 0.1 * dgp.vehicle_group_effects.get(vehicle_group, 0.0)
        + 0.2 * dgp.region_effects.get(region, 0.0)
    )


def sample_severity(
    age_band: int,
    vehicle_group: int,
    region: str,
    n: int,
    dgp: DGPParams = DEFAULT_DGP,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample claim severities from the bimodal DGP.

    Parameters
    ----------
    age_band : int in 1-5
    vehicle_group : int in 1-10
    region : str
    n : int
        Number of samples.
    dgp : DGPParams
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray of shape (n,)
        Claim amounts in GBP.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    p_c = _p_cat(age_band, vehicle_group, region, dgp)
    mu_s = _mu_soft(age_band, vehicle_group, region, dgp)

    is_catastrophic = rng.random(n) < p_c

    # Soft-tissue component
    soft = rng.lognormal(mean=mu_s, sigma=dgp.sigma_soft, size=n)

    # Catastrophic component: Pareto distribution
    # P(X > x) = (scale / x)^alpha for x >= scale
    # Sample: X = scale * U^{-1/alpha} where U ~ Uniform(0, 1)
    u = rng.random(n)
    cat = dgp.scale_cat_base * (u ** (-1.0 / dgp.alpha_cat))

    return np.where(is_catastrophic, cat, soft)


def generate_motor_bi_dataset(
    n_policies: int = 10_000,
    claim_rate: float = 0.05,  # 5% of policies have a claim
    dgp: DGPParams = DEFAULT_DGP,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Generate a synthetic UK motor BI dataset for testing.

    Creates a policy-level dataset and samples claims from the bimodal DGP.
    Only claimed policies are included in the output.

    Parameters
    ----------
    n_policies : int
        Total number of policies (earned).
    claim_rate : float
        Probability each policy has at least one BI claim.
    dgp : DGPParams
        Data generating process parameters.
    seed : int
        Random seed.

    Returns
    -------
    dict with arrays:
        - claim_amount: float array of claim severities
        - age_band: int array (1-5)
        - vehicle_group: int array (1-10)
        - region: str array
        - exposure: float array (earned exposure, all 1.0 in this DGP)

    All arrays have length = number of claims (not policies).
    """
    rng = np.random.default_rng(seed)

    # Generate policy population
    n_claims = rng.binomial(n_policies, claim_rate)
    age_bands = rng.choice([1, 2, 3, 4, 5], size=n_claims, p=[0.15, 0.25, 0.30, 0.20, 0.10])
    vehicle_groups = rng.choice(range(1, 11), size=n_claims)
    regions = rng.choice(REGIONS, size=n_claims, p=[0.20, 0.20, 0.25, 0.25, 0.10])

    # Sample severity for each claim from its own rating factor combination
    severities = np.zeros(n_claims)
    for i in range(n_claims):
        severities[i] = sample_severity(
            age_band=int(age_bands[i]),
            vehicle_group=int(vehicle_groups[i]),
            region=str(regions[i]),
            n=1,
            dgp=dgp,
            rng=rng,
        )[0]

    return {
        "claim_amount": severities,
        "age_band": age_bands,
        "vehicle_group": vehicle_groups,
        "region": regions,
        "exposure": np.ones(n_claims),
    }


def theoretical_tvar(
    p: float,
    age_band: int,
    vehicle_group: int,
    region: str,
    dgp: DGPParams = DEFAULT_DGP,
    n_mc: int = 500_000,
    seed: int = 42,
) -> float:
    """
    Theoretical TVaR from the DGP via Monte Carlo.

    Provides ground truth for validating flow TVaR estimates.
    Uses a large sample to minimise MC error.

    Parameters
    ----------
    p : float
        Probability level.
    age_band, vehicle_group, region : rating factors.
    dgp : DGPParams
    n_mc : int
        MC sample size. 500k gives SE < 1% on TVaR(0.99).
    seed : int

    Returns
    -------
    float
        TVaR estimate from DGP.
    """
    from .actuarial import tvar as _tvar
    samples = sample_severity(age_band, vehicle_group, region, n_mc, dgp, np.random.default_rng(seed))
    return _tvar(samples, p)


def theoretical_ilf(
    limits: list[float],
    basic_limit: float,
    age_band: int,
    vehicle_group: int,
    region: str,
    dgp: DGPParams = DEFAULT_DGP,
    n_mc: int = 500_000,
    seed: int = 42,
) -> dict[float, float]:
    """
    Theoretical ILF from the DGP via Monte Carlo.

    Parameters
    ----------
    limits : list[float]
        Policy limits.
    basic_limit : float
        Reference limit.
    age_band, vehicle_group, region : rating factors.
    dgp, n_mc, seed : as above.

    Returns
    -------
    dict mapping limit -> ILF.
    """
    from .actuarial import ilf as _ilf
    samples = sample_severity(age_band, vehicle_group, region, n_mc, dgp, np.random.default_rng(seed))
    return _ilf(samples, limits=limits, basic_limit=basic_limit)
