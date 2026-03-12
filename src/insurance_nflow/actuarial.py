"""
Actuarial output functions — TVaR, ILF, LEV, reinsurance layer pricing.

These are the outputs pricing teams actually use. The flow gives us a
numerical CDF and PDF; this module uses that to compute standard actuarial
quantities via Monte Carlo integration.

Monte Carlo is preferred over quadrature here because:
1. We already have fast sampling from the flow (coupling architecture).
2. The bimodal distribution shape makes quadrature grid choice fragile.
3. MC uncertainty is quantifiable via bootstrap.
4. Users will understand MC — they run simulations routinely.

Precision note: 1M samples gives SE on TVaR(0.99) of roughly
sigma / (sqrt(1M) * 0.01 * f(q_0.99)) which is adequate for pricing.
50k samples is fine for development/diagnostics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def quantile(
    samples: np.ndarray,
    p: float,
) -> float:
    """
    Empirical quantile (VaR) from flow samples.

    Parameters
    ----------
    samples : np.ndarray
        Samples from the fitted flow (positive values).
    p : float
        Probability level (e.g., 0.99 for 99th percentile).

    Returns
    -------
    float
        Estimated quantile.
    """
    return float(np.quantile(samples, p))


def tvar(
    samples: np.ndarray,
    p: float,
) -> float:
    """
    Tail Value at Risk (TVaR) / Expected Shortfall from flow samples.

    TVaR(p) = E[X | X > VaR(p)] = E[X * 1(X > q_p)] / (1 - p)

    Also known as Conditional Tail Expectation (CTE) or Expected Shortfall (ES).

    Parameters
    ----------
    samples : np.ndarray
        Samples from the fitted flow.
    p : float
        Probability level (e.g., 0.99).

    Returns
    -------
    float
        TVaR estimate.
    """
    q = np.quantile(samples, p)
    tail_samples = samples[samples > q]
    if len(tail_samples) == 0:
        return float(q)
    return float(np.mean(tail_samples))


def limited_expected_value(
    samples: np.ndarray,
    limit: float,
) -> float:
    """
    Limited Expected Value (LEV) at limit u.

    LEV(u) = E[min(X, u)] = integral_0^u P(X > x) dx
           = E[X] - E[(X - u)_+]

    The ground-up expected value capped at u.

    Parameters
    ----------
    samples : np.ndarray
        Samples from the fitted flow.
    limit : float
        Policy limit (pounds).

    Returns
    -------
    float
        Limited expected value estimate.
    """
    return float(np.mean(np.minimum(samples, limit)))


def excess_expected_value(
    samples: np.ndarray,
    deductible: float,
) -> float:
    """
    Expected value excess of deductible d (pure premium above d).

    E[(X - d)_+] = E[X] - LEV(d)

    Parameters
    ----------
    samples : np.ndarray
        Samples from the fitted flow.
    deductible : float
        Deductible / attachment point (pounds).

    Returns
    -------
    float
        Expected excess value.
    """
    return float(np.mean(np.maximum(samples - deductible, 0.0)))


def ilf(
    samples: np.ndarray,
    limits: list[float],
    basic_limit: float,
) -> dict[float, float]:
    """
    Increased Limit Factors (ILF) relative to a basic limit.

    ILF(L) = LEV(L) / LEV(basic_limit)

    ILF is the factor by which the basic limit premium must be multiplied
    to price coverage up to limit L. ILF(basic_limit) = 1.0 by definition.

    The ILF computed from a flow naturally captures:
    - Bimodal body (different body/tail contributions at each limit)
    - Heavy-tail behaviour via TTF

    Parameters
    ----------
    samples : np.ndarray
        Samples from the fitted flow.
    limits : list[float]
        Policy limits to compute ILF for (pounds).
    basic_limit : float
        Reference limit. Typically the lowest standard policy limit
        (e.g., GBP 50,000 for motor BI).

    Returns
    -------
    dict mapping limit -> ILF value.
    """
    lev_basic = limited_expected_value(samples, basic_limit)
    if lev_basic <= 0:
        raise ValueError(
            f"LEV at basic_limit={basic_limit:,.0f} is zero or negative. "
            "Check that samples are positive and basic_limit is sensible."
        )
    result = {}
    for lim in limits:
        lev_lim = limited_expected_value(samples, lim)
        result[float(lim)] = float(lev_lim / lev_basic)
    return result


def reinsurance_layer_cost(
    samples: np.ndarray,
    attachment: float,
    limit: float,
) -> float:
    """
    Expected cost to a reinsurance layer (xs of attachment, up to limit).

    Layer cost = E[min(X - attachment, limit)_+]
               = E[(X - attachment)_+] - E[(X - attachment - limit)_+]
               = LEV(attachment + limit) - LEV(attachment)  (by convention)

    More precisely:
        Layer cost = E[min((X - attachment)_+, limit)]

    This is the pure premium for a per-occurrence XL reinsurance layer
    written as `limit xs attachment`.

    Parameters
    ----------
    samples : np.ndarray
        Samples from the fitted flow.
    attachment : float
        Attachment point (pounds). Layer pays above this.
    limit : float
        Layer limit (pounds). Layer pays up to attachment + limit.

    Returns
    -------
    float
        Expected layer loss per occurrence.
    """
    layer_losses = np.minimum(np.maximum(samples - attachment, 0.0), limit)
    return float(np.mean(layer_losses))


def layer_loss_ratio(
    samples: np.ndarray,
    attachment: float,
    limit: float,
    layer_premium: float,
) -> float:
    """
    Expected loss ratio for a reinsurance layer.

    Parameters
    ----------
    samples : np.ndarray
        Samples from the fitted flow.
    attachment : float
        Attachment point.
    limit : float
        Layer limit.
    layer_premium : float
        Ceded premium for this layer.

    Returns
    -------
    float
        Expected loss ratio (layer cost / layer premium).
    """
    cost = reinsurance_layer_cost(samples, attachment, limit)
    return cost / layer_premium


def burning_cost_summary(
    samples: np.ndarray,
    limits: Optional[list[float]] = None,
    basic_limit: float = 50_000.0,
    tvar_levels: Optional[list[float]] = None,
) -> dict:
    """
    Comprehensive actuarial summary from flow samples.

    Computes the standard outputs a UK motor pricing team would request
    after fitting a severity model.

    Parameters
    ----------
    samples : np.ndarray
        Flow samples.
    limits : list[float]
        Policy limits for ILF curve. Defaults to standard UK motor limits.
    basic_limit : float
        Basic limit for ILF. Default GBP 50,000.
    tvar_levels : list[float]
        TVaR probability levels to compute. Defaults to [0.90, 0.95, 0.99].

    Returns
    -------
    dict with keys:
        mean, median, std, skewness,
        quantiles (dict p -> value),
        tvar (dict p -> value),
        ilf (dict limit -> factor),
        n_samples.
    """
    if limits is None:
        limits = [50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000]
    if tvar_levels is None:
        tvar_levels = [0.90, 0.95, 0.99]

    quantile_levels = [0.5, 0.75, 0.90, 0.95, 0.99, 0.995]

    return {
        "n_samples": len(samples),
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "std": float(np.std(samples)),
        "skewness": float(_skewness(samples)),
        "quantiles": {
            p: float(np.quantile(samples, p)) for p in quantile_levels
        },
        "tvar": {
            p: tvar(samples, p) for p in tvar_levels
        },
        "ilf": ilf(samples, limits=limits, basic_limit=basic_limit),
    }


def _skewness(x: np.ndarray) -> float:
    """Sample skewness (Fisher-Pearson coefficient)."""
    n = len(x)
    if n < 3:
        return float("nan")
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)
    if sigma == 0:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3) * n * (n - 1) / (n - 2))
