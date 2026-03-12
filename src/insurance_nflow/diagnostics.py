"""
Model diagnostics — QQ plots, PIT histograms, tail comparisons, KS test.

The goal is not beautiful plots. The goal is fast, reproducible checks
that a pricing actuary can run in a notebook during model validation.

PIT (Probability Integral Transform) histogram:
    If the model is correctly specified, PIT values U = F(x) are Uniform(0,1).
    Deviations from uniformity indicate model misspecification:
    - U-shaped: tails too light (underestimates tail probability)
    - Hump-shaped: tails too heavy
    - Left skew: model overestimates right tail
    - Right skew: model underestimates right tail

QQ plot in log-scale:
    For heavy-tailed data, a standard QQ plot compresses the tail.
    We plot in log-scale for both axes so the tail is visible.

Tail index comparison:
    Compare flow's implied tail index (from TTF lambda+) against:
    - Empirical Hill estimator on test data
    - Fitted lognormal (has tail index = infinity, i.e., sub-power-law)
    - Fitted Pareto

    If flow's lambda+ is far from Hill estimate, the model is misfitting the tail.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def pit_values(
    cdf_fn,
    x: np.ndarray,
    context: Optional[np.ndarray] = None,
    n_samples: int = 10_000,
) -> np.ndarray:
    """
    Compute PIT values U = F_model(x) via Monte Carlo CDF estimation.

    For each observation x_i, we estimate P_model(X <= x_i) by drawing
    n_samples from the model and computing the empirical CDF.

    For conditional models: context[i] is the covariate row for x_i.
    Each observation gets its own conditional CDF estimate.

    Parameters
    ----------
    cdf_fn : callable
        Function mapping (x_i, context_i, n_samples) -> float or
        (x_i, n_samples) -> float for unconditional models.
    x : np.ndarray of shape (N,)
        Observed claim amounts.
    context : np.ndarray of shape (N, p) or None
    n_samples : int
        MC samples per observation. More = more accurate but slower.

    Returns
    -------
    np.ndarray of shape (N,)
        PIT values in [0, 1].
    """
    pit = np.zeros(len(x))
    for i, xi in enumerate(x):
        ctx = context[i] if context is not None else None
        pit[i] = cdf_fn(xi, ctx, n_samples)
    return pit


def ks_test(pit: np.ndarray) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov test of PIT values against Uniform(0, 1).

    Parameters
    ----------
    pit : np.ndarray
        PIT values from pit_values().

    Returns
    -------
    (statistic, p_value) : tuple[float, float]
        KS test statistic and p-value. Large p-value = no evidence of misfit.
    """
    result = stats.kstest(pit, "uniform")
    return float(result.statistic), float(result.pvalue)


def pit_histogram(
    pit: np.ndarray,
    n_bins: int = 20,
    ax: Optional["plt.Axes"] = None,
    title: str = "PIT Histogram",
) -> "plt.Figure":
    """
    Plot PIT histogram with Uniform(0,1) reference line.

    Parameters
    ----------
    pit : np.ndarray
        PIT values.
    n_bins : int
        Number of histogram bins.
    ax : matplotlib Axes or None
    title : str

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    ax.hist(pit, bins=n_bins, density=True, alpha=0.7, color="#4477AA", label="PIT")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Uniform(0,1)")

    ks_stat, ks_p = ks_test(pit)
    ax.set_xlabel("PIT value U = F(x)")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\nKS stat={ks_stat:.3f}, p={ks_p:.3f}")
    ax.legend()
    fig.tight_layout()
    return fig


def qq_plot_lognormal(
    x: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    title: str = "QQ Plot vs Lognormal",
) -> "plt.Figure":
    """
    QQ plot of observed data against fitted lognormal in log-scale.

    Useful as a baseline: if the lognormal QQ plot shows systematic deviation
    in the upper tail (which it usually does for motor BI), this motivates
    the flow model.

    Parameters
    ----------
    x : np.ndarray
        Observed claim amounts (positive).
    ax : matplotlib Axes or None
    title : str

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    log_x = np.log(x)
    mu, sigma = np.mean(log_x), np.std(log_x, ddof=1)

    n = len(x)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = np.exp(stats.norm.ppf(probs, loc=mu, scale=sigma))
    empirical = np.sort(x)

    ax.scatter(theoretical, empirical, s=3, alpha=0.4, color="#4477AA")
    ax.plot(
        [min(theoretical), max(theoretical)],
        [min(theoretical), max(theoretical)],
        "r--",
        linewidth=1.5,
        label="y = x (perfect fit)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Theoretical quantile (Lognormal)")
    ax.set_ylabel("Empirical quantile")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def qq_plot_flow(
    x_observed: np.ndarray,
    x_simulated: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    title: str = "QQ Plot: Flow vs Observed",
    n_quantiles: int = 200,
) -> "plt.Figure":
    """
    QQ plot comparing flow-simulated distribution against observed data.

    Parameters
    ----------
    x_observed : np.ndarray
        Observed claim amounts.
    x_simulated : np.ndarray
        Samples from fitted flow (large sample, e.g., 100k+).
    n_quantiles : int
        Number of quantile points to plot.
    ax, title : as above.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    probs = np.linspace(0.01, 0.999, n_quantiles)
    q_obs = np.quantile(x_observed, probs)
    q_sim = np.quantile(x_simulated, probs)

    ax.scatter(q_sim, q_obs, s=4, alpha=0.6, color="#4477AA")
    unified = np.array([min(q_sim.min(), q_obs.min()), max(q_sim.max(), q_obs.max())])
    ax.plot(unified, unified, "r--", linewidth=1.5, label="y = x (perfect fit)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Flow quantile")
    ax.set_ylabel("Observed quantile")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def tail_plot(
    x: np.ndarray,
    x_simulated: Optional[np.ndarray] = None,
    ax: Optional["plt.Axes"] = None,
    title: str = "Tail Plot (log-log)",
) -> "plt.Figure":
    """
    Log-log tail probability plot: log(1 - F(x)) vs log(x).

    For a Pareto distribution, this is a straight line with slope = -alpha.
    Useful for visually assessing whether the flow captures the power-law tail.

    Parameters
    ----------
    x : np.ndarray
        Observed claim amounts.
    x_simulated : np.ndarray or None
        Flow-simulated samples (for comparison).
    ax, title : as above.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    x_sorted = np.sort(x)
    n = len(x_sorted)
    surv = 1 - (np.arange(1, n + 1) - 0.5) / n

    ax.plot(x_sorted, surv, color="#4477AA", linewidth=1.5, label="Observed")

    if x_simulated is not None:
        xs_sorted = np.sort(x_simulated)
        ns = len(xs_sorted)
        surv_s = 1 - (np.arange(1, ns + 1) - 0.5) / ns
        ax.plot(xs_sorted, surv_s, color="#EE7733", linewidth=1.5,
                alpha=0.7, label="Flow")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Claim amount (GBP)")
    ax.set_ylabel("P(X > x)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def tail_index_comparison(
    x: np.ndarray,
    flow_lambda_pos: Optional[float] = None,
    k_range: Optional[tuple[int, int]] = None,
) -> dict:
    """
    Compare tail index estimates from multiple methods.

    Returns a summary dict for use in model comparison tables.

    Methods:
    - Hill estimator (empirical, multiple k values)
    - Fitted Pareto MLE
    - Fitted lognormal (tail index = infinity, notionally)
    - Flow TTF lambda+ (if provided)

    Parameters
    ----------
    x : np.ndarray
        Observed claim amounts (positive).
    flow_lambda_pos : float or None
        Flow's TTF right tail parameter (lambda+).
    k_range : (k_min, k_max) or None
        Range of k values for Hill estimator stability plot.

    Returns
    -------
    dict with keys: hill_median, hill_std, pareto_alpha, lognormal_tail_index,
                    flow_lambda_pos, flow_equivalent_alpha.
    """
    from .tail import hill_estimator

    x_sorted = np.sort(x)[::-1]
    n = len(x_sorted)

    if k_range is None:
        k_min = max(5, int(np.sqrt(n) / 2))
        k_max = min(n - 1, int(np.sqrt(n) * 3))
    else:
        k_min, k_max = k_range

    k_values = np.arange(k_min, k_max + 1)
    hill_vals = [hill_estimator(x_sorted, k) for k in k_values]

    # Pareto MLE: alpha_hat = n / sum(log(x/x_min))
    # Use upper 10% as Pareto approximation
    upper = x_sorted[:max(10, n // 10)]
    x_min = upper[-1]
    pareto_alpha = len(upper) / np.sum(np.log(upper / x_min))

    result = {
        "hill_median": float(np.median(hill_vals)),
        "hill_std": float(np.std(hill_vals)),
        "hill_k_range": (int(k_min), int(k_max)),
        "pareto_alpha_upper10pct": float(pareto_alpha),
        "lognormal_tail_index": float("inf"),  # sub-power-law
    }

    if flow_lambda_pos is not None:
        result["flow_lambda_pos"] = float(flow_lambda_pos)
        # TTF lambda+ approximately equals GPD shape xi = 1/alpha
        # So alpha_flow ≈ 1 / lambda+
        result["flow_equivalent_alpha"] = float(1.0 / flow_lambda_pos)

    return result


def model_comparison_table(
    x: np.ndarray,
    log_likelihoods: dict[str, float],
    n_params: dict[str, int],
) -> list[dict]:
    """
    Build AIC/BIC comparison table across models.

    Parameters
    ----------
    x : np.ndarray
        Observed claim amounts (used for n).
    log_likelihoods : dict[str, float]
        Model name -> total log-likelihood.
    n_params : dict[str, int]
        Model name -> number of parameters.

    Returns
    -------
    list of dicts, one per model, sorted by AIC ascending.
    Columns: model, n_params, log_lik, aic, bic, log_lik_per_obs.
    """
    n = len(x)
    rows = []
    for model, ll in log_likelihoods.items():
        k = n_params[model]
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll
        rows.append({
            "model": model,
            "n_params": k,
            "log_lik": float(ll),
            "log_lik_per_obs": float(ll / n),
            "aic": float(aic),
            "bic": float(bic),
        })
    return sorted(rows, key=lambda r: r["aic"])


def fit_parametric_benchmarks(x: np.ndarray) -> tuple[dict[str, float], dict[str, int]]:
    """
    Fit lognormal, gamma, and Pareto to observed data and return log-likelihoods.

    Used as parametric benchmarks in the model comparison table.

    Parameters
    ----------
    x : np.ndarray
        Positive claim amounts.

    Returns
    -------
    (log_likelihoods, n_params) : dicts keyed by model name.
    """
    log_x = np.log(x)
    n = len(x)

    # Lognormal MLE
    mu_ln = np.mean(log_x)
    sigma_ln = np.std(log_x, ddof=1)
    ll_lognormal = float(np.sum(stats.lognorm.logpdf(x, s=sigma_ln, scale=np.exp(mu_ln))))

    # Gamma MLE
    gamma_fit = stats.gamma.fit(x, floc=0)
    ll_gamma = float(np.sum(stats.gamma.logpdf(x, *gamma_fit)))

    # Pareto MLE (Lomax/2-parameter Pareto)
    # Using loc=0, scale = min(x), shape estimated
    x_min = np.min(x)
    pareto_fit = stats.pareto.fit(x, floc=0)
    ll_pareto = float(np.sum(stats.pareto.logpdf(x, *pareto_fit)))

    log_likelihoods = {
        "lognormal": ll_lognormal,
        "gamma": ll_gamma,
        "pareto": ll_pareto,
    }
    n_params = {
        "lognormal": 2,  # mu, sigma
        "gamma": 2,      # shape, scale
        "pareto": 2,     # shape, scale
    }
    return log_likelihoods, n_params
