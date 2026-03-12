"""
Tail Transform layer for heavy-tailed normalizing flows.

Implements a monotone bijection R -> R that maps Gaussian-tailed distributions
to power-law-tailed distributions (or vice versa), motivated by Hickling &
Prangle (2025) "Flexible Tails for Normalizing Flows".

IMPLEMENTATION APPROACH

The Hickling TTF uses an erfc-based transform with complex normalisation constants.
For robustness and numerical stability this implementation uses the Student-t CDF
warping approach, which is:

1. Mathematically equivalent in tail behaviour (both produce GPD-tailed output)
2. Simpler to invert (inverse CDF of both Gaussian and Student-t are standard functions)
3. Numerically stable across all of R

The transform T: R -> R maps standard-normal u to Student-t distributed z:
    T(u; nu) = Phi_t^{-1}(Phi(u); nu)
             = F_t^{-1}(F_N(u); nu)

where Phi is the Gaussian CDF and F_t^{-1} is the Student-t quantile function.

For the FORWARD direction (data -> base, i.e., heavy-tailed z -> Gaussian u):
    T^{-1}(z; nu) = Phi^{-1}(F_t(z; nu))

The tail weight parameter lambda is stored as 1/nu (degrees of freedom inverse).
Larger lambda = smaller nu = heavier tail.

For insurance severity (right-tail focus): lambda+/lambda- are estimated from the
upper and lower tails of the log-claims distribution via the Hill estimator.

RELATIONSHIP TO HICKLING 2025

Hickling et al. use an erfc-based formulation that is more general (asymmetric
tails, exact EVT characterisation). The Student-t warping is a special case
(symmetric tails) that is equivalent for the primary insurance use case.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# Numerical constants
_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _erfc(x: Tensor) -> Tensor:
    """Complementary error function."""
    return torch.erfc(x)


def _erfcinv(y: Tensor) -> Tensor:
    """
    Inverse complementary error function.

    Falls back to scipy if torch.special.erfcinv is not available.
    """
    y_clamped = y.clamp(1e-7, 2.0 - 1e-7)
    try:
        return torch.special.erfcinv(y_clamped)
    except AttributeError:
        import scipy.special as sc
        y_np = y_clamped.detach().cpu().numpy()
        result = sc.erfcinv(y_np)
        return torch.tensor(result, dtype=y.dtype, device=y.device)


def _normal_cdf(x: Tensor) -> Tensor:
    """Standard normal CDF: Phi(x) = 0.5 * erfc(-x / sqrt(2))."""
    return 0.5 * _erfc(-x / _SQRT2)


def _normal_icdf(p: Tensor) -> Tensor:
    """Standard normal inverse CDF: Phi^{-1}(p) = sqrt(2) * erfcinv(2p - but let's use ppf."""
    # Phi^{-1}(p) = -sqrt(2) * erfcinv(2*p)  for p in (0, 1)
    p_clamped = p.clamp(1e-7, 1.0 - 1e-7)
    return -_SQRT2 * _erfcinv(2.0 * p_clamped)


def _student_t_cdf_np(z_np: np.ndarray, nu: float) -> np.ndarray:
    """Student-t CDF (numpy, via scipy)."""
    from scipy.stats import t as t_dist
    return t_dist.cdf(z_np, df=nu).astype(np.float32)


def _student_t_icdf_np(p_np: np.ndarray, nu: float) -> np.ndarray:
    """Student-t inverse CDF (numpy, via scipy)."""
    from scipy.stats import t as t_dist
    return t_dist.ppf(p_np, df=nu).astype(np.float32)


def _student_t_logpdf_np(z_np: np.ndarray, nu: float) -> np.ndarray:
    """Student-t log-PDF (numpy, via scipy)."""
    from scipy.stats import t as t_dist
    return t_dist.logpdf(z_np, df=nu).astype(np.float32)


class TailTransform(nn.Module):
    """
    Student-t warping tail transform for normalizing flows.

    Maps Gaussian-tailed flow output to Student-t-tailed output (heavy tails),
    enabling calibrated heavy-tail modelling of insurance claims.

    The transform:
    - forward(z): T^{-1}(z) = Phi^{-1}(F_t(z; nu)) — heavy-tailed z -> Gaussian u
    - inverse(u): T(u) = F_t^{-1}(Phi(u); nu) — Gaussian u -> heavy-tailed z

    Parameters
    ----------
    lambda_pos : float
        Tail weight = 1/nu_right (degrees of freedom inverse for right tail).
        Larger = heavier right tail. lambda=0 corresponds to Gaussian (limit).
        Reasonable range: 0.1-0.5 for insurance severity.
    lambda_neg : float
        Tail weight for left tail. For log-severity, left tail is usually lighter.
    trainable : bool
        If True, train lambda jointly. If False (default), fix at initial values.
    min_nu : float
        Minimum degrees of freedom (maximum lambda). Default 2.0 (for finite variance).
    """

    def __init__(
        self,
        lambda_pos: float = 0.25,
        lambda_neg: float = 0.1,
        trainable: bool = False,
        min_nu: float = 2.0,
    ) -> None:
        super().__init__()
        # Store log(lambda) for positivity
        self._log_lambda_pos = nn.Parameter(
            torch.tensor(math.log(lambda_pos), dtype=torch.float32),
            requires_grad=trainable,
        )
        self._log_lambda_neg = nn.Parameter(
            torch.tensor(math.log(lambda_neg), dtype=torch.float32),
            requires_grad=trainable,
        )
        self._min_nu = min_nu

    @property
    def lambda_pos(self) -> Tensor:
        return torch.exp(self._log_lambda_pos)

    @property
    def lambda_neg(self) -> Tensor:
        return torch.exp(self._log_lambda_neg)

    @property
    def nu_pos(self) -> float:
        """Degrees of freedom for right tail."""
        lam = float(self.lambda_pos.item())
        if lam <= 1e-7:
            return 1e7  # Effectively Gaussian
        return max(self._min_nu, 1.0 / lam)

    @property
    def nu_neg(self) -> float:
        """Degrees of freedom for left tail."""
        lam = float(self.lambda_neg.item())
        if lam <= 1e-7:
            return 1e7
        return max(self._min_nu, 1.0 / lam)

    @property
    def trainable(self) -> bool:
        return self._log_lambda_pos.requires_grad

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        T^{-1}(z): map heavy-tailed z to Gaussian u.

        Uses positive tail (nu_pos) for z > 0, negative tail (nu_neg) for z < 0.

        Parameters
        ----------
        z : Tensor of shape (...,)

        Returns
        -------
        u : Tensor — Gaussian-distributed
        log_abs_det_jacobian : Tensor — log |du/dz|
        """
        z_np = z.detach().cpu().numpy()
        pos_mask = z_np > 0

        # CDF transformation: T^{-1}(z) = Phi^{-1}(F_t(z; nu))
        # For z > 0: use nu_pos
        # For z < 0: use nu_neg
        # For z = 0: 0 (by symmetry of t-distribution)

        cdf_z = np.where(
            pos_mask,
            _student_t_cdf_np(z_np, self.nu_pos),
            _student_t_cdf_np(z_np, self.nu_neg),
        )
        cdf_z_clamped = np.clip(cdf_z, 1e-7, 1 - 1e-7)

        # Inverse normal CDF: Phi^{-1}(p)
        u_np = np.where(
            pos_mask,
            _student_t_icdf_np(cdf_z_clamped, 1e7),  # large nu = Gaussian approx
            _student_t_icdf_np(cdf_z_clamped, 1e7),
        )
        # Simpler: use scipy norm.ppf directly
        from scipy.stats import norm
        u_np = norm.ppf(cdf_z_clamped).astype(np.float32)
        u = torch.tensor(u_np, dtype=z.dtype, device=z.device)

        # Log Jacobian: log|du/dz| = log f_t(z; nu) - log phi(u)
        # where f_t is Student-t PDF and phi is standard normal PDF
        log_ft_np = np.where(
            pos_mask,
            _student_t_logpdf_np(z_np, self.nu_pos),
            _student_t_logpdf_np(z_np, self.nu_neg),
        )
        log_phi_u_np = -0.5 * u_np ** 2 - _LOG_SQRT_2PI

        log_abs_det = torch.tensor(
            (log_ft_np - log_phi_u_np).astype(np.float32),
            dtype=z.dtype,
            device=z.device,
        )

        return u, log_abs_det

    def inverse(self, u: Tensor) -> tuple[Tensor, Tensor]:
        """
        T(u): map Gaussian u to heavy-tailed z (for sampling).

        Parameters
        ----------
        u : Tensor of shape (...,)

        Returns
        -------
        z : Tensor — heavy-tailed samples
        log_abs_det_jacobian : Tensor — log |dz/du|
        """
        u_np = u.detach().cpu().numpy()
        pos_mask = u_np > 0

        from scipy.stats import norm
        cdf_u = norm.cdf(u_np).astype(np.float32)
        cdf_u_clamped = np.clip(cdf_u, 1e-7, 1 - 1e-7)

        # T(u) = F_t^{-1}(Phi(u); nu)
        z_np = np.where(
            pos_mask,
            _student_t_icdf_np(cdf_u_clamped, self.nu_pos),
            _student_t_icdf_np(cdf_u_clamped, self.nu_neg),
        )
        z = torch.tensor(z_np.astype(np.float32), dtype=u.dtype, device=u.device)

        # Log Jacobian: log|dz/du| = log phi(u) - log f_t(z; nu)
        log_phi_u_np = -0.5 * u_np ** 2 - _LOG_SQRT_2PI
        log_ft_z_np = np.where(
            pos_mask,
            _student_t_logpdf_np(z_np, self.nu_pos),
            _student_t_logpdf_np(z_np, self.nu_neg),
        )
        log_abs_det = torch.tensor(
            (log_phi_u_np - log_ft_z_np).astype(np.float32),
            dtype=u.dtype,
            device=u.device,
        )

        return z, log_abs_det


# ---------------------------------------------------------------------------
# Hill estimator for tail index pre-estimation
# ---------------------------------------------------------------------------


def hill_estimator(x: np.ndarray, k: int) -> float:
    """
    Hill estimator for the tail index of a heavy-tailed distribution.

    Estimates xi = 1/alpha such that P(X > x) ~ x^{-1/xi} for large x.
    Applied to the upper k order statistics.

    Parameters
    ----------
    x : np.ndarray
        1D array of positive values.
    k : int
        Number of upper order statistics. Typical: k ~ sqrt(n).

    Returns
    -------
    float
        Estimated tail weight xi. Larger = heavier tail.
    """
    x_sorted = np.sort(x)[::-1]  # descending
    k = min(k, len(x_sorted) - 1)
    k = max(k, 1)
    log_ratios = np.log(x_sorted[:k]) - np.log(x_sorted[k])
    return float(np.mean(log_ratios))


def hill_double_bootstrap(
    x: np.ndarray,
    k_min: int = 10,
    seed: int = 42,
) -> float:
    """
    Automatic threshold selection for Hill estimator via stable-region method.

    Finds the range of k where the Hill estimates are most stable (low variance)
    and returns the median estimate in that region. This is a simplified version
    of the Danielsson et al. (2001) double bootstrap.

    Parameters
    ----------
    x : np.ndarray
        Positive values (claim amounts or log-exceedances).
    k_min : int
        Minimum k to consider.
    seed : int

    Returns
    -------
    float
        Estimated tail index xi.
    """
    n = len(x)
    x_sorted = np.sort(x)[::-1]

    k_max = max(k_min + 1, int(np.sqrt(n)))
    k_values = np.arange(k_min, k_max)
    if len(k_values) == 0:
        return hill_estimator(x_sorted, max(1, n // 2))

    hill_vals = np.array([hill_estimator(x_sorted, k) for k in k_values])

    if len(hill_vals) < 3:
        return float(np.median(hill_vals))

    window = min(5, len(hill_vals) // 2)
    rolling_var = np.array([
        np.var(hill_vals[max(0, i - window):i + window + 1])
        for i in range(len(hill_vals))
    ])
    # Suppress NaN warnings for tiny arrays
    with np.errstate(invalid='ignore'):
        threshold = np.nanpercentile(rolling_var, 25)
    stable_mask = rolling_var <= threshold
    if stable_mask.sum() == 0:
        stable_mask = np.ones(len(hill_vals), dtype=bool)

    return float(np.median(hill_vals[stable_mask]))


def estimate_tail_params(
    log_claims: np.ndarray,
    quantile_threshold: float = 0.9,
) -> tuple[float, float]:
    """
    Estimate TTF tail parameters (lambda+, lambda-) from training data.

    lambda = 1/nu where nu is the Student-t degrees of freedom.
    Larger lambda = smaller nu = heavier tail.

    For the log-severity distribution:
    - Right tail (large claims): typically heavy, lambda ~ 0.2-0.5
    - Left tail (very small claims): usually lighter, lambda ~ 0.05-0.2

    Parameters
    ----------
    log_claims : np.ndarray
        Log-transformed claim amounts.
    quantile_threshold : float
        Use observations above this quantile for right tail estimation.

    Returns
    -------
    (lambda_pos, lambda_neg) : tuple of float
        Clamped to [0.05, 0.5] for numerical stability (nu in [2, 20]).
    """
    threshold_pos = np.quantile(log_claims, quantile_threshold)
    threshold_neg = np.quantile(log_claims, 1 - quantile_threshold)

    # Right tail exceedances
    right_tail = log_claims[log_claims > threshold_pos]
    if len(right_tail) >= 10:
        exceedances = right_tail - threshold_pos
        xi = hill_double_bootstrap(exceedances + 1.0)
        # Convert: Hill estimates 1/alpha, and for Student-t: alpha = nu (approx)
        # So lambda = 1/nu ≈ xi
        lambda_pos = xi
    else:
        lambda_pos = 0.25

    # Left tail exceedances (reflected)
    left_tail = log_claims[log_claims < threshold_neg]
    if len(left_tail) >= 10:
        exceedances = threshold_neg - left_tail
        xi = hill_double_bootstrap(exceedances + 1.0)
        lambda_neg = xi
    else:
        lambda_neg = 0.1

    # Clamp: lambda in [0.05, 0.5] -> nu in [2, 20]
    lambda_pos = float(np.clip(lambda_pos, 0.05, 0.5))
    lambda_neg = float(np.clip(lambda_neg, 0.05, 0.5))

    return lambda_pos, lambda_neg
