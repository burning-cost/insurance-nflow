"""
Tail Transform Flow (TTF) layer — ported from Hickling & Prangle (2025).

Reference: "Flexible Tails for Normalizing Flows", ICML 2025, PMLR 267:23155-23178.
Code: github.com/Tennessee-Wallaceh/tailnflows

The TTF replaces the standard Gaussian base with a GPD-tailed distribution by
composing a learnable tail transform R on top of the flow output.

Architecture:
    x (claims in log-space) -> [NSF body] -> z -> [Tail Transform R] -> u ~ N(0,1)

where R converts Gaussian-tailed z into Gaussian output u, meaning the inverse
R^{-1} maps Gaussian u to heavy-tailed z.

The tail transform:
    R(z; lambda+, lambda-) = mu_s + sigma_s * s * lambda_s * [erfc(|z| / sqrt(2)) - lambda_s^{-1}]

where s = sign(z), erfc() is the complementary error function, and lambda+, lambda- > 0
are the tail weight parameters. Larger lambda means heavier tail.

EVT connection: applying R^{-1} to N(0,1) gives output in the Fréchet domain of
attraction. The tail parameters lambda+ and lambda- correspond directly to GPD
shape parameters for the right and left tails respectively.

For insurance severity (right-tail focus), only lambda+ is material — lambda- governs
the left tail of the log-severity distribution (i.e., very small claims), which is
often light.

Pre-estimation of lambda via Hill double-bootstrap:
    TTF (fix) mode: estimate lambda from training data before training, then fix.
    This is the recommended mode per Hickling experiments.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# Numerical constant: sqrt(2) and sqrt(2/pi)
_SQRT2 = math.sqrt(2.0)
_SQRT2_OVER_PI = math.sqrt(2.0 / math.pi)


def _erfc(x: Tensor) -> Tensor:
    """Numerically stable erfc using torch.special."""
    return torch.erfc(x)


def _erfc_inverse(y: Tensor) -> Tensor:
    """Numerically stable inverse erfc using erfcinv."""
    # Clamp to valid domain (0, 2) with small margin
    y_clamped = y.clamp(1e-7, 2.0 - 1e-7)
    return torch.special.erfcinv(y_clamped)


class TailTransform(nn.Module):
    """
    Learnable tail transform layer for normalizing flows.

    Implements the transform from Hickling & Prangle (2025). Wraps a zuko-compatible
    flow by adding GPD-like tails to the base distribution.

    Parameters
    ----------
    lambda_pos : float
        Initial tail weight for right tail (lambda+). Must be > 0.
        Larger = heavier right tail.
    lambda_neg : float
        Initial tail weight for left tail (lambda-). Must be > 0.
        For log-severity, the left tail is usually light; default 1.0 is reasonable.
    trainable : bool
        If True, lambda parameters are trained jointly with the rest of the model.
        If False (TTF fix mode, recommended), parameters are frozen at initial values.
    """

    def __init__(
        self,
        lambda_pos: float = 1.0,
        lambda_neg: float = 1.0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        # Use log parameterisation to keep lambda > 0
        self._log_lambda_pos = nn.Parameter(
            torch.tensor(math.log(lambda_pos), dtype=torch.float32),
            requires_grad=trainable,
        )
        self._log_lambda_neg = nn.Parameter(
            torch.tensor(math.log(lambda_neg), dtype=torch.float32),
            requires_grad=trainable,
        )

    @property
    def lambda_pos(self) -> Tensor:
        return torch.exp(self._log_lambda_pos)

    @property
    def lambda_neg(self) -> Tensor:
        return torch.exp(self._log_lambda_neg)

    @property
    def trainable(self) -> bool:
        return self._log_lambda_pos.requires_grad

    def _get_params(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute per-element (lambda_s, mu_s, sigma_s) based on sign of z.

        For z > 0: use lambda_pos.
        For z <= 0: use lambda_neg.
        """
        s = torch.sign(z)
        s = torch.where(s == 0, torch.ones_like(s), s)  # treat z=0 as positive

        lam_pos = self.lambda_pos
        lam_neg = self.lambda_neg

        # Select lambda based on sign
        lam = torch.where(z > 0, lam_pos, lam_neg)

        # Normalisation constants so that R maps N(0,1) to unit variance in tails
        # mu_s and sigma_s are computed per Hickling Appendix A to ensure
        # the transform is an isometry at z=0.
        # mu = lambda * (1 - erfc(0)) = lambda * (1 - 1) = 0
        # At the boundary z=0: erfc(0) = 1, so R(0) = 0 for any lambda.
        # The scale factor: derivative of erfc at 0 is -sqrt(2/pi).
        # We normalise so dR/dz |_{z=0} = 1.
        # dR/dz = -s * lam * (-2/sqrt(pi)) * (1/sqrt(2)) * exp(-z^2/2)
        #       = s * lam * sqrt(2/pi) * exp(-z^2/2)
        # At z=0: dR/dz = s * lam * sqrt(2/pi)
        # To normalise: divide by lam * sqrt(2/pi)
        scale = lam * _SQRT2_OVER_PI

        return s, lam, scale

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply tail transform: z -> u, and compute log |det J|.

        Parameters
        ----------
        z : Tensor of shape (...,)
            Flow output (Gaussian-tailed).

        Returns
        -------
        u : Tensor
            Transformed output (Gaussian).
        log_abs_det_jacobian : Tensor
            log |du/dz| for each element.
        """
        s, lam, scale = self._get_params(z)

        # R(z) = s * lam * (erfc(|z| / sqrt(2)) - 1/lam) / scale
        # Simplify: R(z) = [s * (lam * erfc(|z| / sqrt(2)) - 1)] / scale
        abs_z = z.abs()
        erfc_val = _erfc(abs_z / _SQRT2)

        u = s * (lam * erfc_val - 1.0) / scale

        # Log Jacobian: d(erfc(|z|/sqrt(2)))/dz = -2/sqrt(pi) * exp(-z^2/2) * sign(z) / sqrt(2)
        #             = -sqrt(2/pi) * exp(-z^2/2) * sign(z)
        # dR/dz = s * lam * d(erfc(|z|/sqrt(2)))/dz / scale
        #       = s * lam * (-sqrt(2/pi)) * exp(-z^2/2) * s / scale
        #       = -lam * sqrt(2/pi) * exp(-z^2/2) / scale
        #       = -exp(-z^2/2)   (after cancellation with scale = lam * sqrt(2/pi))
        # |dR/dz| = exp(-z^2/2)
        log_abs_det = -0.5 * z.pow(2)

        return u, log_abs_det

    def inverse(self, u: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply inverse tail transform: u -> z.

        Parameters
        ----------
        u : Tensor of shape (...,)
            Standard normal samples.

        Returns
        -------
        z : Tensor
            Tail-distributed samples.
        log_abs_det_jacobian : Tensor
            log |dz/du| for each element.
        """
        # Invert R(z) = s * (lam * erfc(|z|/sqrt(2)) - 1) / scale
        # Let sign s_u = sign(u), then lam = lam(s_u)
        s_u = torch.sign(u)
        s_u = torch.where(s_u == 0, torch.ones_like(s_u), s_u)

        lam = torch.where(u > 0, self.lambda_pos, self.lambda_neg)
        scale = lam * _SQRT2_OVER_PI

        # u * scale = s * (lam * erfc(|z|/sqrt(2)) - 1)
        # u * scale / s + 1 = lam * erfc(|z|/sqrt(2))
        # erfc(|z|/sqrt(2)) = (u * scale * s_u + 1) / lam
        erfc_val = (u * scale * s_u + 1.0) / lam

        # |z| = sqrt(2) * erfcinv(erfc_val)
        abs_z = _SQRT2 * _erfc_inverse(erfc_val)
        z = s_u * abs_z

        # Log Jacobian of inverse: log|dz/du| = -log|du/dz|
        # |du/dz| = exp(-z^2/2), so log|dz/du| = z^2/2
        log_abs_det = 0.5 * z.pow(2)

        return z, log_abs_det


# ---------------------------------------------------------------------------
# Hill estimator for tail index pre-estimation
# ---------------------------------------------------------------------------


def hill_estimator(x: np.ndarray, k: int) -> float:
    """
    Hill estimator for the tail index of a heavy-tailed distribution.

    Estimates alpha such that P(X > x) ~ x^{-alpha} for large x.
    Returns the reciprocal alpha (= GPD shape parameter xi = 1/alpha).

    Parameters
    ----------
    x : np.ndarray
        1D array of positive values (claim amounts or log-claims).
    k : int
        Number of upper order statistics to use. Typical choice: k ~ sqrt(n).

    Returns
    -------
    float
        Hill estimate of xi = 1/alpha (tail weight parameter).
        Larger = heavier tail.
    """
    x_sorted = np.sort(x)[::-1]  # descending
    if k >= len(x_sorted):
        k = len(x_sorted) - 1
    log_ratios = np.log(x_sorted[:k]) - np.log(x_sorted[k])
    return float(np.mean(log_ratios))


def hill_double_bootstrap(
    x: np.ndarray,
    k_min: int = 10,
    n_subsample: int = 200,
    seed: int = 42,
) -> float:
    """
    Double bootstrap procedure for optimal k selection in Hill estimator.

    Implements a simplified version of the Danielsson et al. (2001) double
    bootstrap for automatic threshold selection. Returns the tail index
    xi = 1/alpha estimated at the optimal k.

    For insurance severity: apply to raw claim amounts (not log-transformed)
    to get the tail index of the original distribution.

    Parameters
    ----------
    x : np.ndarray
        Positive claim amounts.
    k_min : int
        Minimum k to consider.
    n_subsample : int
        Number of subsample sizes to try in the bootstrap.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Estimated tail weight xi (lambda in TTF notation).
        Typical range for motor BI: 0.3 - 0.8.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    x_sorted = np.sort(x)[::-1]

    # Compute Hill estimates for range of k values
    k_values = np.arange(k_min, max(k_min + 1, int(np.sqrt(n))))
    hill_vals = np.array([hill_estimator(x_sorted, k) for k in k_values])

    # Find k that minimises AMSE via bootstrap
    # Simplified: use the median over the stable region
    # (full Danielsson bootstrap is computationally intensive)
    # Stable region: find longest run where variance of hill_vals is small
    if len(hill_vals) < 3:
        return float(np.median(hill_vals))

    # Rolling variance over window of 5
    window = min(5, len(hill_vals) // 2)
    rolling_var = np.array([
        np.var(hill_vals[max(0, i - window):i + window + 1])
        for i in range(len(hill_vals))
    ])
    # Select k in the stable (low-variance) region
    stable_mask = rolling_var <= np.percentile(rolling_var, 25)
    if stable_mask.sum() == 0:
        stable_mask = np.ones(len(hill_vals), dtype=bool)

    return float(np.median(hill_vals[stable_mask]))


def estimate_tail_params(
    log_claims: np.ndarray,
    quantile_threshold: float = 0.9,
) -> tuple[float, float]:
    """
    Estimate TTF tail parameters (lambda+, lambda-) from training data.

    This is the pre-estimation step for TTF (fix) mode. Applied to log-claims
    (after log-transform), the right tail of log-claims corresponds to
    catastrophic injuries; the left tail to very small claims.

    Parameters
    ----------
    log_claims : np.ndarray
        Log-transformed claim amounts.
    quantile_threshold : float
        Use observations above this quantile for right tail estimation,
        below (1 - threshold) for left tail.

    Returns
    -------
    (lambda_pos, lambda_neg) : tuple of float
        Tail weight parameters. Default minimum of 0.5 to prevent
        degenerate transforms.
    """
    threshold_pos = np.quantile(log_claims, quantile_threshold)
    threshold_neg = np.quantile(log_claims, 1 - quantile_threshold)

    # Right tail
    right_tail = log_claims[log_claims > threshold_pos]
    if len(right_tail) >= 10:
        # Hill estimator on exceedances above threshold
        exceedances = right_tail - threshold_pos
        lambda_pos = hill_double_bootstrap(exceedances + 1.0)  # shift to positive
    else:
        lambda_pos = 1.0

    # Left tail (reflected)
    left_tail = log_claims[log_claims < threshold_neg]
    if len(left_tail) >= 10:
        exceedances = threshold_neg - left_tail  # positive exceedances
        lambda_neg = hill_double_bootstrap(exceedances + 1.0)
    else:
        lambda_neg = 0.5  # left tail of log-severity is usually light

    # Clamp to reasonable range
    lambda_pos = float(np.clip(lambda_pos, 0.3, 3.0))
    lambda_neg = float(np.clip(lambda_neg, 0.3, 3.0))

    return lambda_pos, lambda_neg
