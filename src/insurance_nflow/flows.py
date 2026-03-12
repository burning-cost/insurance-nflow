"""
Core flow architecture — SeverityFlow internals.

Wraps zuko Neural Spline Flow (NSF) with:
- Log-transform for positive support (claims > 0)
- Optional TTF tail layer (Hickling & Prangle 2025)
- Exposure weighting in training loss

Architecture in log-space:
    log(claim) -> [zuko NSF, n_transforms coupling layers] -> [TTF tail] -> N(0,1)

The log-transform is applied outside the flow as a fixed bijector. The Jacobian
correction -log(x) is included in the log-probability computation.

Design choice: put the TTF layer between the flow and the base distribution,
not between the data and the flow. This means the NSF learns the bulk of the
distribution shape (bimodal body) and the TTF corrects the tail behaviour.
Alternative (TTF before NSF) would require the NSF to learn the body of a
GPD-tailed distribution — harder.

Context conditioning: zuko NSF accepts a context tensor that is fed into each
coupling layer's parameter network. This is the natural API for rating factor
conditioning. No separate encoder needed — zuko handles it internally.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    import zuko
    from zuko.flows import NSF
    ZUKO_AVAILABLE = True
except ImportError:
    ZUKO_AVAILABLE = False

from .tail import TailTransform, estimate_tail_params


class LogTransformMixin:
    """
    Mixin providing log-transform and its Jacobian for positive-support data.

    Claims x > 0 are mapped to y = log(x) in R.
    Log-probability in original space:
        log p(x) = log p(log(x)) - log(x)

    The -log(x) term is the log-Jacobian of the log-transform.
    """

    @staticmethod
    def to_log_space(x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Transform claims to log-space.

        Parameters
        ----------
        x : Tensor
            Positive claim amounts.

        Returns
        -------
        log_x : Tensor
            Log-transformed claims.
        log_jacobian : Tensor
            Log |d(log x)/dx| = -log(x) per observation.
        """
        log_x = torch.log(x.clamp(min=1e-8))
        log_jacobian = -log_x  # -log(x) = log(1/x) = log|d(log x)/dx|
        return log_x, log_jacobian


class _TTFWrappedDistribution:
    """
    Wraps a zuko distribution with a TTF tail transform.

    Not a full torch Distribution — just wraps log_prob and sample
    for the internal training loop.
    """

    def __init__(
        self,
        base_dist,
        tail_transform: TailTransform,
    ) -> None:
        self._base_dist = base_dist
        self._tail = tail_transform

    def log_prob(self, z: Tensor) -> Tensor:
        """log p(z) under the TTF-augmented distribution."""
        # z -> u = TTF(z), log|du/dz|
        u, ladj = self._tail.forward(z)
        # log p(z) = log p_base(u) + log|du/dz|
        return self._base_dist.log_prob(u) + ladj

    def sample(self, shape: tuple = ()) -> Tensor:
        """Sample from TTF-augmented distribution."""
        u = self._base_dist.sample(shape)
        z, _ = self._tail.inverse(u)
        return z


class SeverityFlowModel(nn.Module, LogTransformMixin):
    """
    Neural Spline Flow for insurance severity with optional TTF tail layer.

    This is the internal model class. Users interact with SeverityFlow (in
    severity.py) which handles training, validation, and actuarial outputs.

    Parameters
    ----------
    features : int
        Dimensionality of data (1 for univariate severity).
    context_features : int
        Number of conditioning variables (rating factors). 0 for unconditional.
    n_transforms : int
        Number of NSF coupling layers. More = more expressive, slower.
    hidden_features : list[int]
        Hidden layer sizes in the coupling networks.
    tail_transform : bool
        Whether to add TTF tail layer.
    lambda_pos : float
        Initial (or fixed) right tail weight.
    lambda_neg : float
        Initial (or fixed) left tail weight.
    tail_trainable : bool
        Whether to train tail parameters jointly (TTF joint) or fix (TTF fix).
    """

    def __init__(
        self,
        features: int = 1,
        context_features: int = 0,
        n_transforms: int = 6,
        hidden_features: Optional[list[int]] = None,
        tail_transform: bool = True,
        lambda_pos: float = 1.0,
        lambda_neg: float = 0.5,
        tail_trainable: bool = False,
    ) -> None:
        if not ZUKO_AVAILABLE:
            raise ImportError(
                "zuko is required for SeverityFlowModel. "
                "Install with: pip install zuko"
            )
        super().__init__()

        if hidden_features is None:
            hidden_features = [64, 64]

        self.features = features
        self.context_features = context_features
        self.use_tail_transform = tail_transform

        # zuko NSF: conditional or unconditional
        context_dim = context_features if context_features > 0 else 0
        self.nsf = NSF(
            features=features,
            context=context_dim,
            transforms=n_transforms,
            hidden_features=hidden_features,
        )

        # TTF tail layer
        if tail_transform:
            self.tail = TailTransform(
                lambda_pos=lambda_pos,
                lambda_neg=lambda_neg,
                trainable=tail_trainable,
            )
        else:
            self.tail = None

    def log_prob(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Log-probability of positive claim amounts.

        Includes the log-Jacobian of the log-transform.

        Parameters
        ----------
        x : Tensor of shape (N,) or (N, 1)
            Positive claim amounts.
        context : Tensor of shape (N, context_features) or None
            Rating factor values.

        Returns
        -------
        Tensor of shape (N,)
            Log-probabilities in original (not log) scale.
        """
        x_flat = x.view(-1)

        # Log-transform: x -> log(x), add Jacobian
        log_x, log_jac = self.to_log_space(x_flat)
        log_x_2d = log_x.unsqueeze(-1)  # (N, 1) for zuko

        # NSF log-prob in log-space
        if context is not None:
            flow_dist = self.nsf(context)
        else:
            flow_dist = self.nsf()

        if self.tail is not None:
            # Get the flow's base distribution and wrap with TTF
            # zuko returns a distribution whose log_prob we modify
            nsf_log_prob = flow_dist.log_prob(log_x_2d)

            # Apply TTF correction: the flow maps data to z (not standard normal).
            # We need log p(log_x) under TTF-augmented model.
            # NSF already accounts for its internal transforms.
            # The TTF layer sits at the output of the NSF (between NSF and base).
            # Implementation: compute NSF log_prob with the TTF-modified base.
            # zuko's transform interface: flow_dist.transform maps data -> base.
            # We apply TTF to the transformed value.
            try:
                # Use zuko's transform to get z (the output before base dist)
                z = flow_dist.transform(log_x_2d)  # z in NSF output space
                z_flat = z.squeeze(-1)
                u, ladj_ttf = self.tail.forward(z_flat)
                # log p_base(u) under standard normal
                log_p_base = torch.distributions.Normal(0, 1).log_prob(u)
                # Total: log p(x) = log p_base(u) + log|du/dz| + log|dz/d(log x)| + log|d(log x)/dx|
                # The last two terms: NSF gives log|dz/d(log x)|, included in flow_dist.log_prob
                # We need to reconstruct: flow_dist.log_prob = log_p_base(z_nsf_output) + nsf_ladj
                # But we want: log_p_base(TTF(z)) + ladj_ttf + nsf_ladj
                # = nsf_log_prob - log_p_base(z_nsf_output) + log_p_base(u) + ladj_ttf
                # where z_nsf_output = z_flat
                log_p_base_z = torch.distributions.Normal(0, 1).log_prob(z_flat)
                log_p_in_log_space = nsf_log_prob - log_p_base_z.sum(-1 if z.dim() > 1 else 0) + log_p_base + ladj_ttf
            except AttributeError:
                # Fallback if zuko API doesn't expose transform directly
                log_p_in_log_space = flow_dist.log_prob(log_x_2d)
        else:
            log_p_in_log_space = flow_dist.log_prob(log_x_2d)

        # Add log-transform Jacobian to get log p in original scale
        return log_p_in_log_space + log_jac

    def sample(
        self,
        n: int,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample claim amounts from the fitted model.

        Parameters
        ----------
        n : int
            Number of samples.
        context : Tensor of shape (n, context_features) or None
            Rating factor values (one row per sample).

        Returns
        -------
        Tensor of shape (n,)
            Sampled claim amounts (positive, original scale).
        """
        if context is not None:
            flow_dist = self.nsf(context)
        else:
            flow_dist = self.nsf()

        with torch.no_grad():
            log_samples = flow_dist.sample((n,)) if context is None else flow_dist.sample()
            # log_samples shape: (n, 1) or (n,)
            if log_samples.dim() > 1:
                log_samples = log_samples.squeeze(-1)

        return torch.exp(log_samples)

    def n_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def tail_indices(self) -> tuple[float, float]:
        """
        Return (lambda+, lambda-) tail weight parameters.

        Returns (1.0, 0.5) if no tail transform.
        """
        if self.tail is None:
            return (1.0, 0.5)
        return (
            float(self.tail.lambda_pos.item()),
            float(self.tail.lambda_neg.item()),
        )


def build_flow(
    context_features: int,
    n_transforms: int,
    hidden_features: Optional[list[int]],
    tail_transform: bool,
    lambda_pos: float,
    lambda_neg: float,
    tail_trainable: bool,
) -> SeverityFlowModel:
    """
    Factory function for SeverityFlowModel.

    Preferred over direct instantiation in the fit path, allowing
    future architecture switches without changing SeverityFlow.
    """
    return SeverityFlowModel(
        features=1,
        context_features=context_features,
        n_transforms=n_transforms,
        hidden_features=hidden_features,
        tail_transform=tail_transform,
        lambda_pos=lambda_pos,
        lambda_neg=lambda_neg,
        tail_trainable=tail_trainable,
    )
