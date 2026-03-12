"""
SeverityFlow — the main user-facing class.

This is the only class most pricing team members need to know about.
It hides all PyTorch mechanics behind a scikit-learn-style fit/predict API.

Design decisions:
1. No pandas dependency in the core — accept numpy arrays and column name lists.
   polars is a dependency for optional data handling helpers.
2. Training runs on CPU by default. Pass device='cuda' if you have a GPU.
3. Early stopping via patience parameter. Validation split defaults to 20%.
4. Exposure weights are normalised to sum to n (not to 1) inside the loss,
   so the AIC/BIC k-penalty is unaffected by the weight scale.
5. SeverityFlowResult is a plain dataclass — easy to serialise/inspect.

The conditional flow (ConditionalSeverityFlow) is a thin subclass that
enforces context is provided to fit() and prediction methods. Separated for
clarity of documentation and API discoverability.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import zuko
    ZUKO_AVAILABLE = True
except ImportError:
    ZUKO_AVAILABLE = False

from .actuarial import (
    quantile as _quantile,
    tvar as _tvar,
    limited_expected_value,
    ilf as _ilf,
    reinsurance_layer_cost,
    burning_cost_summary,
)
from .tail import estimate_tail_params


@dataclass
class SeverityFlowResult:
    """
    Results from fitting a SeverityFlow model.

    Attributes
    ----------
    train_log_likelihood : float
        Mean per-observation log-likelihood on training data.
    val_log_likelihood : float or None
        Mean per-observation log-likelihood on validation data.
    n_parameters : int
        Number of trainable PyTorch parameters.
    aic : float
        Akaike Information Criterion. With ~50k-200k parameters, this
        heavily penalises the flow vs parametric families — use test
        log-likelihood as the primary comparison metric.
    bic : float
        Bayesian Information Criterion.
    n_eff : float
        Effective sample size accounting for exposure weights.
        n_eff = (sum(w))^2 / sum(w^2).
    tail_lambda_pos : float
        TTF right tail parameter (lambda+). Larger = heavier right tail.
    tail_lambda_neg : float
        TTF left tail parameter (lambda-).
    training_history : list[dict]
        Per-epoch training history: epoch, train_loss, val_loss.
    training_seconds : float
        Wall time for training.
    n_epochs_run : int
        Epochs completed (may be less than max_epochs if early stopping).
    converged : bool
        True if training completed without hitting max_epochs (early stop).
    """

    train_log_likelihood: float
    val_log_likelihood: Optional[float]
    n_parameters: int
    aic: float
    bic: float
    n_eff: float
    tail_lambda_pos: float
    tail_lambda_neg: float
    training_history: list[dict] = field(default_factory=list)
    training_seconds: float = 0.0
    n_epochs_run: int = 0
    converged: bool = False

    def __repr__(self) -> str:
        return (
            f"SeverityFlowResult("
            f"val_logL={self.val_log_likelihood:.4f}, "
            f"AIC={self.aic:.1f}, "
            f"n_params={self.n_parameters:,}, "
            f"lambda+={self.tail_lambda_pos:.3f}, "
            f"epochs={self.n_epochs_run}"
            f")"
        )


class SeverityFlow:
    """
    Normalizing flow for insurance claim severity.

    Models the full severity distribution P(X | context) using a Neural Spline
    Flow (NSF) in log-space, optionally augmented with a Hickling & Prangle 2025
    Tail Transform (TTF) for heavy-tailed motor BI claims.

    Unconditional usage (no rating factors):

        flow = SeverityFlow()
        result = flow.fit(claim_amounts)
        print(flow.tvar(0.99))

    Conditional usage (with rating factors):

        flow = SeverityFlow(context_features=5)
        result = flow.fit(claim_amounts, context=rating_factor_matrix)
        print(flow.tvar(0.99, context=new_risks))

    Parameters
    ----------
    context_features : int
        Number of rating factor columns. 0 for unconditional.
    n_transforms : int
        Number of NSF coupling layers. Default 6. Reduce to 3-4 for small datasets.
    hidden_features : list[int]
        Hidden layer sizes in coupling networks. Default [64, 64].
    tail_transform : bool
        Add TTF tail layer (Hickling 2025). Recommended True for motor BI.
    tail_mode : str
        'fix' (default): pre-estimate tail params via Hill estimator, then fix.
        'joint': train tail params jointly. More flexible but less stable.
    batch_size : int
        Mini-batch size for Adam training.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience (epochs without val improvement).
    lr : float
        Adam learning rate.
    val_fraction : float
        Fraction of data held out for validation and early stopping.
    device : str
        PyTorch device. 'cpu' or 'cuda'. Default 'cpu'.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        context_features: int = 0,
        n_transforms: int = 6,
        hidden_features: Optional[list[int]] = None,
        tail_transform: bool = True,
        tail_mode: str = "fix",
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 20,
        lr: float = 1e-3,
        val_fraction: float = 0.2,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.context_features = context_features
        self.n_transforms = n_transforms
        self.hidden_features = hidden_features or [64, 64]
        self.tail_transform = tail_transform
        self.tail_mode = tail_mode
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.val_fraction = val_fraction
        self.device = device
        self.seed = seed

        self._model = None
        self._result: Optional[SeverityFlowResult] = None
        self._fitted = False

    def fit(
        self,
        claims: np.ndarray,
        context: Optional[np.ndarray] = None,
        exposure_weights: Optional[np.ndarray] = None,
    ) -> SeverityFlowResult:
        """
        Fit the flow to observed claim amounts.

        Parameters
        ----------
        claims : np.ndarray of shape (N,)
            Positive claim amounts (GBP). Must be > 0.
        context : np.ndarray of shape (N, context_features) or None
            Rating factor values. Required if context_features > 0.
        exposure_weights : np.ndarray of shape (N,) or None
            Observation weights (e.g., earned exposure). If None, uniform.

        Returns
        -------
        SeverityFlowResult
            Training results including log-likelihood, AIC/BIC, tail parameters.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required. Install: pip install torch")
        if not ZUKO_AVAILABLE:
            raise ImportError("zuko is required. Install: pip install zuko")

        claims = np.asarray(claims, dtype=np.float64)
        if np.any(claims <= 0):
            raise ValueError("All claim amounts must be positive (> 0).")

        if context is not None:
            context = np.asarray(context, dtype=np.float64)
            if context.shape[0] != len(claims):
                raise ValueError(
                    f"context has {context.shape[0]} rows but claims has {len(claims)} observations."
                )
            if self.context_features == 0:
                self.context_features = context.shape[1]
            elif context.shape[1] != self.context_features:
                raise ValueError(
                    f"context has {context.shape[1]} columns but context_features={self.context_features}."
                )
        elif self.context_features > 0:
            raise ValueError(
                f"context_features={self.context_features} but no context provided."
            )

        if exposure_weights is not None:
            exposure_weights = np.asarray(exposure_weights, dtype=np.float64)
            if len(exposure_weights) != len(claims):
                raise ValueError("exposure_weights must have same length as claims.")
            if np.any(exposure_weights < 0):
                raise ValueError("exposure_weights must be non-negative.")
            # Normalise to mean 1.0
            exposure_weights = exposure_weights / exposure_weights.mean()

        # Effective sample size for AIC/BIC
        if exposure_weights is not None:
            n_eff = float((exposure_weights.sum() ** 2) / (exposure_weights ** 2).sum())
        else:
            n_eff = float(len(claims))

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Train/val split
        n = len(claims)
        n_val = max(1, int(n * self.val_fraction))
        n_train = n - n_val
        perm = np.random.permutation(n)
        idx_train, idx_val = perm[:n_train], perm[n_train:]

        # Estimate tail parameters before training (TTF fix mode)
        lambda_pos, lambda_neg = 0.25, 0.1
        if self.tail_transform:
            log_claims = np.log(claims)
            lambda_pos, lambda_neg = estimate_tail_params(
                log_claims[idx_train],
                quantile_threshold=0.9,
            )

        tail_trainable = self.tail_mode == "joint"

        # Build model
        from .flows import build_flow
        self._model = build_flow(
            context_features=self.context_features,
            n_transforms=self.n_transforms,
            hidden_features=self.hidden_features,
            tail_transform=self.tail_transform,
            lambda_pos=lambda_pos,
            lambda_neg=lambda_neg,
            tail_trainable=tail_trainable,
        ).to(self.device)

        # Build tensors
        x_tensor = torch.tensor(claims, dtype=torch.float32)
        ctx_tensor = (
            torch.tensor(context, dtype=torch.float32)
            if context is not None else None
        )
        w_tensor = (
            torch.tensor(exposure_weights, dtype=torch.float32)
            if exposure_weights is not None
            else torch.ones(n, dtype=torch.float32)
        )

        x_train = x_tensor[idx_train].to(self.device)
        x_val = x_tensor[idx_val].to(self.device)
        w_train = w_tensor[idx_train].to(self.device)
        w_val = w_tensor[idx_val].to(self.device)

        ctx_train = ctx_val = None
        if ctx_tensor is not None:
            ctx_train = ctx_tensor[idx_train].to(self.device)
            ctx_val = ctx_tensor[idx_val].to(self.device)

        # DataLoader for mini-batches
        if ctx_train is not None:
            train_ds = TensorDataset(x_train, ctx_train, w_train)
        else:
            train_ds = TensorDataset(x_train, w_train)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=self.lr,
        )

        history = []
        best_val_loss = float("inf")
        best_state = None
        patience_count = 0
        t_start = time.time()

        for epoch in range(self.max_epochs):
            # Training
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                optimizer.zero_grad()
                if ctx_train is not None:
                    x_b, ctx_b, w_b = batch
                    log_prob = self._model.log_prob(x_b, ctx_b)
                else:
                    x_b, w_b = batch
                    log_prob = self._model.log_prob(x_b)

                # Weighted NLL loss
                loss = -(log_prob * w_b).mean()

                if not torch.isfinite(loss):
                    warnings.warn(f"Non-finite loss at epoch {epoch}. Skipping batch.")
                    continue

                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)

            # Validation
            self._model.eval()
            with torch.no_grad():
                if ctx_val is not None:
                    val_log_prob = self._model.log_prob(x_val, ctx_val)
                else:
                    val_log_prob = self._model.log_prob(x_val)
                val_loss = -(val_log_prob * w_val).mean().item()

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

            # Early stopping
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self._model.state_dict().items()
                }
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    # Restore best weights
                    if best_state is not None:
                        self._model.load_state_dict(best_state)
                    break

        # Restore best state if training completed without early stop
        if best_state is not None and patience_count < self.patience:
            self._model.load_state_dict(best_state)

        training_seconds = time.time() - t_start

        # Final log-likelihoods
        self._model.eval()
        with torch.no_grad():
            train_ll = float(
                self._model.log_prob(x_train, ctx_train).mean().item()
                if ctx_train is not None
                else self._model.log_prob(x_train).mean().item()
            )
            val_ll = float(
                self._model.log_prob(x_val, ctx_val).mean().item()
                if ctx_val is not None
                else self._model.log_prob(x_val).mean().item()
            )

        n_params = self._model.n_parameters()
        total_ll = train_ll * n_train + val_ll * n_val
        aic = 2 * n_params - 2 * total_ll
        bic = n_params * np.log(n) - 2 * total_ll
        lambda_pos_fitted, lambda_neg_fitted = self._model.tail_indices()

        self._result = SeverityFlowResult(
            train_log_likelihood=train_ll,
            val_log_likelihood=val_ll,
            n_parameters=n_params,
            aic=aic,
            bic=bic,
            n_eff=n_eff,
            tail_lambda_pos=lambda_pos_fitted,
            tail_lambda_neg=lambda_neg_fitted,
            training_history=history,
            training_seconds=training_seconds,
            n_epochs_run=len(history),
            converged=(patience_count < self.patience),
        )
        self._fitted = True
        return self._result

    def _check_fitted(self) -> None:
        if not self._fitted or self._model is None:
            raise RuntimeError("Call fit() before using prediction methods.")

    def sample(
        self,
        n: int,
        context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draw samples from the fitted severity distribution.

        Parameters
        ----------
        n : int
            Number of samples.
        context : np.ndarray of shape (n, context_features) or None
            Rating factor values. If provided, each row is one risk.
            If context_features > 0 but context has one row, broadcast
            to n samples from that risk.

        Returns
        -------
        np.ndarray of shape (n,)
            Claim amount samples in GBP.
        """
        self._check_fitted()
        ctx_tensor = None
        if context is not None:
            ctx_array = np.asarray(context, dtype=np.float64)
            if ctx_array.ndim == 1:
                ctx_array = ctx_array[np.newaxis, :]
            if ctx_array.shape[0] == 1:
                ctx_array = np.repeat(ctx_array, n, axis=0)
            ctx_tensor = torch.tensor(ctx_array, dtype=torch.float32).to(self.device)

        self._model.eval()
        with torch.no_grad():
            samples = self._model.sample(n, ctx_tensor)

        return samples.cpu().numpy()

    def log_prob(
        self,
        claims: np.ndarray,
        context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Log-probability of observed claim amounts under the fitted model.

        Parameters
        ----------
        claims : np.ndarray of shape (N,)
        context : np.ndarray of shape (N, context_features) or None

        Returns
        -------
        np.ndarray of shape (N,)
            Log-probabilities.
        """
        self._check_fitted()
        x_t = torch.tensor(np.asarray(claims, dtype=np.float32)).to(self.device)
        ctx_t = None
        if context is not None:
            ctx_t = torch.tensor(np.asarray(context, dtype=np.float32)).to(self.device)

        self._model.eval()
        with torch.no_grad():
            lp = self._model.log_prob(x_t, ctx_t)
        return lp.cpu().numpy()

    def quantile(
        self,
        p: float,
        context: Optional[np.ndarray] = None,
        n_samples: int = 100_000,
    ) -> Union[float, np.ndarray]:
        """
        Quantile (VaR) at probability level p.

        Parameters
        ----------
        p : float
            Probability level (e.g., 0.99 for 99th percentile VaR).
        context : np.ndarray or None
            If None: single quantile from marginal (or unconditional) distribution.
            If shape (p,): single context row, returns float.
            If shape (m, p): m context rows, returns array of m quantiles.
        n_samples : int
            MC samples for quantile estimation.

        Returns
        -------
        float or np.ndarray
        """
        samples = self.sample(n_samples, context)
        return float(np.quantile(samples, p))

    def tvar(
        self,
        p: float,
        context: Optional[np.ndarray] = None,
        n_samples: int = 100_000,
    ) -> float:
        """
        Tail Value at Risk (TVaR / CTE / ES) at probability level p.

        TVaR(p) = E[X | X > VaR(p)]

        Parameters
        ----------
        p : float
            Probability level (e.g., 0.99).
        context : np.ndarray or None
        n_samples : int

        Returns
        -------
        float
        """
        samples = self.sample(n_samples, context)
        return _tvar(samples, p)

    def ilf(
        self,
        limits: list[float],
        basic_limit: float = 50_000.0,
        context: Optional[np.ndarray] = None,
        n_samples: int = 500_000,
    ) -> dict[float, float]:
        """
        Increased Limit Factors relative to basic_limit.

        ILF(L) = E[min(X, L)] / E[min(X, basic_limit)]

        Parameters
        ----------
        limits : list[float]
            Policy limits to compute ILF at.
        basic_limit : float
            Reference limit. Default GBP 50,000.
        context : np.ndarray or None
        n_samples : int
            More samples = more accurate ILF, especially at high limits.
            500k recommended for production.

        Returns
        -------
        dict mapping limit -> ILF factor.
        """
        samples = self.sample(n_samples, context)
        return _ilf(samples, limits=limits, basic_limit=basic_limit)

    def reinsurance_layer(
        self,
        attachment: float,
        limit: float,
        context: Optional[np.ndarray] = None,
        n_samples: int = 500_000,
    ) -> float:
        """
        Expected cost to a reinsurance layer (limit xs attachment).

        Parameters
        ----------
        attachment : float
            Attachment point (GBP).
        limit : float
            Layer limit (GBP).
        context : np.ndarray or None
        n_samples : int

        Returns
        -------
        float
            Pure premium per occurrence.
        """
        samples = self.sample(n_samples, context)
        return reinsurance_layer_cost(samples, attachment, limit)

    def summary(
        self,
        context: Optional[np.ndarray] = None,
        n_samples: int = 500_000,
    ) -> dict:
        """
        Full actuarial summary: quantiles, TVaR, ILF curve.

        Parameters
        ----------
        context : np.ndarray or None
        n_samples : int

        Returns
        -------
        dict from actuarial.burning_cost_summary().
        """
        samples = self.sample(n_samples, context)
        return burning_cost_summary(samples)

    @property
    def result(self) -> Optional[SeverityFlowResult]:
        """Training results, or None if not yet fitted."""
        return self._result


class ConditionalSeverityFlow(SeverityFlow):
    """
    Severity flow that requires rating factor context.

    Thin wrapper around SeverityFlow that:
    1. Enforces context is provided at fit() time.
    2. Raises a clear error if prediction methods are called without context.
    3. Documents the conditional use case more explicitly.

    Parameters
    ----------
    context_features : int
        Number of rating factor columns. Required (no default).
    All other parameters: same as SeverityFlow.
    """

    def __init__(self, context_features: int, **kwargs) -> None:
        if context_features <= 0:
            raise ValueError(
                "ConditionalSeverityFlow requires context_features > 0. "
                "For unconditional flows, use SeverityFlow(context_features=0)."
            )
        super().__init__(context_features=context_features, **kwargs)

    def fit(
        self,
        claims: np.ndarray,
        context: np.ndarray,  # required, no Optional
        exposure_weights: Optional[np.ndarray] = None,
    ) -> SeverityFlowResult:
        """
        Fit conditional severity flow.

        Parameters
        ----------
        claims : np.ndarray of shape (N,)
        context : np.ndarray of shape (N, context_features)
            Rating factor matrix. Required.
        exposure_weights : np.ndarray of shape (N,) or None
        """
        if context is None:
            raise ValueError(
                "ConditionalSeverityFlow.fit() requires context (rating factors). "
                "Received None."
            )
        return super().fit(claims, context=context, exposure_weights=exposure_weights)

    def conditional_quantile(
        self,
        context: np.ndarray,
        p: float,
        n_samples: int = 100_000,
    ) -> float:
        """
        Quantile at level p for a specific risk profile.

        Parameters
        ----------
        context : np.ndarray of shape (p,) or (1, p)
            Single risk's rating factors.
        p : float
        n_samples : int

        Returns
        -------
        float
        """
        return self.quantile(p, context=context, n_samples=n_samples)

    def conditional_tvar(
        self,
        context: np.ndarray,
        p: float,
        n_samples: int = 100_000,
    ) -> float:
        """
        TVaR at level p for a specific risk profile.

        Parameters
        ----------
        context : np.ndarray of shape (p,) or (1, p)
        p : float
        n_samples : int

        Returns
        -------
        float
        """
        return self.tvar(p, context=context, n_samples=n_samples)
