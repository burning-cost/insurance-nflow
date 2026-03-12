# insurance-nflow

Normalizing flows for insurance severity distribution modelling.

## The problem

The standard approach to severity modelling — lognormal GLM, gamma GLM — makes a strong bet on the distribution family. The family governs the tail, the shape, the response to rating factors. Most of that bet is untestable on training data and wrong in the upper tail, where the money is.

For UK motor bodily injury, this matters acutely. BI severity is bimodal: soft-tissue claims cluster around GBP 5,000; catastrophic injury claims (spinal, brain injury) follow a power law from GBP 100,000 upward. A lognormal fits neither mode well and dramatically misrepresents the tail. A pricing team using lognormal TVaR(0.99) to price their catastrophic injury layer is using the wrong number.

Normalizing flows drop the family assumption entirely. A Neural Spline Flow (NSF) learns the full conditional distribution p(severity | rating factors) from data. The [Hickling & Prangle (ICML 2025)](https://proceedings.mlr.press/v267/hickling25a.html) Tail Transform Flow (TTF) adds GPD-like heavy tails on top, with tail weight parameters estimated from the data via the Hill estimator.

The result is a model that can represent bimodality, shape changes by rating factor, and calibrated heavy tails — without choosing a parametric family.

## What this library provides

- `SeverityFlow` — unconditional severity distribution
- `ConditionalSeverityFlow` — p(severity | rating factors)
- TTF tail layer (Hickling & Prangle 2025), optional
- TVaR, ILF curves, LEV, reinsurance layer pricing from flow samples
- Diagnostics: PIT histogram, QQ plot, tail index comparison, AIC/BIC table vs parametric benchmarks
- Synthetic UK motor BI data generator (bimodal DGP, known tail index) for testing

## Install

```bash
pip install insurance-nflow
```

PyTorch is required. For CPU-only (recommended for most pricing teams):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install insurance-nflow
```

This is roughly 900MB of PyTorch plus the library. The API hides all PyTorch from the user; you interact with numpy arrays throughout.

## Quickstart

### Unconditional severity

```python
import numpy as np
from insurance_nflow import SeverityFlow

# Load your claims data
claims = np.array([...])  # positive GBP amounts

# Fit a flow with heavy-tail extension
flow = SeverityFlow(
    n_transforms=6,
    tail_transform=True,   # Hickling 2025 TTF
    tail_mode='fix',       # pre-estimate tail params via Hill, then fix
    max_epochs=200,
    patience=20,
)
result = flow.fit(claims)

print(result)
# SeverityFlowResult(val_logL=-9.2341, AIC=1234567.8, n_params=97412, lambda+=0.612, epochs=143)

# Actuarial outputs
print(flow.tvar(0.99))          # TVaR(99%) in GBP
print(flow.quantile(0.995))     # VaR(99.5%) in GBP

# ILF curve
ilf = flow.ilf(
    limits=[50_000, 100_000, 250_000, 500_000, 1_000_000],
    basic_limit=50_000,
)
# {50000.0: 1.0, 100000.0: 1.31, 250000.0: 1.72, ...}
```

### Conditional severity (with rating factors)

```python
from insurance_nflow import ConditionalSeverityFlow
import numpy as np

claims = np.array([...])          # shape (N,)
context = np.array([...])         # shape (N, n_factors) — e.g. age_band, vehicle_group, region

flow = ConditionalSeverityFlow(
    context_features=5,
    n_transforms=6,
    tail_transform=True,
)
result = flow.fit(claims, context=context, exposure_weights=exposure)

# Price a specific risk
young_london = np.array([[1, 8, 1, 3, 0]])  # one row = one risk profile
print(flow.conditional_tvar(young_london, 0.99))

# Compare against a base risk
mid_north = np.array([[3, 5, 3, 5, 5]])
print(flow.conditional_tvar(mid_north, 0.99))
```

### Reinsurance layer pricing

```python
# Expected cost to 200k xs 50k per-occurrence XL layer
cost = flow.reinsurance_layer(
    attachment=50_000,
    limit=200_000,
    context=portfolio_factors,
    n_samples=500_000,
)
```

### Model comparison vs parametric families

```python
from insurance_nflow.diagnostics import fit_parametric_benchmarks, model_comparison_table

# Fit lognormal, gamma, Pareto as baselines
ll_benchmarks, k_benchmarks = fit_parametric_benchmarks(claims)

# Add flow
ll_benchmarks["flow"] = float(result.train_log_likelihood * len(claims))
k_benchmarks["flow"] = result.n_parameters

table = model_comparison_table(claims, ll_benchmarks, k_benchmarks)
# Sorted by AIC. Note: AIC heavily penalises the flow's ~100k params.
# Use test log-likelihood per observation as the primary metric.
```

## Architecture

```
log(claim) ─→ [NSF: n coupling layers] ─→ [TTF tail layer] ─→ N(0,1)
                  ↑ context encoder
              rating factors feed into
              each coupling layer's
              parameter network (zuko API)
```

The log-transform maps positive claims to the real line. The NSF (Neural Spline Flow, Durkan et al. 2019) learns the residual non-Gaussianity — including the bimodal body. The TTF layer (Hickling & Prangle 2025) corrects the tail to GPD-like behaviour, with tail weight parameters lambda+, lambda- estimated from the upper and lower tails of the training data via Hill double-bootstrap.

**Why NSF over MAF?** Coupling architecture means sampling parallelises. Drawing 1M scenario claims takes seconds, not minutes.

**Why TTF (fix) over (joint)?** Hickling's experiments show TTF (fix) — pre-estimating tail parameters and fixing them — outperforms joint training on heavy-tailed targets. Less flexibility, more stability.

**Why zuko?** It's the only actively-maintained Python normalizing flow library (v1.6.0, March 2026). The `flow(context).log_prob(x)` API is clean. nflows is abandoned; normflows is slow to update.

## Actuarial functions

These work on any numpy array of positive values — not just flow samples:

```python
from insurance_nflow import tvar, ilf, limited_expected_value, reinsurance_layer_cost

samples = np.array([...])

tvar(samples, 0.99)
ilf(samples, limits=[100_000, 250_000], basic_limit=50_000)
limited_expected_value(samples, 100_000)
reinsurance_layer_cost(samples, attachment=50_000, limit=200_000)
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests that require torch/zuko use `pytest.importorskip` and skip gracefully if the packages aren't installed. The data, actuarial, and diagnostic tests have no heavy dependencies.

## Limitations and roadmap

**v0.1.0 (this release):**
- No truncation/censoring support. Policies with deductibles require the CDF for each observation, which zuko does not compute analytically. Feasible but doubles training time. Planned for v0.2.0.
- CPU training only (GPU opt-in via `device='cuda'`). For 100k claims, 200 epochs takes ~50 minutes on a modern CPU. Subsample to 100k for initial model search.
- Univariate only (one severity dimension). Joint modelling of claim count and severity is not in scope.

**v0.2.0 planned:**
- Truncated/censored likelihood
- Aggregate loss simulation (Panjer + flow severity)
- Model persistence (save/load)

## References

- Hickling & Prangle (2025), [Flexible Tails for Normalizing Flows](https://proceedings.mlr.press/v267/hickling25a.html), ICML 2025, PMLR 267:23155-23178
- Durkan et al. (2019), [Neural Spline Flows](https://arxiv.org/abs/1906.04032), NeurIPS 2019
- Papamakarios et al. (2021), [Normalizing Flows for Probabilistic Modeling and Inference](https://jmlr.org/papers/v22/19-1028.html), JMLR 22(57)
- Winkler et al. (2019), [Learning Likelihoods with Conditional Normalizing Flows](https://arxiv.org/abs/1912.00042)
