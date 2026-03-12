"""
insurance-nflow: Normalizing flows for insurance severity distribution modelling.

The standard actuarial approach to severity modelling — lognormal GLM, gamma GLM —
fixes the distribution family and models the mean as a function of rating factors.
This library drops the family assumption entirely. A normalizing flow learns the
full conditional distribution p(severity | rating factors), including shape
changes, bimodality, and heavy tails.

Primary use case: UK motor bodily injury severity, which is bimodal (soft-tissue
claims ~GBP 5k and catastrophic claims GBP 100k+) with a power-law tail that
lognormal misrepresents. The Tail Transform Flow (TTF) of Hickling & Prangle
(ICML 2025) is integrated as an optional heavy-tail layer.

Quickstart — unconditional severity:

    from insurance_nflow import SeverityFlow
    import numpy as np

    claims = np.array([...])  # positive claim amounts
    flow = SeverityFlow(tail_transform=True)
    result = flow.fit(claims)

    print(result)
    print(flow.tvar(0.99))
    print(flow.ilf(limits=[100_000, 250_000, 500_000], basic_limit=50_000))

Quickstart — conditional (rating factors):

    from insurance_nflow import ConditionalSeverityFlow
    import numpy as np

    claims = np.array([...])
    context = np.array([...])  # shape (n_claims, n_factors)

    flow = ConditionalSeverityFlow(context_features=5)
    result = flow.fit(claims, context=context)

    # TVaR for a specific risk profile
    new_risk = np.array([[1, 3, 2, 0, 1]])  # one row = one risk
    print(flow.conditional_tvar(new_risk, 0.99))

For testing without a GPU or full PyTorch install, use the synthetic data
generator which has no PyTorch dependency:

    from insurance_nflow.data import generate_motor_bi_dataset
    dataset = generate_motor_bi_dataset(n_policies=10_000)
    claims = dataset['claim_amount']

Dependencies:
    torch>=2.0 (CPU install is sufficient: pip install torch --index-url ...)
    zuko>=1.0 (normalizing flow library)
    scipy, numpy, matplotlib

References:
    Hickling & Prangle (2025), "Flexible Tails for Normalizing Flows",
        ICML 2025, PMLR 267:23155-23178, arXiv:2406.16971.
    Durkan et al. (2019), "Neural Spline Flows", NeurIPS 2019.
    Papamakarios et al. (2021), "Normalizing Flows for Probabilistic
        Modeling and Inference", JMLR 22(57).
"""

from .severity import SeverityFlow, ConditionalSeverityFlow, SeverityFlowResult
from .actuarial import (
    tvar,
    quantile,
    limited_expected_value,
    ilf,
    reinsurance_layer_cost,
    burning_cost_summary,
)
from .tail import TailTransform, estimate_tail_params, hill_estimator
from .data import generate_motor_bi_dataset, DGPParams, sample_severity
from .diagnostics import (
    pit_values,
    ks_test,
    pit_histogram,
    qq_plot_lognormal,
    qq_plot_flow,
    tail_plot,
    tail_index_comparison,
    model_comparison_table,
    fit_parametric_benchmarks,
)

__version__ = "0.1.0"
__all__ = [
    # Main classes
    "SeverityFlow",
    "ConditionalSeverityFlow",
    "SeverityFlowResult",
    # Actuarial functions (callable without fitting a flow)
    "tvar",
    "quantile",
    "limited_expected_value",
    "ilf",
    "reinsurance_layer_cost",
    "burning_cost_summary",
    # Tail transform
    "TailTransform",
    "estimate_tail_params",
    "hill_estimator",
    # Data generation
    "generate_motor_bi_dataset",
    "DGPParams",
    "sample_severity",
    # Diagnostics
    "pit_values",
    "ks_test",
    "pit_histogram",
    "qq_plot_lognormal",
    "qq_plot_flow",
    "tail_plot",
    "tail_index_comparison",
    "model_comparison_table",
    "fit_parametric_benchmarks",
]
