# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-nflow: Motor BI Severity Distribution Modelling
# MAGIC
# MAGIC This notebook demonstrates the full workflow for fitting a normalizing flow
# MAGIC to UK motor bodily injury severity data. We use the synthetic bimodal DGP
# MAGIC included in the library, which has a known tail index and known TVaR/ILF
# MAGIC values for validation.
# MAGIC
# MAGIC **Workflow:**
# MAGIC 1. Generate synthetic UK motor BI dataset (bimodal, heavy-tailed)
# MAGIC 2. Fit parametric benchmarks (lognormal, gamma)
# MAGIC 3. Fit unconditional SeverityFlow with TTF tail
# MAGIC 4. Compare AIC/BIC and test log-likelihood
# MAGIC 5. Compute TVaR, ILF, reinsurance layer costs
# MAGIC 6. Fit conditional SeverityFlow (rating factors)
# MAGIC 7. Diagnostics: PIT, QQ plot, tail index comparison

# COMMAND ----------

# MAGIC %pip install insurance-nflow torch --index-url https://download.pytorch.org/whl/cpu

# COMMAND ----------

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_nflow import (
    SeverityFlow,
    ConditionalSeverityFlow,
    generate_motor_bi_dataset,
    tvar,
    ilf,
    fit_parametric_benchmarks,
    model_comparison_table,
    tail_index_comparison,
    burning_cost_summary,
)
from insurance_nflow.data import theoretical_tvar, theoretical_ilf, DGPParams
from insurance_nflow.diagnostics import (
    pit_histogram,
    qq_plot_lognormal,
    qq_plot_flow,
    tail_plot,
    ks_test,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic UK Motor BI Dataset

# COMMAND ----------

# Generate 10,000 policies with 5% claim rate -> ~500 claims
dataset = generate_motor_bi_dataset(n_policies=50_000, claim_rate=0.05, seed=42)
claims = dataset["claim_amount"]
age_band = dataset["age_band"].astype(float)
vehicle_group = dataset["vehicle_group"].astype(float)
region_encoded = np.array([
    {"london": 0, "south_east": 1, "midlands": 2, "north": 3, "scotland": 4}[r]
    for r in dataset["region"]
], dtype=float)

print(f"Number of claims: {len(claims)}")
print(f"Mean severity: GBP {np.mean(claims):,.0f}")
print(f"Median severity: GBP {np.median(claims):,.0f}")
print(f"95th percentile: GBP {np.quantile(claims, 0.95):,.0f}")
print(f"99th percentile: GBP {np.quantile(claims, 0.99):,.0f}")
print(f"Max: GBP {np.max(claims):,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribution overview

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram on log scale
axes[0].hist(np.log10(claims), bins=50, alpha=0.7, color="#4477AA")
axes[0].set_xlabel("log10(Claim amount GBP)")
axes[0].set_ylabel("Count")
axes[0].set_title("Severity distribution (log10 scale)")
axes[0].axvline(np.log10(5_000), color="red", linestyle="--", alpha=0.5, label="5k")
axes[0].axvline(np.log10(100_000), color="orange", linestyle="--", alpha=0.5, label="100k")
axes[0].legend()

# Tail plot
x_sorted = np.sort(claims)
surv = 1 - (np.arange(1, len(x_sorted) + 1) - 0.5) / len(x_sorted)
axes[1].loglog(x_sorted, surv, color="#4477AA")
axes[1].set_xlabel("Claim amount GBP")
axes[1].set_ylabel("P(X > x)")
axes[1].set_title("Survival function (log-log)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Parametric Benchmarks

# COMMAND ----------

ll_benchmarks, k_benchmarks = fit_parametric_benchmarks(claims)
print("Parametric benchmark log-likelihoods:")
for model, ll in sorted(ll_benchmarks.items(), key=lambda x: -x[1]):
    aic = 2 * k_benchmarks[model] - 2 * ll
    print(f"  {model:<12}: log-lik/obs = {ll/len(claims):.4f}, AIC = {aic:.1f}")

# COMMAND ----------

# Lognormal QQ plot (baseline: should show tail deviation)
fig = qq_plot_lognormal(claims, title="QQ Plot: Observed vs Lognormal")
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit Unconditional SeverityFlow

# COMMAND ----------

flow = SeverityFlow(
    n_transforms=6,
    hidden_features=[64, 64],
    tail_transform=True,
    tail_mode="fix",       # Hill estimator then fix
    max_epochs=200,
    patience=20,
    batch_size=256,
    lr=1e-3,
    seed=42,
)

print("Fitting unconditional SeverityFlow...")
result = flow.fit(claims)
print(f"\n{result}")
print(f"Training time: {result.training_seconds:.1f}s")
print(f"Tail lambda+: {result.tail_lambda_pos:.3f} (right tail weight)")
print(f"Tail lambda-: {result.tail_lambda_neg:.3f} (left tail weight)")

# COMMAND ----------

# Training curve
history = result.training_history
epochs = [h["epoch"] for h in history]
train_losses = [h["train_loss"] for h in history]
val_losses = [h["val_loss"] for h in history]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, train_losses, label="Train NLL", color="#4477AA")
ax.plot(epochs, val_losses, label="Val NLL", color="#EE7733")
ax.set_xlabel("Epoch")
ax.set_ylabel("Negative log-likelihood")
ax.set_title("Training curve")
ax.legend()
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Model Comparison

# COMMAND ----------

# Add flow to comparison
ll_benchmarks["flow (NSF+TTF)"] = float(result.train_log_likelihood * len(claims))
k_benchmarks["flow (NSF+TTF)"] = result.n_parameters

table = model_comparison_table(claims, ll_benchmarks, k_benchmarks)

print(f"\nModel comparison (sorted by AIC, n={len(claims)}):")
print(f"{'Model':<20} {'k':>8} {'logL/obs':>10} {'AIC':>15} {'BIC':>15}")
print("-" * 70)
for row in table:
    print(
        f"{row['model']:<20} {row['n_params']:>8,} "
        f"{row['log_lik_per_obs']:>10.4f} "
        f"{row['aic']:>15,.1f} "
        f"{row['bic']:>15,.1f}"
    )

print("\nNote: AIC heavily penalises the flow's ~100k parameters.")
print("Use test log-likelihood per observation as the primary metric.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Actuarial Outputs

# COMMAND ----------

print("=" * 50)
print("Actuarial Summary — Unconditional Severity Flow")
print("=" * 50)

# Sample a large portfolio
print("\nSampling 500k claims from flow...")
samples = flow.sample(500_000)

summary = burning_cost_summary(samples)
print(f"\nMean severity:    GBP {summary['mean']:>12,.0f}")
print(f"Median severity:  GBP {summary['median']:>12,.0f}")
print(f"Std deviation:    GBP {summary['std']:>12,.0f}")
print(f"Skewness:              {summary['skewness']:>12.2f}")

print("\nQuantiles:")
for p, q in summary["quantiles"].items():
    print(f"  VaR({100*p:.1f}%):  GBP {q:>12,.0f}")

print("\nTVaR:")
for p, tv in summary["tvar"].items():
    print(f"  TVaR({100*p:.1f}%): GBP {tv:>12,.0f}")

print("\nILF Curve (basic limit = GBP 50,000):")
for limit, factor in sorted(summary["ilf"].items()):
    print(f"  GBP {limit:>10,.0f}: {factor:.4f}")

# COMMAND ----------

# Reinsurance layer pricing
print("\nReinsurance Layer Pricing:")
layers = [
    (50_000, 100_000),
    (100_000, 150_000),
    (150_000, 350_000),
    (500_000, 500_000),
]
from insurance_nflow.actuarial import reinsurance_layer_cost
for att, lim in layers:
    cost = reinsurance_layer_cost(samples, att, lim)
    print(f"  GBP {lim:>7,.0f} xs {att:>7,.0f}: GBP {cost:>8,.2f} per occurrence")

# COMMAND ----------

# Compare flow TVaR against DGP theoretical TVaR
print("\nValidation against DGP theoretical TVaR:")
for p in [0.90, 0.95, 0.99]:
    tv_flow = tvar(samples, p)
    tv_dgp = theoretical_tvar(p, 3, 5, "midlands", n_mc=200_000, seed=42)
    rel_err = abs(tv_flow - tv_dgp) / tv_dgp
    print(f"  TVaR({100*p:.0f}%): Flow={tv_flow:>10,.0f}  DGP={tv_dgp:>10,.0f}  RelErr={rel_err:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Diagnostics

# COMMAND ----------

# QQ plot: flow vs observed
fig = qq_plot_flow(
    claims,
    samples,
    title="QQ Plot: Flow vs Observed (motor BI)",
)
display(fig)
plt.close()

# COMMAND ----------

# Tail plot comparison
fig = tail_plot(
    claims,
    x_simulated=samples[:10_000],
    title="Tail Plot: Flow vs Observed",
)
display(fig)
plt.close()

# COMMAND ----------

# Tail index comparison
tail_result = tail_index_comparison(
    claims,
    flow_lambda_pos=result.tail_lambda_pos,
)
print("\nTail Index Comparison:")
print(f"  Hill estimator (median):     {tail_result['hill_median']:.3f}")
print(f"  Hill estimator (std):        {tail_result['hill_std']:.3f}")
print(f"  Pareto MLE (upper 10%):      alpha={tail_result['pareto_alpha_upper10pct']:.3f}")
print(f"  Flow TTF lambda+:            {tail_result['flow_lambda_pos']:.3f}")
print(f"  Flow equivalent alpha:       {tail_result['flow_equivalent_alpha']:.3f}")
print(f"  Lognormal (tail index):      {tail_result['lognormal_tail_index']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Conditional Severity Flow

# COMMAND ----------

# Build context matrix: age_band, vehicle_group, region (encoded)
context = np.column_stack([age_band, vehicle_group, region_encoded])
print(f"Context shape: {context.shape}")
print("Columns: age_band (1-5), vehicle_group (1-10), region (0-4)")

# COMMAND ----------

cflow = ConditionalSeverityFlow(
    context_features=3,
    n_transforms=4,
    hidden_features=[64, 64],
    tail_transform=True,
    tail_mode="fix",
    max_epochs=150,
    patience=15,
    batch_size=256,
    lr=1e-3,
    seed=42,
)

print("Fitting conditional SeverityFlow...")
cresult = cflow.fit(claims, context=context, exposure_weights=dataset["exposure"])
print(f"\n{cresult}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conditional TVaR comparison by risk profile

# COMMAND ----------

# Define risk profiles to compare
risk_profiles = {
    "Young (17-25), high vehicle group, London":     np.array([[1, 9, 0]]),
    "Mid-age (36-50), mid vehicle group, Midlands":  np.array([[3, 5, 2]]),
    "Senior (65+), low vehicle group, North":        np.array([[5, 3, 3]]),
    "Mid-age (36-50), high vehicle group, South-East": np.array([[3, 8, 1]]),
}

print(f"\n{'Risk Profile':<55} {'TVaR(90%)':>10} {'TVaR(95%)':>10} {'TVaR(99%)':>10}")
print("-" * 90)
for name, ctx in risk_profiles.items():
    tv90 = cflow.conditional_tvar(ctx, 0.90, n_samples=50_000)
    tv95 = cflow.conditional_tvar(ctx, 0.95, n_samples=50_000)
    tv99 = cflow.conditional_tvar(ctx, 0.99, n_samples=50_000)
    print(f"{name:<55} {tv90:>10,.0f} {tv95:>10,.0f} {tv99:>10,.0f}")

# COMMAND ----------

# Conditional ILF for high-risk vs low-risk
print("\nConditional ILF Curves (basic limit = GBP 50,000):")
limits = [25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]

young_london = np.array([[1, 9, 0]])
mid_north = np.array([[3, 5, 3]])

ilf_young = cflow.ilf(limits, basic_limit=50_000, context=young_london, n_samples=200_000)
ilf_mid = cflow.ilf(limits, basic_limit=50_000, context=mid_north, n_samples=200_000)

print(f"\n{'Limit':>12} {'Young-London ILF':>18} {'Mid-North ILF':>15}")
print("-" * 50)
for lim in limits:
    print(f"GBP {lim:>9,.0f}   {ilf_young[float(lim)]:>10.4f}            {ilf_mid[float(lim)]:>10.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The normalizing flow with TTF tail:
# MAGIC
# MAGIC 1. Captures the bimodal shape of motor BI severity (soft-tissue + catastrophic)
# MAGIC 2. Provides calibrated heavy-tail behaviour via the TTF layer
# MAGIC 3. Conditions naturally on rating factors without requiring a parametric family
# MAGIC 4. Produces TVaR, ILF, and reinsurance layer costs that can be compared against
# MAGIC    parametric benchmarks
# MAGIC
# MAGIC **Primary limitation**: with ~100k parameters, AIC/BIC heavily penalise the flow
# MAGIC vs 2-parameter families. Use held-out test log-likelihood per observation as the
# MAGIC primary comparison metric. The flow wins on test log-lik when the sample is large
# MAGIC enough for the neural network to generalise.
# MAGIC
# MAGIC **Recommended use**: deploy alongside parametric families, not instead of them.
# MAGIC The conditional TVaR and ILF outputs are the pricing killer feature — no parametric
# MAGIC family in a GLM framework provides these directly.
