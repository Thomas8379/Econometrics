# Bootstrap Inference — econtools

## Overview

The `econtools.uncertainty.bootstrap` module provides reproducible
bootstrap inference for OLS and 2SLS (IV) regression estimators. It
computes bootstrap standard errors, confidence intervals, and p-values,
and supports four resampling schemes suitable for different data
structures.

## Quick Start

```python
import pandas as pd
from econtools.uncertainty.bootstrap import run_bootstrap

df = pd.read_stata("data_lake/raw/wooldridge_and_oleg/wage1.dta", ...)

result = run_bootstrap(
    data=df,
    y="lwage",
    X=["educ", "exper", "tenure"],
    estimator="ols",
    bootstrap_method="iid_pairs",
    B=1999,
    seed=42,
)

# Point estimates
print(result["point_estimate"]["params"])

# Bootstrap standard errors
print(result["bootstrap"]["se"])

# 95% confidence intervals (percentile method)
print(result["bootstrap"]["ci"]["percentile"])

# Bootstrap p-values
print(result["bootstrap"]["pvalues"])
```

---

## Bootstrap Methods

### `iid_pairs` — i.i.d. pairs bootstrap

**When to use:** Cross-sectional data with i.i.d. observations. The
default for non-panel, non-clustered data.

**How it works:** Sample _n_ rows with replacement from the estimation
sample. Refit the estimator on each bootstrap sample.

**Consistency:** Consistent under i.i.d. sampling. SE converges to the
heteroskedasticity-robust SE as _n → ∞_ under misspecification.

---

### `wild` — wild bootstrap

**When to use:** Cross-sectional data with heteroskedasticity. Superior
to `iid_pairs` when errors are heteroskedastic but observations are
independent.

**How it works:**

1. Fit the estimator on the original data → β̂, residuals û.
2. For each replication b, draw auxiliary weights v_b (Rademacher or
   Mammen) and form y*_b = X β̂ + û ⊙ v_b.
3. Refit the estimator on (y*_b, X) → β̂*_b.

For **2SLS wild bootstrap**, X and Z are held fixed; y* is formed using
structural residuals from the second stage.

**Distributions:**
- `rademacher` (default): v ~ {−1, +1} each with probability 0.5.
  Recommended for most cases.
- `mammen`: Two-point Mammen (1993) distribution; preserves first three
  moments of N(0,1). Can improve size in small samples.

---

### `cluster_pairs` — cluster pairs bootstrap

**When to use:** Data with one-way clustering (e.g., by state, school,
or firm). Accounts for arbitrary within-cluster correlation.

**How it works:** Let G = number of unique clusters. Sample G clusters
with replacement; include all observations from each sampled cluster.
Refit on the resampled dataset.

**Warning:** Bootstrap inference is unreliable when G < 30. The module
issues a warning and records it in the manifest.

**Choosing the cluster variable:** The cluster should correspond to the
unit of potential dependence (e.g., the treatment assignment unit).

---

### `panel_cluster_id` — panel bootstrap by entity id

**When to use:** Panel data (repeated observations per entity) with
potential within-entity correlation. The default safe bootstrap for
panels when `id_col` is provided.

**How it works:** Equivalent to `cluster_pairs` with `cluster = id_col`.
Resamples entity ids with replacement, keeping all time observations for
each sampled entity intact. If `time_col` is provided, within-id
observations are sorted by time in the resampled dataset.

**Note:** This bootstrap is valid for pooled OLS and pooled 2SLS. It is
not designed for fixed-effects estimators (which difference out the
entity-level variation).

---

## Not Yet Implemented

The following methods raise `NotImplementedError`:

- `block_time_series` — moving-block bootstrap for time series
- `stationary_bootstrap` — Politis–Romano stationary bootstrap

---

## Confidence Intervals

### Percentile CI

Quantiles of the bootstrap coefficient draws:

    CI = [Q(β̂*, α/2), Q(β̂*, 1−α/2)]

### Basic CI (reverse percentile)

Uses the distribution of β̂* − β̂ to correct for bias in the percentile CI:

    CI = [2β̂ − Q(β̂*, 1−α/2), 2β̂ − Q(β̂*, α/2)]

---

## Bootstrap p-values

Two-sided p-values based on the centred bootstrap distribution. Let
d_b = β̂*_b − β̂ (centred draws). Then:

    p = 2 · min(
        (#{d_b ≤ 0} + 1) / (B + 1),
        (#{d_b ≥ 0} + 1) / (B + 1)
    )

Adding 1 to numerator and denominator is the finite-sample adjustment of
Davidson & MacKinnon (2000), which avoids zero p-values. The p-value
tests H₀: β = 0.

---

## "Bootstrap as Estimation" (Bagged Coefficients)

The output includes:

- `bagged_mean`: mean of the B bootstrap coefficient draws
- `bagged_median`: median of the B bootstrap coefficient draws

**Important:** These change the *estimator*, not just its inference. The
mean of bootstrap draws is the "bagged" estimator (Breiman 1996), which
can reduce variance at the cost of some bias. Use these for
exploratory purposes only; standard inference should use `point_estimate`.

---

## Reproducibility

All randomness is controlled via a single integer `seed`:

1. A `numpy.random.SeedSequence(seed)` is created.
2. `B` child sequences are spawned deterministically.
3. Each replication b uses `numpy.random.default_rng(child_b)`.

This means:
- Results are identical across repeated runs with the same seed.
- Results are identical for `n_jobs=1` and `n_jobs>1` (parallel threads
  each receive their own independent RNG).

---

## Manifest

Every run produces a manifest (JSON) capturing:

- Timestamp and git commit hash
- Package versions (numpy, pandas, statsmodels, econtools, Python)
- Full configuration and its SHA-256 hash
- Sample statistics (n_obs, n_dropped, column lists)
- Cluster count and any warnings

The config hash is computed as SHA-256 of the JSON-serialised config
with sorted keys, so it is stable regardless of Python dict ordering.

---

## CLI

```
econtools bootstrap --config path/to/config.yaml [--manifest-path PATH] [--quiet]
```

The YAML config file controls all parameters. See `bootstrap_default_config.yaml`
in the repo root for a fully annotated example.

---

## Examples

### OLS with iid pairs and wild bootstrap

```python
result_iid = run_bootstrap(df, y="lwage", X=["educ", "exper"],
                            estimator="ols", bootstrap_method="iid_pairs",
                            B=1999, seed=42)

result_wild = run_bootstrap(df, y="lwage", X=["educ", "exper"],
                             estimator="ols", bootstrap_method="wild",
                             wild_dist="rademacher", B=1999, seed=42)
```

### Clustered bootstrap

```python
result_cl = run_bootstrap(df, y="score", X=["class_size", "income"],
                           estimator="ols", bootstrap_method="cluster_pairs",
                           cluster="school_id", B=1999, seed=42)
```

### Panel bootstrap by entity id

```python
result_panel = run_bootstrap(df, y="lwage", X=["exper", "tenure"],
                              estimator="ols",
                              bootstrap_method="panel_cluster_id",
                              id_col="person_id", time_col="year",
                              B=1999, seed=42)
```

### 2SLS with iid pairs bootstrap

```python
result_iv = run_bootstrap(
    df, y="lwage",
    X=["exper", "tenure"],        # exogenous regressors
    endog=["educ"],               # endogenous
    Z=["fatheduc", "motheduc"],   # excluded instruments
    estimator="2sls",
    bootstrap_method="iid_pairs",
    B=1999, seed=42,
)
print(result_iv["point_estimate"]["first_stage_diagnostics"])
# {'first_stage_F': 55.4, 'weak_iv_flag': False}
```

### Saving draws and reading results

```python
result = run_bootstrap(..., save_draws_path="results/draws.npy")
draws = result["bootstrap"]["draws"]  # ndarray (B, k)

import numpy as np
draws_loaded = np.load("results/draws.npy")  # same array

# Read manifest
import json
manifest = result["_manifest"]
print(manifest["config_hash"])   # sha256:...
print(manifest["git_commit"])    # abc1234
```

---

## References

- Davidson, R. & MacKinnon, J.G. (2000). Bootstrap tests: How many
  bootstraps? *Econometric Reviews*, 19(1), 55–68.
- Mammen, E. (1993). Bootstrap and wild bootstrap for high-dimensional
  linear models. *Annals of Statistics*, 21(1), 255–285.
- Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap.
  *JASA*, 89(428), 1303–1313.
- Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2),
  123–140.
