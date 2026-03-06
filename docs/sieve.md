# Sieve Infrastructure

The sieve subsystem provides systematic, **honest** search over candidate model specifications — functional forms, control sets, and instrument choices — with built-in anti-p-hacking guardrails.

## Quick start

```bash
econtools sieve --config sieve.yaml
econtools sieve-report --run ./sieve_output
```

```python
from econtools.sieve import run_sieve

results = run_sieve(
    data=df,
    y="log_wage",
    base_X=["educ", "exper", "tenure"],
    estimator="ols",
    sieve_spec={
        "generators": {"features": {"polynomial": {"enabled": True, "degree": 2}}},
        "protocol": {"mode": "holdout", "test_frac": 0.30},
        "selection": {"primary_metric": "rmse", "top_k": 3},
    },
    seed=12345,
    output_dir="./sieve_output",
)

for cand in results["selected"]:
    print(cand.candidate_hash, cand.X_terms)

print(results["leaderboard"].head())
```

---

## Core concepts

### Sieve

A **sieve** is a search space definition + evaluation protocol + selection rule applied to a dataset. The goal is to find promising model specifications while maintaining statistical honesty.

### Candidate

A **candidate** is a fully-specified model: which variables are included (after transforms), which estimator is used, and (for IV) which instruments are selected. Every candidate has a stable `candidate_hash` — the same specification always produces the same hash.

### Protocol

The **protocol** controls *how* candidates are evaluated:

| Protocol | When to use |
|---|---|
| `holdout` (default) | Any goal; honest by default — selection on train, confirm on test |
| `cv` | Predictive modeling with many candidates |
| `crossfit` | IV / causal inference with instrument or functional-form search |

> **Rule of thumb:** for causal inference, use `crossfit` or `holdout`. Never use `allow_in_sample_selection: true` and report results as confirmatory.

### Guardrails

Candidates are automatically rejected if they violate hard constraints:

- **`weak_iv`**: first-stage F < `min_first_stage_f` (default: 10)
- **`too_many_terms`**: exceeds `max_terms` complexity limit
- **`too_many_instruments`**: exceeds `max_instruments`
- **`sign_constraint`**: coefficient sign violates a user-specified constraint
- **`fit_failed`**: numerical failure during estimation

Every rejected candidate is logged with its reason code — the full leaderboard is always saved.

---

## Protocol choice: prediction vs causal inference

### Prediction (OLS functional-form search)

```yaml
protocol:
  mode: cv
  k: 5
scoring:
  primary_metric: rmse
```

The goal is out-of-sample prediction error. Use k-fold CV. AIC/BIC can be used as secondary metrics but are in-sample and labeled accordingly.

### Causal inference (IV instrument search)

```yaml
protocol:
  mode: crossfit
  k: 5
scoring:
  primary_metric: first_stage_f
constraints:
  min_first_stage_f: 10.0
```

Cross-fitting prevents selection from leaking into the outcome equation. The first-stage F statistic is the primary strength criterion; the overidentification test p-value is a diagnostic (not a selection criterion by itself).

> **Warning on overidentification tests:** Sargan/Hansen tests are informative but should not be used as the sole selection criterion. A candidate can appear valid in-sample while being invalid out-of-sample. Always use holdout or cross-fitting for IV sieves.

### Exploratory mode

If you need in-sample selection for pure exploration:

```yaml
protocol:
  allow_in_sample_selection: true
```

All outputs will be stamped `EXPLORATORY ONLY`. **Do not report exploratory results as confirmatory.**

---

## Post-selection inference

After sieve selection, standard p-values are **not valid** — they ignore the selection process. Valid approaches:

1. **Holdout confirmation** (default): select on train, report on held-out test set. P-values on the test set are valid *for the selected model* (but not for the selection procedure itself).
2. **Pre-registered shortlist**: if you pre-register ≤ 3 models before seeing any results, those model's p-values are unaffected by the sieve.
3. **Cross-fitting with sample splitting**: Chernozhukov et al. (2018) approach for causal parameters.

The sieve always reports the search size and selection protocol. Both must be disclosed in published work.

---

## Determinism guarantee

Same `data` + same `sieve_spec` + same `seed` ⇒ identical:
- candidate set (same hashes)
- train/test splits (same fold indices)
- scores and rankings
- selected models

Parallel execution (`n_jobs > 1`) produces the same results as sequential when `deterministic_parallel: true`.

---

## Reproducing a run from the manifest

```python
from econtools.sieve.api import load_sieve_results
from econtools.sieve.manifest import load_manifest

# Load everything
results = load_sieve_results("./sieve_output")
manifest = results["manifest"]
print(manifest["run_id"], manifest["config_hash"], manifest["dataset_fingerprint"])

# The manifest contains the full config hash; re-running with the same
# config and seed on data with the same fingerprint reproduces the run.
```

---

## Feature generators

### Polynomial terms

```yaml
generators:
  features:
    polynomial:
      enabled: true
      degree: 2
      vars: ["educ", "exper"]   # null = all base_X
```

Adds `educ_pow2`, `exper_pow2`. Setting `orthogonalize: true` reduces multicollinearity.

### Interaction terms

```yaml
generators:
  features:
    interactions:
      enabled: true
      max_order: 2
      whitelist: [["educ", "exper"]]  # only this pair
```

Adds `educ_x_exper` (alphabetically ordered names).

### Log transforms

```yaml
generators:
  features:
    log:
      enabled: true
      vars: ["wage"]
      shift: 0.0   # use shift > 0 if zeros present
```

Variables with non-positive values (after shift) are skipped with a warning.

### Restricted cubic splines

```yaml
generators:
  features:
    splines:
      enabled: true
      n_knots: 4
      vars: ["age"]
```

Adds `age_rcs4_b1`, `age_rcs4_b2` (k-2 basis columns for k knots). Knots at evenly-spaced quantiles.

---

## Instrument generators (IV)

### Lags

```yaml
generators:
  instruments:
    lags:
      enabled: true
      lag_ks: [1, 2]
      time_col: year
      id_col: id          # panel: lag within entity
```

Adds `lag_z_k1`, `lag_z_k2`. Lagging is done after sorting by `(id_col, time_col)`.

### LOO group means (preferred over simple group means)

```yaml
generators:
  instruments:
    loo_group_means:
      enabled: true
      group_col: industry
      vars: ["z_baseline"]
```

Computes leave-one-out group mean to avoid reflection bias.

---

## Output artifacts

After `run_sieve(..., output_dir="./out")`:

```
./out/
├── run_manifest.json        # Full provenance: config hash, data fingerprint, splits
├── leaderboard.parquet      # All candidates: scores, rejection reasons
├── leaderboard.csv          # Same, CSV format
└── selected_models/
    ├── model_<hash1>.json   # Model card: coefficients, scores, transforms
    └── model_<hash2>.json
```

---

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M. et al. (2018). "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). "Robust inference with multiway clustering." *Journal of Business & Economic Statistics*.
- Harrell, F. E. (2015). *Regression Modeling Strategies*. Springer.
