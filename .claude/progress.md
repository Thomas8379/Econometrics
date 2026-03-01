# econtools — Project Progress

## Current status

**Phase 1 complete.** 143 tests passing.

| Phase | Status | Tests |
|-------|--------|-------|
| 0 — Data layer | ✅ Complete | 77 |
| 1 — OLS foundations | ✅ Complete | 66 new (143 total) |
| 2 — IV + Panel | ⬜ Not started | — |
| 3 — Binary + extensions | ⬜ Not started | — |
| 4 — System + asset pricing | ⬜ Not started | — |

Last commit: `ae30df1` — "Add econtools Phase 1: OLS foundations, diagnostics, plots, and tables"

---

## Phase 1 deliverables (done)

### Models
- `econtools/models/_results.py` — `FitMetrics`, `RegressionResult` frozen dataclasses
- `econtools/models/ols.py` — `fit_ols`, `fit_wls`, `fit_ols_formula`

### Inference
- `econtools/inference/se_types.py` — `resolve_cov_args`, `VALID_COV_TYPES`
- `econtools/inference/hypothesis.py` — `TestResult`, `wald_test`, `f_test`, `t_test_coeff`, `conf_int`

### Diagnostics
- `econtools/diagnostics/heteroskedasticity.py` — `breusch_pagan`, `white_test`
- `econtools/diagnostics/normality.py` — `jarque_bera`
- `econtools/diagnostics/specification.py` — `reset_test`
- `econtools/diagnostics/multicollinearity.py` — `compute_vif`, `condition_number`

### Plots (always return Figure, never call plt.show)
- `econtools/plots/residual_plots.py` — `plot_residuals_vs_fitted`, `plot_scale_location`, `plot_qq`
- `econtools/plots/coefficient_plots.py` — `plot_coef_forest`

### Tables
- `econtools/tables/reg_table.py` — `reg_table(result, stars, se_in_parens, digits, format)`
  - Formats: `"text"`, `"latex"`, `"html"`

---

## Phase 2 — next up (spec §3, §4, §5)

### IV estimators (`econtools/iv/`)
- `fit_iv2sls(df, dep_var, exog_vars, endog_vars, instruments)` — via linearmodels
- `fit_liml(...)` — via linearmodels
- First-stage diagnostics: F-stat, Cragg-Donald, Anderson-Rubin
- SE types: HC0–HC3, HAC, cluster (linearmodels `cov_type` labels differ from statsmodels)

### Panel estimators (`econtools/panel/`)
- `fit_fe(df, dep_var, exog_vars, entity, time)` — Fixed Effects via linearmodels
- `fit_re(...)` — Random Effects
- `fit_pooled(...)` — Pooled OLS
- `fit_fd(...)` — First Differences
- Driscoll-Kraay SEs (`DK`), clustered SEs

### Inference additions
- Multi-model comparison table (`compare_table` in `econtools/tables/`)
- Hausman test (FE vs RE)

### Spec reference: `rough spec plan.txt` §3–5

---

## Phase 1 known gotchas (avoid repeating)

- `jarque_bera` → `statsmodels.stats.stattools`, NOT `statsmodels.stats.diagnostic`
- `wald_test(use_f=False)` result has `.df_denom` but NOT `.df_num`
- Suppress statsmodels 0.14 FutureWarning: `wald_test(..., scalar=False)`
- Always wrap comparison in `bool()` or use Python `float` before `< 0.05` to avoid `numpy.bool_`
- linearmodels result attributes differ: `.std_errors` (not `.bse`), `.tstats` (not `.tvalues`)
- Plot test files need `matplotlib.use("Agg")` at module level
