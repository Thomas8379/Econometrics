# Active Memory — econtools session context

Use this file to record in-flight decisions, open questions, and
short-term context that should carry across conversations.

---

## Current focus

Phase 1 just completed and committed. Ready to begin Phase 2.

No in-flight work — clean slate.

---

## Architecture decisions locked in

### RegressionResult normalisation layer
`fit_ols` / `fit_wls` return a frozen `RegressionResult` dataclass.
Phase 2 linearmodels-backed estimators **must** also return `RegressionResult`
so diagnostics / plots / tables work unchanged.
Attribute mapping for linearmodels results:
- `.std_errors` → `bse`
- `.tstats` → `tvalues`
- `.pvalues` → `pvalues`
- `.resids` → `resid`
- `.fitted_values` → `fitted`

### SE label convention
Friendly labels in `VALID_COV_TYPES`; `resolve_cov_args()` maps to library kwargs.
Phase 2 will need a **separate resolver** for linearmodels (different kwarg names).
Add `resolve_linearmodels_cov_args()` to `econtools/inference/se_types.py`.

### TestResult
Frozen dataclass in `econtools/inference/hypothesis.py`.
All diagnostic functions return `TestResult` — do not create new result types.

---

## Open decisions for Phase 2

- [ ] Which linearmodels version is installed? Check `import linearmodels; print(linearmodels.__version__)`
- [ ] Confirm linearmodels IV2SLS constructor signature: `IV2SLS(dependent, exog, endog, instruments)`
- [ ] Decide entity/time handling: accept column names and set index internally, or require pre-set MultiIndex?
  - Recommendation: accept column names, set MultiIndex internally (consistent with Phase 0 `set_panel_index`)
- [ ] `compare_table`: row-per-regressor, columns = models — design before implementing

---

## Test fixtures added in Phase 1 (in tests/conftest.py)

| Fixture | Description |
|---------|-------------|
| `ols_data` | y = 2 + 3x1 + 0.5x2 + N(0,1), n=200, seed=42 |
| `ols_result` | `fit_ols(ols_data, "y", ["x1","x2"])` |
| `heteroskedastic_data` | e = x·N(0,1), n=500, seed=1 |
| `non_normal_data` | e ~ chi2(2)−2, n=500, seed=2 |
| `misspecified_data` | y = 1+2x+1.5x², n=300, seed=3 |

Phase 2 will need: `panel_data`, `iv_data` (with valid instruments).

---

## File locations cheatsheet

```
econtools/
  models/     _results.py  ols.py  __init__.py
  inference/  se_types.py  hypothesis.py  __init__.py
  diagnostics/ heteroskedasticity.py  normality.py  specification.py  multicollinearity.py  __init__.py
  plots/      residual_plots.py  coefficient_plots.py  __init__.py
  tables/     reg_table.py  __init__.py
  iv/         (empty — Phase 2)
  panel/      (empty — Phase 2)
tests/
  conftest.py          (shared fixtures)
  models/              test_results.py  test_ols.py
  inference/           test_se_types.py  test_hypothesis.py
  diagnostics/         test_heteroskedasticity.py  test_normality.py  test_specification.py  test_multicollinearity.py
  plots/               test_residual_plots.py  test_coefficient_plots.py
  tables/              test_reg_table.py
rough spec plan.txt    (ground truth — consult before implementing anything)
```
