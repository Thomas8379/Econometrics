# CLAUDE.md — econtools project rules

## Project: econtools

A Python econometrics toolkit wrapping `statsmodels`, `linearmodels`, `scipy`, `pandas`, and `pyarrow`.

**Spec reference:** `rough spec plan.txt` is the ground-truth capability inventory. Consult it before implementing any model, estimator, SE type, diagnostic, or plot function.

---

## Package conventions

- Package lives at `econtools/` inside `C:\Econometrics\`
- Install in editable mode: `pip install -e .` (pyproject.toml at repo root)
- Python available as `python` (not `python3`)

### Module layout

The package has two layers: a **new architecture** (authoritative) and **legacy modules** kept as re-export shims for backward compatibility. Shim removal is planned for the end of Phase D.

```
econtools/
├── _core/                    # Shared internals — import from here, not legacy modules
│   ├── types.py              # Estimate, FitMetrics, TestResult (canonical definitions)
│   ├── formatting.py         # _star, _fmt, _latex_star, _latex_escape (one copy)
│   └── cov_mapping.py        # resolve_cov_args(cov_type, *, backend="sm"|"lm")
│
├── model/
│   └── spec.py               # ModelSpec frozen dataclass (declarative model spec)
│
├── fit/                      # Estimation — the only public fitting API long-term
│   ├── estimators.py         # fit_model(spec, df) -> Estimate  ← use this
│   ├── _sm_adapter.py        # statsmodels backend (OLS, WLS, Probit)
│   ├── _lm_adapter.py        # linearmodels backend (IV2SLS, panel FE/RE/FD)
│   └── _builders.py          # build_sm_result, build_lm_iv_result, build_lm_panel_result
│
├── evaluation/               # All statistical tests and diagnostics
│   ├── hypothesis.py         # wald_test, f_test, t_test_coeff, lr_test, score_test, conf_int
│   ├── heteroskedasticity.py # breusch_pagan, white_test
│   ├── normality.py          # jarque_bera
│   ├── specification.py      # reset_test
│   ├── multicollinearity.py  # compute_vif, condition_number
│   ├── serial_correlation.py # durbin_watson, breusch_godfrey, ljung_box
│   ├── stationarity.py       # adf_test, kpss_test
│   ├── time_series.py        # granger_causality
│   ├── iv_checks.py          # wu_hausman, sargan_test, weak_iv_f
│   ├── panel_checks.py       # hausman_test, bp_lm_test
│   ├── influence.py          # Phase 3 stub (Cook's D, DFFITS, DFBETAs)
│   └── binary_metrics.py     # Binary classification metrics (extracted from reg_table)
│
├── output/                   # Rendering only — no statistical logic here
│   ├── tables/
│   │   └── content.py        # TableContent intermediate repr (decouples content from render)
│   ├── figures/              # (stubs)
│   ├── latex/
│   │   ├── engine.py         # find_engine, compile (no hardcoded paths; uses ECON_LATEX_FALLBACK_PATH)
│   │   ├── journal_profiles.py  # JournalProfile dataclass; ECONOMETRICA, AER instances
│   │   └── document.py       # assemble_document, write_document
│   └── knowledge_base/
│       ├── registry.py       # load_entry(id), render_entry(entry, **subs), list_entries()
│       └── entries/          # YAML: breusch_pagan, white_test, ols, hausman, iv_2sls
│
├── uncertainty/
│   ├── cov_estimators.py     # re-exports from _core/cov_mapping; HC0-3, HAC, cluster, kernel
│   └── bootstrap.py          # Phase 3 stub (pairs, wild, cluster bootstrap)
│
├── cli/
│   ├── main.py               # delegates to _cli_monolith.py
│   └── commands/             # stubs: data.py, analysis.py, output_cmds.py
├── _cli_monolith.py          # original CLI (1160 lines); preserved during migration
│
├── data/                     # Phase 0 — unchanged
│   ├── io.py                 # load_dta, save_curated, verify_hash
│   ├── inspect.py
│   ├── clean.py
│   ├── transform.py
│   ├── construct.py
│   ├── provenance.py
│   └── trend.py
│
│  ── Legacy shims (re-export from new modules; do not add new logic here) ──
├── models/                   # → _core/types + fit/_sm_adapter + fit/_lm_adapter
├── inference/                # → _core/types + _core/cov_mapping + evaluation/hypothesis
├── diagnostics/              # → evaluation/*
├── plots/                    # → output/figures (still has implementations; not yet moved)
└── tables/                   # → output/tables (still has implementations; not yet moved)
```

### The public fitting API

Prefer `fit_model` over the legacy convenience functions:

```python
from econtools.fit import fit_model
from econtools.model.spec import ModelSpec

spec = ModelSpec(dep_var="y", exog_vars=["x1", "x2"], estimator="ols")
result = fit_model(spec, df)  # returns Estimate
```

Legacy functions (`fit_ols`, `fit_iv_2sls`, `fit_panel`) remain available as thin wrappers and will be removed at the end of Phase D.

### Core types

All in `econtools._core.types`:

- **`Estimate`** (alias: `RegressionResult`) — point estimates, SEs, fit metrics, raw result
- **`FitMetrics`** — all optional fields (`r_squared`, `aic`, `r_squared_within`, etc.) default to `float("nan")`; safe for any model type
- **`TestResult`** — `(statistic, pvalue, df, distribution, null_hypothesis, reject, details)`

### Function signature conventions

- **Pure functions:** `(DataFrame, **kwargs) → DataFrame` for all pipeline steps. No in-place mutation.
- Every pipeline function logs its call to the provenance sidecar via `provenance.log_step(...)`.
- Plot functions always return `matplotlib.Figure`. Never call `plt.show()` inside a plot function.
- All public functions must have type annotations.
- Statistical logic never lives in rendering code (`output/` receives pre-computed content).

### Naming conventions

- Module-level: `snake_case` for functions, `PascalCase` for classes
- Derived column names: `log_<col>`, `lag_<col>_k<n>`, `d_<col>`, `<col>_sq`, `<col1>_x_<col2>`
- Curated Parquet files: `<name>_v<N>.parquet` (e.g. `wages_panel_v1.parquet`)
- Sidecar metadata: `<name>_v<N>_meta.json`

---

## Data layer rules

- `data_lake/raw/` is **immutable** — never write to it
- All raw `.dta` loads must call `verify_hash()` against `manifest.json`
- Save curated outputs with `save_curated()` which auto-writes the `_meta.json` sidecar
- Parquet settings: `engine='pyarrow'`, `compression='snappy'`, `index=True`

---

## Testing rules

- Tests live in `tests/` using `pytest`
- Run tests with: `python -m pytest tests/ -v`
- Test files mirror module structure: `tests/data/test_io.py` ↔ `econtools/data/io.py`
- Fixtures in `tests/conftest.py`; use small synthetic DataFrames, not real `.dta` files
- Every public function in the data layer must have at least one test

### Special test directories

- `tests/fit/` — tests for the `fit_model` dispatcher and new architecture
- `tests/validation/` — numerical validation against Wooldridge textbook datasets
  - Skip gracefully if `.dta` files are absent (use `@require_dta("name")` from `conftest`)
  - Do not use these for CI; they require the local data lake

### Phase 3 quarantine

Tests that depend on unimplemented Phase 3 features are marked:

```python
@pytest.mark.phase3
def test_something_binary(): ...
```

Run only non-Phase-3 tests: `python -m pytest tests/ -v -m "not phase3"`

The `phase3` marker is registered in `pyproject.toml`.

---

## SE and estimator choice

- For cross-section models: prefer `statsmodels`
- For panel/IV models: prefer `linearmodels`
- `cov_type` labels → library args resolved by `_core/cov_mapping.resolve_cov_args(cov_type, backend=...)`
- Valid labels: `classical`, `HC0`, `HC1`, `HC2`, `HC3`, `HAC`, `newey_west`, `cluster`
- Never invent SE types not in the spec

---

## TODO convention

Stubs use a standardised comment format:

```python
# TODO(econtools): <category> — <description>
```

Categories: `adapter`, `kb-entry`, `test`, `render`, `validate`, `cli-cmd`, `profile`

Collect all TODOs: `python scripts/collect_todos.py`

---

## Implementation phases

- **Phase 0 (data):** `econtools/data/` — complete
- **Phase 1 (foundations):** OLS, WLS, HC0–HC3, NW, F/t/Wald, fit metrics, RESET, BP, White, JB, VIF, residual plots, coef forest plot — complete
- **Phase 2 (IV + panel):** IV2SLS, LIML, first-stage diagnostics, FE/RE/Pooled OLS, clustered SEs, DK SEs, multi-model table
- **Phase 3 (binary + extensions):** Probit, Logit, marginal effects, influence suite, structural breaks, bootstrap SEs, binscatter
- **Phase 4 (system + asset pricing):** SUR, 3SLS, Fama-MacBeth, linear factor models, GMM-IV

Do not implement Phase N+1 functionality while Phase N is incomplete.

---

## What NOT to do

- Do not add `rich` pretty-printing inside library functions — only in CLI/notebook helpers
- Do not call `plt.show()` or `plt.savefig()` inside plot functions
- Do not mutate DataFrames in-place
- Do not skip `verify_hash()` for raw data loads
- Do not use `fastparquet` — always use `engine='pyarrow'`
- Do not add statistical logic to `output/` modules — rendering only
- Do not add new logic to legacy shim modules (`models/`, `inference/`, `diagnostics/`)
