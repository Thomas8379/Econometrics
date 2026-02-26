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

```
econtools/
├── data/          # Phase 0 — data pipeline (io, inspect, clean, transform, construct, provenance)
├── models/        # Phase 1+ — estimators (OLS, WLS, GLS, GLM)
├── iv/            # Phase 2  — IV estimators (IV2SLS, LIML, GMM-IV)
├── panel/         # Phase 2  — panel estimators (FE, RE, PooledOLS, FD)
├── inference/     # Phase 1+ — SE types, Wald/F/LM tests, bootstrap
├── diagnostics/   # Phase 1+ — assumption tests (BP, White, JB, RESET, VIF, influence)
├── plots/         # Phase 1+ — Figure-returning plot functions (never call plt.show())
└── tables/        # Phase 1+ — reg_table, compare_table, LaTeX/HTML export
```

### Function signature conventions

- **Pure functions:** `(DataFrame, **kwargs) → DataFrame` for all pipeline steps. No in-place mutation.
- Every pipeline function logs its call to the provenance sidecar via `provenance.log_step(...)`.
- Plot functions always return `matplotlib.Figure`. Never call `plt.show()` inside a plot function.
- All public functions must have type annotations.

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

---

## SE and estimator choice

- For cross-section models: prefer `statsmodels`
- For panel/IV models: prefer `linearmodels`
- `cov_type` labels → library args are mapped in spec §2
- Never invent SE types not in the spec

---

## Implementation phases

- **Phase 0 (data):** `econtools/data/` — fully implement per spec §1.0–1.10
- **Phase 1 (foundations):** OLS, WLS, HC0–HC3, NW, F/t/Wald, fit metrics, RESET, BP, White, JB, VIF, residual plots, coef forest plot
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
