# econtools — Documentation Index

## Module guides

| Doc | What it covers |
|-----|----------------|
| [bootstrap.md](bootstrap.md) | Bootstrap inference for OLS and 2SLS: resampling methods, CI and p-value formulas, CLI, reproducibility |
| [binary_model_output.md](binary_model_output.md) | Extended output for Logit/Probit: likelihood metrics, prediction quality, marginal effects |
| [pub_latex.md](pub_latex.md) | Publication-quality LaTeX tables: `ResultsTable`, `SummaryTable`, `DiagnosticsTable` — panels, booktabs, notes |
| [sieve.md](sieve.md) | Sieve infrastructure: functional-form search, instrument search, honest evaluation, reproducibility |

## Quick links

- **Run a regression:** `fit_model(spec, df)` in `econtools.fit`
- **Publication LaTeX table:** `ResultsTable([res1, res2], panels=[...])` in `econtools.output.tables`
- **Summary statistics table:** `SummaryTable(df, stats=["mean","std","N"])` in `econtools.output.tables`
- **Bootstrap SEs:** `run_bootstrap(data, y, X, ...)` in `econtools.uncertainty.bootstrap`
- **Run a sieve:** `run_sieve(data, y, base_X, estimator, sieve_spec=...)` in `econtools.sieve`
- **CLI — bootstrap:** `econtools bootstrap --config cfg.yaml`
- **CLI — sieve:** `econtools sieve --config sieve.yaml`
- **Tests (fast):** `python -m pytest tests/ -v -m "not phase3 and not slow"`

## Implementation phases

| Phase | Status | Scope |
|-------|--------|-------|
| 0 — Data layer | Complete | `econtools/data/` — load, clean, transform, provenance |
| 1 — OLS foundations | Complete | OLS/WLS, HC0–HC3/HAC/cluster SEs, diagnostics, plots, tables |
| 2 — IV + panel | Complete | IV-2SLS, LIML, FE/RE/FD, clustered SEs, multi-model tables |
| 3 — Binary + extensions | In progress | Probit (stub), bootstrap (complete), influence (stub) |
| — Publication output | Complete | `pub_latex`: `ResultsTable`, `SummaryTable`, `DiagnosticsTable` |
| — Sieve infrastructure | Complete | Functional-form and instrument search with anti-p-hacking guardrails |
| 4 — System + asset pricing | Planned | SUR, 3SLS, Fama–MacBeth, GMM-IV |
