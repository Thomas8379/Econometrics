# Stress Test Findings

## Summary

Three full research analyses were conducted using 518 cryptocurrencies,
1M+ daily observations, and the econtools pipeline. This document records
bugs found, scalability issues, and tools needed.

## Bugs Fixed

### 1. `ljung_box_q` and `box_pierce_q` — DataFrame return format (FIXED)
- **File:** `econtools/diagnostics/serial_correlation.py`
- **Issue:** `acorr_ljungbox(..., return_df=False)` is ignored in recent
  statsmodels (>= 0.14). It always returns a DataFrame, but the old code
  unpacked it as a tuple, leading to `ValueError: could not convert string to float`.
- **Fix:** Check `isinstance(out, pd.DataFrame)` and extract via `.iloc[-1]`.

### 2. `RegressionResult.se` vs `.bse` attribute name
- **Issue:** Bootstrap result access used `res.se['var']` but the attribute
  is actually `res.bse['var']`.
- **Note:** This is a naming inconsistency in our types — `bse` follows
  statsmodels convention but `se` would be more intuitive.

## Scalability Issues Discovered

### 3. White test infeasible for N > 100K
- **Issue:** `white_test()` computes cross-products of all regressors,
  creating an O(N * K^2) matrix. For N=1M+ and K=8, this requires
  multi-GB memory and takes >30 minutes.
- **Recommendation:** Add a sample size guard or automatic subsampling
  option to `white_test()`.

### 4. HC1/HC3 extremely slow for large N with heavy-tailed data
- **Issue:** HC1 and HC3 covariance estimators become very slow (>10 min)
  on datasets with N > 100K when the dependent variable has extreme
  kurtosis (e.g., squared returns with kurtosis > 100).
- **Root cause:** The sandwich matrix computation `X' * diag(e^2) * X`
  with very large residuals causes numerical issues.
- **Recommendation:** Document performance characteristics; consider
  chunked computation or subsampling for diagnostics.

### 5. LOWESS in residual plots is O(N^2)
- **Issue:** `plot_residuals_vs_fitted(result, lowess=True)` calls
  `sm_lowess()` which is O(N^2). For N=1M+ this is hours.
- **Recommendation:** Auto-disable LOWESS or subsample when N > 50K.
  Add a `max_points` parameter to residual plot functions.

### 6. Bootstrap `run_bootstrap()` API — `add_intercept` vs `add_constant`
- **Issue:** Inconsistent naming between `fit_ols(add_constant=True)` and
  `run_bootstrap(add_intercept=True)`. Easy to confuse.
- **Recommendation:** Standardize on one name across the package.

## Tools We Need

### High Priority
1. **Clustered standard errors for OLS** — panel data needs `cluster` SE
   for pooled regressions (currently only available via `linearmodels`).
2. **Fama-MacBeth estimator** — a proper `fit_fama_macbeth()` function
   rather than manual month-by-month loops.
3. **Large-N guards** — automatic warnings/subsampling for diagnostics
   when N > threshold.

### Medium Priority
4. **Granger causality test** integration with `TestResult` format.
5. **Rolling regression** utility for time-varying coefficients.
6. **Summary statistics table** from raw DataFrame (not just from a
   pre-computed describe() output).

### Nice to Have
7. **GARCH model wrapper** for volatility modeling.
8. **Quantile regression** for distributional analysis.
9. **Panel data diagnostics** (cross-sectional dependence, etc.).

## Research Outputs

### Report 1: BTC Fundamentals (btc_fundamentals_report.tex)
- 5 OLS specifications with HAC SEs
- Stationarity tests (ADF, KPSS)
- Full diagnostic battery (BP, White, JB, RESET, LB)
- VIF analysis, coefficient forest plots
- Finding: MVRV ratio marginally significant; NVT weakly negative

### Report 2: Halving Event Study (halving_event_study.tex)
- CAR analysis across 4 halvings
- Regime regressions with HC3 SEs
- Wild bootstrap (B=2000) for halving coefficient
- Finding: Post-halving dummy ~0.4% daily premium (significant)

### Report 3: Cross-Section & Spillovers (cross_section_report.tex)
- 1M+ obs pooled OLS with HC1/HC3/HAC comparison
- Fama-MacBeth monthly cross-sectional regressions (135 months)
- BTC-to-altcoin volatility spillover test
- Finding: Strong BTC co-movement; short-term reversal; volatility spillover

## Curated Datasets

| File | Rows | Cols | Description |
|------|------|------|-------------|
| crypto_panel_v1.parquet | 1,049,754 | 14 | 433 coins daily price/returns |
| crypto_btc_onchain_v1.parquet | 5,428 | 25 | BTC fundamentals 2010-2025 |
| crypto_majors_v1.parquet | 31,522 | 42 | Top 10 coins with vol measures |
| crypto_btc_halving_v1.parquet | 5,428 | 29 | BTC with halving event windows |
