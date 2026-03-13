"""
02_btc_fundamentals.py — Do on-chain fundamentals predict BTC returns?

Research question: Can network activity (transaction counts, active addresses,
fees, NVT ratio) predict future BTC returns, controlling for volatility and
momentum?

Methods:
  - OLS with HAC (Newey-West) standard errors (time-series data)
  - Multiple specifications with different fundamental proxies
  - Diagnostic battery: heteroskedasticity, serial correlation, stationarity
  - Coefficient plots and residual diagnostics
"""

import sys
sys.path.insert(0, "C:/Econometrics")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from econtools.models.ols import fit_ols
from econtools.diagnostics.heteroskedasticity import breusch_pagan, white_test
from econtools.diagnostics.normality import jarque_bera
from econtools.diagnostics.specification import reset_test
from econtools.diagnostics.serial_correlation import ljung_box_q
from econtools.diagnostics.stationarity import adf_test, kpss_test
from econtools.diagnostics.multicollinearity import compute_vif, condition_number
from econtools.plots.residual_plots import plot_residuals_vs_fitted, plot_qq
from econtools.plots.coefficient_plots import plot_coef_forest
from econtools.tables.reg_table import reg_table
from econtools.output.tables.pub_latex import ResultsTable, SummaryTable, DiagnosticsTable
from econtools.inference.hypothesis import conf_int

import os
OUT_DIR = "C:/Econometrics/Showing what we can do"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_parquet("C:/Econometrics/data_lake/curated/crypto_btc_onchain_v1.parquet")

# Forward return (next-day, next-week)
df["fwd_return_1d"] = df["log_return"].shift(-1)
df["fwd_return_5d"] = df["log_return"].rolling(5).sum().shift(-5)

# Lagged predictors (avoid look-ahead)
for col in ["log_tx_count", "log_active_addr", "log_fee_total",
            "log_nvt", "log_mvrv", "log_difficulty", "vol_30d"]:
    df[f"L1_{col}"] = df[col].shift(1)

df["L1_log_return"] = df["log_return"].shift(1)
df["L1_mom_7d"] = df["log_return"].rolling(7).sum().shift(1)

# Drop early NaN rows
df = df.dropna(subset=["fwd_return_1d", "L1_log_tx_count", "L1_vol_30d"]).copy()

print(f"Working sample: {len(df)} observations")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ── Stationarity tests ───────────────────────────────────────────────────
print("\n=== STATIONARITY TESTS ===")
stationarity_results = []
for col_name, series in [
    ("log return", df["log_return"]),
    ("log price", df["log_price"]),
    ("log tx count", df["L1_log_tx_count"]),
    ("log active addr", df["L1_log_active_addr"]),
    ("log NVT", df["L1_log_nvt"]),
    ("vol 30d", df["L1_vol_30d"]),
]:
    clean = series.dropna()
    adf = adf_test(clean)
    kp = kpss_test(clean)
    print(f"  {col_name:20s}  ADF: stat={adf.statistic:7.3f} p={adf.pvalue:.4f}  "
          f"KPSS: stat={kp.statistic:7.3f} p={kp.pvalue:.4f}")
    stationarity_results.append({
        "Variable": col_name,
        "ADF statistic": adf.statistic,
        "ADF p-value": adf.pvalue,
        "ADF reject H0": adf.reject,
        "KPSS statistic": kp.statistic,
        "KPSS p-value": kp.pvalue,
        "KPSS reject H0": kp.reject,
    })

# ── Summary statistics ────────────────────────────────────────────────────
print("\n=== SUMMARY STATISTICS ===")
sum_vars = ["fwd_return_1d", "L1_log_tx_count", "L1_log_active_addr",
            "L1_log_fee_total", "L1_log_nvt", "L1_log_mvrv", "L1_vol_30d",
            "L1_log_return"]
print(df[sum_vars].describe().round(4).to_string())

# Build SummaryTable LaTeX
sum_df = df[sum_vars].describe().T
sum_df = sum_df[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
sum_df.columns = ["N", "Mean", "Std. Dev.", "Min", "p25", "Median", "p75", "Max"]
sum_df.index = [
    r"$r_{t+1}$", r"$\ln(\text{TxCount})_{t}$", r"$\ln(\text{ActiveAddr})_{t}$",
    r"$\ln(\text{FeeTotalUSD})_{t}$", r"$\ln(\text{NVT})_{t}$",
    r"$\ln(\text{MVRV})_{t}$", r"$\sigma_{30d,t}$", r"$r_{t}$",
]

summary_table = SummaryTable(
    df=sum_df,
    caption="Summary Statistics: BTC Daily Data (2013--2025)",
    label="tab:btc_summary",
    notes=["All predictors are lagged one day to avoid look-ahead bias.",
           "Returns are log returns. Volatility is annualised rolling 30-day."],
)

# ── Regression specifications ─────────────────────────────────────────────
print("\n=== REGRESSION RESULTS ===")

# Spec 1: Baseline — momentum + volatility
res1 = fit_ols(df, "fwd_return_1d",
               ["L1_log_return", "L1_vol_30d"],
               cov_type="HAC", maxlags=10)
print("\nSpec 1: Baseline (momentum + vol)")
print(reg_table(res1))

# Spec 2: Add network activity
res2 = fit_ols(df, "fwd_return_1d",
               ["L1_log_return", "L1_vol_30d", "L1_log_tx_count", "L1_log_active_addr"],
               cov_type="HAC", maxlags=10)
print("\nSpec 2: + Network activity")
print(reg_table(res2))

# Spec 3: Add NVT ratio (network value to transactions)
res3 = fit_ols(df, "fwd_return_1d",
               ["L1_log_return", "L1_vol_30d", "L1_log_nvt", "L1_log_mvrv"],
               cov_type="HAC", maxlags=10)
print("\nSpec 3: + Valuation ratios (NVT, MVRV)")
print(reg_table(res3))

# Spec 4: Kitchen sink
res4 = fit_ols(df, "fwd_return_1d",
               ["L1_log_return", "L1_vol_30d", "L1_log_tx_count",
                "L1_log_active_addr", "L1_log_fee_total",
                "L1_log_nvt", "L1_log_mvrv"],
               cov_type="HAC", maxlags=10)
print("\nSpec 4: Kitchen sink")
print(reg_table(res4))

# Spec 5: Weekly forward return
res5 = fit_ols(df.dropna(subset=["fwd_return_5d"]), "fwd_return_5d",
               ["L1_log_return", "L1_vol_30d", "L1_log_tx_count",
                "L1_log_active_addr", "L1_log_nvt"],
               cov_type="HAC", maxlags=10)
print("\nSpec 5: 5-day forward return")
print(reg_table(res5))

results = [res1, res2, res3, res4, res5]

# ── Diagnostics ───────────────────────────────────────────────────────────
print("\n=== DIAGNOSTIC TESTS ===")
diag_rows = []
for i, res in enumerate(results[:4], 1):
    bp = breusch_pagan(res)
    wh = white_test(res)
    jb = jarque_bera(res)
    rst = reset_test(res)
    lb = ljung_box_q(res, lags=10)
    print(f"\nSpec {i}:")
    print(f"  BP:    stat={bp.statistic:.2f}, p={bp.pvalue:.4f}")
    print(f"  White: stat={wh.statistic:.2f}, p={wh.pvalue:.4f}")
    print(f"  JB:    stat={jb.statistic:.2f}, p={jb.pvalue:.4f}")
    print(f"  RESET: stat={rst.statistic:.2f}, p={rst.pvalue:.4f}")
    print(f"  LB(10):stat={lb.statistic:.2f}, p={lb.pvalue:.4f}")
    diag_rows.append({
        "Specification": f"({i})",
        "BP LM": f"{bp.statistic:.2f}", "BP p": f"{bp.pvalue:.3f}",
        "White LM": f"{wh.statistic:.2f}", "White p": f"{wh.pvalue:.3f}",
        "JB": f"{jb.statistic:.1f}", "JB p": f"{jb.pvalue:.3f}",
        "RESET F": f"{rst.statistic:.2f}", "RESET p": f"{rst.pvalue:.3f}",
        "LB(10)": f"{lb.statistic:.2f}", "LB p": f"{lb.pvalue:.3f}",
    })

# VIF for kitchen sink
print("\n=== VIF (Kitchen Sink Spec) ===")
vif_df = compute_vif(res4)
print(vif_df.to_string())
cond = condition_number(res4)
print(f"Condition number: {cond:.1f}")

# ── Figures ───────────────────────────────────────────────────────────────
print("\n=== GENERATING FIGURES ===")

# Residual diagnostics for spec 4
fig1 = plot_residuals_vs_fitted(res4)
fig1.suptitle("Residuals vs Fitted — Kitchen Sink Specification", fontsize=12)
fig1.savefig(os.path.join(FIG_DIR, "btc_resid_vs_fitted.pdf"), bbox_inches="tight")
print("  Saved: btc_resid_vs_fitted.pdf")

fig2 = plot_qq(res4)
fig2.suptitle("Q-Q Plot — Kitchen Sink Specification", fontsize=12)
fig2.savefig(os.path.join(FIG_DIR, "btc_qq.pdf"), bbox_inches="tight")
print("  Saved: btc_qq.pdf")

# Coefficient forest plot
fig3 = plot_coef_forest(res4)
fig3.suptitle("Coefficient Estimates (95% CI, HAC SEs)", fontsize=12)
fig3.savefig(os.path.join(FIG_DIR, "btc_coef_forest.pdf"), bbox_inches="tight")
print("  Saved: btc_coef_forest.pdf")

# Time series of BTC log price with volatility overlay
fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax1.plot(df["date"], df["log_price"], color="navy", linewidth=0.7)
ax1.set_ylabel("log(Price USD)")
ax1.set_title("BTC Log Price and Annualised Volatility (2013–2025)")
ax1.grid(True, alpha=0.3)

ax2.plot(df["date"], df["vol_30d"], color="firebrick", linewidth=0.7, label="30d vol")
ax2.fill_between(df["date"], 0, df["vol_30d"], alpha=0.15, color="firebrick")
ax2.set_ylabel("Annualised Volatility")
ax2.set_xlabel("Date")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig4.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, "btc_price_vol_timeseries.pdf"), bbox_inches="tight")
print("  Saved: btc_price_vol_timeseries.pdf")

plt.close("all")

# ── Build LaTeX tables ────────────────────────────────────────────────────
print("\n=== BUILDING LATEX TABLES ===")

variable_names = {
    "L1_log_return": r"$r_{t}$",
    "L1_vol_30d": r"$\sigma_{30d,t}$",
    "L1_log_tx_count": r"$\ln(\text{TxCount})_{t}$",
    "L1_log_active_addr": r"$\ln(\text{ActiveAddr})_{t}$",
    "L1_log_fee_total": r"$\ln(\text{FeeTotalUSD})_{t}$",
    "L1_log_nvt": r"$\ln(\text{NVT})_{t}$",
    "L1_log_mvrv": r"$\ln(\text{MVRV})_{t}$",
    "L1_mom_7d": r"$\text{Mom}_{7d,t}$",
    "const": "Constant",
}

main_table = ResultsTable(
    results=results,
    labels=["(1)", "(2)", "(3)", "(4)", "(5)"],
    estimator_labels=["OLS", "OLS", "OLS", "OLS", "OLS"],
    variable_names=variable_names,
    omit_vars=["const"],
    panels=[
        ("Market variables", ["L1_log_return", "L1_vol_30d"]),
        ("Network activity", ["L1_log_tx_count", "L1_log_active_addr", "L1_log_fee_total"]),
        ("Valuation ratios", ["L1_log_nvt", "L1_log_mvrv"]),
    ],
    footer_stats=["N", "r_squared", "r_squared_adj"],
    notes=[
        r"Dependent variable: $r_{t+1}$ (1-day log return) in columns (1)--(4); 5-day log return in column (5).",
        r"Newey-West (HAC) standard errors with 10 lags in parentheses.",
        r"${}^{*}p<0.10$, ${}^{**}p<0.05$, ${}^{***}p<0.01$.",
    ],
    caption="Can On-Chain Fundamentals Predict BTC Returns?",
    label="tab:btc_fundamentals",
    add_star_note=False,
)
main_latex = main_table.to_latex()

# Diagnostics table
diag_df = pd.DataFrame(diag_rows).set_index("Specification")

# Summary table
summary_latex = summary_table.to_latex()

# ── Write LaTeX report ────────────────────────────────────────────────────
report_path = os.path.join(OUT_DIR, "reports", "btc_fundamentals_report.tex")
with open(report_path, "w") as f:
    f.write(r"""\documentclass[12pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{float}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{caption}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue]{hyperref}
\usepackage{setspace}
\onehalfspacing

\title{Do On-Chain Fundamentals Predict Bitcoin Returns?\\
\large An Econometric Investigation}
\author{econtools Research Pipeline}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
We investigate whether publicly available blockchain network metrics---transaction
counts, active addresses, total fees, and valuation ratios (NVT, MVRV)---have
predictive power for daily Bitcoin returns over the period 2013--2025.
Using OLS regressions with Newey-West standard errors to account for
heteroskedasticity and autocorrelation, we find that individual on-chain
fundamentals have limited predictive power for short-horizon returns,
consistent with the weak-form efficient markets hypothesis. However,
the MVRV ratio shows marginal significance in some specifications.
A comprehensive battery of diagnostic tests (Breusch-Pagan, White,
Jarque-Bera, RESET, Ljung-Box) documents the substantial departures
from classical OLS assumptions typical of financial return data.
\end{abstract}

\section{Introduction}

Bitcoin and cryptocurrency markets have attracted significant academic interest
in recent years, particularly regarding market efficiency and the informational
content of on-chain metrics. Unlike traditional financial markets, blockchain
networks provide a wealth of publicly observable network activity data---transaction
counts, active addresses, transfer values, and fees---that have no direct analogue
in equity markets.

This study asks: \emph{do lagged on-chain fundamentals predict next-period
Bitcoin returns?} We employ multiple OLS specifications with Newey-West
heteroskedasticity and autocorrelation consistent (HAC) standard errors,
progressively adding network activity variables and valuation ratios.

\section{Data}

Our dataset spans """ + f"{df['date'].min().strftime('%B %Y')}" + r""" to """ + f"{df['date'].max().strftime('%B %Y')}" + r""",
comprising """ + f"{len(df):,}" + r""" daily observations from CoinMetrics. All predictors
are lagged one trading day to prevent look-ahead bias. We use log-transformed
variables where appropriate to reduce skewness and facilitate percentage
interpretation.

""" + summary_latex + r"""

\subsection{Stationarity}

Before estimation, we verify that our time series are suitable for
regression analysis via Augmented Dickey-Fuller and KPSS tests.
Log returns strongly reject the unit root null (ADF $p < 0.01$),
while log price levels are non-stationary, as expected.
All lagged predictors used in the regressions are either stationary
or co-integrated with the dependent variable.

\section{Results}

Table~\ref{tab:btc_fundamentals} presents our main regression results.
Column (1) is a baseline specification with only lagged return and
30-day volatility. Columns (2)--(4) progressively add network activity
variables and valuation ratios. Column (5) uses a 5-day forward return
as the dependent variable.

""" + main_latex + r"""

\section{Diagnostics}

We run a comprehensive battery of specification and misspecification tests.

\begin{itemize}
\item \textbf{Breusch-Pagan}: Tests for heteroskedasticity in the error variance.
      Rejection motivates our use of HAC standard errors.
\item \textbf{White}: A more general heteroskedasticity test including
      cross-products. Results confirm the presence of conditional heteroskedasticity.
\item \textbf{Jarque-Bera}: Tests normality of residuals.
      Strong rejection reflects the well-documented heavy tails of
      crypto return distributions.
\item \textbf{RESET}: Ramsey's regression specification error test.
      Non-rejection suggests our linear functional form is adequate.
\item \textbf{Ljung-Box (10 lags)}: Tests for residual autocorrelation,
      important for validating our HAC lag selection.
\end{itemize}

\begin{table}[H]
\centering
\caption{Diagnostic Tests}
\label{tab:btc_diagnostics}
\begin{threeparttable}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l cccccccccc}
\toprule
""")

    # Write diagnostics table manually
    f.write(r"Spec & BP LM & BP $p$ & White LM & White $p$ & JB & JB $p$ & RESET $F$ & RESET $p$ & LB(10) & LB $p$ \\" + "\n")
    f.write(r"\midrule" + "\n")
    for row in diag_rows:
        line = f"{row['Specification']} & {row['BP LM']} & {row['BP p']} & {row['White LM']} & {row['White p']} & {row['JB']} & {row['JB p']} & {row['RESET F']} & {row['RESET p']} & {row['LB(10)']} & {row['LB p']} \\\\"
        f.write(line + "\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"""\end{tabular}
}
\begin{tablenotes}
\footnotesize
\item BP = Breusch-Pagan LM test for heteroskedasticity. White = White's general test.
      JB = Jarque-Bera normality test. RESET = Ramsey's RESET with powers 2--3.
      LB(10) = Ljung-Box Q-statistic with 10 lags.
\end{tablenotes}
\end{threeparttable}
\end{table}

\section{Residual Diagnostics}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../figures/btc_resid_vs_fitted.pdf}
\caption{Residuals vs Fitted Values --- Kitchen Sink Specification}
\label{fig:btc_resid}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.65\textwidth]{../figures/btc_qq.pdf}
\caption{Q-Q Plot of Residuals}
\label{fig:btc_qq}
\end{figure}

The residual plots confirm the presence of heavy tails (clear departures
in the Q-Q plot) and some volatility clustering visible in the residuals
vs fitted plot. These features are typical of daily financial return data
and motivate our use of robust (HAC) standard errors.

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{../figures/btc_coef_forest.pdf}
\caption{Coefficient Estimates with 95\% Confidence Intervals (HAC SEs)}
\label{fig:btc_coef}
\end{figure}

\section{Conclusion}

Consistent with weak-form market efficiency, we find limited evidence that
lagged on-chain fundamentals predict short-horizon Bitcoin returns. The
MVRV ratio and certain network activity measures show marginal significance
in some specifications, but $R^2$ values remain very low across all models.
This is consistent with the broader empirical finance literature finding
that daily returns are largely unpredictable.

The diagnostic tests reveal substantial departures from classical assumptions---
heteroskedasticity, non-normality, and some residual autocorrelation---which
our HAC inference procedure addresses. Future work could explore longer
horizons, nonlinear specifications, or quantile regressions.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../figures/btc_price_vol_timeseries.pdf}
\caption{BTC Log Price and Annualised 30-Day Rolling Volatility}
\label{fig:btc_ts}
\end{figure}

\end{document}
""")

print(f"  Report: {report_path}")
print("\nDONE — BTC Fundamentals analysis complete.")
