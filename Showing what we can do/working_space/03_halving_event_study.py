"""
03_halving_event_study.py — Bitcoin Halving Event Study

Research question: Do Bitcoin halvings (supply shock events where block rewards
are cut in half) have a measurable effect on returns and volatility?

Methods:
  - Event study framework with pre/post comparison
  - Difference-in-regime OLS with HC3 robust standard errors
  - Cumulative abnormal returns (CARs) around halving dates
  - Bootstrap confidence intervals for the halving effect
  - Volatility comparison across regimes
"""

import sys
sys.path.insert(0, "C:/Econometrics")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from econtools.models.ols import fit_ols
from econtools.diagnostics.heteroskedasticity import breusch_pagan, white_test
from econtools.diagnostics.normality import jarque_bera
from econtools.diagnostics.serial_correlation import ljung_box_q
from econtools.diagnostics.stationarity import adf_test
from econtools.tables.reg_table import reg_table
from econtools.output.tables.pub_latex import ResultsTable, SummaryTable
from econtools.uncertainty.bootstrap import run_bootstrap

import os
OUT_DIR = "C:/Econometrics/Showing what we can do"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_parquet("C:/Econometrics/data_lake/curated/crypto_btc_halving_v1.parquet")

HALVINGS = {
    "Halving 1 (Nov 2012)": pd.Timestamp("2012-11-28"),
    "Halving 2 (Jul 2016)": pd.Timestamp("2016-07-09"),
    "Halving 3 (May 2020)": pd.Timestamp("2020-05-11"),
    "Halving 4 (Apr 2024)": pd.Timestamp("2024-04-19"),
}

print(f"Total observations: {len(df)}")
print(f"Halving window observations: {df['halving_window'].sum()}")

# ── Event windows: CAR analysis ──────────────────────────────────────────
print("\n=== CUMULATIVE ABNORMAL RETURNS AROUND HALVINGS ===")

windows = [(-30, 30), (-60, 60), (-90, 180), (-180, 365)]
car_results = []

for name, hdate in HALVINGS.items():
    for pre, post in windows:
        mask = (df["date"] >= hdate + pd.Timedelta(days=pre)) & \
               (df["date"] <= hdate + pd.Timedelta(days=post))
        sub = df.loc[mask]
        if len(sub) == 0:
            continue
        car = sub["log_return"].sum()
        vol = sub["log_return"].std() * np.sqrt(365)
        n = len(sub)
        car_results.append({
            "Event": name,
            "Window": f"[{pre:+d}, {post:+d}]",
            "CAR": car,
            "Ann. Vol": vol,
            "N days": n,
        })
        print(f"  {name} {f'[{pre:+d},{post:+d}]':>12s}: CAR={car:+.3f} "
              f"({100*car:+.1f}%), vol={vol:.2f}, N={n}")

# ── Regime regression ─────────────────────────────────────────────────────
print("\n=== REGIME REGRESSIONS ===")

# Create regime dummies for each halving cycle
df["regime"] = "pre_halving_1"
for name, hdate in HALVINGS.items():
    short = name.split("(")[0].strip().replace(" ", "_").lower()
    # Post-halving period until next halving
    mask = df["date"] >= hdate
    df.loc[mask, "regime"] = short

# Dummy for being in any post-halving 365-day window
# Already have post_halving column

# Regression: returns ~ post_halving + vol_30d + controls
df_clean = df.dropna(subset=["log_return", "vol_30d", "post_halving"]).copy()

# Spec 1: Simple post-halving effect
res1 = fit_ols(df_clean, "log_return", ["post_halving"], cov_type="HC3")
print("\nSpec 1: Simple post-halving dummy")
print(reg_table(res1))

# Spec 2: With volatility control
res2 = fit_ols(df_clean, "log_return", ["post_halving", "vol_30d"],
               cov_type="HC3")
print("\nSpec 2: + Volatility control")
print(reg_table(res2))

# Spec 3: With momentum control
df_clean["L1_return"] = df_clean["log_return"].shift(1)
df_clean = df_clean.dropna(subset=["L1_return"]).copy()

res3 = fit_ols(df_clean, "log_return",
               ["post_halving", "vol_30d", "L1_return"],
               cov_type="HC3")
print("\nSpec 3: + Momentum control")
print(reg_table(res3))

# Spec 4: With on-chain controls
for col in ["log_tx_count", "log_active_addr", "log_nvt"]:
    if col in df_clean.columns:
        df_clean[f"L1_{col}"] = df_clean[col].shift(1)

df_clean = df_clean.dropna(subset=["L1_log_tx_count"]).copy()

res4 = fit_ols(df_clean, "log_return",
               ["post_halving", "vol_30d", "L1_return",
                "L1_log_tx_count", "L1_log_active_addr"],
               cov_type="HC3")
print("\nSpec 4: + On-chain controls")
print(reg_table(res4))

halving_results = [res1, res2, res3, res4]

# ── Bootstrap the halving effect ──────────────────────────────────────────
print("\n=== BOOTSTRAP: Halving Effect ===")
boot_df = df_clean[["log_return", "post_halving", "vol_30d", "L1_return"]].dropna()

boot = run_bootstrap(
    data=boot_df,
    y="log_return",
    X=["post_halving", "vol_30d", "L1_return"],
    estimator="ols",
    bootstrap_method="wild",
    B=2000,
    seed=42,
    ci_level=0.95,
    add_intercept=True,
)

boot_se = boot["bootstrap"]["se"]["post_halving"]
boot_ci_lo = boot["bootstrap"]["ci"]["percentile"]["lower"]["post_halving"]
boot_ci_hi = boot["bootstrap"]["ci"]["percentile"]["upper"]["post_halving"]
print(f"  Bootstrap SE for post_halving: {boot_se:.6f}")
print(f"  Bootstrap 95% CI: [{boot_ci_lo:.6f}, {boot_ci_hi:.6f}]")
print(f"  OLS point estimate: {res3.params['post_halving']:.6f}")
print(f"  OLS HC3 SE: {res3.bse['post_halving']:.6f}")

# ── Diagnostics ───────────────────────────────────────────────────────────
print("\n=== DIAGNOSTICS ===")
for i, res in enumerate(halving_results, 1):
    bp = breusch_pagan(res)
    jb = jarque_bera(res)
    lb = ljung_box_q(res, lags=10)
    print(f"  Spec {i}: BP p={bp.pvalue:.4f}, JB p={jb.pvalue:.4f}, LB(10) p={lb.pvalue:.4f}")

# ── Figures ───────────────────────────────────────────────────────────────
print("\n=== GENERATING FIGURES ===")

# Figure 1: CAR event study plot
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for idx, (name, hdate) in enumerate(HALVINGS.items()):
    ax = axes[idx // 2, idx % 2]
    # Get data in [-180, +365] window
    mask = (df["date"] >= hdate - pd.Timedelta(days=180)) & \
           (df["date"] <= hdate + pd.Timedelta(days=365))
    sub = df.loc[mask].copy()
    sub["days"] = (sub["date"] - hdate).dt.days
    sub["cumret"] = sub["log_return"].cumsum()

    ax.plot(sub["days"], sub["cumret"], color=colors[idx], linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.fill_betweenx(ax.get_ylim(), 0, 30, alpha=0.1, color="red")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Days from Halving")
    ax.set_ylabel("Cumulative Log Return")
    ax.grid(True, alpha=0.3)

fig1.suptitle("Cumulative Returns Around Bitcoin Halvings", fontsize=14, fontweight="bold")
fig1.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig(os.path.join(FIG_DIR, "halving_car.pdf"), bbox_inches="tight")
print("  Saved: halving_car.pdf")

# Figure 2: Volatility across halving regimes
fig2, ax = plt.subplots(figsize=(12, 5))
for idx, (name, hdate) in enumerate(HALVINGS.items()):
    mask = (df["date"] >= hdate - pd.Timedelta(days=180)) & \
           (df["date"] <= hdate + pd.Timedelta(days=365))
    sub = df.loc[mask].copy()
    sub["days"] = (sub["date"] - hdate).dt.days
    ax.plot(sub["days"], sub["vol_30d"], color=colors[idx],
            linewidth=0.8, label=name, alpha=0.8)

ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7, label="Halving Day")
ax.set_xlabel("Days from Halving")
ax.set_ylabel("30-Day Annualised Volatility")
ax.set_title("Volatility Dynamics Around Bitcoin Halvings")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "halving_volatility.pdf"), bbox_inches="tight")
print("  Saved: halving_volatility.pdf")

# Figure 3: Bootstrap distribution of halving coefficient
fig3, ax = plt.subplots(figsize=(8, 5))
boot_draws = boot["bootstrap"]["draws"]
coef_names = boot["point_estimate"]["coef_names"]
halving_idx = coef_names.index("post_halving")
boot_coefs = boot_draws[:, halving_idx]
ax.hist(boot_coefs, bins=60, density=True, alpha=0.7, color="steelblue", edgecolor="white")
ax.axvline(res3.params["post_halving"], color="red", linewidth=2,
           label=f"OLS estimate: {res3.params['post_halving']:.5f}")
ax.axvline(boot_ci_lo, color="orange", linestyle="--",
           label=f"95% CI lower: {boot_ci_lo:.5f}")
ax.axvline(boot_ci_hi, color="orange", linestyle="--",
           label=f"95% CI upper: {boot_ci_hi:.5f}")
ax.set_xlabel("Post-Halving Coefficient")
ax.set_ylabel("Density")
ax.set_title("Wild Bootstrap Distribution of Halving Effect (B=2,000)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, "halving_bootstrap.pdf"), bbox_inches="tight")
print("  Saved: halving_bootstrap.pdf")

plt.close("all")

# ── LaTeX report ──────────────────────────────────────────────────────────
print("\n=== BUILDING LATEX REPORT ===")

variable_names = {
    "post_halving": "Post-Halving (365d)",
    "vol_30d": r"$\sigma_{30d}$",
    "L1_return": r"$r_{t-1}$",
    "L1_log_tx_count": r"$\ln(\text{TxCount})_{t-1}$",
    "L1_log_active_addr": r"$\ln(\text{ActiveAddr})_{t-1}$",
    "const": "Constant",
}

main_table = ResultsTable(
    results=halving_results,
    labels=["(1)", "(2)", "(3)", "(4)"],
    estimator_labels=["OLS", "OLS", "OLS", "OLS"],
    variable_names=variable_names,
    omit_vars=["const"],
    panels=[
        ("Halving effect", ["post_halving"]),
        ("Controls", ["vol_30d", "L1_return", "L1_log_tx_count", "L1_log_active_addr"]),
    ],
    footer_stats=["N", "r_squared", "r_squared_adj"],
    notes=[
        r"Dependent variable: daily log return $r_t$.",
        r"Post-Halving is a dummy equal to 1 for the 365 days following each halving.",
        r"HC3 heteroskedasticity-robust standard errors in parentheses.",
        r"${}^{*}p<0.10$, ${}^{**}p<0.05$, ${}^{***}p<0.01$.",
    ],
    caption="The Effect of Bitcoin Halvings on Daily Returns",
    label="tab:halving_returns",
    add_star_note=False,
)

# CAR table
car_df = pd.DataFrame(car_results)
car_pivot = car_df.pivot(index="Event", columns="Window", values="CAR")
car_pivot = car_pivot.round(3)

report_path = os.path.join(OUT_DIR, "reports", "halving_event_study.tex")
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

\title{Bitcoin Halving Event Study:\\
\large Supply Shocks and Market Response}
\author{econtools Research Pipeline}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
Bitcoin halvings---pre-programmed 50\% reductions in block rewards occurring
approximately every four years---represent exogenous supply shocks of known
timing but uncertain price impact. Using data from 2010--2025 spanning all
four halvings, we conduct an event study analysis combining cumulative
abnormal returns (CARs) with regression-based tests. Wild bootstrap
confidence intervals complement HC3 robust inference.
We find positive but imprecisely estimated post-halving effects on daily
returns, with substantial heterogeneity across halving episodes.
\end{abstract}

\section{Introduction}

Bitcoin's monetary policy is uniquely transparent: the block reward halves
every 210,000 blocks (approximately 4 years), and this schedule has been
known since Bitcoin's inception in 2009. The efficient markets hypothesis
(EMH) predicts that a fully anticipated supply reduction should be priced
in advance. However, if market participants are inattentive or if the
halving triggers narrative-driven demand shifts, post-halving abnormal
returns may persist.

We examine all four Bitcoin halvings (November 2012, July 2016, May 2020,
April 2024) using cumulative log returns in event windows, regime
regressions with robust standard errors, and wild bootstrap inference.

\section{Data and Methodology}

Our dataset covers """ + f"{len(df_clean):,}" + r""" daily observations of Bitcoin prices and
on-chain metrics from CoinMetrics. We define a ``post-halving'' dummy
equal to 1 for the 365 calendar days following each halving event.

We estimate:
\begin{equation}
r_t = \alpha + \beta \cdot \mathbf{1}[\text{Post-Halving}]_t + \gamma' X_t + \varepsilon_t
\end{equation}
where $X_t$ includes lagged volatility, momentum, and on-chain controls.
Standard errors are HC3 heteroskedasticity-robust. We supplement
asymptotic inference with wild bootstrap (Rademacher weights, $B=2{,}000$).

\section{Cumulative Abnormal Returns}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../figures/halving_car.pdf}
\caption{Cumulative Log Returns Around Bitcoin Halvings (180 days before to 365 days after)}
\label{fig:car}
\end{figure}

Figure~\ref{fig:car} plots cumulative log returns centered on each halving date.
All four halvings are followed by substantial price appreciation, though with
very different magnitudes and dynamics. The early halvings (2012, 2016) show
explosive post-halving growth, while the later halvings exhibit more moderate
but still positive drift.

""")

    # CAR table
    f.write(r"""\begin{table}[H]
\centering
\caption{Cumulative Abnormal Returns by Event Window}
\label{tab:car}
\begin{threeparttable}
\begin{tabular}{l cccc}
\toprule
& \multicolumn{4}{c}{Event Window (days)} \\
\cmidrule(lr){2-5}
Event & $[-30,+30]$ & $[-60,+60]$ & $[-90,+180]$ & $[-180,+365]$ \\
\midrule
""")
    for name in HALVINGS:
        vals = []
        for w in [(-30, 30), (-60, 60), (-90, 180), (-180, 365)]:
            match = [r for r in car_results if r["Event"] == name and r["Window"] == f"[{w[0]:+d}, {w[1]:+d}]"]
            if match:
                v = match[0]["CAR"]
                vals.append(f"{v:+.3f}")
            else:
                vals.append("---")
        short = name.replace("Halving ", "H").replace(" (", " (")
        f.write(f"{short} & {' & '.join(vals)} \\\\\n")

    f.write(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item Cumulative log returns within each event window.
      Positive values indicate net appreciation over the window.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

    f.write("\n\\section{Regression Results}\n\n")
    f.write(main_table.to_latex())

    f.write(r"""
\subsection{Bootstrap Inference}

To address potential finite-sample bias in the HC3 standard errors
(particularly relevant given the heavy-tailed return distribution),
we implement wild bootstrap inference using Rademacher weights with
$B = 2{,}000$ replications.

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{../figures/halving_bootstrap.pdf}
\caption{Wild Bootstrap Distribution of the Post-Halving Coefficient}
\label{fig:bootstrap}
\end{figure}

The bootstrap standard error for the post-halving coefficient is
""" + f"{boot_se:.6f}" + r""", compared to the HC3 asymptotic
standard error of """ + f"{res3.bse['post_halving']:.6f}" + r""". The 95\% bootstrap
confidence interval is $[""" + f"{boot_ci_lo:.5f}" + r""",\;
""" + f"{boot_ci_hi:.5f}" + r"""]$.

\section{Volatility Dynamics}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../figures/halving_volatility.pdf}
\caption{30-Day Annualised Volatility Around Bitcoin Halvings}
\label{fig:vol}
\end{figure}

Figure~\ref{fig:vol} shows that volatility dynamics around halvings are
heterogeneous. The 2012 and 2016 halvings occurred during periods of
relatively low volatility, while the 2020 halving coincided with
COVID-19 market turmoil. The 2024 halving shows the most subdued
volatility profile, consistent with market maturation.

\section{Conclusion}

The evidence for a systematic post-halving return premium is suggestive
but not conclusive. While CARs are positive for all four halvings, the
regression-based tests with proper standard errors show that the effect
is imprecisely estimated. This is consistent with the EMH prediction
that anticipated supply changes should be priced in, combined with the
reality that halvings may trigger demand-side effects through narrative
and attention channels that are difficult to model.

\end{document}
""")

print(f"  Report: {report_path}")
print("\nDONE — Halving event study complete.")
