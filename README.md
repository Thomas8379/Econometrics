# Econometrics

Personal econometrics workspace for coursework, supervisions, and independent research.

---

## Folder Structure

```
Econometrics/
├── README.md
├── CLAUDE.md                  # project rules for Claude Code
├── pyproject.toml             # econtools package definition
├── .gitignore
│
├── econtools/                 # Python econometrics toolkit (see below)
│
├── tests/                     # pytest suite (336 tests)
│   ├── data/
│   ├── models/
│   ├── inference/
│   ├── diagnostics/
│   ├── plots/
│   ├── tables/
│   ├── fit/                   # tests for fit_model() dispatcher
│   ├── uncertainty/           # bootstrap inference tests
│   ├── sieve/                 # sieve infrastructure tests (85 tests)
│   └── validation/            # numerical validation (Wooldridge datasets + external fixtures)
│       └── fixtures/          # catalogue.json, nist_embedded.py, results_files/
│
├── data_lake/
│   ├── raw/
│   │   └── wooldridge_and_oleg/   # 92 Wooldridge textbook datasets + 3 local = 95 .dta files
│   │       ├── manifest.json      # tracked: names, sources, SHA-256 hashes
│   │       └── *.dta              # gitignored binary files
│   └── curated/                   # cleaned / merged datasets (Parquet, versioned)
│
├── projects/                      # self-contained analysis projects
│   └── <project-name>/
│       ├── README.md
│       └── notebooks/
│
├── docs/                          # module documentation
│   ├── bootstrap.md               # Bootstrap inference — methods, CIs, p-values, CLI
│   ├── binary_model_output.md     # Logit/Probit output statistics
│   ├── pub_latex.md               # Publication LaTeX tables (ResultsTable, SummaryTable, DiagnosticsTable)
│   └── sieve.md                   # Sieve infrastructure — functional-form and instrument search
│
├── bootstrap_default_config.yaml  # Annotated config schema for econtools bootstrap
├── sieve_default_config.yaml      # Annotated config schema for econtools sieve
│
├── scripts/
│   └── collect_todos.py           # grep codebase for TODO(econtools) by category
│
└── Supervisions/                  # supervision work
```

---

## econtools Package

A Python econometrics toolkit. Install in editable mode:

```bash
pip install -e .
```

### Quick start

```python
from econtools.fit import fit_model
from econtools.model.spec import ModelSpec

spec = ModelSpec(dep_var="lwage", exog_vars=["educ", "exper"], estimator="ols")
result = fit_model(spec, df)

print(result.params)
print(result.fit.r_squared)
```

### Package layout

```
econtools/
├── _core/              # Shared internals
│   ├── types.py        # Estimate, FitMetrics, TestResult
│   ├── formatting.py   # _star, _fmt, _latex_star, _latex_escape
│   └── cov_mapping.py  # Unified SE-type resolver (statsmodels + linearmodels)
│
├── model/
│   └── spec.py         # ModelSpec — declarative model specification
│
├── fit/                # Estimation layer
│   └── estimators.py   # fit_model(spec, df) → Estimate
│
├── evaluation/         # Statistical tests and diagnostics
│   ├── hypothesis.py       # wald_test, f_test, t_test_coeff, conf_int
│   ├── heteroskedasticity.py  # breusch_pagan, white_test
│   ├── normality.py        # jarque_bera
│   ├── specification.py    # reset_test
│   ├── multicollinearity.py   # compute_vif, condition_number
│   ├── serial_correlation.py  # durbin_watson, breusch_godfrey, ljung_box
│   ├── stationarity.py     # adf_test, kpss_test
│   ├── iv_checks.py        # wu_hausman, sargan_test, weak_iv_f
│   └── panel_checks.py     # hausman_test, bp_lm_test
│
├── output/             # Reporting — no statistical logic
│   ├── tables/         # reg_table, compare_table, TableContent repr
│   │   └── pub_latex.py   # ResultsTable, SummaryTable, DiagnosticsTable (booktabs + panels)
│   ├── latex/          # Journal profiles (ECONOMETRICA, AER), document assembly
│   └── knowledge_base/ # YAML entries: OLS, IV, BP test, Hausman, etc.
│
├── sieve/              # Specification search (functional-form + instrument sieves)
│   ├── api.py          # run_sieve(), load_sieve_results()
│   ├── candidates.py   # Candidate dataclass + stable hashing
│   ├── generators/     # Feature and instrument generators
│   ├── fitters.py      # Unified fit_candidate() interface
│   ├── scorers.py      # OLS/IV scoring metrics
│   ├── protocols.py    # Holdout / k-fold / cross-fitting splits
│   ├── selection.py    # Guardrails, Pareto frontier, top-k selection
│   ├── manifest.py     # Run provenance (config hash, data fingerprint)
│   └── reporting.py    # Model cards, leaderboard summaries, LaTeX output
│
├── uncertainty/        # Variance estimation and bootstrap inference
│   ├── cov_estimators.py  # HC0–HC3, HAC, cluster, kernel SE types
│   ├── bootstrap.py       # run_bootstrap() — reproducible bootstrap inference
│   ├── _bootstrap_estimators.py  # Standalone numpy OLS/2SLS for bootstrap speed
│   └── _bootstrap_manifest.py    # Config hashing, manifest builder/writer
│
├── data/               # Data pipeline (Phase 0)
│   ├── io.py           # load_dta, save_curated, verify_hash
│   ├── inspect.py      # summarise, dtypes, missing
│   ├── clean.py        # drop_missing, winsorise, standardise
│   ├── transform.py    # log, lag, diff, interact
│   ├── construct.py    # dummies, polynomials, date features
│   └── provenance.py   # log_step, load_log
│
└── cli/                # Command-line interface
    └── commands/
        ├── bootstrap_cli.py  # econtools bootstrap --config cfg.yaml
        └── sieve_cli.py      # econtools sieve --config sieve.yaml
```

### Supported estimators

| `estimator=` | Backend | Notes |
|---|---|---|
| `"ols"` | statsmodels | Ordinary least squares |
| `"wls"` | statsmodels | Weighted least squares; requires `weights_col` |
| `"2sls"` | linearmodels | IV-2SLS; requires `endog_vars` + `instruments` |
| `"fe"` | linearmodels | Fixed effects panel; requires `entity_col` + `time_col` |
| `"re"` | linearmodels | Random effects panel |
| `"fd"` | linearmodels | First-difference panel |

### Supported SE types (`cov_type=`)

`"classical"`, `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"`, `"HAC"` / `"newey_west"`, `"cluster"`

### Publication LaTeX tables

```python
from econtools.output.tables import ResultsTable, SummaryTable, DiagnosticsTable

# Multi-model results table with panels (AER/Econometrica style)
t = ResultsTable(
    results=[ols_result, iv_result],
    labels=["(1)", "(2)"],
    estimator_labels=["OLS", "2SLS"],
    variable_names={"educ": "Education", "exper": "Experience"},
    omit_vars=["const"],
    panels=[
        ("Panel A: Baseline",  ["educ", "exper"]),
        ("Panel B: Extended",  ["educ", "exper", "tenure"]),
    ],
    footer_stats=["N", "r_squared", "first_stage_f"],
    notes=["HC3 standard errors in parentheses."],
    caption="Returns to Schooling",
    label="tab:returns",
)
print(t.to_latex())   # paste directly into .tex

# Summary statistics
s = SummaryTable(df, vars=["wage", "educ", "exper"], stats=["mean", "std", "N"])
print(s.to_latex())

# Diagnostic test results
from econtools.evaluation.heteroskedasticity import breusch_pagan, white_test
d = DiagnosticsTable([breusch_pagan(res), white_test(res)], caption="Diagnostics")
print(d.to_latex())
```

All tables use `booktabs` rules (`\toprule / \midrule / \bottomrule`), `threeparttable` for notes, and optional `\resizebox{\textwidth}{!}{}` for A4 fitting.  See [docs/pub_latex.md](docs/pub_latex.md) for the full API.

### Sieve: systematic specification search

```python
from econtools.sieve import run_sieve

# Functional-form sieve (OLS)
result = run_sieve(
    data=df,
    y="log_wage",
    base_X=["educ", "exper"],
    estimator="ols",
    sieve_spec={
        "generators": {
            "features": {
                "polynomial":   {"enabled": True, "degree": 2},
                "interactions": {"enabled": True},
                "log":          {"enabled": True, "vars": ["exper"]},
            }
        },
        "protocol":   {"mode": "holdout", "test_frac": 0.30},
        "selection":  {"primary_metric": "rmse", "top_k": 3},
        "constraints": {"max_terms": 8},
    },
    seed=12345,
    output_dir="./sieve_output",
)

# Inspect results
for cand in result["selected"]:
    print(cand.X_terms, cand.candidate_hash)

print(result["leaderboard"].head())

# IV instrument sieve (2SLS)
result_iv = run_sieve(
    data=df, y="y", base_X=["x1"], estimator="2sls",
    endog=["w"], base_Z=["z_baseline"],
    sieve_spec={
        "generators": {"instruments": {
            "z_polynomial":  {"enabled": True, "degree": 2},
            "z_interactions": {"enabled": True},
        }},
        "protocol":    {"mode": "crossfit", "k": 5},
        "selection":   {"primary_metric": "first_stage_f", "higher_is_better": True},
        "constraints": {"min_first_stage_f": 10.0},
    },
    seed=12345,
)
```

**CLI:**

```bash
# Copy and edit the annotated config
cp sieve_default_config.yaml my_sieve.yaml
econtools sieve --config my_sieve.yaml --output ./sieve_runs/run1

# Inspect a saved run
econtools sieve-report --run ./sieve_runs/run1
```

**Anti-p-hacking by default:** selection is done on a training split; final performance is reported on a held-out test set.  Every rejected candidate is logged with a reason code.  In-sample selection requires `allow_in_sample_selection: true` and stamps all outputs as **EXPLORATORY ONLY**.

See [docs/sieve.md](docs/sieve.md) for full documentation.

### Bootstrap inference

```python
from econtools.uncertainty.bootstrap import run_bootstrap

result = run_bootstrap(
    data=df,
    y="lwage",
    X=["exper", "tenure"],        # exogenous regressors (constant added automatically)
    estimator="ols",              # "ols" | "2sls"
    bootstrap_method="iid_pairs", # see table below
    B=1999,
    seed=42,
)

# Point estimates, bootstrap SEs, CIs, p-values
print(result["point_estimate"]["params"])
print(result["bootstrap"]["se"])
print(result["bootstrap"]["ci"]["percentile"])
print(result["bootstrap"]["pvalues"])
```

**Bootstrap methods**

| `bootstrap_method=` | When to use |
|---|---|
| `"iid_pairs"` | i.i.d. cross-section data (default for non-panel) |
| `"wild"` | Heteroskedastic cross-section; Rademacher (default) or Mammen weights |
| `"cluster_pairs"` | One-way clustered data; requires `cluster=` column |
| `"panel_cluster_id"` | Panel data; resamples entity ids, keeps full within-id histories; requires `id_col=` |

For 2SLS, also pass `endog=["endog_var"]` (endogenous regressors) and `Z=["instrument"]` (excluded instruments).

**CLI**

```bash
econtools bootstrap --config analysis/bootstrap_config.yaml
```

Copy `bootstrap_default_config.yaml` from the repo root and fill in your dataset path and column names. The CLI writes a CSV results table, an optional LaTeX table, and a `manifest.json` recording the full reproducibility record (seed, config hash, git commit, package versions).

See [docs/bootstrap.md](docs/bootstrap.md) for full documentation.

### Running tests

```bash
python -m pytest tests/ -v -m "not phase3 and not slow"  # fast tests (336 passing)
python -m pytest tests/ -v -m "not phase3"               # includes slow simulations
python -m pytest tests/test_pub_latex.py -v              # publication table tests (29)
python -m pytest tests/sieve/ -v -m "not slow"           # sieve tests (85, fast subset)
python -m pytest tests/validation/ -v                    # Wooldridge numerical validation
```

**pytest markers**

| Marker | Meaning |
|--------|---------|
| `phase3` | Phase 3 features not yet implemented (binary models, influence suite) |
| `slow` | Simulation-based tests with B=999–1999 replications; correct but takes a few seconds |

### Validation fixtures

`tests/validation/fixtures/` holds reference material for numerical validation tests. It is **not tracked by git** (the cloned repos are gitignored), but the derived artefacts are:

| File | Description |
|------|-------------|
| `catalogue.json` | 31 structured fixture entries from statsmodels and linearmodels test suites, annotated with estimator type, dataset, expected quantities, and priority |
| `nist_embedded.py` | Self-contained NIST StRD data — importable with no statsmodels dependency; exposes `DATASETS["Longley"]` with certified coefficients, SEs, R², and F-statistic |
| `external_results_files.json` | 64-entry index of CSV/txt reference files used by statsmodels and linearmodels tests, with provenance and quantities |
| `results_files/` | Copies of all 64 referenced data files, mirroring the upstream directory structure |

**26 of the 31 catalogue fixtures are high-priority** (explicitly cross-checked against Stata, Gretl, R, or NIST). Key sources:

- `statsmodels/regression/tests/test_robustcov.py` — HC0/HC1 and HAC Newey-West vs Stata (`macrodata`)
- `statsmodels/regression/tests/test_glsar_stata.py` — Cochrane-Orcutt GLSAR vs Stata
- `statsmodels/regression/tests/test_glsar_gretl.py` — GLSAR vs Gretl (includes RESET, ARCH, HAC)
- `linearmodels/tests/iv/test_against_stata.py` — IV2SLS, LIML, GMM vs Stata (`housing.csv`, simulated)
- `linearmodels/tests/panel/test_simulated_against_stata.py` — FE, RE, Pooled, Between OLS vs Stata

To rebuild the fixtures directory from scratch:

```bash
cd tests/validation/fixtures

git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/statsmodels/statsmodels.git statsmodels_tests
cd statsmodels_tests && git sparse-checkout set \
  statsmodels/regression/tests statsmodels/sandbox/regression statsmodels/tsa/tests
cd ..

git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/bashtage/linearmodels.git linearmodels_tests
cd linearmodels_tests && git sparse-checkout set \
  linearmodels/tests linearmodels/iv/tests linearmodels/panel/tests
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/bootstrap.md](docs/bootstrap.md) | Bootstrap inference — resampling methods, CI and p-value formulas, CLI, reproducibility |
| [docs/binary_model_output.md](docs/binary_model_output.md) | Logit/Probit output — likelihood metrics, prediction quality, marginal effects |
| [docs/pub_latex.md](docs/pub_latex.md) | Publication LaTeX tables — `ResultsTable`, `SummaryTable`, `DiagnosticsTable`; panels, booktabs, notes |
| [docs/sieve.md](docs/sieve.md) | Sieve infrastructure — functional-form and instrument search; honest evaluation; reproducibility |

---

## Data Sources

### `data_lake/raw/wooldridge_and_oleg/`

| Source | Files | Description |
|--------|-------|-------------|
| Wooldridge | 92 | All datasets from *Introductory Econometrics: A Modern Approach* (Wooldridge). Sourced from [yiyuezhuo/Introductory-Econometrics](https://github.com/yiyuezhuo/Introductory-Econometrics). |
| Local | 3 | `fatkids_db692.dta`, `boat_race.dta`, `wheat_india.dta` — original project data. |

The `manifest.json` in that folder records every file's standardised name, original name, source tag, and SHA-256 hash.

---

## Data Conventions

### Raw is immutable

Files in `data_lake/raw/` are **never modified**. They are the ground-truth source of record. New data is added by appending new files or new source subdirectories — never by overwriting.

### Curated is versioned

`data_lake/curated/` holds cleaned, merged, or feature-engineered datasets derived from raw sources. Files here are versioned by date or semantic version in the filename, e.g. `wages_panel_v1.parquet`.

### Preferred intermediate format: Parquet

Parquet is the standard format for curated datasets and intermediate outputs:
- Efficient columnar storage with built-in compression
- Preserves column types (avoids CSV round-trip issues)
- Compatible with pandas, polars, R (`arrow`), Stata (via Python bridge)

`.dta` files live only in `raw/`. Once loaded into an analysis pipeline, save derived data as Parquet.

### Referencing shared data with hashes

To pin a dataset version in a project, record the filename and SHA-256 hash from `manifest.json`. Example:

```python
import hashlib, pathlib

def verify(path, expected_sha256):
    h = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    assert h == expected_sha256, f"Hash mismatch: {path}"
```

---

## Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| Raw data files | `snake_case.dta` | `cps78_85.dta`, `wheat_india.dta` |
| Curated datasets | `snake_case_vN.parquet` | `wages_panel_v1.parquet` |
| Notebooks | `NN_description.ipynb` | `01_eda_wages.ipynb` |
| Projects | `kebab-case/` | `returns-to-education/` |
| Scripts | `snake_case.py` | `clean_wages.py` |

---

## Git Strategy

- `data_lake/raw/` is **gitignored** — binary `.dta` files are not tracked
- `data_lake/raw/wooldridge_and_oleg/manifest.json` **is tracked** — preserves the file inventory and hashes
- `data_lake/curated/` Parquet files are gitignored by default unless small enough to commit
- Notebooks: strip outputs before committing (`nbstripout` recommended)

---

## Reproducing the Raw Data Lake

```bash
# Re-download Wooldridge datasets
git clone --depth=1 --filter=blob:none --sparse \
  https://github.com/yiyuezhuo/Introductory-Econometrics.git /tmp/woo
cd /tmp/woo && git sparse-checkout set "stata data"

# Copy and lowercase
for f in "/tmp/woo/stata data"/*.{dta,DTA}; do
  dest=$(basename "$f" | tr '[:upper:]' '[:lower:]')
  cp "$f" "data_lake/raw/wooldridge_and_oleg/$dest"
done
rm -rf /tmp/woo
```

Local files (`fatkids_db692.dta`, `boat_race.dta`, `wheat_india.dta`) must be sourced separately.
