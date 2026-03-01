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
├── tests/                     # pytest suite (164 tests)
│   ├── data/
│   ├── models/
│   ├── inference/
│   ├── diagnostics/
│   ├── plots/
│   ├── tables/
│   ├── fit/                   # tests for fit_model() dispatcher
│   └── validation/            # numerical validation against Wooldridge datasets
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
│   ├── latex/          # Journal profiles (ECONOMETRICA, AER), document assembly
│   └── knowledge_base/ # YAML entries: OLS, IV, BP test, Hausman, etc.
│
├── uncertainty/        # Variance estimation
│   └── cov_estimators.py  # HC0–HC3, HAC, cluster, kernel
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

### Running tests

```bash
python -m pytest tests/ -v                    # all tests (164 passing)
python -m pytest tests/ -v -m "not phase3"   # exclude Phase 3 stubs
python -m pytest tests/validation/ -v        # Wooldridge numerical validation
```

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
