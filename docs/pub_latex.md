# Publication-quality LaTeX Tables

`econtools.output.tables` provides three classes for generating journal-ready LaTeX tables directly from fitted model results, DataFrames, and test results.

## Required LaTeX packages

Add to your document preamble:

```latex
\usepackage{booktabs}        % \toprule / \midrule / \bottomrule
\usepackage{threeparttable}  % notes block below the table
\usepackage{float}           % [H] float placement
% optional:
\usepackage{siunitx}         % aligned number columns (use_siunitx=True)
```

---

## ResultsTable

Journal-style multi-model coefficient table.

```python
from econtools.output.tables import ResultsTable

t = ResultsTable(
    results=[res_ols, res_iv],           # list of Estimate objects
    labels=["(1)", "(2)"],               # column headers
    estimator_labels=["OLS", "2SLS"],    # optional second header row
    variable_names={"educ": "Education", "exper": "Experience"},
    omit_vars=["const"],                 # suppress from body
    footer_stats=["N", "r_squared", "r_squared_adj", "first_stage_f"],
    notes=["Robust standard errors in parentheses."],
    caption="Returns to Schooling",
    label="tab:returns",
)
print(t.to_latex())
```

### Panels

Group variables under labelled panel headers (Panel A / Panel B style):

```python
t = ResultsTable(
    results=[res1, res2, res3],
    panels=[
        ("Panel A: Baseline",  ["educ", "exper"]),
        ("Panel B: Extended",  ["educ", "exper", "tenure", "married"]),
    ],
    omit_vars=["const"],
    ...
)
```

Each panel is separated by `\addlinespace` and its header spans all columns in italics.

### Column groups

Span a label over multiple model columns with `\cmidrule`:

```python
t = ResultsTable(
    results=[r1, r2, r3, r4],
    column_groups=[
        ("Cross-section", [1, 2]),    # columns 1–2 (1-based among data cols)
        ("Panel",         [3, 4]),
    ],
    ...
)
```

### Footer statistics

Built-in keys:

| Key | Label |
|-----|-------|
| `"N"` | $N$ |
| `"r_squared"` | $R^2$ |
| `"r_squared_adj"` | Adj.\ $R^2$ |
| `"r_squared_within"` | Within $R^2$ |
| `"f_stat"` | $F$-statistic |
| `"rmse"` | RMSE |
| `"aic"` | AIC |
| `"bic"` | BIC |
| `"first_stage_f"` | First-stage $F$ |
| `"cov_type"` | SE type |
| `"log_likelihood"` | Log-likelihood |

Custom rows: `footer_stats=[("Fixed effects", ["Yes", "No", "Yes"])]`.
The values list must have the same length as `results`.

### Full parameter reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `results` | required | List of `Estimate` objects |
| `labels` | `["(1)", ...]` | Column header labels |
| `estimator_labels` | `None` | Optional second header row |
| `variable_names` | `{}` | `{col: display_name}` overrides |
| `omit_vars` | `[]` | Variables to suppress |
| `panels` | `None` | `[(label, [vars])]` grouping |
| `column_groups` | `None` | `[(label, [col_indices])]` spanning headers |
| `footer_stats` | `["N","r_squared","r_squared_adj"]` | Statistics below separator |
| `notes` | `[]` | Note strings in `tablenotes` |
| `add_star_note` | `True` | Auto-append p-value key to notes |
| `stars` | `True` | Significance stars on coefficients |
| `digits` | `3` | Decimal places |
| `fit_to_page` | `True` | Wrap in `\resizebox{\textwidth}{!}{}` |
| `use_siunitx` | `False` | Use `S` columns for number alignment |
| `caption` | `None` | Table caption |
| `label` | `None` | LaTeX `\label` key |
| `float_position` | `"htbp"` | Float specifier |
| `font_size` | `"small"` | Font size command (`None` to omit) |

---

## SummaryTable

Descriptive statistics for a set of variables.

```python
from econtools.output.tables import SummaryTable

t = SummaryTable(
    df,
    vars=["wage", "educ", "exper", "tenure"],
    stats=["mean", "std", "min", "p25", "median", "p75", "max", "N"],
    var_names={"wage": "Wage (\\$/hr)", "educ": "Education (yrs)"},
    caption="Summary Statistics",
    label="tab:summary",
)
print(t.to_latex())
```

### Grouped summary table (panels)

```python
t = SummaryTable(
    df,
    panels=[
        ("Outcome variables",   ["wage", "log_wage"]),
        ("Individual controls", ["educ", "exper", "tenure"]),
        ("Demographic",         ["married", "black", "south"]),
    ],
    stats=["mean", "std", "N"],
)
```

### Available statistics

| Key | Description |
|-----|-------------|
| `"N"` | Non-missing count |
| `"mean"` | Mean |
| `"std"` | Standard deviation |
| `"min"` | Minimum |
| `"p25"` | 25th percentile |
| `"median"` | Median (50th percentile) |
| `"p75"` | 75th percentile |
| `"max"` | Maximum |

---

## DiagnosticsTable

Compact table of hypothesis test results.

```python
from econtools.output.tables import DiagnosticsTable
from econtools.evaluation.heteroskedasticity import breusch_pagan, white_test
from econtools.evaluation.normality import jarque_bera

tests = [breusch_pagan(result), white_test(result), jarque_bera(result)]

t = DiagnosticsTable(
    tests,
    caption="Specification Diagnostics",
    label="tab:diag",
    show_h0=False,   # put H₀ in notes instead of inline rows
)
print(t.to_latex())
```

### Grouped diagnostics

```python
t = DiagnosticsTable(
    tests=[bp, wh, jb, dw, bg],
    groups=[
        ("Heteroskedasticity", [0, 1]),
        ("Normality",          [2]),
        ("Serial correlation", [3, 4]),
    ],
)
```

### Output columns

| Column | Content |
|--------|---------|
| Test | Test name |
| Stat | Test statistic |
| $p$-value | Asymptotic p-value |
| df | Degrees of freedom |
| Dist. | Reference distribution (F, Chi2, t, …) |
| Reject | Yes / No at 5% |

H₀ text appears either as inline sub-rows (`show_h0=True`) or in the `tablenotes` block (`show_h0=False`, the default).

---

## Integration with `assemble_document`

Combine multiple tables into a full LaTeX document:

```python
from econtools.output.latex.document import write_document
from econtools.output.latex.journal_profiles import AER

write_document(
    fragments=[summary_table.to_latex(), results_table.to_latex(), diag_table.to_latex()],
    out_path="output/paper_tables.tex",
    profile=AER,
    title="Returns to Schooling",
)
```

---

## Example: AER-style results table

```python
from econtools.output.tables import ResultsTable

t = ResultsTable(
    results=[ols_result, iv_result_q1, iv_result_qs],
    labels=["(1)", "(2)", "(3)"],
    estimator_labels=["OLS", "IV", "IV"],
    variable_names={"educ": "Education", "exper": "Experience (yrs)"},
    omit_vars=["const"],
    panels=[
        ("Panel A: Main results",     ["educ", "exper"]),
        ("Panel B: Robustness",       ["educ", "exper", "tenure"]),
    ],
    footer_stats=[
        "N",
        "r_squared",
        "first_stage_f",
        ("Quarter-of-birth IVs", ["No", "Yes", "Yes"]),
        ("State FE",             ["No", "No", "Yes"]),
    ],
    notes=["Robust (HC3) standard errors in parentheses. Sample: males aged 30–49."],
    caption="OLS and IV Estimates of Returns to Schooling",
    label="tab:returns",
    float_position="htbp",
)
print(t.to_latex())
```

The output is ready to paste directly into a `.tex` file — no manual editing of `&` separators or rule placement needed.
