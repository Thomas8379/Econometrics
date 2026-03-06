"""Publication-quality LaTeX table builder.

Produces journal-style tables using booktabs, threeparttable, and optional
siunitx.  Output is designed to fit a standard A4 page cleanly.

Required LaTeX packages in the document preamble::

    \\usepackage{booktabs}
    \\usepackage{threeparttable}
    \\usepackage{float}
    % optional for number alignment:
    \\usepackage{siunitx}

Public API
----------
ResultsTable      – multi-model regression results with optional panels
SummaryTable      – descriptive statistics (mean/std/min/max/N …)
DiagnosticsTable  – statistical test results with optional grouping

Typical usage::

    from econtools.output.tables.pub_latex import ResultsTable
    t = ResultsTable(
        results=[res1, res2, res3],
        labels=["(1)", "(2)", "(3)"],
        estimator_labels=["OLS", "2SLS", "2SLS"],
        panels=[("Baseline", ["educ", "exper"]), ("Extended", ["educ", "exper", "tenure"])],
        footer_stats=["N", "r_squared", "f_stat"],
        notes=["Standard errors in parentheses."],
        caption="Returns to Schooling",
        label="tab:returns",
    )
    print(t.to_latex())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from econtools._core.formatting import _fmt, _latex_star, _star
from econtools._core.types import Estimate, TestResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """Minimal LaTeX escape for variable/column names in table cells."""
    return (
        text.replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def _coef_cell(coef: float, pval: float, digits: int, stars: bool) -> str:
    s = _star(pval) if stars else ""
    return _fmt(coef, digits) + _latex_star(s)


def _se_cell(se: float, digits: int) -> str:
    return rf"$({_fmt(se, digits)})$"


def _nan_or(val: float, digits: int) -> str:
    if math.isnan(val):
        return ""
    return _fmt(val, digits)


def _first_stage_f(result: Estimate) -> float:
    """Extract first-stage F from a linearmodels IV result, or return nan."""
    raw = result.raw
    # linearmodels stores first_stage result object
    for attr in ("first_stage", "_first_stage"):
        fs = getattr(raw, attr, None)
        if fs is not None:
            try:
                # linearmodels: first_stage.diagnostics["f.stat"]
                diag = getattr(fs, "diagnostics", None)
                if diag is not None:
                    return float(diag.loc["f.stat", "stat"])
            except Exception:
                pass
    return float("nan")


def _tabular_spec(n_cols: int, use_siunitx: bool) -> str:
    if use_siunitx:
        return "l" + " S[table-format=3.3]" * n_cols
    return "l" + " c" * n_cols


def _font_cmd(font_size: str | None) -> tuple[str, str]:
    """Return (open_cmd, close_cmd) for font size wrapping."""
    if font_size is None:
        return ("", "")
    valid = {"tiny", "scriptsize", "footnotesize", "small", "normalsize", "large"}
    if font_size not in valid:
        raise ValueError(f"font_size must be one of {valid}")
    return (rf"\{font_size}{{%", r"%}")


def _build_table_env(
    body_lines: list[str],
    caption: str | None,
    label: str | None,
    float_position: str,
    fit_to_page: bool,
    font_size: str | None,
    notes: list[str],
) -> str:
    """Wrap body_lines in table / threeparttable / resizebox environment."""
    fp = f"[{float_position}]" if float_position else ""
    lines: list[str] = [rf"\begin{{table}}{fp}", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")

    if notes:
        lines.append(r"\begin{threeparttable}")

    if fit_to_page:
        lines.append(r"\resizebox{\textwidth}{!}{%")

    font_open, font_close = _font_cmd(font_size)
    if font_open:
        lines.append(font_open)

    lines.extend(body_lines)

    if font_open:
        lines.append(font_close)
    if fit_to_page:
        lines.append(r"}")  # closes resizebox

    if notes:
        lines.append(r"\begin{tablenotes}")
        lines.append(r"\footnotesize")
        for note in notes:
            lines.append(rf"\item {note}")
        lines.append(r"\end{tablenotes}")
        lines.append(r"\end{threeparttable}")

    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ResultsTable
# ---------------------------------------------------------------------------

# Footer stat label lookup
_FOOTER_LABELS: dict[str, str] = {
    "N": r"$N$",
    "r_squared": r"$R^2$",
    "r_squared_adj": r"Adj.\ $R^2$",
    "r_squared_within": r"Within $R^2$",
    "f_stat": r"$F$-statistic",
    "rmse": "RMSE",
    "aic": "AIC",
    "bic": "BIC",
    "first_stage_f": r"First-stage $F$",
    "cov_type": "SE type",
    "log_likelihood": "Log-likelihood",
}


def _footer_value(stat: str, result: Estimate, digits: int) -> str:
    if stat == "N":
        return str(int(result.fit.nobs))
    if stat == "first_stage_f":
        v = _first_stage_f(result)
        return _nan_or(v, digits)
    if stat == "cov_type":
        return _esc(result.cov_type)
    attr_map = {
        "r_squared": "r_squared",
        "r_squared_adj": "r_squared_adj",
        "r_squared_within": "r_squared_within",
        "f_stat": "f_stat",
        "rmse": "rmse",
        "aic": "aic",
        "bic": "bic",
        "log_likelihood": "log_likelihood",
    }
    if stat in attr_map:
        v = getattr(result.fit, attr_map[stat], float("nan"))
        return _nan_or(float(v), digits)
    return ""


class ResultsTable:
    """Publication-quality multi-model results table.

    Parameters
    ----------
    results:
        List of fitted :class:`~econtools._core.types.Estimate` objects.
    labels:
        Column header labels e.g. ``["(1)", "(2)", "(3)"]``.  Defaults to
        ``(1)``, ``(2)``, …
    estimator_labels:
        Optional second header row with estimator names (e.g. "OLS", "2SLS").
    variable_names:
        Mapping from parameter name to display name.
    omit_vars:
        Parameter names to suppress from the table body (e.g. ``["const"]``).
    panels:
        List of ``(panel_label, [var_names])`` tuples.  Variables are shown
        under their panel header.  Variables not in any panel are ignored.
        If ``None`` all variables are shown in a flat list.
    column_groups:
        List of ``(group_label, [col_indices])`` tuples for spanning column
        headers, rendered with ``\\cmidrule``.  Indices are 1-based relative
        to the data columns (not including the label column).
    footer_stats:
        Statistics to show below the separator.  Supports built-in keys
        (``"N"``, ``"r_squared"``, ``"r_squared_adj"``, ``"r_squared_within"``,
        ``"f_stat"``, ``"rmse"``, ``"aic"``, ``"bic"``, ``"first_stage_f"``,
        ``"cov_type"``) and custom ``(label, [values])`` tuples.
    notes:
        List of note strings rendered in the ``tablenotes`` block.
    add_star_note:
        Automatically append a significance-level note.
    stars:
        Whether to append significance stars to coefficients.
    digits:
        Decimal places.
    fit_to_page:
        Wrap the tabular in ``\\resizebox{\\textwidth}{!}{}``.
    use_siunitx:
        Use ``siunitx`` ``S`` columns for number alignment.
    caption:
        Table caption.
    label:
        LaTeX ``\\label`` key.
    float_position:
        Float specifier e.g. ``"htbp"`` or ``"H"``.
    font_size:
        Font size command (``"small"``, ``"footnotesize"``, etc.) or ``None``.
    """

    def __init__(
        self,
        results: list[Estimate],
        labels: list[str] | None = None,
        estimator_labels: list[str] | None = None,
        variable_names: dict[str, str] | None = None,
        omit_vars: list[str] | None = None,
        panels: list[tuple[str, list[str]]] | None = None,
        column_groups: list[tuple[str, list[int]]] | None = None,
        footer_stats: list[str | tuple] | None = None,
        notes: list[str] | None = None,
        add_star_note: bool = True,
        stars: bool = True,
        digits: int = 3,
        fit_to_page: bool = True,
        use_siunitx: bool = False,
        caption: str | None = None,
        label: str | None = None,
        float_position: str = "htbp",
        font_size: str | None = "small",
    ) -> None:
        if not results:
            raise ValueError("results must be a non-empty list.")
        self.results = results
        self.labels = labels or [f"({i + 1})" for i in range(len(results))]
        self.estimator_labels = estimator_labels
        self.variable_names = variable_names or {}
        self.omit_vars = set(omit_vars or [])
        self.panels = panels
        self.column_groups = column_groups
        self.footer_stats = footer_stats or ["N", "r_squared", "r_squared_adj"]
        self.notes = list(notes or [])
        if add_star_note and stars:
            self.notes.append(
                r"$^{*}$ $p<0.10$,\ $^{**}$ $p<0.05$,\ $^{***}$ $p<0.01$."
            )
        self.stars = stars
        self.digits = digits
        self.fit_to_page = fit_to_page
        self.use_siunitx = use_siunitx
        self.caption = caption
        self.label = label
        self.float_position = float_position
        self.font_size = font_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_vars(self) -> list[str]:
        """Union of all param names across results, preserving first-seen order."""
        seen: set[str] = set()
        out: list[str] = []
        for res in self.results:
            for name in res.params.index:
                if name not in seen:
                    seen.add(name)
                    out.append(name)
        return [v for v in out if v not in self.omit_vars]

    def _display_name(self, var: str) -> str:
        raw = self.variable_names.get(var, var)
        return _esc(raw)

    def _coef_row(self, var: str) -> list[str]:
        """Return the coefficient row cells (one per result)."""
        cells = []
        for res in self.results:
            if var in res.params.index:
                cells.append(
                    _coef_cell(
                        float(res.params[var]),
                        float(res.pvalues[var]),
                        self.digits,
                        self.stars,
                    )
                )
            else:
                cells.append("")
        return cells

    def _se_row(self, var: str) -> list[str]:
        cells = []
        for res in self.results:
            if var in res.bse.index:
                cells.append(_se_cell(float(res.bse[var]), self.digits))
            else:
                cells.append("")
        return cells

    def _tabular_row(self, label: str, cells: list[str]) -> str:
        return " & ".join([label] + cells) + r" \\"

    def _panel_header(self, panel_label: str) -> str:
        n = len(self.results) + 1
        return (
            rf"\multicolumn{{{n}}}{{l}}{{\textit{{{panel_label}}}}} \\"
            + r"[2pt]"
        )

    def _body_lines(self) -> list[str]:
        lines: list[str] = []
        if self.panels:
            for panel_label, var_list in self.panels:
                lines.append(self._panel_header(panel_label))
                for var in var_list:
                    if var in self.omit_vars:
                        continue
                    lines.append(
                        self._tabular_row(self._display_name(var), self._coef_row(var))
                    )
                    lines.append(
                        self._tabular_row("", self._se_row(var))
                    )
                lines.append(r"\addlinespace[2pt]")
        else:
            for var in self._all_vars():
                lines.append(
                    self._tabular_row(self._display_name(var), self._coef_row(var))
                )
                lines.append(self._tabular_row("", self._se_row(var)))
        return lines

    def _footer_lines(self) -> list[str]:
        lines: list[str] = []
        for stat in self.footer_stats:
            if isinstance(stat, tuple):
                label, values = stat
                assert len(values) == len(self.results), (
                    f"Custom footer row '{label}' must have {len(self.results)} values."
                )
                lines.append(self._tabular_row(label, [str(v) for v in values]))
            else:
                lbl = _FOOTER_LABELS.get(stat, stat)
                vals = [_footer_value(stat, res, self.digits) for res in self.results]
                lines.append(self._tabular_row(lbl, vals))
        return lines

    def _column_group_lines(self) -> list[str]:
        """Render column group spanning headers with cmidrule."""
        if not self.column_groups:
            return []
        header_cells = [""]  # empty label column
        cmidrules = []
        n_model_cols = len(self.results)

        for group_label, col_indices in self.column_groups:
            # col_indices are 1-based among data columns
            lo = min(col_indices) + 1  # +1 for label col
            hi = max(col_indices) + 1
            cmidrules.append(rf"\cmidrule(lr){{{lo}-{hi}}}")
            span = hi - lo + 1
            header_cells.append(
                rf"\multicolumn{{{span}}}{{c}}{{{group_label}}}"
            )

        line = " & ".join(header_cells) + r" \\"
        return [line] + cmidrules

    def to_latex(self) -> str:
        """Return the full LaTeX table string."""
        n = len(self.results)
        spec = _tabular_spec(n, self.use_siunitx)

        body: list[str] = [rf"\begin{{tabular}}{{{spec}}}"]
        body.append(r"\toprule")

        # Column group spanning headers (if any)
        if self.column_groups:
            body.extend(self._column_group_lines())

        # Primary header row (model numbers)
        body.append(self._tabular_row("", self.labels))

        # Optional estimator name row
        if self.estimator_labels:
            body.append(self._tabular_row("", self.estimator_labels))

        body.append(r"\midrule")

        # Body: coefficients (with panels or flat)
        body.extend(self._body_lines())

        body.append(r"\midrule")

        # Footer statistics
        body.extend(self._footer_lines())

        body.append(r"\bottomrule")
        body.append(r"\end{tabular}")

        return _build_table_env(
            body,
            caption=self.caption,
            label=self.label,
            float_position=self.float_position,
            fit_to_page=self.fit_to_page,
            font_size=self.font_size,
            notes=self.notes,
        )

    def __str__(self) -> str:
        return self.to_latex()


# ---------------------------------------------------------------------------
# SummaryTable
# ---------------------------------------------------------------------------

_STAT_LABELS: dict[str, str] = {
    "N": r"$N$",
    "mean": "Mean",
    "std": r"Std.\ Dev.",
    "min": "Min",
    "p25": "P25",
    "median": "Median",
    "p75": "P75",
    "max": "Max",
}

_STAT_FUNCS: dict[str, Any] = {
    "N": lambda s: int(s.count()),
    "mean": lambda s: s.mean(),
    "std": lambda s: s.std(),
    "min": lambda s: s.min(),
    "p25": lambda s: s.quantile(0.25),
    "median": lambda s: s.median(),
    "p75": lambda s: s.quantile(0.75),
    "max": lambda s: s.max(),
}


class SummaryTable:
    """Publication-quality descriptive statistics table.

    Parameters
    ----------
    df:
        Source :class:`pandas.DataFrame`.
    vars:
        Columns to include.  ``None`` selects all numeric columns.
    stats:
        Statistics to compute per variable.  Supported: ``"N"``, ``"mean"``,
        ``"std"``, ``"min"``, ``"p25"``, ``"median"``, ``"p75"``, ``"max"``.
    var_names:
        Display name overrides (column name → display name).
    panels:
        ``[(panel_label, [var_names])]`` for grouped rows.
    caption / label / digits / fit_to_page / float_position / font_size:
        Standard table options.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vars: list[str] | None = None,
        stats: list[str] = ("mean", "std", "min", "max", "N"),
        var_names: dict[str, str] | None = None,
        panels: list[tuple[str, list[str]]] | None = None,
        notes: list[str] | None = None,
        caption: str | None = None,
        label: str | None = None,
        digits: int = 3,
        fit_to_page: bool = True,
        float_position: str = "htbp",
        font_size: str | None = "small",
    ) -> None:
        self.df = df
        self.vars = vars or list(df.select_dtypes("number").columns)
        self.stats = list(stats)
        self.var_names = var_names or {}
        self.panels = panels
        self.notes = list(notes or [])
        self.caption = caption
        self.label = label
        self.digits = digits
        self.fit_to_page = fit_to_page
        self.float_position = float_position
        self.font_size = font_size

    def _display_name(self, col: str) -> str:
        return _esc(self.var_names.get(col, col))

    def _stat_val(self, col: str, stat: str) -> str:
        series = self.df[col].dropna()
        fn = _STAT_FUNCS[stat]
        v = fn(series)
        if stat == "N":
            return str(int(v))
        return _fmt(float(v), self.digits)

    def _tabular_row(self, label: str, cells: list[str]) -> str:
        return " & ".join([label] + cells) + r" \\"

    def _panel_header(self, panel_label: str) -> str:
        n = len(self.stats) + 1
        return rf"\multicolumn{{{n}}}{{l}}{{\textit{{{panel_label}}}}} \\[2pt]"

    def _var_row(self, col: str) -> str:
        cells = [self._stat_val(col, s) for s in self.stats]
        return self._tabular_row(self._display_name(col), cells)

    def to_latex(self) -> str:
        spec = "l" + " r" * len(self.stats)
        header_cells = [_STAT_LABELS.get(s, s) for s in self.stats]

        body: list[str] = [rf"\begin{{tabular}}{{{spec}}}"]
        body.append(r"\toprule")
        body.append(self._tabular_row("Variable", header_cells))
        body.append(r"\midrule")

        if self.panels:
            for panel_label, var_list in self.panels:
                body.append(self._panel_header(panel_label))
                for col in var_list:
                    if col not in self.df.columns:
                        continue
                    body.append(self._var_row(col))
                body.append(r"\addlinespace[2pt]")
        else:
            for col in self.vars:
                if col not in self.df.columns:
                    continue
                body.append(self._var_row(col))

        body.append(r"\bottomrule")
        body.append(r"\end{tabular}")

        return _build_table_env(
            body,
            caption=self.caption,
            label=self.label,
            float_position=self.float_position,
            fit_to_page=self.fit_to_page,
            font_size=self.font_size,
            notes=self.notes,
        )

    def __str__(self) -> str:
        return self.to_latex()


# ---------------------------------------------------------------------------
# DiagnosticsTable
# ---------------------------------------------------------------------------


class DiagnosticsTable:
    """Publication-quality diagnostic test results table.

    Parameters
    ----------
    tests:
        List of :class:`~econtools._core.types.TestResult` objects.
    groups:
        ``[(group_label, [indices])]`` where indices are 0-based into ``tests``.
        If provided, tests are rendered under group headers.
    show_h0:
        Include H₀ text as a sub-row below each test.
    notes / caption / label / digits / fit_to_page / float_position / font_size:
        Standard table options.
    """

    def __init__(
        self,
        tests: list[TestResult],
        groups: list[tuple[str, list[int]]] | None = None,
        show_h0: bool = False,
        notes: list[str] | None = None,
        caption: str | None = None,
        label: str | None = None,
        digits: int = 3,
        fit_to_page: bool = False,
        float_position: str = "htbp",
        font_size: str | None = None,
    ) -> None:
        self.tests = tests
        self.groups = groups
        self.show_h0 = show_h0
        self.notes = list(notes or [])
        self.caption = caption
        self.label = label
        self.digits = digits
        self.fit_to_page = fit_to_page
        self.float_position = float_position
        self.font_size = font_size

    def _df_str(self, df: float | tuple | None) -> str:
        if df is None:
            return ""
        if isinstance(df, tuple):
            return f"({df[0]:.0f}, {df[1]:.0f})"
        return f"{df:.0f}"

    def _test_rows(self, test_indices: list[int]) -> list[str]:
        rows = []
        for i in test_indices:
            r = self.tests[i]
            reject = "Yes" if r.reject else "No"
            rows.append(
                " & ".join([
                    _esc(r.test_name),
                    _fmt(r.statistic, self.digits),
                    _fmt(r.pvalue, self.digits),
                    self._df_str(r.df),
                    r.distribution,
                    reject,
                ]) + r" \\"
            )
            if self.show_h0:
                rows.append(
                    rf"\multicolumn{{6}}{{l}}{{\footnotesize{{H$_0$: {_esc(r.null_hypothesis)}}}}} \\"
                )
        return rows

    def _group_header(self, label: str) -> str:
        return rf"\multicolumn{{6}}{{l}}{{\textit{{{label}}}}} \\[2pt]"

    def to_latex(self) -> str:
        spec = "l r r r l c"
        header = r"Test & Stat & $p$-value & df & Dist. & Reject \\"

        body: list[str] = [rf"\begin{{tabular}}{{{spec}}}"]
        body.append(r"\toprule")
        body.append(header)
        body.append(r"\midrule")

        if self.groups:
            for group_label, indices in self.groups:
                body.append(self._group_header(group_label))
                body.extend(self._test_rows(indices))
                body.append(r"\addlinespace[2pt]")
        else:
            body.extend(self._test_rows(list(range(len(self.tests)))))

        body.append(r"\bottomrule")
        body.append(r"\end{tabular}")

        # Put H0s in notes if not showing inline
        all_notes = list(self.notes)
        if not self.show_h0:
            for r in self.tests:
                all_notes.append(rf"\textit{{{_esc(r.test_name)}}}: H$_0$: {_esc(r.null_hypothesis)}")

        return _build_table_env(
            body,
            caption=self.caption,
            label=self.label,
            float_position=self.float_position,
            fit_to_page=self.fit_to_page,
            font_size=self.font_size,
            notes=all_notes,
        )

    def __str__(self) -> str:
        return self.to_latex()
