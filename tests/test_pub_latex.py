"""Tests for publication-quality LaTeX table builder."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from econtools._core.types import Estimate, FitMetrics, TestResult
from econtools.output.tables.pub_latex import DiagnosticsTable, ResultsTable, SummaryTable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_estimate(
    dep_var: str = "y",
    params: dict | None = None,
    model_type: str = "OLS",
    n: int = 100,
    r2: float = 0.35,
) -> Estimate:
    params = params or {"const": 1.5, "x1": 0.42, "x2": -0.17}
    idx = list(params.keys())
    p = pd.Series(params)
    bse = pd.Series({k: abs(v) * 0.15 + 0.05 for k, v in params.items()})
    tvals = p / bse
    pvals = pd.Series({k: 0.001 if abs(t) > 2 else 0.15 for k, t in tvals.items()})

    fit = FitMetrics(
        nobs=n, df_model=float(len(params) - 1), df_resid=float(n - len(params)),
        r_squared=r2, r_squared_adj=r2 - 0.01, f_stat=45.0, rmse=0.5,
    )
    return Estimate(
        model_type=model_type,
        dep_var=dep_var,
        params=p,
        bse=bse,
        tvalues=tvals,
        pvalues=pvals,
        conf_int_lower=p - 1.96 * bse,
        conf_int_upper=p + 1.96 * bse,
        resid=pd.Series(np.zeros(n)),
        fitted=pd.Series(np.ones(n)),
        cov_params=pd.DataFrame(),
        cov_type="HC3",
        fit=fit,
        raw=None,
    )


def _make_test_result(
    name: str = "Breusch-Pagan",
    stat: float = 5.2,
    pval: float = 0.022,
    reject: bool = True,
) -> TestResult:
    return TestResult(
        test_name=name,
        statistic=stat,
        pvalue=pval,
        df=2.0,
        distribution="Chi2",
        null_hypothesis="Homoskedasticity",
        reject=reject,
    )


# ---------------------------------------------------------------------------
# ResultsTable
# ---------------------------------------------------------------------------


class TestResultsTable:
    def test_basic_output(self):
        res = _make_estimate()
        t = ResultsTable([res], caption="Test table", label="tab:test")
        latex = t.to_latex()
        assert r"\begin{table}" in latex
        assert r"\toprule" in latex
        assert r"\midrule" in latex
        assert r"\bottomrule" in latex
        assert r"\end{table}" in latex

    def test_booktabs_not_hline(self):
        res = _make_estimate()
        t = ResultsTable([res])
        latex = t.to_latex()
        assert r"\hline" not in latex

    def test_threeparttable_with_notes(self):
        res = _make_estimate()
        t = ResultsTable([res], notes=["A note.", "Another note."], add_star_note=False)
        latex = t.to_latex()
        assert r"\begin{threeparttable}" in latex
        assert r"\begin{tablenotes}" in latex
        assert r"\item A note." in latex

    def test_no_notes_no_threeparttable(self):
        res = _make_estimate()
        t = ResultsTable([res], notes=[], add_star_note=False)
        latex = t.to_latex()
        assert r"\begin{threeparttable}" not in latex

    def test_resizebox_when_fit_to_page(self):
        res = _make_estimate()
        t = ResultsTable([res], fit_to_page=True)
        latex = t.to_latex()
        assert r"\resizebox{\textwidth}{!}" in latex

    def test_no_resizebox_when_not_fit_to_page(self):
        res = _make_estimate()
        t = ResultsTable([res], fit_to_page=False)
        latex = t.to_latex()
        assert r"\resizebox" not in latex

    def test_multiple_models(self):
        res1 = _make_estimate(params={"const": 1.0, "x1": 0.5})
        res2 = _make_estimate(params={"const": 0.8, "x1": 0.3, "x2": 0.1})
        t = ResultsTable([res1, res2], labels=["(1)", "(2)"])
        latex = t.to_latex()
        assert "(1)" in latex
        assert "(2)" in latex
        # Both x1 and x2 should appear
        assert "x1" in latex
        assert "x2" in latex

    def test_omit_const(self):
        res = _make_estimate(params={"const": 1.5, "x1": 0.42})
        t = ResultsTable([res], omit_vars=["const"])
        latex = t.to_latex()
        # const should not appear
        # (it may appear in _esc form, but the column itself should not)
        # Check that const rows are omitted
        assert r"const" not in latex

    def test_panels(self):
        res = _make_estimate(params={"const": 1.5, "x1": 0.42, "x2": -0.17})
        t = ResultsTable(
            [res],
            panels=[("Panel A: Baseline", ["x1"]), ("Panel B: Extended", ["x2"])],
            omit_vars=["const"],
        )
        latex = t.to_latex()
        assert "Panel A" in latex
        assert "Panel B" in latex

    def test_variable_names(self):
        res = _make_estimate(params={"x1": 0.5})
        t = ResultsTable([res], variable_names={"x1": "Education"}, omit_vars=["const"])
        latex = t.to_latex()
        assert "Education" in latex

    def test_estimator_labels(self):
        res1 = _make_estimate(model_type="OLS")
        res2 = _make_estimate(model_type="2SLS")
        t = ResultsTable([res1, res2], estimator_labels=["OLS", "2SLS"])
        latex = t.to_latex()
        assert "OLS" in latex
        assert "2SLS" in latex

    def test_star_note_included_by_default(self):
        res = _make_estimate()
        t = ResultsTable([res], stars=True, add_star_note=True, notes=[])
        latex = t.to_latex()
        assert "p<0" in latex

    def test_footer_stats(self):
        res = _make_estimate()
        t = ResultsTable([res], footer_stats=["N", "r_squared", "f_stat"])
        latex = t.to_latex()
        assert r"$N$" in latex
        assert r"$R^2$" in latex
        assert r"$F$-statistic" in latex

    def test_custom_footer_row(self):
        res = _make_estimate()
        t = ResultsTable([res], footer_stats=[("My custom row", ["Yes"])])
        latex = t.to_latex()
        assert "My custom row" in latex

    def test_caption_and_label(self):
        res = _make_estimate()
        t = ResultsTable([res], caption="Returns to Schooling", label="tab:main")
        latex = t.to_latex()
        assert r"\caption{Returns to Schooling}" in latex
        assert r"\label{tab:main}" in latex

    def test_str_returns_latex(self):
        res = _make_estimate()
        t = ResultsTable([res])
        assert str(t) == t.to_latex()

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ResultsTable([])

    def test_column_groups(self):
        res1 = _make_estimate()
        res2 = _make_estimate()
        res3 = _make_estimate()
        t = ResultsTable(
            [res1, res2, res3],
            column_groups=[("Group A", [1, 2]), ("Group B", [3])],
        )
        latex = t.to_latex()
        assert "Group A" in latex
        assert r"\cmidrule" in latex

    def test_se_in_parens(self):
        res = _make_estimate(params={"x1": 0.42})
        t = ResultsTable([res], omit_vars=["const"])
        latex = t.to_latex()
        # SE rows should be present (shown as $(0.xxx)$)
        assert "$(0." in latex or "$(0" in latex.replace(" ", "")


# ---------------------------------------------------------------------------
# SummaryTable
# ---------------------------------------------------------------------------


class TestSummaryTable:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "wage": rng.lognormal(2.0, 0.5, 100),
            "educ": rng.integers(8, 20, 100).astype(float),
            "exper": rng.integers(0, 40, 100).astype(float),
        })

    def test_basic(self, sample_df):
        t = SummaryTable(sample_df, stats=["mean", "std", "N"])
        latex = t.to_latex()
        assert r"\toprule" in latex
        assert "Mean" in latex
        assert "Std" in latex
        assert r"$N$" in latex

    def test_all_vars_included(self, sample_df):
        t = SummaryTable(sample_df, vars=["wage", "educ"])
        latex = t.to_latex()
        assert "wage" in latex
        assert "educ" in latex
        assert "exper" not in latex

    def test_var_names_override(self, sample_df):
        t = SummaryTable(sample_df, var_names={"wage": "Wage Rate"})
        latex = t.to_latex()
        assert "Wage Rate" in latex

    def test_panels(self, sample_df):
        t = SummaryTable(
            sample_df,
            panels=[("Panel A", ["wage"]), ("Panel B", ["educ", "exper"])],
        )
        latex = t.to_latex()
        assert "Panel A" in latex
        assert "Panel B" in latex

    def test_caption_label(self, sample_df):
        t = SummaryTable(sample_df, caption="Summary Stats", label="tab:sum")
        latex = t.to_latex()
        assert r"\caption{Summary Stats}" in latex
        assert r"\label{tab:sum}" in latex

    def test_missing_col_skipped(self, sample_df):
        t = SummaryTable(sample_df, vars=["wage", "nonexistent"])
        # Should not raise, just skip
        latex = t.to_latex()
        assert "wage" in latex


# ---------------------------------------------------------------------------
# DiagnosticsTable
# ---------------------------------------------------------------------------


class TestDiagnosticsTable:
    def test_basic(self):
        tests = [_make_test_result(), _make_test_result("White test", 8.1, 0.017)]
        t = DiagnosticsTable(tests)
        latex = t.to_latex()
        assert r"\toprule" in latex
        assert "Breusch-Pagan" in latex
        assert "White test" in latex

    def test_h0_in_notes(self):
        tests = [_make_test_result()]
        t = DiagnosticsTable(tests, show_h0=False)
        latex = t.to_latex()
        # H0 should appear in notes (threeparttable)
        assert "Homoskedasticity" in latex

    def test_grouped_tests(self):
        t1 = _make_test_result("Breusch-Pagan")
        t2 = _make_test_result("Jarque-Bera", stat=2.1, pval=0.35, reject=False)
        t = DiagnosticsTable(
            [t1, t2],
            groups=[("Heteroskedasticity", [0]), ("Normality", [1])],
        )
        latex = t.to_latex()
        assert "Heteroskedasticity" in latex
        assert "Normality" in latex

    def test_caption(self):
        t = DiagnosticsTable([_make_test_result()], caption="Diagnostic Tests", label="tab:d")
        latex = t.to_latex()
        assert r"\caption{Diagnostic Tests}" in latex
