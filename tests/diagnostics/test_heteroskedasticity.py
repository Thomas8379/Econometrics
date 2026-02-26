"""Tests for econtools.diagnostics.heteroskedasticity."""

from __future__ import annotations

from econtools.diagnostics import breusch_pagan, white_test
from econtools.inference.hypothesis import TestResult
from econtools.models import fit_ols


def test_breusch_pagan_returns_test_result(ols_result) -> None:
    tr = breusch_pagan(ols_result)
    assert isinstance(tr, TestResult)
    assert tr.test_name == "Breusch-Pagan"


def test_breusch_pagan_rejects_heteroskedastic(heteroskedastic_data) -> None:
    result = fit_ols(heteroskedastic_data, "y", ["x"])
    tr = breusch_pagan(result)
    assert tr.reject is True
    assert tr.pvalue < 0.01


def test_breusch_pagan_no_reject_homoskedastic(ols_result) -> None:
    """Well-specified homoskedastic DGP — BP should not reject."""
    tr = breusch_pagan(ols_result)
    assert tr.pvalue > 0.05


def test_white_test_returns_test_result(ols_result) -> None:
    tr = white_test(ols_result)
    assert isinstance(tr, TestResult)
    assert tr.test_name == "White"


def test_white_test_rejects_heteroskedastic(heteroskedastic_data) -> None:
    result = fit_ols(heteroskedastic_data, "y", ["x"])
    tr = white_test(result)
    assert tr.reject is True
    assert tr.pvalue < 0.01
