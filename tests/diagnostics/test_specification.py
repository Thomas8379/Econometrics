"""Tests for econtools.diagnostics.specification."""

from __future__ import annotations

from econtools.diagnostics import reset_test
from econtools.inference.hypothesis import TestResult
from econtools.models import fit_ols


def test_reset_returns_test_result(ols_result) -> None:
    tr = reset_test(ols_result)
    assert isinstance(tr, TestResult)
    assert tr.test_name == "RESET"


def test_reset_rejects_misspecified(misspecified_data) -> None:
    """Linear model on quadratic DGP — RESET should reject."""
    result = fit_ols(misspecified_data, "y", ["x"])
    tr = reset_test(result)
    assert tr.reject is True
    assert tr.pvalue < 0.001


def test_reset_no_reject_well_specified(ols_result) -> None:
    """Correctly specified linear DGP — RESET should not reject."""
    tr = reset_test(ols_result)
    assert tr.pvalue > 0.05
