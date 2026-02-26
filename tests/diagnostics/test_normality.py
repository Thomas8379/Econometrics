"""Tests for econtools.diagnostics.normality."""

from __future__ import annotations

from econtools.diagnostics import jarque_bera
from econtools.inference.hypothesis import TestResult
from econtools.models import fit_ols


def test_jarque_bera_returns_test_result(ols_result) -> None:
    tr = jarque_bera(ols_result)
    assert isinstance(tr, TestResult)
    assert tr.test_name == "Jarque-Bera"
    assert tr.df == 2


def test_jarque_bera_rejects_non_normal(non_normal_data) -> None:
    """Chi²-distributed errors — JB should strongly reject normality."""
    result = fit_ols(non_normal_data, "y", ["x"])
    tr = jarque_bera(result)
    assert tr.reject is True
    assert tr.pvalue < 0.001


def test_jarque_bera_no_reject_normal(ols_result) -> None:
    """Normally-distributed errors — JB should not reject."""
    tr = jarque_bera(ols_result)
    assert tr.pvalue > 0.05
