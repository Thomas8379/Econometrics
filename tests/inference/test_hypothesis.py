"""Tests for econtools.inference.hypothesis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from econtools.inference.hypothesis import (
    TestResult,
    t_test_coeff,
    f_test,
    wald_test,
    conf_int,
)


def test_t_test_returns_test_result(ols_result) -> None:
    tr = t_test_coeff(ols_result, "x1")
    assert isinstance(tr, TestResult)


def test_t_test_x1_rejects_zero(ols_result) -> None:
    """x1 true coeff = 3 — H0: x1=0 should be rejected."""
    tr = t_test_coeff(ols_result, "x1", value=0.0)
    assert tr.reject is True
    assert tr.pvalue < 0.001


def test_t_test_invalid_var_raises(ols_result) -> None:
    with pytest.raises(ValueError, match="not found"):
        t_test_coeff(ols_result, "nonexistent_var")


def test_conf_int_returns_dataframe(ols_result) -> None:
    ci = conf_int(ols_result)
    assert isinstance(ci, pd.DataFrame)
    assert "lower" in ci.columns
    assert "upper" in ci.columns


def test_conf_int_width_increases_with_alpha(ols_result) -> None:
    """A 90% CI is narrower than a 99% CI."""
    ci_90 = conf_int(ols_result, alpha=0.10)
    ci_99 = conf_int(ols_result, alpha=0.01)
    width_90 = (ci_90["upper"] - ci_90["lower"]).mean()
    width_99 = (ci_99["upper"] - ci_99["lower"]).mean()
    assert width_90 < width_99


def test_f_test_joint_restriction(ols_result) -> None:
    """Test x1=0 AND x2=0 jointly — should reject strongly."""
    # params: [const, x1, x2]
    n_params = len(ols_result.params)
    # Restrict x1 and x2 to zero (exclude const)
    R = np.zeros((2, n_params))
    idx = list(ols_result.params.index)
    R[0, idx.index("x1")] = 1.0
    R[1, idx.index("x2")] = 1.0
    tr = f_test(ols_result, R)
    assert isinstance(tr, TestResult)
    assert tr.distribution == "F"
    assert tr.reject is True


def test_wald_test_chi2(ols_result) -> None:
    """Wald test with use_f=False should report Chi2 distribution."""
    n_params = len(ols_result.params)
    R = np.zeros((1, n_params))
    idx = list(ols_result.params.index)
    R[0, idx.index("x1")] = 1.0
    tr = wald_test(ols_result, R, use_f=False)
    assert tr.distribution == "Chi2"
    assert tr.reject is True
