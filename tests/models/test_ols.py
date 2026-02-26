"""Tests for econtools.models.ols — fit_ols, fit_wls, fit_ols_formula."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from econtools.models import fit_ols, fit_wls, fit_ols_formula, RegressionResult
from econtools.inference.se_types import VALID_COV_TYPES


# ---------------------------------------------------------------------------
# fit_ols
# ---------------------------------------------------------------------------


def test_fit_ols_returns_result(ols_data) -> None:
    result = fit_ols(ols_data, "y", ["x1", "x2"])
    assert isinstance(result, RegressionResult)


def test_fit_ols_model_type(ols_data) -> None:
    result = fit_ols(ols_data, "y", ["x1", "x2"])
    assert result.model_type == "OLS"


def test_fit_ols_coefficients_close_to_true(ols_data) -> None:
    """DGP: y = 2 + 3*x1 + 0.5*x2 + e — should recover these parameters."""
    result = fit_ols(ols_data, "y", ["x1", "x2"])
    assert abs(result.params["const"] - 2.0) < 0.3
    assert abs(result.params["x1"] - 3.0) < 0.3
    assert abs(result.params["x2"] - 0.5) < 0.3


def test_fit_ols_r_squared_in_range(ols_result) -> None:
    assert 0.0 <= ols_result.fit.r_squared <= 1.0


def test_fit_ols_resid_length(ols_data, ols_result) -> None:
    assert len(ols_result.resid) == len(ols_data)


def test_fit_ols_hc3(ols_data) -> None:
    result = fit_ols(ols_data, "y", ["x1", "x2"], cov_type="HC3")
    assert result.cov_type == "HC3"
    assert isinstance(result.params, pd.Series)


def test_fit_ols_hac(ols_data) -> None:
    result = fit_ols(ols_data, "y", ["x1", "x2"], cov_type="HAC", maxlags=2)
    assert result.cov_type == "HAC"


def test_fit_ols_drops_nan(ols_data) -> None:
    df_with_nan = ols_data.copy()
    df_with_nan.loc[0, "y"] = float("nan")
    df_with_nan.loc[1, "x1"] = float("nan")
    result = fit_ols(df_with_nan, "y", ["x1", "x2"])
    assert len(result.resid) == len(ols_data) - 2


def test_fit_ols_invalid_cov_type(ols_data) -> None:
    with pytest.raises(ValueError, match="Unknown cov_type"):
        fit_ols(ols_data, "y", ["x1", "x2"], cov_type="HCCM99")


# ---------------------------------------------------------------------------
# fit_wls
# ---------------------------------------------------------------------------


def test_fit_wls_returns_result(ols_data) -> None:
    df = ols_data.copy()
    df["w"] = np.abs(df["x1"]) + 0.1
    result = fit_wls(df, "y", ["x1", "x2"], weights="w")
    assert isinstance(result, RegressionResult)
    assert result.model_type == "WLS"


def test_fit_wls_differs_from_ols(ols_data) -> None:
    """WLS with non-uniform weights should give different coefficients."""
    df = ols_data.copy()
    df["w"] = np.abs(df["x1"]) + 0.1
    ols = fit_ols(df, "y", ["x1", "x2"])
    wls = fit_wls(df, "y", ["x1", "x2"], weights="w")
    # Params should differ (they could coincidentally be equal, but very unlikely)
    assert not np.allclose(ols.params.values, wls.params.values, atol=1e-6)


# ---------------------------------------------------------------------------
# fit_ols_formula
# ---------------------------------------------------------------------------


def test_fit_ols_formula_returns_result(ols_data) -> None:
    result = fit_ols_formula(ols_data, "y ~ x1 + x2")
    assert isinstance(result, RegressionResult)


def test_fit_ols_formula_dep_var(ols_data) -> None:
    result = fit_ols_formula(ols_data, "y ~ x1 + x2")
    assert result.dep_var == "y"


def test_fit_ols_formula_has_intercept(ols_data) -> None:
    """Patsy formula adds 'Intercept' constant."""
    result = fit_ols_formula(ols_data, "y ~ x1 + x2")
    assert "Intercept" in result.params.index
