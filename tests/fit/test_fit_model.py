"""Tests for the new fit_model() dispatcher."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from econtools.fit import fit_model
from econtools.model.spec import ModelSpec


@pytest.fixture
def simple_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_fit_model_ols_returns_estimate(simple_df: pd.DataFrame) -> None:
    from econtools._core.types import Estimate

    spec = ModelSpec(dep_var="y", exog_vars=["x1", "x2"], estimator="ols")
    result = fit_model(spec, simple_df)

    assert isinstance(result, Estimate)
    assert result.model_type == "OLS"
    assert result.dep_var == "y"
    assert "x1" in result.params.index
    assert "x2" in result.params.index
    assert result.fit.nobs == 50
    assert result.fit.r_squared > 0.8  # strong signal


def test_fit_model_ols_coefficients_correct(simple_df: pd.DataFrame) -> None:
    spec = ModelSpec(dep_var="y", exog_vars=["x1", "x2"], estimator="ols")
    result = fit_model(spec, simple_df)

    # DGP: y = 1 + 2*x1 - 0.5*x2 + noise
    assert abs(float(result.params["x1"]) - 2.0) < 0.3
    assert abs(float(result.params["x2"]) - (-0.5)) < 0.3


def test_fit_model_wls(simple_df: pd.DataFrame) -> None:
    rng = np.random.default_rng(99)
    simple_df = simple_df.copy()
    simple_df["w"] = rng.uniform(0.5, 2.0, size=len(simple_df))

    spec = ModelSpec(
        dep_var="y",
        exog_vars=["x1", "x2"],
        weights_col="w",
        estimator="wls",
    )
    result = fit_model(spec, simple_df)
    assert result.model_type == "WLS"
    assert result.fit.nobs == 50


def test_fit_model_unknown_estimator_raises(simple_df: pd.DataFrame) -> None:
    spec = ModelSpec(dep_var="y", exog_vars=["x1"], estimator="garch")
    with pytest.raises(ValueError, match="Unknown estimator"):
        fit_model(spec, simple_df)


def test_fit_model_hc3_se(simple_df: pd.DataFrame) -> None:
    spec = ModelSpec(dep_var="y", exog_vars=["x1", "x2"], cov_type="HC3")
    result = fit_model(spec, simple_df)
    assert result.cov_type == "HC3"
    assert all(result.bse > 0)
