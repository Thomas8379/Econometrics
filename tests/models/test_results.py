"""Tests for econtools.models._results dataclasses."""

from __future__ import annotations

import pytest
import pandas as pd

from econtools.models._results import FitMetrics, RegressionResult


def _make_fit_metrics() -> FitMetrics:
    return FitMetrics(
        r_squared=0.9,
        r_squared_adj=0.89,
        aic=-100.0,
        bic=-90.0,
        log_likelihood=-50.0,
        f_stat=200.0,
        f_pvalue=0.0,
        rmse=0.5,
        ssr=10.0,
        ess=90.0,
        tss=100.0,
        nobs=100,
        df_model=2.0,
        df_resid=97.0,
    )


def _make_series(vals: list[float], index: list[str]) -> pd.Series:
    return pd.Series(vals, index=index)


def test_fit_metrics_instantiation() -> None:
    fm = _make_fit_metrics()
    assert fm.r_squared == 0.9
    assert fm.nobs == 100


def test_fit_metrics_frozen() -> None:
    fm = _make_fit_metrics()
    with pytest.raises((AttributeError, TypeError)):
        fm.r_squared = 0.5  # type: ignore[misc]


def test_regression_result_instantiation(ols_result) -> None:
    from econtools.models._results import RegressionResult
    assert isinstance(ols_result, RegressionResult)


def test_regression_result_frozen(ols_result) -> None:
    with pytest.raises((AttributeError, TypeError)):
        ols_result.model_type = "WLS"  # type: ignore[misc]


def test_regression_result_field_types(ols_result) -> None:
    assert isinstance(ols_result.params, pd.Series)
    assert isinstance(ols_result.bse, pd.Series)
    assert isinstance(ols_result.resid, pd.Series)
    assert isinstance(ols_result.fitted, pd.Series)
    assert isinstance(ols_result.cov_params, pd.DataFrame)
    assert isinstance(ols_result.fit, FitMetrics)
