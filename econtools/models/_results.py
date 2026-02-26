"""Frozen dataclasses for regression results.

Public API
----------
FitMetrics
RegressionResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FitMetrics:
    """Goodness-of-fit statistics for a fitted regression model."""

    r_squared: float
    r_squared_adj: float
    aic: float
    bic: float
    log_likelihood: float
    f_stat: float
    f_pvalue: float
    rmse: float
    ssr: float
    ess: float
    tss: float
    nobs: int
    df_model: float
    df_resid: float


@dataclass(frozen=True)
class RegressionResult:
    """Normalised result object for OLS, WLS, and future estimators.

    The raw statsmodels (or linearmodels) result is exposed via ``.raw``
    for escape-hatch access to library-specific attributes.
    """

    model_type: str
    dep_var: str
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    conf_int_lower: pd.Series
    conf_int_upper: pd.Series
    resid: pd.Series
    fitted: pd.Series
    cov_params: pd.DataFrame
    cov_type: str
    fit: FitMetrics
    raw: Any
    formula: str | None = None
