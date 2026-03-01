"""Linearmodels backend for the fit/ layer.

All functions take a :class:`~econtools.model.spec.ModelSpec` and a
:class:`pandas.DataFrame`, call the appropriate linearmodels estimator,
and return an :class:`~econtools._core.types.Estimate`.

Internal API.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from linearmodels.panel import FirstDifferenceOLS

from econtools._core.cov_mapping import resolve_cov_args
from econtools._core.types import Estimate
from econtools.fit._builders import build_lm_iv_result, build_lm_panel_result
from econtools.model.spec import ModelSpec


def fit_iv_from_spec(spec: ModelSpec, df: pd.DataFrame) -> Estimate:
    """Fit IV-2SLS from a :class:`ModelSpec`.

    TODO(econtools): adapter — add LIML support (spec.estimator == 'liml')
    TODO(econtools): adapter — add GMM-IV support (Phase 4)
    """
    cols = (
        [spec.dep_var]
        + list(spec.exog_vars)
        + list(spec.endog_vars)
        + list(spec.instruments)
    )
    data = df[cols].dropna()

    y: pd.Series = data[spec.dep_var]
    exog: pd.DataFrame = data[list(spec.exog_vars)]
    endog: pd.DataFrame = data[list(spec.endog_vars)]
    instr: pd.DataFrame = data[list(spec.instruments)]

    if spec.add_constant:
        exog = sm.add_constant(exog)

    cov_args = resolve_cov_args(
        spec.cov_type,
        backend="lm",
        maxlags=spec.cov_kwargs.get("maxlags"),
        groups=spec.cov_kwargs.get("groups"),
    )
    sm_result = IV2SLS(y, exog, endog, instr).fit(**cov_args)

    return build_lm_iv_result(sm_result, "2SLS", spec.dep_var, spec.cov_type, spec.alpha, y)


def fit_panel_from_spec(spec: ModelSpec, df: pd.DataFrame) -> Estimate:
    """Fit a panel model from a :class:`ModelSpec`.

    Currently supports ``estimator='fd'`` (FirstDifference).

    TODO(econtools): adapter — add FE (BetweenOLS + within transform) support
    TODO(econtools): adapter — add RE (RandomEffects) support
    TODO(econtools): adapter — add PooledOLS support
    """
    if spec.entity_col is None or spec.time_col is None:
        raise ValueError(
            "Panel models require ModelSpec.entity_col and ModelSpec.time_col."
        )

    cols = (
        [spec.dep_var]
        + list(spec.exog_vars)
        + [spec.entity_col, spec.time_col]
    )
    data = df[cols].dropna()
    panel = data.set_index([spec.entity_col, spec.time_col]).sort_index()

    y: pd.Series = panel[spec.dep_var]
    X: pd.DataFrame = panel[list(spec.exog_vars)]
    if spec.add_constant:
        X = X.assign(const=1.0)

    cov_type_lm = resolve_cov_args(spec.cov_type, backend="lm").get(
        "cov_type", "unadjusted"
    )

    if spec.estimator in ("fd", "first_difference"):
        res = FirstDifferenceOLS(y, X).fit(cov_type=cov_type_lm)
        model_type = "FirstDifference"
    else:
        raise ValueError(
            f"Panel estimator '{spec.estimator}' is not yet implemented. "
            "Available: 'fd'. FE/RE/Pooled require Phase 2+ completion."
        )

    return build_lm_panel_result(res, model_type, spec.dep_var, spec.cov_type, spec.alpha)
