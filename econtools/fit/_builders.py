"""Consolidated result builders for all backends.

Four thin functions replace the four duplicate ``_build_result`` copies
scattered across ``models/ols.py``, ``models/iv.py``, ``models/panel.py``,
and ``models/probit.py``.

Internal API — not part of the public surface.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from econtools._core.types import Estimate, FitMetrics


def build_sm_result(
    sm_result: object,
    model_type: str,
    dep_var: str,
    cov_type_label: str,
    alpha: float,
    *,
    formula: str | None = None,
) -> Estimate:
    """Build an :class:`Estimate` from a fitted statsmodels OLS/WLS result."""
    ci = sm_result.conf_int(alpha=alpha)

    fvalue = sm_result.fvalue
    f_pvalue = sm_result.f_pvalue
    f_stat = float(np.squeeze(fvalue)) if fvalue is not None else float("nan")
    f_pval = float(np.squeeze(f_pvalue)) if f_pvalue is not None else float("nan")

    fit = FitMetrics(
        nobs=int(sm_result.nobs),
        df_model=float(sm_result.df_model),
        df_resid=float(sm_result.df_resid),
        r_squared=float(sm_result.rsquared),
        r_squared_adj=float(sm_result.rsquared_adj),
        aic=float(sm_result.aic),
        bic=float(sm_result.bic),
        log_likelihood=float(sm_result.llf),
        f_stat=f_stat,
        f_pvalue=f_pval,
        rmse=float(np.sqrt(sm_result.mse_resid)),
        ssr=float(sm_result.ssr),
        ess=float(sm_result.ess),
        tss=float(sm_result.centered_tss),
    )

    return Estimate(
        model_type=model_type,
        dep_var=dep_var,
        params=sm_result.params,
        bse=sm_result.bse,
        tvalues=sm_result.tvalues,
        pvalues=sm_result.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=sm_result.resid,
        fitted=sm_result.fittedvalues,
        cov_params=sm_result.cov_params(),
        cov_type=cov_type_label,
        fit=fit,
        raw=sm_result,
        formula=formula,
    )


def build_lm_iv_result(
    sm_result: object,
    model_type: str,
    dep_var: str,
    cov_type_label: str,
    alpha: float,
    y: pd.Series,
) -> Estimate:
    """Build an :class:`Estimate` from a fitted linearmodels IV result."""
    try:
        ci = sm_result.conf_int(level=1 - alpha)
    except Exception:
        ci = sm_result.conf_int()

    f_stat = float("nan")
    f_pval = float("nan")
    f_obj = getattr(sm_result, "f_statistic", None)
    if f_obj is not None:
        f_stat = float(getattr(f_obj, "stat", f_stat))
        f_pval = float(getattr(f_obj, "pval", f_pval))

    resid = sm_result.resids
    ssr = float(np.sum(np.asarray(resid) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    ess = float(tss - ssr)
    rmse = math.sqrt(ssr / len(y))

    fit = FitMetrics(
        nobs=int(sm_result.nobs),
        df_model=float(sm_result.df_model),
        df_resid=float(sm_result.df_resid),
        r_squared=float(sm_result.rsquared),
        r_squared_adj=float(sm_result.rsquared_adj),
        aic=float(getattr(sm_result, "aic", float("nan"))),
        bic=float(getattr(sm_result, "bic", float("nan"))),
        log_likelihood=float(getattr(sm_result, "loglik", float("nan"))),
        f_stat=f_stat,
        f_pvalue=f_pval,
        rmse=rmse,
        ssr=ssr,
        ess=ess,
        tss=tss,
    )

    return Estimate(
        model_type=model_type,
        dep_var=dep_var,
        params=sm_result.params,
        bse=sm_result.std_errors,
        tvalues=sm_result.tstats,
        pvalues=sm_result.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=sm_result.resids,
        fitted=sm_result.fitted_values,
        cov_params=sm_result.cov,
        cov_type=cov_type_label,
        fit=fit,
        raw=sm_result,
        formula=None,
    )


def build_lm_panel_result(
    res: object,
    model_type: str,
    dep_var: str,
    cov_type_label: str,
    alpha: float,
) -> Estimate:
    """Build an :class:`Estimate` from a fitted linearmodels panel result."""
    ci = res.conf_int(level=1 - alpha)

    f_stat = float("nan")
    f_pval = float("nan")
    f_obj = getattr(res, "f_statistic", None)
    if f_obj is not None:
        f_stat = float(getattr(f_obj, "stat", f_stat))
        f_pval = float(getattr(f_obj, "pval", f_pval))

    resid = res.resids
    ssr = float(np.sum(np.asarray(resid) ** 2))
    y = np.asarray(res.model.dependent.values2d).squeeze()
    tss = float(np.sum((y - y.mean()) ** 2))
    ess = float(tss - ssr)
    rmse = math.sqrt(ssr / len(y))

    fit = FitMetrics(
        nobs=int(res.nobs),
        df_model=float(res.df_model),
        df_resid=float(res.df_resid),
        r_squared=float(res.rsquared),
        f_stat=f_stat,
        f_pvalue=f_pval,
        rmse=rmse,
        ssr=ssr,
        ess=ess,
        tss=tss,
    )

    return Estimate(
        model_type=model_type,
        dep_var=dep_var,
        params=res.params,
        bse=res.std_errors,
        tvalues=res.tstats,
        pvalues=res.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=res.resids,
        fitted=res.fitted_values,
        cov_params=res.cov,
        cov_type=cov_type_label,
        fit=fit,
        raw=res,
        formula=None,
    )


def build_binary_result(
    sm_result: object,
    dep_var: str,
    cov_type_label: str,
    alpha: float,
) -> Estimate:
    """Build an :class:`Estimate` from a fitted statsmodels Probit/Logit result."""
    ci = sm_result.conf_int(alpha=alpha)

    resid = getattr(sm_result, "resid_response", None)
    if resid is None:
        resid = getattr(sm_result, "resid", None)
    if resid is None:
        resid = pd.Series(dtype=float)

    fitted = sm_result.predict()

    fit = FitMetrics(
        nobs=int(sm_result.nobs),
        df_model=float(sm_result.df_model),
        df_resid=float(sm_result.df_resid),
        r_squared=float(getattr(sm_result, "prsquared", math.nan)),
        aic=float(sm_result.aic),
        bic=float(sm_result.bic),
        log_likelihood=float(sm_result.llf),
        pseudo_r_squared=float(getattr(sm_result, "prsquared", math.nan)),
    )

    # model_type is inferred from the class name
    class_name = type(sm_result.model).__name__.lower()
    model_type = "Logit (LDP)" if "logit" in class_name else "Probit (LDP)"

    return Estimate(
        model_type=model_type,
        dep_var=dep_var,
        params=sm_result.params,
        bse=sm_result.bse,
        tvalues=sm_result.tvalues,
        pvalues=sm_result.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=resid,
        fitted=fitted,
        cov_params=sm_result.cov_params(),
        cov_type=cov_type_label,
        fit=fit,
        raw=sm_result,
        formula=None,
    )
