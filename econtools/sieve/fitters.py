"""Fitting adapters for sieve candidates.

Provides a unified interface: ``fit_candidate(candidate, data) -> FitResult``.
Internally delegates to :func:`~econtools.fit.estimators.fit_model` via
:class:`~econtools.model.spec.ModelSpec`.

Public API
----------
FitResult
fit_candidate(candidate, data, *, drop_na) -> FitResult | None
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from econtools.sieve.candidates import Candidate


@dataclass
class FitResult:
    """Result of fitting a single :class:`~econtools.sieve.candidates.Candidate`.

    Parameters
    ----------
    candidate:
        The candidate that was fitted.
    params:
        Estimated coefficients (Series indexed by variable name).
    bse:
        Standard errors.
    pvalues:
        p-values.
    resid:
        In-sample residuals.
    fitted:
        In-sample fitted values.
    n_obs:
        Number of observations used in fitting.
    r_squared:
        In-sample R² (NaN for IV/panel).
    aic:
        AIC (NaN where not applicable).
    bic:
        BIC (NaN where not applicable).
    first_stage_f:
        First-stage partial F statistic (IV only; NaN otherwise).
    overid_pvalue:
        Sargan overidentification test p-value (NaN if exactly identified).
    warnings:
        List of warning messages (singularity, dropped columns, etc.).
    raw:
        Underlying library result object (for escape-hatch access).
    """

    candidate: Candidate
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    resid: pd.Series
    fitted: pd.Series
    n_obs: int
    r_squared: float = float("nan")
    aic: float = float("nan")
    bic: float = float("nan")
    first_stage_f: float = float("nan")
    overid_pvalue: float = float("nan")
    warnings: list[str] = field(default_factory=list)
    raw: Any = None


def _extract_first_stage_f(raw_result: Any) -> float:
    """Pull first-stage F from a linearmodels IV result object."""
    for attr in ("first_stage", "_first_stage"):
        fs = getattr(raw_result, attr, None)
        if fs is None:
            continue
        try:
            diag = getattr(fs, "diagnostics", None)
            if diag is not None:
                return float(diag.loc["f.stat", "stat"])
        except Exception:
            pass
    return float("nan")


def _extract_overid_pvalue(raw_result: Any) -> float:
    """Pull Sargan overidentification p-value from a linearmodels IV result."""
    for method in ("sargan", "basmann"):
        obj = getattr(raw_result, method, None)
        if obj is None:
            continue
        try:
            return float(getattr(obj, "pval", float("nan")))
        except Exception:
            pass
    return float("nan")


def fit_candidate(
    candidate: Candidate,
    data: pd.DataFrame,
    *,
    drop_na: bool = True,
) -> FitResult | None:
    """Fit *candidate* on *data*.

    Parameters
    ----------
    candidate:
        The model specification to fit.
    data:
        DataFrame containing all required columns.
    drop_na:
        Drop rows with any NaN in y, X, or Z before fitting.

    Returns
    -------
    :class:`FitResult` or ``None`` if fitting failed (singularity, etc.).
    """
    from econtools.fit.estimators import fit_model
    from econtools.model.spec import ModelSpec
    from econtools._core.cov_mapping import resolve_cov_args

    # Collect required columns
    cols_needed = [candidate.y] + list(candidate.X_terms)
    if candidate.endog:
        cols_needed += list(candidate.endog)
    if candidate.Z_terms:
        cols_needed += list(candidate.Z_terms)

    missing = [c for c in cols_needed if c not in data.columns]
    if missing:
        return FitResult(
            candidate=candidate,
            params=pd.Series(dtype=float),
            bse=pd.Series(dtype=float),
            pvalues=pd.Series(dtype=float),
            resid=pd.Series(dtype=float),
            fitted=pd.Series(dtype=float),
            n_obs=0,
            warnings=[f"Missing columns: {missing}"],
        )

    df = data[cols_needed].copy()
    if drop_na:
        df = df.dropna()

    if len(df) == 0:
        return FitResult(
            candidate=candidate,
            params=pd.Series(dtype=float),
            bse=pd.Series(dtype=float),
            pvalues=pd.Series(dtype=float),
            resid=pd.Series(dtype=float),
            fitted=pd.Series(dtype=float),
            n_obs=0,
            warnings=["No observations after dropping NaN."],
        )

    # Map estimator name to ModelSpec estimator
    estimator = candidate.estimator.lower()
    if estimator == "fe_ols":
        estimator = "fe"
    elif estimator == "fe_2sls":
        estimator = "2sls"  # linearmodels handles FE+IV via spec

    # Build cov_type kwargs
    cov_kwargs: dict = {}
    try:
        backend = "lm" if estimator in ("2sls", "iv2sls", "fe", "re", "fd") else "sm"
        cov_kwargs = resolve_cov_args(candidate.cov_type, backend=backend)
        if candidate.cluster_var and candidate.cluster_var in data.columns:
            cov_kwargs["groups"] = data.loc[df.index, candidate.cluster_var]
    except Exception:
        pass

    spec = ModelSpec(
        dep_var=candidate.y,
        exog_vars=list(candidate.X_terms),
        estimator=estimator,
        endog_vars=list(candidate.endog) if candidate.endog else None,
        instrument_vars=list(candidate.Z_terms) if candidate.Z_terms else None,
        add_const=candidate.intercept,
        cov_type=candidate.cov_type,
    )

    fit_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est = fit_model(spec, df)
        for w in caught:
            fit_warnings.append(str(w.message))
    except Exception as exc:
        return FitResult(
            candidate=candidate,
            params=pd.Series(dtype=float),
            bse=pd.Series(dtype=float),
            pvalues=pd.Series(dtype=float),
            resid=pd.Series(dtype=float),
            fitted=pd.Series(dtype=float),
            n_obs=len(df),
            warnings=[f"Fitting failed: {exc}"],
        )

    first_stage_f = _extract_first_stage_f(est.raw)
    overid_pval = _extract_overid_pvalue(est.raw)

    return FitResult(
        candidate=candidate,
        params=est.params,
        bse=est.bse,
        pvalues=est.pvalues,
        resid=est.resid,
        fitted=est.fitted,
        n_obs=int(est.fit.nobs),
        r_squared=float(est.fit.r_squared),
        aic=float(est.fit.aic),
        bic=float(est.fit.bic),
        first_stage_f=first_stage_f,
        overid_pvalue=overid_pval,
        warnings=fit_warnings,
        raw=est.raw,
    )
