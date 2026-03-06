"""Scoring functions for sieve candidates.

Scorers take a :class:`~econtools.sieve.fitters.FitResult` plus (optionally)
held-out evaluation data and return a ``dict[str, float]`` of named metrics.

Constraint violation is represented as ``math.inf`` for metrics where lower is
better, or ``-math.inf`` where higher is better.

Public API
----------
score_ols(fit, eval_data) -> dict
score_iv(fit, eval_data) -> dict
HIGHER_IS_BETTER      – set of metric names where larger values are preferred
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from econtools.sieve.fitters import FitResult


# ---------------------------------------------------------------------------
# Metric direction
# ---------------------------------------------------------------------------

HIGHER_IS_BETTER: frozenset[str] = frozenset({
    "r_squared",
    "first_stage_f",
    "overid_pvalue",   # not used for selection but higher = better fit
    "log_likelihood",
})

LOWER_IS_BETTER: frozenset[str] = frozenset({
    "mse",
    "rmse",
    "mae",
    "aic",
    "bic",
    "n_terms",
})


# ---------------------------------------------------------------------------
# OLS scorer
# ---------------------------------------------------------------------------


def _predict_ols(fit: FitResult, df: pd.DataFrame) -> np.ndarray:
    """Predict using OLS coefficients (intercept handled automatically)."""
    X_cols = [c for c in fit.params.index if c in df.columns]
    if "const" not in X_cols and "const" in fit.params.index:
        # const not in df — add it
        X = df[X_cols].values
        coefs = fit.params[X_cols].values
        return X @ coefs + float(fit.params.get("const", 0.0))
    elif all(c in df.columns for c in fit.params.index):
        X = df[fit.params.index].values
        return X @ fit.params.values
    else:
        X = df[X_cols].values
        coefs = fit.params[X_cols].values
        return X @ coefs


def score_ols(
    fit: FitResult,
    eval_data: pd.DataFrame | None = None,
    *,
    include_aic_bic: bool = True,
) -> dict[str, float]:
    """Compute OLS scoring metrics.

    Parameters
    ----------
    fit:
        Result from :func:`~econtools.sieve.fitters.fit_candidate`.
    eval_data:
        Held-out evaluation data.  If ``None``, in-sample metrics are used
        (labeled ``'in_sample'`` in warnings).
    include_aic_bic:
        Whether to include AIC/BIC (in-sample information criteria).

    Returns
    -------
    dict of metric_name -> float value
    """
    scores: dict[str, float] = {}

    if fit.n_obs == 0 or len(fit.params) == 0:
        return {
            "mse": math.inf,
            "rmse": math.inf,
            "r_squared": -math.inf,
            "n_terms": float(fit.candidate.n_terms),
        }

    if eval_data is not None and len(eval_data) > 0:
        y_eval = eval_data[fit.candidate.y].values
        try:
            y_pred = _predict_ols(fit, eval_data)
            resid = y_eval - y_pred
            mse = float(np.mean(resid ** 2))
            scores["mse"] = mse
            scores["rmse"] = math.sqrt(mse)
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y_eval - y_eval.mean()) ** 2))
            scores["r_squared_eval"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            scores["mae"] = float(np.mean(np.abs(resid)))
        except Exception:
            scores["mse"] = math.inf
            scores["rmse"] = math.inf
    else:
        # In-sample
        resid = fit.resid.values
        mse = float(np.mean(resid ** 2))
        scores["mse"] = mse
        scores["rmse"] = math.sqrt(mse)
        scores["r_squared"] = float(fit.r_squared) if not math.isnan(fit.r_squared) else float("nan")

    if include_aic_bic:
        scores["aic"] = float(fit.aic) if not math.isnan(fit.aic) else float("nan")
        scores["bic"] = float(fit.bic) if not math.isnan(fit.bic) else float("nan")

    scores["n_terms"] = float(fit.candidate.n_terms)
    scores["n_obs"] = float(fit.n_obs)

    return scores


# ---------------------------------------------------------------------------
# IV scorer
# ---------------------------------------------------------------------------


def score_iv(
    fit: FitResult,
    eval_data: pd.DataFrame | None = None,
    *,
    min_first_stage_f: float = 10.0,
    max_instruments: int | None = None,
) -> dict[str, float]:
    """Compute IV scoring metrics.

    Parameters
    ----------
    fit:
        Result from :func:`~econtools.sieve.fitters.fit_candidate`.
    eval_data:
        Held-out evaluation data for out-of-sample scoring.
    min_first_stage_f:
        Minimum acceptable first-stage F statistic (hard constraint).
        If violated, ``first_stage_f`` is returned as-is but the caller
        (selection layer) should reject the candidate.
    max_instruments:
        Maximum number of instruments (hard constraint; None = no limit).

    Returns
    -------
    dict of metric_name -> float
    """
    scores: dict[str, float] = {}

    if fit.n_obs == 0 or len(fit.params) == 0:
        return {
            "first_stage_f": -math.inf,
            "mse": math.inf,
            "rmse": math.inf,
            "overid_pvalue": float("nan"),
            "n_terms": float(fit.candidate.n_terms),
            "n_instruments": float(len(fit.candidate.Z_terms or [])),
        }

    scores["first_stage_f"] = float(fit.first_stage_f)
    scores["overid_pvalue"] = float(fit.overid_pvalue)
    scores["n_instruments"] = float(len(fit.candidate.Z_terms or []))
    scores["n_terms"] = float(fit.candidate.n_terms)
    scores["n_obs"] = float(fit.n_obs)

    # Predictive score (2SLS residuals on eval set)
    if eval_data is not None and len(eval_data) > 0:
        try:
            y_eval = eval_data[fit.candidate.y].values
            y_pred = _predict_ols(fit, eval_data)
            resid = y_eval - y_pred
            mse = float(np.mean(resid ** 2))
            scores["mse"] = mse
            scores["rmse"] = math.sqrt(mse)
        except Exception:
            scores["mse"] = math.inf
            scores["rmse"] = math.inf
    else:
        resid = fit.resid.values
        mse = float(np.mean(resid ** 2))
        scores["mse"] = mse
        scores["rmse"] = math.sqrt(mse)

    return scores


# ---------------------------------------------------------------------------
# Stability scorer (cross-fold coefficient variance)
# ---------------------------------------------------------------------------


def stability_score(
    params_per_fold: list[pd.Series],
    primary_var: str | None = None,
) -> dict[str, float]:
    """Compute coefficient stability across folds.

    Parameters
    ----------
    params_per_fold:
        List of coefficient Series, one per fold.
    primary_var:
        If provided, also report the sign-stability and IQR for this variable.

    Returns
    -------
    dict with ``"coef_cv_mean"`` (mean CV across all params) and optionally
    ``"primary_sign_stable"`` (fraction of folds with the same sign).
    """
    if not params_per_fold:
        return {}

    common_vars = set(params_per_fold[0].index)
    for p in params_per_fold[1:]:
        common_vars &= set(p.index)

    scores: dict[str, float] = {}
    cvs: list[float] = []

    for var in common_vars:
        vals = np.array([float(p[var]) for p in params_per_fold])
        std = float(np.std(vals))
        mean = float(np.mean(vals))
        cv = std / abs(mean) if abs(mean) > 1e-12 else float("nan")
        if not math.isnan(cv):
            cvs.append(cv)

    scores["coef_cv_mean"] = float(np.mean(cvs)) if cvs else float("nan")

    if primary_var and primary_var in common_vars:
        vals = np.array([float(p[primary_var]) for p in params_per_fold])
        signs = np.sign(vals)
        scores["primary_sign_stable"] = float(np.mean(signs == signs[0]))
        scores["primary_coef_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))

    return scores
