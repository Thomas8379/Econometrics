"""Declarative model specification.

A :class:`ModelSpec` fully describes what the economist wants to estimate.
Pass it to :func:`econtools.fit.fit_model` to produce an :class:`Estimate`.

Public API
----------
ModelSpec
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """Declarative specification of an econometric model.

    Parameters
    ----------
    dep_var:
        Name of the dependent variable column.
    exog_vars:
        Names of exogenous regressors (constant added unless
        ``add_constant=False``).
    endog_vars:
        Endogenous regressors (IV models only).
    instruments:
        Excluded instruments (IV models only).
    entity_col:
        Entity identifier column for panel models.
    time_col:
        Time identifier column for panel models.
    weights_col:
        Column of observation weights (WLS only).
    effects:
        Fixed-effects specification: ``'entity'``, ``'time'``, or
        ``'both'``.  ``None`` for pooled/cross-section models.
    estimator:
        One of ``'ols'``, ``'wls'``, ``'2sls'``, ``'liml'``,
        ``'fe'``, ``'re'``, ``'fd'``, ``'pooled'``, ``'probit'``,
        ``'logit'``.
    cov_type:
        Covariance estimator — one of :data:`econtools._core.cov_mapping.VALID_COV_TYPES`.
    cov_kwargs:
        Extra keyword arguments forwarded to the covariance estimator
        (e.g. ``maxlags``, ``groups``).
    add_constant:
        Prepend a constant column to the design matrix.
    alpha:
        Significance level for confidence intervals (default 0.05 → 95% CI).
    """

    dep_var: str
    exog_vars: list[str]
    endog_vars: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    entity_col: str | None = None
    time_col: str | None = None
    weights_col: str | None = None
    effects: str | None = None       # "entity" | "time" | "both"
    estimator: str = "ols"           # "ols" | "wls" | "2sls" | "fe" | "re" | "fd" | "probit"
    cov_type: str = "classical"
    cov_kwargs: dict = field(default_factory=dict)
    add_constant: bool = True
    alpha: float = 0.05
