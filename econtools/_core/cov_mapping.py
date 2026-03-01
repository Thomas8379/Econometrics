"""Unified covariance-type resolver for both statsmodels and linearmodels.

Single source of truth — supersedes ``inference.se_types.resolve_cov_args``
and the private ``_resolve_iv_cov_args`` in ``models.iv``.

Public API
----------
VALID_COV_TYPES
resolve_cov_args(cov_type, *, backend, maxlags, groups) -> dict
"""

from __future__ import annotations

from typing import Any


VALID_COV_TYPES: tuple[str, ...] = (
    "classical",
    "HC0",
    "HC1",
    "HC2",
    "HC3",
    "HAC",
    "newey_west",
    "cluster",
)


def resolve_cov_args(
    cov_type: str,
    *,
    backend: str = "sm",
    maxlags: int | None = None,
    groups: Any = None,
) -> dict[str, Any]:
    """Map a friendly *cov_type* label to backend ``.fit()`` kwargs.

    Parameters
    ----------
    cov_type:
        One of :data:`VALID_COV_TYPES`.
    backend:
        ``'sm'`` (statsmodels) or ``'lm'`` (linearmodels).
    maxlags:
        Maximum lags for HAC/Newey-West SEs.
    groups:
        Cluster variable array for clustered SEs.

    Returns
    -------
    dict
        Keyword arguments suitable for passing to the backend's
        ``.fit()`` method.

    Raises
    ------
    ValueError
        If *cov_type* is unknown or required args are missing.
    """
    if cov_type not in VALID_COV_TYPES:
        raise ValueError(
            f"Unknown cov_type {cov_type!r}. Choose from: {VALID_COV_TYPES}."
        )

    if backend == "sm":
        return _resolve_sm(cov_type, maxlags=maxlags, groups=groups)
    if backend == "lm":
        return _resolve_lm(cov_type, maxlags=maxlags, groups=groups)
    raise ValueError(f"Unknown backend {backend!r}. Choose 'sm' or 'lm'.")


# ---------------------------------------------------------------------------
# Private per-backend resolvers
# ---------------------------------------------------------------------------


def _resolve_sm(
    cov_type: str,
    maxlags: int | None,
    groups: Any,
) -> dict[str, Any]:
    """Resolve cov_type for statsmodels."""
    if cov_type == "classical":
        return {"cov_type": "nonrobust"}

    if cov_type in ("HC0", "HC1", "HC2", "HC3"):
        return {"cov_type": cov_type}

    if cov_type in ("HAC", "newey_west"):
        kwds: dict[str, Any] = {}
        if maxlags is not None:
            kwds["maxlags"] = maxlags
        result: dict[str, Any] = {"cov_type": "HAC"}
        if kwds:
            result["cov_kwds"] = kwds
        return result

    # cluster
    if groups is None:
        raise ValueError("cov_type='cluster' requires the ``groups`` argument.")
    return {"cov_type": "cluster", "cov_kwds": {"groups": groups}}


def _resolve_lm(
    cov_type: str,
    maxlags: int | None,
    groups: Any,
) -> dict[str, Any]:
    """Resolve cov_type for linearmodels."""
    if cov_type == "classical":
        return {"cov_type": "unadjusted"}

    if cov_type in ("HC0", "HC1", "HC2", "HC3"):
        return {"cov_type": "robust"}

    if cov_type in ("HAC", "newey_west"):
        cov: dict[str, Any] = {"cov_type": "kernel", "kernel": "bartlett"}
        if maxlags is not None:
            cov["bandwidth"] = maxlags
        return cov

    # cluster
    if groups is None:
        raise ValueError("cov_type='cluster' requires the ``groups`` argument.")
    return {"cov_type": "clustered", "clusters": groups}
