"""SE-type resolution helpers.

Maps friendly cov_type labels to statsmodels keyword arguments.

Public API
----------
VALID_COV_TYPES
resolve_cov_args(cov_type, maxlags=None, groups=None) -> dict
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
    maxlags: int | None = None,
    groups: Any = None,
) -> dict[str, Any]:
    """Map a friendly cov_type label to statsmodels .fit() kwargs.

    Parameters
    ----------
    cov_type:
        One of VALID_COV_TYPES.
    maxlags:
        Maximum lags for HAC/Newey-West SEs.
    groups:
        Cluster variable array for clustered SEs.

    Returns
    -------
    dict with keys ``cov_type`` and optionally ``cov_kwds``.

    Raises
    ------
    ValueError
        If ``cov_type`` is not in VALID_COV_TYPES or required args are missing.
    """
    if cov_type not in VALID_COV_TYPES:
        raise ValueError(
            f"Unknown cov_type '{cov_type}'. "
            f"Choose from: {VALID_COV_TYPES}."
        )

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
        raise ValueError(
            "cov_type='cluster' requires the ``groups`` argument."
        )
    return {"cov_type": "cluster", "cov_kwds": {"groups": groups}}
