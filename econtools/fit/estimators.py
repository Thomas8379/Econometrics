"""Central dispatch for econtools estimation.

The single public entry point is :func:`fit_model`, which takes a fully
specified :class:`~econtools.model.spec.ModelSpec` and a
:class:`pandas.DataFrame` and returns an :class:`~econtools._core.types.Estimate`.

Public API
----------
fit_model(spec, df) -> Estimate
"""

from __future__ import annotations

import pandas as pd

from econtools._core.types import Estimate
from econtools.model.spec import ModelSpec


def fit_model(spec: ModelSpec, df: pd.DataFrame) -> Estimate:
    """Fit a model defined by *spec* on *df* and return an :class:`Estimate`.

    Parameters
    ----------
    spec:
        Fully specified :class:`~econtools.model.spec.ModelSpec`.
    df:
        Input data.

    Returns
    -------
    Estimate
        Normalised result object; access ``.raw`` for library-specific
        attributes.

    Raises
    ------
    ValueError
        If ``spec.estimator`` is not supported.
    """
    estimator = spec.estimator.lower()

    if estimator in ("ols", "wls"):
        from econtools.fit._sm_adapter import fit_ols_from_spec
        return fit_ols_from_spec(spec, df)

    if estimator in ("2sls", "iv2sls", "liml"):
        from econtools.fit._lm_adapter import fit_iv_from_spec
        return fit_iv_from_spec(spec, df)

    if estimator in ("fe", "re", "fd", "first_difference", "pooled"):
        from econtools.fit._lm_adapter import fit_panel_from_spec
        return fit_panel_from_spec(spec, df)

    if estimator == "probit":
        from econtools.fit._sm_adapter import fit_probit_from_spec
        return fit_probit_from_spec(spec, df)

    if estimator == "logit":
        # TODO(econtools): adapter — add Logit support to _sm_adapter.py
        raise NotImplementedError(
            "Logit is Phase 3.  Use estimator='probit' for now."
        )

    raise ValueError(
        f"Unknown estimator '{spec.estimator}'. "
        "Choose from: 'ols', 'wls', '2sls', 'fe', 're', 'fd', 'probit'."
    )
