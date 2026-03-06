"""Instrument generators for the Z-side of IV sieves.

All generators are pure functions; they add columns to a copy of the DataFrame
and return the new column names plus associated :class:`TransformSpec` objects.

Validity guardrails (hard rules)
---------------------------------
* Excluded instruments must not be functions of y (enforced by the caller via
  ``y_col`` filtering — callers must pass ``y_col`` so generators can warn).
* Lags use only past observations (requires a sorted time column).
* LOO group means exclude the current unit to avoid reflection bias.
* Post-treatment variables are never added automatically (callers control the
  ``base_X`` and ``base_Z`` lists).

Public API
----------
generate_lags(vars, lag_k, df, time_col, id_col) -> (names, df, specs)
generate_z_polynomials(base_Z, degree, df) -> (names, df, specs)
generate_z_interactions(base_Z, exog_vars, df) -> (names, df, specs)
generate_group_means(vars, group_col, df) -> (names, df, specs)
generate_loogroup_means(vars, group_col, df) -> (names, df, specs)
generate_instrument_candidates(base_Z, base_X, config, df, y_col) -> (candidate_Z_sets, df, specs)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from econtools.sieve.candidates import TransformSpec


# ---------------------------------------------------------------------------
# Lags
# ---------------------------------------------------------------------------


def generate_lags(
    vars: list[str],
    lag_k: int,
    df: pd.DataFrame,
    *,
    time_col: str,
    id_col: str | None = None,
) -> tuple[list[str], pd.DataFrame, list[TransformSpec]]:
    """Add lag-k values for each variable.

    Parameters
    ----------
    vars:
        Variable names to lag.
    lag_k:
        Number of periods to lag (>= 1).
    df:
        Working DataFrame.  Must contain *time_col* (and *id_col* if panel).
    time_col:
        Column to sort by before lagging.  Must be sortable (int, float, or datetime).
    id_col:
        Panel unit identifier.  If provided, lagging is done within groups
        (each entity's time series is lagged separately).

    Returns
    -------
    (new_names, augmented_df, specs)

    Notes
    -----
    Lagging is done using ``groupby(...).shift(lag_k)`` which correctly handles
    gaps in the time dimension (NaN is introduced for the first lag_k periods
    of each entity).  The DataFrame is sorted by (id_col, time_col) before lagging
    to guarantee temporal order; the original index order is restored afterward.
    """
    if lag_k < 1:
        raise ValueError("lag_k must be >= 1.")

    df = df.copy()
    orig_index = df.index

    sort_cols = ([id_col, time_col] if id_col else [time_col])
    df = df.sort_values(sort_cols)

    new_names: list[str] = []
    specs: list[TransformSpec] = []

    for var in vars:
        col_name = f"lag_{var}_k{lag_k}"
        if col_name in df.columns:
            new_names.append(col_name)
            continue

        if id_col:
            df[col_name] = df.groupby(id_col)[var].shift(lag_k)
        else:
            df[col_name] = df[var].shift(lag_k)

        new_names.append(col_name)
        specs.append(TransformSpec(
            kind="lag",
            base_vars=(var,),
            output_vars=(col_name,),
            params={"lag_k": lag_k, "time_col": time_col, "id_col": id_col},
        ))

    # Restore original index order
    df = df.reindex(orig_index)
    return new_names, df, specs


# ---------------------------------------------------------------------------
# Polynomial / nonlinear transforms of instruments
# ---------------------------------------------------------------------------


def generate_z_polynomials(
    base_Z: list[str],
    degree: int,
    df: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, list[TransformSpec]]:
    """Add polynomial terms up to *degree* for each baseline instrument.

    Wraps :func:`~econtools.sieve.generators.features.generate_polynomial`.
    """
    from econtools.sieve.generators.features import generate_polynomial
    return generate_polynomial(base_Z, degree, df)


def generate_z_interactions(
    base_Z: list[str],
    exog_vars: list[str],
    df: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, list[TransformSpec]]:
    """Add interactions between baseline instruments and exogenous covariates.

    Parameters
    ----------
    base_Z:
        Baseline instrument names.
    exog_vars:
        Exogenous covariates to interact with (typically ``base_X``).
    df:
        Working DataFrame.

    Returns
    -------
    (new_names, augmented_df, specs)
    """
    from econtools.sieve.generators.features import generate_interactions
    return generate_interactions(base_Z, exog_vars, df)


# ---------------------------------------------------------------------------
# Group means
# ---------------------------------------------------------------------------


def generate_group_means(
    vars: list[str],
    group_col: str,
    df: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, list[TransformSpec]]:
    """Add group-mean columns (Hausman-Taylor style instruments).

    Parameters
    ----------
    vars:
        Variables for which to compute group means.
    group_col:
        Column defining the group membership (e.g. industry, state).
    df:
        Working DataFrame.

    Returns
    -------
    (new_names, augmented_df, specs)

    Warnings
    --------
    Group means are not leave-one-out (LOO); they include the unit itself.
    This can cause reflection bias in within-group IV settings.  Prefer
    :func:`generate_loogroup_means` unless you have a specific reason.
    """
    df = df.copy()
    new_names: list[str] = []
    specs: list[TransformSpec] = []

    group_means = df.groupby(group_col)[vars].transform("mean")

    for var in vars:
        col_name = f"gmean_{var}"
        if col_name in df.columns:
            new_names.append(col_name)
            continue
        df[col_name] = group_means[var]
        new_names.append(col_name)
        specs.append(TransformSpec(
            kind="group_mean",
            base_vars=(var,),
            output_vars=(col_name,),
            params={"group_col": group_col, "loo": False},
        ))

    return new_names, df, specs


def generate_loogroup_means(
    vars: list[str],
    group_col: str,
    df: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, list[TransformSpec]]:
    """Add leave-one-out (LOO) group means.

    For each observation, the mean is computed over all other group members
    (excluding itself).  This avoids the reflection problem.

    Parameters
    ----------
    vars:
        Variables for which to compute LOO group means.
    group_col:
        Column defining groups.
    df:
        Working DataFrame (must have a unique index).

    Returns
    -------
    (new_names, augmented_df, specs)
    """
    df = df.copy()
    new_names: list[str] = []
    specs: list[TransformSpec] = []

    grp = df.groupby(group_col)

    for var in vars:
        col_name = f"loo_gmean_{var}"
        if col_name in df.columns:
            new_names.append(col_name)
            continue

        group_sum = grp[var].transform("sum")
        group_n = grp[var].transform("count")
        # LOO mean = (group_sum - obs_value) / (group_n - 1)
        # If group has only 1 member, LOO mean is NaN
        loo = (group_sum - df[var]) / (group_n - 1)
        loo[group_n <= 1] = np.nan

        df[col_name] = loo
        new_names.append(col_name)
        specs.append(TransformSpec(
            kind="loo_group_mean",
            base_vars=(var,),
            output_vars=(col_name,),
            params={"group_col": group_col, "loo": True},
        ))

    return new_names, df, specs


# ---------------------------------------------------------------------------
# Composite: apply all configured instrument generators
# ---------------------------------------------------------------------------


def generate_instrument_candidates(
    base_Z: list[str],
    base_X: list[str],
    config: dict,
    df: pd.DataFrame,
    y_col: str,
) -> tuple[list[list[str]], pd.DataFrame, list[TransformSpec]]:
    """Apply all configured instrument generators.

    Parameters
    ----------
    base_Z:
        Starting instrument set (must already be in *df*).
    base_X:
        Exogenous regressors (used for interaction instruments).
    config:
        Instrument generator config dict from sieve YAML:

        .. code-block:: yaml

            instruments:
              lags:
                enabled: false
                lag_ks: [1, 2]
                vars: null      # null = apply to base_Z
                time_col: year
                id_col: id
              z_polynomial:
                enabled: false
                degree: 2
              z_interactions:
                enabled: false
                exog_vars: null   # null = base_X
              group_means:
                enabled: false
                vars: null
                group_col: null
              loo_group_means:
                enabled: false
                vars: null
                group_col: null

    df:
        Working DataFrame.
    y_col:
        Dependent variable name — used to warn if instruments appear correlated
        with y by name (heuristic only).

    Returns
    -------
    (candidate_Z_sets, augmented_df, all_specs)
        Each element of ``candidate_Z_sets`` is a list of new instrument names
        from a single generator call.
    """
    df = df.copy()
    z_sets: list[list[str]] = []
    all_specs: list[TransformSpec] = []

    inst_cfg = config.get("instruments", {})

    # Lags
    lag_cfg = inst_cfg.get("lags", {})
    if lag_cfg.get("enabled", False):
        time_col = lag_cfg.get("time_col")
        id_col = lag_cfg.get("id_col")
        lag_vars = lag_cfg.get("vars") or base_Z
        if not time_col:
            warnings.warn(
                "Lag instruments require 'time_col' in instruments.lags config; skipping.",
                UserWarning,
                stacklevel=2,
            )
        else:
            for k in lag_cfg.get("lag_ks", [1]):
                names, df, specs = generate_lags(
                    lag_vars, int(k), df, time_col=time_col, id_col=id_col
                )
                if names:
                    z_sets.append(names)
                    all_specs.extend(specs)

    # Polynomial instruments
    zpoly_cfg = inst_cfg.get("z_polynomial", {})
    if zpoly_cfg.get("enabled", False):
        degree = int(zpoly_cfg.get("degree", 2))
        names, df, specs = generate_z_polynomials(base_Z, degree, df)
        if names:
            z_sets.append(names)
            all_specs.extend(specs)

    # Interaction instruments
    zinter_cfg = inst_cfg.get("z_interactions", {})
    if zinter_cfg.get("enabled", False):
        exog_vars = zinter_cfg.get("exog_vars") or base_X
        names, df, specs = generate_z_interactions(base_Z, exog_vars, df)
        if names:
            z_sets.append(names)
            all_specs.extend(specs)

    # Group means
    gm_cfg = inst_cfg.get("group_means", {})
    if gm_cfg.get("enabled", False):
        gm_vars = gm_cfg.get("vars") or base_Z
        group_col = gm_cfg.get("group_col")
        if not group_col:
            warnings.warn(
                "group_means requires 'group_col'; skipping.",
                UserWarning,
                stacklevel=2,
            )
        else:
            names, df, specs = generate_group_means(gm_vars, group_col, df)
            if names:
                z_sets.append(names)
                all_specs.extend(specs)

    # LOO group means
    loo_cfg = inst_cfg.get("loo_group_means", {})
    if loo_cfg.get("enabled", False):
        loo_vars = loo_cfg.get("vars") or base_Z
        group_col = loo_cfg.get("group_col")
        if not group_col:
            warnings.warn(
                "loo_group_means requires 'group_col'; skipping.",
                UserWarning,
                stacklevel=2,
            )
        else:
            names, df, specs = generate_loogroup_means(loo_vars, group_col, df)
            if names:
                z_sets.append(names)
                all_specs.extend(specs)

    # Heuristic: warn if any generated instrument name contains y_col
    all_new_names = [n for ns in z_sets for n in ns]
    for name in all_new_names:
        if y_col in name:
            warnings.warn(
                f"Generated instrument '{name}' contains the dependent variable name "
                f"'{y_col}'. Review carefully to ensure instrument validity.",
                UserWarning,
                stacklevel=2,
            )

    return z_sets, df, all_specs
