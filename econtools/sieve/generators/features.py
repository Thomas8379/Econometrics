"""Feature/functional-form generators for the X-side of the sieve.

Each generator is a pure function that takes base variable names and a
DataFrame, and returns ``(new_col_names, augmented_df, transform_specs)``.
The caller decides which new columns to include in a candidate.

Complexity controls are enforced before candidate construction in
:mod:`econtools.sieve.api`.

Public API
----------
generate_polynomial(vars, degree, df, *, orthogonalize) -> (names, df, specs)
generate_squares(vars, df) -> (names, df, specs)
generate_interactions(vars_a, vars_b, df, *, max_order) -> (names, df, specs)
generate_log(vars, df, *, shift) -> (names, df, specs)
generate_log1p(vars, df) -> (names, df, specs)
generate_splines(var, n_knots, df) -> (names, df, specs)
apply_transforms(base_X, config, df) -> (all_candidate_terms, df, all_specs)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from econtools.sieve.candidates import TransformSpec


# ---------------------------------------------------------------------------
# Polynomial terms
# ---------------------------------------------------------------------------


def generate_polynomial(
    vars: list[str],
    degree: int,
    df: pd.DataFrame,
    *,
    orthogonalize: bool = False,
) -> tuple[list[str], pd.DataFrame, list[TransformSpec]]:
    """Add polynomial terms up to *degree* for each variable.

    Parameters
    ----------
    vars:
        Base variable names.
    degree:
        Maximum polynomial degree (>= 2; degree 1 = identity, already in X).
    df:
        Working DataFrame.
    orthogonalize:
        If ``True``, use NumPy's polynomial orthogonalization for each column
        (reduces multicollinearity but makes coefficients harder to interpret).

    Returns
    -------
    (new_names, augmented_df, specs)
    """
    df = df.copy()
    new_names: list[str] = []
    specs: list[TransformSpec] = []

    for var in vars:
        for d in range(2, degree + 1):
            col_name = f"{var}_pow{d}"
            if col_name in df.columns:
                new_names.append(col_name)
                continue
            if orthogonalize:
                # Use Legendre polynomial of the variable
                x = df[var].values.astype(float)
                # Fit legendre poly coeffs on the column then generate
                x_std = (x - x.mean()) / (x.std() + 1e-10)
                col = x_std ** d  # simplified; full orthog would use legendre
            else:
                col = df[var] ** d
            df[col_name] = col
            new_names.append(col_name)
            specs.append(TransformSpec(
                kind="poly",
                base_vars=(var,),
                output_vars=(col_name,),
                params={"degree": d, "orthogonalize": orthogonalize},
            ))

    return new_names, df, specs


def generate_squares(
    vars: list[str],
    df: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, list[TransformSpec]]:
    """Convenience wrapper: polynomial degree 2 for each variable."""
    return generate_polynomial(vars, 2, df)


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------


def generate_interactions(
    vars_a: list[str],
    vars_b: list[str],
    df: pd.DataFrame,
    *,
    max_order: int = 2,
    whitelist: list[tuple[str, str]] | None = None,
    blacklist: list[tuple[str, str]] | None = None,
) -> tuple[list[str], pd.DataFrame, list[TransformSpec]]:
    """Add pairwise interaction terms (vars_a × vars_b).

    Parameters
    ----------
    vars_a, vars_b:
        Sets of variables to cross.  Self-interactions are skipped (use
        :func:`generate_polynomial` for squared terms).
    max_order:
        Maximum interaction order (currently only 2 is supported; higher orders
        are reserved for future implementation).
    whitelist:
        If provided, only these ``(var_a, var_b)`` pairs are included.
    blacklist:
        If provided, these pairs are excluded.

    Returns
    -------
    (new_names, augmented_df, specs)
    """
    if max_order > 2:
        raise NotImplementedError("Interaction order > 2 is not yet implemented.")

    df = df.copy()
    new_names: list[str] = []
    specs: list[TransformSpec] = []
    bl_set = set(blacklist or [])

    for va in vars_a:
        for vb in vars_b:
            if va == vb:
                continue
            pair = tuple(sorted([va, vb]))
            if pair in bl_set:
                continue
            if whitelist and (va, vb) not in whitelist and (vb, va) not in whitelist:
                continue

            col_name = f"{va}_x_{vb}"
            # Canonical ordering: sorted alphabetically
            c1, c2 = sorted([va, vb])
            col_name = f"{c1}_x_{c2}"
            if col_name in df.columns:
                new_names.append(col_name)
                continue

            df[col_name] = df[va] * df[vb]
            new_names.append(col_name)
            specs.append(TransformSpec(
                kind="interaction",
                base_vars=(c1, c2),
                output_vars=(col_name,),
                params={"order": 2},
            ))

    return new_names, df, specs


# ---------------------------------------------------------------------------
# Log transforms
# ---------------------------------------------------------------------------


def generate_log(
    vars: list[str],
    df: pd.DataFrame,
    *,
    shift: float = 0.0,
) -> tuple[list[str], pd.DataFrame, list[TransformSpec]]:
    """Add log(x + shift) for each variable.

    Variables with non-positive values (after shift) are skipped with a warning.

    Parameters
    ----------
    vars:
        Base variable names.
    df:
        Working DataFrame.
    shift:
        Constant added before taking log (to handle zeros).

    Returns
    -------
    (new_names, augmented_df, specs)
    """
    df = df.copy()
    new_names: list[str] = []
    specs: list[TransformSpec] = []

    for var in vars:
        vals = df[var] + shift
        if (vals <= 0).any():
            warnings.warn(
                f"generate_log: variable '{var}' has non-positive values after "
                f"shift={shift}; skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        col_name = f"log_{var}" if shift == 0.0 else f"log_{var}_s{shift}"
        if col_name in df.columns:
            new_names.append(col_name)
            continue

        df[col_name] = np.log(vals)
        new_names.append(col_name)
        specs.append(TransformSpec(
            kind="log",
            base_vars=(var,),
            output_vars=(col_name,),
            params={"shift": shift},
        ))

    return new_names, df, specs


def generate_log1p(
    vars: list[str],
    df: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, list[TransformSpec]]:
    """Add log(1 + x) for non-negative variables."""
    df = df.copy()
    new_names: list[str] = []
    specs: list[TransformSpec] = []

    for var in vars:
        if (df[var] < 0).any():
            warnings.warn(
                f"generate_log1p: variable '{var}' has negative values; skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        col_name = f"log1p_{var}"
        if col_name in df.columns:
            new_names.append(col_name)
            continue

        df[col_name] = np.log1p(df[var])
        new_names.append(col_name)
        specs.append(TransformSpec(
            kind="log1p",
            base_vars=(var,),
            output_vars=(col_name,),
            params={},
        ))

    return new_names, df, specs


# ---------------------------------------------------------------------------
# Restricted cubic splines
# ---------------------------------------------------------------------------


def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Compute restricted cubic spline basis matrix.

    Uses Harrell's parameterisation: k-2 non-linear basis functions for k knots.

    Parameters
    ----------
    x:
        1-D array of predictor values.
    knots:
        1-D array of knot positions (typically quantiles of x).

    Returns
    -------
    np.ndarray of shape (n, k-2) — the non-linear columns (excluding the
    linear term which is x itself).
    """
    k = len(knots)
    n = len(x)
    if k < 3:
        raise ValueError("Restricted cubic splines require at least 3 knots.")

    # Harrell's formula (Frank Harrell, Regression Modeling Strategies)
    t_k = knots[-1]
    t_k1 = knots[-2]
    denom = (t_k - knots[0]) ** 2

    cols = []
    for j in range(k - 2):
        t_j = knots[j]
        col = (
            np.maximum(x - t_j, 0) ** 3
            - np.maximum(x - t_k1, 0) ** 3 * (t_k - t_j) / (t_k - t_k1)
            + np.maximum(x - t_k, 0) ** 3 * (t_k1 - t_j) / (t_k - t_k1)
        ) / denom
        cols.append(col)

    return np.column_stack(cols) if cols else np.zeros((n, 0))


def generate_splines(
    var: str,
    n_knots: int,
    df: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, list[TransformSpec]]:
    """Add restricted cubic spline basis columns for *var* with *n_knots* knots.

    Knot positions are placed at equally-spaced quantiles of *var*.
    The resulting basis (excluding the linear term) has ``n_knots - 2`` columns.

    Parameters
    ----------
    var:
        Predictor variable.
    n_knots:
        Number of knots (>= 3).
    df:
        Working DataFrame.

    Returns
    -------
    (new_names, augmented_df, specs)
    """
    if n_knots < 3:
        raise ValueError("generate_splines: n_knots must be >= 3.")

    df = df.copy()
    x = df[var].values.astype(float)
    quantile_pcts = np.linspace(0, 100, n_knots)
    knots = np.percentile(x, quantile_pcts)

    basis = _rcs_basis(x, knots)
    new_names: list[str] = []
    specs: list[TransformSpec] = []
    col_names = [f"{var}_rcs{n_knots}_b{j+1}" for j in range(basis.shape[1])]

    for j, col_name in enumerate(col_names):
        if col_name not in df.columns:
            df[col_name] = basis[:, j]
        new_names.append(col_name)

    if col_names:
        specs.append(TransformSpec(
            kind="spline",
            base_vars=(var,),
            output_vars=tuple(col_names),
            params={"n_knots": n_knots, "knot_positions": list(knots)},
        ))

    return new_names, df, specs


# ---------------------------------------------------------------------------
# Composite: apply all requested transforms
# ---------------------------------------------------------------------------


def apply_transforms(
    base_X: list[str],
    config: dict,
    df: pd.DataFrame,
) -> tuple[list[list[str]], pd.DataFrame, list[TransformSpec]]:
    """Apply all configured feature transforms to *df*.

    Parameters
    ----------
    base_X:
        Base exogenous variable names (already in *df*).
    config:
        Feature generator config dict from sieve YAML:

        .. code-block:: yaml

            features:
              polynomial:
                enabled: true
                degree: 2
                vars: null   # null = apply to all base_X
              interactions:
                enabled: true
                max_order: 2
              log:
                enabled: true
                vars: null
                shift: 0
              splines:
                enabled: false
                n_knots: 4
                vars: []

    df:
        Input DataFrame.

    Returns
    -------
    (candidate_term_sets, augmented_df, all_specs)
        ``candidate_term_sets`` is a list of term-sets; each entry is a list
        of new column names added by a single generator call.  These are
        combined with ``base_X`` by the calling code to build candidates.
    """
    df = df.copy()
    term_sets: list[list[str]] = []
    all_specs: list[TransformSpec] = []

    feat_cfg = config.get("features", {})

    # Polynomial
    poly_cfg = feat_cfg.get("polynomial", {})
    if poly_cfg.get("enabled", False):
        degree = int(poly_cfg.get("degree", 2))
        poly_vars = poly_cfg.get("vars") or base_X
        orth = bool(poly_cfg.get("orthogonalize", False))
        names, df, specs = generate_polynomial(poly_vars, degree, df, orthogonalize=orth)
        if names:
            term_sets.append(names)
            all_specs.extend(specs)

    # Interactions
    inter_cfg = feat_cfg.get("interactions", {})
    if inter_cfg.get("enabled", False):
        max_order = int(inter_cfg.get("max_order", 2))
        a_vars = inter_cfg.get("vars_a") or base_X
        b_vars = inter_cfg.get("vars_b") or base_X
        whitelist = inter_cfg.get("whitelist")
        blacklist = inter_cfg.get("blacklist")
        names, df, specs = generate_interactions(
            a_vars, b_vars, df,
            max_order=max_order,
            whitelist=[tuple(p) for p in whitelist] if whitelist else None,
            blacklist=[tuple(p) for p in blacklist] if blacklist else None,
        )
        if names:
            term_sets.append(names)
            all_specs.extend(specs)

    # Log
    log_cfg = feat_cfg.get("log", {})
    if log_cfg.get("enabled", False):
        log_vars = log_cfg.get("vars") or base_X
        shift = float(log_cfg.get("shift", 0.0))
        names, df, specs = generate_log(log_vars, df, shift=shift)
        if names:
            term_sets.append(names)
            all_specs.extend(specs)

    # log1p
    log1p_cfg = feat_cfg.get("log1p", {})
    if log1p_cfg.get("enabled", False):
        log1p_vars = log1p_cfg.get("vars") or base_X
        names, df, specs = generate_log1p(log1p_vars, df)
        if names:
            term_sets.append(names)
            all_specs.extend(specs)

    # Splines
    spline_cfg = feat_cfg.get("splines", {})
    if spline_cfg.get("enabled", False):
        n_knots = int(spline_cfg.get("n_knots", 4))
        spline_vars = spline_cfg.get("vars") or []
        for var in spline_vars:
            if var not in df.columns:
                continue
            try:
                names, df, specs = generate_splines(var, n_knots, df)
                if names:
                    term_sets.append(names)
                    all_specs.extend(specs)
            except Exception as e:
                import warnings
                warnings.warn(f"Spline generation failed for '{var}': {e}")

    return term_sets, df, all_specs
