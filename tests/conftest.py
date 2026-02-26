"""Shared pytest fixtures for econtools tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Small cross-sectional DataFrame for basic tests."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "wage": rng.uniform(5.0, 30.0, n),
            "educ": rng.integers(8, 18, n).astype(float),
            "exper": rng.integers(0, 30, n).astype(float),
            "female": rng.integers(0, 2, n).astype(float),
            "label_col": ["foo" if i % 2 == 0 else "bar" for i in range(n)],
        }
    )


@pytest.fixture
def simple_df_missing(simple_df: pd.DataFrame) -> pd.DataFrame:
    """simple_df with ~10% of wage values set to NaN."""
    df = simple_df.copy()
    rng = np.random.default_rng(7)
    mask = rng.random(len(df)) < 0.10
    df.loc[mask, "wage"] = float("nan")
    return df


@pytest.fixture
def panel_df() -> pd.DataFrame:
    """Balanced panel: 10 entities × 5 periods."""
    n_entities = 10
    n_periods = 5
    rng = np.random.default_rng(99)
    rows = []
    for entity in range(1, n_entities + 1):
        for t in range(1, n_periods + 1):
            rows.append(
                {
                    "id": entity,
                    "year": 1990 + t,
                    "y": rng.normal(2.0, 1.0),
                    "x": rng.normal(0.0, 1.0),
                    "w": rng.uniform(0.5, 5.0),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def panel_df_multiindex(panel_df: pd.DataFrame) -> pd.DataFrame:
    """panel_df with (id, year) MultiIndex."""
    return panel_df.set_index(["id", "year"]).sort_index()


# ---------------------------------------------------------------------------
# Phase 1 fixtures — models / inference / diagnostics / plots / tables
# ---------------------------------------------------------------------------


@pytest.fixture
def ols_data() -> pd.DataFrame:
    """DGP: y = 2 + 3*x1 + 0.5*x2 + N(0,1), seed=42, n=200."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    y = 2.0 + 3.0 * x1 + 0.5 * x2 + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture
def ols_result(ols_data: pd.DataFrame):
    """Pre-fitted OLS result on ols_data."""
    from econtools.models import fit_ols
    return fit_ols(ols_data, "y", ["x1", "x2"])


@pytest.fixture
def heteroskedastic_data() -> pd.DataFrame:
    """Heteroskedastic DGP: e = x * N(0,1), n=500."""
    rng = np.random.default_rng(1)
    n = 500
    x = rng.uniform(1.0, 10.0, n)
    e = x * rng.normal(0.0, 1.0, n)
    y = 1.0 + 2.0 * x + e
    return pd.DataFrame({"y": y, "x": x})


@pytest.fixture
def non_normal_data() -> pd.DataFrame:
    """Non-normal errors: e ~ chi2(2) - 2, n=500."""
    rng = np.random.default_rng(2)
    n = 500
    x = rng.normal(0.0, 1.0, n)
    e = rng.chisquare(2, n) - 2.0
    y = 1.0 + 2.0 * x + e
    return pd.DataFrame({"y": y, "x": x})


@pytest.fixture
def misspecified_data() -> pd.DataFrame:
    """Quadratic DGP fitted linearly: y = 1 + 2x + 1.5x², n=300."""
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(-3.0, 3.0, n)
    e = rng.normal(0.0, 0.5, n)
    y = 1.0 + 2.0 * x + 1.5 * x**2 + e
    return pd.DataFrame({"y": y, "x": x})
