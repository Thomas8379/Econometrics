"""Tests for econtools.data.transform."""

import numpy as np
import pandas as pd
import pytest

from econtools.data.transform import (
    demean_within,
    diff_col,
    dummies,
    growth_rate,
    interact,
    lag,
    lead,
    log1p_col,
    log_col,
    poly,
    rolling_mean,
    standardise,
    time_trend,
)


# ---------------------------------------------------------------------------
# log_col / log1p_col
# ---------------------------------------------------------------------------


def test_log_col_adds_column(simple_df: pd.DataFrame) -> None:
    result = log_col(simple_df, "wage")
    assert "log_wage" in result.columns
    np.testing.assert_allclose(result["log_wage"], np.log(simple_df["wage"]))


def test_log_col_custom_name(simple_df: pd.DataFrame) -> None:
    result = log_col(simple_df, "wage", name="ln_w")
    assert "ln_w" in result.columns


def test_log_col_no_mutation(simple_df: pd.DataFrame) -> None:
    _ = log_col(simple_df, "wage")
    assert "log_wage" not in simple_df.columns


def test_log1p_col(simple_df: pd.DataFrame) -> None:
    result = log1p_col(simple_df, "exper")
    assert "log1p_exper" in result.columns
    np.testing.assert_allclose(result["log1p_exper"], np.log1p(simple_df["exper"]))


# ---------------------------------------------------------------------------
# lag / lead
# ---------------------------------------------------------------------------


def test_lag_k1(panel_df: pd.DataFrame) -> None:
    df_sorted = panel_df.sort_values(["id", "year"])
    result = lag(df_sorted, "y", "id", k=1)
    assert "lag_y_k1" in result.columns
    # Entity 1, year 1992 should have lag = entity 1 year 1991
    e1 = result[result["id"] == 1].sort_values("year")
    assert np.isnan(e1.iloc[0]["lag_y_k1"])  # first obs has no lag
    assert e1.iloc[1]["lag_y_k1"] == pytest.approx(e1.iloc[0]["y"])


def test_lead_k1(panel_df: pd.DataFrame) -> None:
    df_sorted = panel_df.sort_values(["id", "year"])
    result = lead(df_sorted, "y", "id", k=1)
    assert "lead_y_k1" in result.columns
    e1 = result[result["id"] == 1].sort_values("year")
    assert np.isnan(e1.iloc[-1]["lead_y_k1"])  # last obs has no lead
    assert e1.iloc[0]["lead_y_k1"] == pytest.approx(e1.iloc[1]["y"])


def test_lag_no_bleed(panel_df: pd.DataFrame) -> None:
    """Lag should not carry last obs of entity N to first obs of entity N+1."""
    df_sorted = panel_df.sort_values(["id", "year"])
    result = lag(df_sorted, "y", "id", k=1)
    # First year for entity 2 should be NaN, not entity 1's last value
    e2_first = result[(result["id"] == 2) & (result["year"] == result[result["id"] == 2]["year"].min())]
    assert np.isnan(e2_first.iloc[0]["lag_y_k1"])


# ---------------------------------------------------------------------------
# diff_col / growth_rate
# ---------------------------------------------------------------------------


def test_diff_col(panel_df: pd.DataFrame) -> None:
    df = panel_df.sort_values(["id", "year"])
    result = diff_col(df, "y", "id")
    assert "d_y" in result.columns
    e1 = result[result["id"] == 1].sort_values("year")
    assert np.isnan(e1.iloc[0]["d_y"])
    assert e1.iloc[1]["d_y"] == pytest.approx(e1.iloc[1]["y"] - e1.iloc[0]["y"])


def test_growth_rate(panel_df: pd.DataFrame) -> None:
    df = panel_df.sort_values(["id", "year"])
    result = growth_rate(df, "w", "id")  # w is always > 0
    assert "g_w" in result.columns


# ---------------------------------------------------------------------------
# dummies
# ---------------------------------------------------------------------------


def test_dummies_drop_first(simple_df: pd.DataFrame) -> None:
    result = dummies(simple_df, "label_col", drop_first=True)
    # 2 categories → 1 dummy column with drop_first
    dummy_cols = [c for c in result.columns if c.startswith("label_col_")]
    assert len(dummy_cols) == 1


def test_dummies_no_drop_first(simple_df: pd.DataFrame) -> None:
    result = dummies(simple_df, "label_col", drop_first=False)
    dummy_cols = [c for c in result.columns if c.startswith("label_col_")]
    assert len(dummy_cols) == 2


def test_dummies_removes_original_col(simple_df: pd.DataFrame) -> None:
    result = dummies(simple_df, "label_col")
    assert "label_col" not in result.columns


# ---------------------------------------------------------------------------
# interact / poly
# ---------------------------------------------------------------------------


def test_interact(simple_df: pd.DataFrame) -> None:
    result = interact(simple_df, "educ", "exper")
    assert "educ_x_exper" in result.columns
    np.testing.assert_allclose(
        result["educ_x_exper"], simple_df["educ"] * simple_df["exper"]
    )


def test_poly_degree2(simple_df: pd.DataFrame) -> None:
    result = poly(simple_df, "educ", degree=2)
    assert "educ_sq" in result.columns
    np.testing.assert_allclose(result["educ_sq"], simple_df["educ"] ** 2)


def test_poly_degree3(simple_df: pd.DataFrame) -> None:
    result = poly(simple_df, "educ", degree=3)
    assert "educ_d3" in result.columns


# ---------------------------------------------------------------------------
# standardise / demean_within
# ---------------------------------------------------------------------------


def test_standardise(simple_df: pd.DataFrame) -> None:
    result = standardise(simple_df, "wage")
    assert "z_wage" in result.columns
    assert abs(result["z_wage"].mean()) < 1e-10
    assert abs(result["z_wage"].std() - 1.0) < 1e-10


def test_demean_within(panel_df: pd.DataFrame) -> None:
    result = demean_within(panel_df, "y", "id")
    assert "dm_y" in result.columns
    # Within-entity mean of demeaned variable should be ~0
    entity_means = result.groupby("id")["dm_y"].mean()
    np.testing.assert_allclose(entity_means, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# time_trend / rolling_mean
# ---------------------------------------------------------------------------


def test_time_trend(panel_df: pd.DataFrame) -> None:
    df = panel_df.sort_values(["id", "year"])
    result = time_trend(df, "id")
    assert "t" in result.columns
    e1 = result[result["id"] == 1].sort_values("year")
    assert list(e1["t"]) == [0, 1, 2, 3, 4]


def test_rolling_mean(panel_df: pd.DataFrame) -> None:
    df = panel_df.sort_values(["id", "year"])
    result = rolling_mean(df, "y", "id", k=3)
    assert "roll3_y" in result.columns
    # First k-1 obs within each entity should be NaN
    e1 = result[result["id"] == 1].sort_values("year")
    assert np.isnan(e1.iloc[0]["roll3_y"])
    assert np.isnan(e1.iloc[1]["roll3_y"])
    assert not np.isnan(e1.iloc[2]["roll3_y"])
