"""Tests for feature and instrument generators."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from econtools.sieve.generators.features import (
    generate_interactions,
    generate_log,
    generate_log1p,
    generate_polynomial,
    generate_splines,
    generate_squares,
)
from econtools.sieve.generators.instruments import (
    generate_group_means,
    generate_lags,
    generate_loogroup_means,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.uniform(0.5, 5.0, n),
        "x3": rng.integers(1, 10, n).astype(float),
        "y": rng.standard_normal(n),
    })


@pytest.fixture
def panel_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_entities = 5
    n_periods = 10
    ids = np.repeat(np.arange(n_entities), n_periods)
    times = np.tile(np.arange(n_periods), n_entities)
    z = rng.standard_normal(n_entities * n_periods)
    y = z + rng.standard_normal(n_entities * n_periods) * 0.5
    return pd.DataFrame({"id": ids, "t": times, "z": z, "y": y})


# ---------------------------------------------------------------------------
# Polynomial
# ---------------------------------------------------------------------------


class TestGeneratePolynomial:
    def test_basic(self, base_df):
        names, df2, specs = generate_polynomial(["x1", "x2"], 3, base_df)
        assert "x1_pow2" in names
        assert "x1_pow3" in names
        assert "x2_pow2" in names
        assert "x2_pow3" in names
        # Correct values
        np.testing.assert_allclose(df2["x1_pow2"].values, base_df["x1"].values ** 2)
        np.testing.assert_allclose(df2["x1_pow3"].values, base_df["x1"].values ** 3)

    def test_degree_2_convenience(self, base_df):
        names_sq, df_sq, _ = generate_squares(["x1"], base_df)
        names_p, df_p, _ = generate_polynomial(["x1"], 2, base_df)
        assert names_sq == names_p
        np.testing.assert_allclose(df_sq["x1_pow2"].values, df_p["x1_pow2"].values)

    def test_term_count(self, base_df):
        names, _, specs = generate_polynomial(["x1", "x2"], 2, base_df)
        # degree 2 for 2 vars = 2 new columns
        assert len(names) == 2
        assert len(specs) == 2

    def test_no_duplicate_columns(self, base_df):
        # Calling twice should not duplicate
        _, df2, _ = generate_polynomial(["x1"], 2, base_df)
        names2, df3, specs2 = generate_polynomial(["x1"], 2, df2)
        # Specs for already-present columns should be empty (column skipped)
        assert "x1_pow2" in df3.columns
        assert len(specs2) == 0


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------


class TestGenerateInteractions:
    def test_basic(self, base_df):
        names, df2, specs = generate_interactions(["x1"], ["x2"], base_df)
        assert "x1_x_x2" in names
        np.testing.assert_allclose(
            df2["x1_x_x2"].values, base_df["x1"].values * base_df["x2"].values
        )

    def test_self_interaction_skipped(self, base_df):
        names, _, _ = generate_interactions(["x1"], ["x1"], base_df)
        assert len(names) == 0

    def test_canonical_ordering(self, base_df):
        # Both orderings should produce the same column
        names1, df1, _ = generate_interactions(["x1"], ["x2"], base_df)
        names2, df2, _ = generate_interactions(["x2"], ["x1"], base_df)
        assert "x1_x_x2" in names1
        assert "x1_x_x2" in names2
        np.testing.assert_allclose(df1["x1_x_x2"].values, df2["x1_x_x2"].values)

    def test_whitelist(self, base_df):
        names, _, _ = generate_interactions(
            ["x1", "x3"], ["x2"], base_df, whitelist=[("x1", "x2")]
        )
        # Only x1×x2 should be included
        assert len(names) == 1
        assert "x1_x_x2" in names

    def test_blacklist(self, base_df):
        names, _, _ = generate_interactions(
            ["x1", "x3"], ["x2"], base_df, blacklist=[("x1", "x2")]
        )
        # x1×x2 excluded, x2×x3 included
        assert "x1_x_x2" not in names
        assert "x2_x_x3" in names


# ---------------------------------------------------------------------------
# Log transforms
# ---------------------------------------------------------------------------


class TestGenerateLog:
    def test_basic(self, base_df):
        # x2 is always positive
        names, df2, specs = generate_log(["x2"], base_df)
        assert "log_x2" in names
        np.testing.assert_allclose(df2["log_x2"].values, np.log(base_df["x2"].values))

    def test_negative_values_skipped(self, base_df):
        # x1 has negative values — should be skipped with a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            names, df2, specs = generate_log(["x1"], base_df)
        assert len(names) == 0
        assert any("non-positive" in str(warn.message) for warn in w)

    def test_shift(self, base_df):
        # shift=1 makes x1+1 positive
        min_x1 = base_df["x1"].min()
        shift = -min_x1 + 0.01
        names, df2, specs = generate_log(["x1"], base_df, shift=shift)
        assert len(names) == 1
        col = names[0]
        expected = np.log(base_df["x1"].values + shift)
        np.testing.assert_allclose(df2[col].values, expected)

    def test_log1p_nonnegative(self, base_df):
        # x2 is positive
        names, df2, specs = generate_log1p(["x2"], base_df)
        assert "log1p_x2" in names
        np.testing.assert_allclose(df2["log1p_x2"].values, np.log1p(base_df["x2"].values))

    def test_log1p_negative_skipped(self, base_df):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            names, _, _ = generate_log1p(["x1"], base_df)
        assert len(names) == 0
        assert any("negative" in str(warn.message) for warn in w)


# ---------------------------------------------------------------------------
# Splines
# ---------------------------------------------------------------------------


class TestGenerateSplines:
    def test_basic(self, base_df):
        names, df2, specs = generate_splines("x1", 4, base_df)
        # k=4 knots => k-2=2 basis columns
        assert len(names) == 2
        assert all(n.startswith("x1_rcs4_b") for n in names)

    def test_values_finite(self, base_df):
        names, df2, _ = generate_splines("x2", 5, base_df)
        for n in names:
            assert np.all(np.isfinite(df2[n].values)), f"Column {n} has non-finite values"

    def test_min_knots(self, base_df):
        with pytest.raises(ValueError, match="n_knots"):
            generate_splines("x1", 2, base_df)

    def test_shape(self, base_df):
        for n_knots in [3, 4, 5, 6]:
            names, df2, _ = generate_splines("x1", n_knots, base_df)
            assert len(names) == n_knots - 2


# ---------------------------------------------------------------------------
# Lags
# ---------------------------------------------------------------------------


class TestGenerateLags:
    def test_basic_panel(self, panel_df):
        names, df2, specs = generate_lags(
            ["z"], 1, panel_df, time_col="t", id_col="id"
        )
        assert "lag_z_k1" in names
        # First period of each entity should be NaN
        first_periods = df2[df2["t"] == 0]["lag_z_k1"]
        assert first_periods.isna().all()

    def test_lag_values_correct(self, panel_df):
        names, df2, _ = generate_lags(["z"], 1, panel_df, time_col="t", id_col="id")
        # For entity 0, t=1: lag should equal value at t=0
        e0 = df2[df2["id"] == 0].sort_values("t")
        assert float(e0.iloc[1]["lag_z_k1"]) == pytest.approx(float(e0.iloc[0]["z"]))

    def test_no_lookahead(self, panel_df):
        """Lags must use only past data — lag at t=k uses t=0..k-1."""
        names, df2, _ = generate_lags(["z"], 2, panel_df, time_col="t", id_col="id")
        e0 = df2[df2["id"] == 0].sort_values("t")
        # t=2: lag_k2 should equal t=0 value
        assert float(e0.iloc[2]["lag_z_k2"]) == pytest.approx(float(e0.iloc[0]["z"]))

    def test_invalid_lag_k(self, panel_df):
        with pytest.raises(ValueError):
            generate_lags(["z"], 0, panel_df, time_col="t")


# ---------------------------------------------------------------------------
# Group means
# ---------------------------------------------------------------------------


class TestGroupMeans:
    @pytest.fixture
    def group_df(self):
        rng = np.random.default_rng(7)
        n = 60
        groups = np.repeat(["A", "B", "C"], 20)
        z = rng.standard_normal(n)
        return pd.DataFrame({"group": groups, "z": z})

    def test_group_mean_values(self, group_df):
        names, df2, _ = generate_group_means(["z"], "group", group_df)
        assert "gmean_z" in names
        for g in ["A", "B", "C"]:
            mask = group_df["group"] == g
            expected = float(group_df.loc[mask, "z"].mean())
            actual = df2.loc[mask, "gmean_z"].iloc[0]
            assert float(actual) == pytest.approx(expected, abs=1e-10)

    def test_loo_group_mean_excludes_self(self, group_df):
        names, df2, _ = generate_loogroup_means(["z"], "group", group_df)
        assert "loo_gmean_z" in names
        # For each unit, the LOO mean should not include itself
        for idx in range(len(group_df)):
            g = group_df.iloc[idx]["group"]
            self_val = group_df.iloc[idx]["z"]
            mask = group_df["group"] == g
            group_vals = group_df.loc[mask, "z"]
            expected_loo = (group_vals.sum() - self_val) / (len(group_vals) - 1)
            actual = df2.iloc[idx]["loo_gmean_z"]
            assert float(actual) == pytest.approx(float(expected_loo), abs=1e-10)

    def test_loo_single_member_is_nan(self):
        """Groups with only one member must return NaN for LOO mean."""
        df = pd.DataFrame({"group": ["A", "B", "B"], "z": [1.0, 2.0, 3.0]})
        _, df2, _ = generate_loogroup_means(["z"], "group", df)
        # Group A has only one member → NaN
        a_loo = df2[df["group"] == "A"]["loo_gmean_z"]
        assert a_loo.isna().all()
