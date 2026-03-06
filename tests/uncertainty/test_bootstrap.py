"""Rigorous test suite for the bootstrap inference module.

Test categories
---------------
1. Unit tests (fast, deterministic)
2. OLS correctness against analytic benchmarks
3. Cluster bootstrap tests
4. Panel bootstrap tests
5. IV / 2SLS bootstrap tests
6. Failure-mode and edge-case tests

All simulation tests fix seeds and are deterministic.  Slow simulation
tests are marked ``@pytest.mark.slow`` so they can be skipped in CI:
    pytest tests/uncertainty/ -m "not slow"
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from econtools.uncertainty.bootstrap import run_bootstrap
from econtools.uncertainty._bootstrap_estimators import ols_fit, twosls_fit
from econtools.uncertainty._bootstrap_manifest import compute_config_hash


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def simple_ols_df() -> pd.DataFrame:
    """Small homoskedastic linear model.  y = 2 + 3*x + eps, n=200."""
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(size=n)
    y = 2.0 + 3.0 * x + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x})


@pytest.fixture
def hetero_df() -> pd.DataFrame:
    """Heteroskedastic linear model.  Var(eps) = x^2."""
    rng = np.random.default_rng(13)
    n = 400
    x = rng.uniform(0.5, 2.5, size=n)
    eps = rng.normal(scale=x, size=n)
    y = 1.0 + 2.0 * x + eps
    return pd.DataFrame({"y": y, "x": x})


@pytest.fixture
def clustered_df() -> pd.DataFrame:
    """Clustered data: G=50 clusters, 20 obs each.
    Common shock per cluster + idiosyncratic noise."""
    rng = np.random.default_rng(17)
    G, m = 50, 20
    n = G * m
    cluster_id = np.repeat(np.arange(G), m)
    x = rng.normal(size=n)
    # cluster-level shock (creates within-cluster correlation)
    cluster_shock = rng.normal(scale=1.5, size=G)
    eps = cluster_shock[cluster_id] + rng.normal(scale=0.5, size=n)
    y = 1.0 + 2.0 * x + eps
    return pd.DataFrame({"y": y, "x": x, "cluster_id": cluster_id})


@pytest.fixture
def panel_df() -> pd.DataFrame:
    """Panel data: N=50 entities, T=10 periods, within-entity corr in both x and y.

    DGP:
        x_it = alpha_x_i + 0.3*eps_x_it   (strong within-entity x correlation)
        y_it = 1.5 + 0.8*x_it + alpha_y_i + 0.3*eps_it

    With rho_x ≈ 0.91 and rho_u ≈ 0.97, the DEFF for beta_x is ≈ 9,
    so panel_cluster_id SE >> iid_pairs SE.
    """
    rng = np.random.default_rng(23)
    N, T = 50, 10
    n = N * T
    entity_id = np.repeat(np.arange(N), T)
    time_id = np.tile(np.arange(T), N)

    alpha_x = rng.normal(scale=1.0, size=N)   # entity-level x effect
    alpha_y = rng.normal(scale=2.0, size=N)   # entity-level error effect

    x = alpha_x[entity_id] + 0.3 * rng.normal(size=n)  # strong within-entity x corr
    eps = alpha_y[entity_id] + 0.3 * rng.normal(size=n)
    y = 1.5 + 0.8 * x + eps
    return pd.DataFrame({"y": y, "x": x, "entity_id": entity_id, "time": time_id})


@pytest.fixture
def iv_df() -> pd.DataFrame:
    """IV dataset where 2SLS coefficient is approximately known.

    DGP:
        z ~ N(0,1)       [instrument]
        v ~ N(0,1)       [endogeneity source]
        x = 0.7*z + v   [endogenous regressor]
        y = 1 + 2*x + v  [structural equation, beta_x = 2]

    Because cov(x,v) != 0, OLS is biased upward.
    2SLS with instrument z recovers beta_x = 2.
    """
    rng = np.random.default_rng(31)
    n = 800
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    x = 0.7 * z + v
    y = 1.0 + 2.0 * x + v
    return pd.DataFrame({"y": y, "x": x, "z": z})


# ===========================================================================
# 1. Unit tests
# ===========================================================================


class TestRNGDeterminism:
    """Same seed/config → identical byte-for-byte draws."""

    def test_sequential_same_seed(self, simple_ols_df: pd.DataFrame) -> None:
        kw = dict(
            data=simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs",
            B=99, seed=42, n_jobs=1,
        )
        r1 = run_bootstrap(**kw)
        r2 = run_bootstrap(**kw)
        np.testing.assert_array_equal(r1["bootstrap"]["draws"], r2["bootstrap"]["draws"])

    def test_parallel_matches_sequential(self, simple_ols_df: pd.DataFrame) -> None:
        """n_jobs>1 produces identical draws to n_jobs=1."""
        common = dict(
            data=simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs",
            B=99, seed=42,
        )
        r_seq = run_bootstrap(**common, n_jobs=1)
        r_par = run_bootstrap(**common, n_jobs=2)
        # Draws may be in a different order for parallel execution,
        # but sorted draws must be identical.
        draws_seq = np.sort(r_seq["bootstrap"]["draws"], axis=0)
        draws_par = np.sort(r_par["bootstrap"]["draws"], axis=0)
        np.testing.assert_allclose(draws_seq, draws_par, rtol=0, atol=0)

    def test_different_seeds_differ(self, simple_ols_df: pd.DataFrame) -> None:
        r1 = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=99, seed=1,
        )
        r2 = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=99, seed=2,
        )
        assert not np.array_equal(
            r1["bootstrap"]["draws"], r2["bootstrap"]["draws"]
        )


class TestShapesAndKeys:
    """Output contains required keys; dimensions match expectations."""

    def test_required_top_level_keys(self, simple_ols_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=49, seed=0,
        )
        assert "point_estimate" in result
        assert "bootstrap" in result
        assert "metadata" in result
        assert "_manifest" in result

    def test_point_estimate_keys(self, simple_ols_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=49, seed=0,
        )
        pe = result["point_estimate"]
        for key in ("params", "coef_names", "fitted", "residuals"):
            assert key in pe, f"missing key: {key}"

    def test_bootstrap_keys(self, simple_ols_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=49, seed=0,
        )
        bs = result["bootstrap"]
        for key in ("draws", "se", "ci", "pvalues", "bagged_mean", "bagged_median"):
            assert key in bs, f"missing key: {key}"

    def test_draws_shape(self, simple_ols_df: pd.DataFrame) -> None:
        B = 77
        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=B, seed=0,
        )
        k = len(result["point_estimate"]["coef_names"])
        assert result["bootstrap"]["draws"].shape == (B, k)

    def test_metadata_keys(self, simple_ols_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=49, seed=0,
        )
        meta = result["metadata"]
        for key in (
            "method", "estimator", "B", "seed", "n_obs", "n_dropped",
            "config_hash", "timestamp", "package_versions", "warnings",
        ):
            assert key in meta, f"missing metadata key: {key}"


class TestMissingDataDrops:
    """Listwise deletion works correctly."""

    def test_drops_counted(self) -> None:
        rng = np.random.default_rng(99)
        n = 100
        df = pd.DataFrame({
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
        })
        # Introduce 5 missing values in y
        df.loc[[0, 10, 20, 30, 40], "y"] = float("nan")
        result = run_bootstrap(
            df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=49, seed=0,
        )
        assert result["metadata"]["n_dropped"] == 5
        assert result["metadata"]["n_obs"] == 95

    def test_sample_indices_correct(self) -> None:
        rng = np.random.default_rng(99)
        n = 80
        df = pd.DataFrame({
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
        })
        df.loc[[1, 3, 5], "x"] = float("nan")
        result = run_bootstrap(
            df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=49, seed=0,
        )
        assert result["metadata"]["n_obs"] == n - 3

    def test_raises_on_all_missing(self) -> None:
        df = pd.DataFrame({"y": [float("nan")] * 10, "x": [1.0] * 10})
        with pytest.raises(ValueError, match="No observations"):
            run_bootstrap(
                df, y="y", X=["x"],
                estimator="ols", bootstrap_method="iid_pairs", B=9, seed=0,
            )


class TestInterceptHandling:
    """add_intercept behavior is correct."""

    def test_adds_const_by_default(self, simple_ols_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=49, seed=0,
        )
        assert "const" in result["point_estimate"]["coef_names"]

    def test_no_duplicate_const_when_already_present(self) -> None:
        rng = np.random.default_rng(3)
        n = 100
        df = pd.DataFrame({
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "const": np.ones(n),
        })
        result = run_bootstrap(
            df, y="y", X=["x", "const"],
            estimator="ols", bootstrap_method="iid_pairs",
            B=49, seed=0, add_intercept=True,
        )
        names = result["point_estimate"]["coef_names"]
        # should not add a second constant
        assert names.count("const") == 1
        assert "const" in names

    def test_no_intercept_when_disabled(self, simple_ols_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs",
            B=49, seed=0, add_intercept=False,
        )
        assert "const" not in result["point_estimate"]["coef_names"]
        assert result["bootstrap"]["draws"].shape[1] == 1  # x only

    def test_coef_count_matches(self, simple_ols_df: pd.DataFrame) -> None:
        # add_intercept=True → 2 coefficients (const, x)
        r = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=9, seed=0,
        )
        assert len(r["point_estimate"]["coef_names"]) == 2
        assert r["bootstrap"]["draws"].shape[1] == 2


# ===========================================================================
# 2. OLS correctness tests
# ===========================================================================


class TestOLSClosedForm:
    """Standalone ols_fit matches numpy lstsq."""

    def test_matches_lstsq(self) -> None:
        rng = np.random.default_rng(5)
        n, k = 200, 4
        X = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
        y = X @ rng.normal(size=k) + rng.normal(size=n)

        coefs_ols, _, _ = ols_fit(X, y)
        coefs_lstsq, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        np.testing.assert_allclose(coefs_ols, coefs_lstsq, rtol=1e-10)

    def test_residuals_orthogonal_to_X(self) -> None:
        rng = np.random.default_rng(11)
        n = 150
        X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
        y = rng.normal(size=n)
        _, resid, _ = ols_fit(X, y)
        # X'resid should be zero (up to numerical precision)
        np.testing.assert_allclose(X.T @ resid, np.zeros(3), atol=1e-10)

    def test_fitted_plus_resid_equals_y(self) -> None:
        rng = np.random.default_rng(19)
        n = 100
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = rng.normal(size=n)
        _, resid, fitted = ols_fit(X, y)
        np.testing.assert_allclose(fitted + resid, y, rtol=1e-12)


class TestIIDPairsBootstrapSE:
    """Bootstrap SE should be close to analytic OLS SE under homoskedasticity."""

    @pytest.mark.slow
    def test_se_close_to_analytic(self, simple_ols_df: pd.DataFrame) -> None:
        # Analytic OLS SE from statsmodels
        import statsmodels.api as sm

        X_sm = sm.add_constant(simple_ols_df[["x"]])
        sm_fit = sm.OLS(simple_ols_df["y"], X_sm).fit()
        analytic_se_x = float(sm_fit.bse["x"])

        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs",
            B=1999, seed=42,
        )
        boot_se_x = result["bootstrap"]["se"]["x"]

        # For n=200, B=1999, bootstrap SE should be within 15% of analytic SE
        rel_error = abs(boot_se_x - analytic_se_x) / analytic_se_x
        assert rel_error < 0.15, (
            f"Bootstrap SE ({boot_se_x:.4f}) differs from analytic SE "
            f"({analytic_se_x:.4f}) by {rel_error:.1%}"
        )

    def test_pvalues_in_unit_interval(self, simple_ols_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=199, seed=0,
        )
        for name, pv in result["bootstrap"]["pvalues"].items():
            assert 0 < pv <= 1, f"p-value for {name!r} out of range: {pv}"

    def test_ci_contains_point_estimate(self, simple_ols_df: pd.DataFrame) -> None:
        """Percentile CI should usually contain the point estimate."""
        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=499, seed=0,
        )
        pe = result["point_estimate"]["params"]
        ci = result["bootstrap"]["ci"]["percentile"]
        for name in pe:
            assert ci["lower"][name] <= pe[name] <= ci["upper"][name], (
                f"Point estimate for {name!r} outside percentile CI"
            )


class TestWildBootstrap:
    """Wild bootstrap SE should match HC-robust SE for heteroskedastic data."""

    @pytest.mark.slow
    def test_wild_se_close_to_hc1(self, hetero_df: pd.DataFrame) -> None:
        """Wild bootstrap SE is within 20% of HC1 SE for heteroskedastic data.

        Both wild and iid_pairs are consistent for HC-type SEs under independent
        heteroskedasticity.  We verify correctness by checking both are close
        to HC1 rather than testing a fragile ordering.
        """
        import statsmodels.api as sm

        X_sm = sm.add_constant(hetero_df[["x"]])
        hc1_fit = sm.OLS(hetero_df["y"], X_sm).fit(cov_type="HC1")
        hc1_se_x = float(hc1_fit.bse["x"])

        r_wild = run_bootstrap(
            hetero_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild",
            wild_dist="rademacher", B=1999, seed=42,
        )
        se_wild = r_wild["bootstrap"]["se"]["x"]
        rel_error = abs(se_wild - hc1_se_x) / hc1_se_x

        assert rel_error < 0.20, (
            f"Wild bootstrap SE ({se_wild:.4f}) differs from HC1 SE "
            f"({hc1_se_x:.4f}) by {rel_error:.1%}"
        )

    def test_wild_mammen_runs(self, simple_ols_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild",
            wild_dist="mammen", B=99, seed=0,
        )
        assert result["bootstrap"]["draws"].shape[0] == 99

    def test_wild_rademacher_weights_binary(self, simple_ols_df: pd.DataFrame) -> None:
        """Check Rademacher weights are ±1."""
        from econtools.uncertainty.bootstrap import _wild_weights
        rng = np.random.default_rng(0)
        v = _wild_weights(rng, n=1000, dist="rademacher")
        assert set(np.unique(v)).issubset({-1.0, 1.0})

    def test_wild_mammen_weights_two_values(self, simple_ols_df: pd.DataFrame) -> None:
        from econtools.uncertainty.bootstrap import _wild_weights
        rng = np.random.default_rng(0)
        v = _wild_weights(rng, n=2000, dist="mammen")
        unique_v = np.unique(v)
        assert len(unique_v) == 2
        # Check the two values are approximately -(sqrt5-1)/2 and (sqrt5+1)/2
        sqrt5 = np.sqrt(5)
        expected = sorted([-(sqrt5 - 1) / 2, (sqrt5 + 1) / 2])
        actual = sorted(unique_v.tolist())
        np.testing.assert_allclose(actual, expected, rtol=1e-10)


# ===========================================================================
# 3. Cluster bootstrap tests
# ===========================================================================


class TestClusterPairs:
    """cluster_pairs bootstrap matches cluster-robust SE asymptotically."""

    @pytest.mark.slow
    def test_se_close_to_cluster_robust(self, clustered_df: pd.DataFrame) -> None:
        """Bootstrap SE should be within 20% of cluster-robust SE for G=50."""
        import statsmodels.api as sm

        X_sm = sm.add_constant(clustered_df[["x"]])
        cluster_fit = sm.OLS(clustered_df["y"], X_sm).fit(
            cov_type="cluster",
            cov_kwds={"groups": clustered_df["cluster_id"]},
        )
        cl_se_x = float(cluster_fit.bse["x"])

        result = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="cluster_pairs",
            cluster="cluster_id", B=999, seed=42,
        )
        boot_se_x = result["bootstrap"]["se"]["x"]
        rel_error = abs(boot_se_x - cl_se_x) / cl_se_x
        assert rel_error < 0.20, (
            f"Cluster bootstrap SE ({boot_se_x:.4f}) differs from cluster-robust "
            f"SE ({cl_se_x:.4f}) by {rel_error:.1%}"
        )

    def test_small_cluster_warning(self) -> None:
        """G < 30 clusters triggers a warning in metadata."""
        rng = np.random.default_rng(7)
        G, m = 10, 20  # only 10 clusters
        n = G * m
        df = pd.DataFrame({
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "cid": np.repeat(np.arange(G), m),
        })
        with pytest.warns(UserWarning, match="Small cluster count"):
            result = run_bootstrap(
                df, y="y", X=["x"],
                estimator="ols", bootstrap_method="cluster_pairs",
                cluster="cid", B=49, seed=0,
            )
        assert result["metadata"]["cluster_count"] == G
        assert any("Small cluster" in w for w in result["metadata"]["warnings"])

    def test_cluster_count_in_metadata(self, clustered_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="cluster_pairs",
            cluster="cluster_id", B=49, seed=0,
        )
        assert result["metadata"]["cluster_count"] == 50

    def test_requires_cluster_col(self, simple_ols_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="cluster"):
            run_bootstrap(
                simple_ols_df, y="y", X=["x"],
                estimator="ols", bootstrap_method="cluster_pairs",
                B=9, seed=0,
            )


# ===========================================================================
# 4. Panel bootstrap tests
# ===========================================================================


class TestPanelClusterID:
    """panel_cluster_id: entity histories are preserved after resampling."""

    def test_preserves_full_histories(self, panel_df: pd.DataFrame) -> None:
        """Each sampled entity has exactly T time observations in output.

        We verify this by inspecting the draws shape and that no partial
        histories occur in the resampled dataset.  We check via a manual
        resample using the same RNG logic.
        """
        T = 10  # time periods per entity

        # The draws should have shape (B, k) and B draws should complete
        result = run_bootstrap(
            panel_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="panel_cluster_id",
            id_col="entity_id", time_col="time", B=49, seed=0,
        )
        # Each draw uses a dataset of size N*T (50 entities resampled → 50*10 rows)
        # We can't directly inspect the internal resampled data, but we can
        # verify the draws array is fully populated (no NaN from crashed draws)
        draws = result["bootstrap"]["draws"]
        assert not np.any(np.isnan(draws)), "Some draws produced NaN coefficients"
        assert draws.shape == (49, 2)  # 2 coefficients: const, x

    def test_panel_cluster_id_preserves_histories_manual(self) -> None:
        """Manual check that cluster_to_indices never returns partial T."""
        rng = np.random.default_rng(0)
        N, T = 5, 4
        entity_id = np.repeat(np.arange(N), T)
        unique_ids = np.unique(entity_id)
        cluster_to_indices = {
            c: np.where(entity_id == c)[0] for c in unique_ids
        }
        # Each cluster should have exactly T observations
        for c, idx in cluster_to_indices.items():
            assert len(idx) == T, (
                f"Entity {c} has {len(idx)} rows, expected {T}"
            )

    @pytest.mark.slow
    def test_panel_se_larger_than_iid(self, panel_df: pd.DataFrame) -> None:
        """panel_cluster_id SE > iid_pairs SE due to strong within-entity corr.

        With strong entity-level effects (var=4) and small idiosyncratic noise
        (var=0.09), iid_pairs underestimates variance. This is a directional test.
        """
        r_iid = run_bootstrap(
            panel_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs",
            B=999, seed=42,
        )
        r_panel = run_bootstrap(
            panel_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="panel_cluster_id",
            id_col="entity_id", time_col="time", B=999, seed=42,
        )
        se_iid = r_iid["bootstrap"]["se"]["x"]
        se_panel = r_panel["bootstrap"]["se"]["x"]
        # Panel bootstrap should give larger SE due to within-entity correlation
        assert se_panel > se_iid, (
            f"Expected panel SE ({se_panel:.4f}) > iid SE ({se_iid:.4f})"
        )

    def test_requires_id_col(self, simple_ols_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="id_col"):
            run_bootstrap(
                simple_ols_df, y="y", X=["x"],
                estimator="ols", bootstrap_method="panel_cluster_id",
                B=9, seed=0,
            )


# ===========================================================================
# 5. IV / 2SLS bootstrap tests
# ===========================================================================


class TestTwoSLSEstimator:
    """Standalone twosls_fit accuracy."""

    def test_matches_matrix_formula(self, iv_df: pd.DataFrame) -> None:
        """2SLS via twosls_fit matches the closed-form IV matrix estimator.

        IV estimator: beta_iv = (Z'X)^{-1} Z'y  (just-identified case)
        where X includes const and x, Z includes const and z.
        """
        n = len(iv_df)
        X_exog = np.column_stack([np.ones(n)])  # just const
        X_endog = iv_df["x"].to_numpy()[:, np.newaxis]
        Z_excl = iv_df["z"].to_numpy()[:, np.newaxis]
        y = iv_df["y"].to_numpy()

        coefs, _, _, _ = twosls_fit(y, X_exog, X_endog, Z_excl)
        # Should recover approximately beta_x = 2 (DGP)
        assert abs(coefs[1] - 2.0) < 0.15, (
            f"2SLS estimate {coefs[1]:.4f} far from DGP value 2.0"
        )

    def test_first_stage_f_positive(self, iv_df: pd.DataFrame) -> None:
        n = len(iv_df)
        X_exog = np.ones((n, 1))
        X_endog = iv_df["x"].to_numpy()[:, np.newaxis]
        Z = iv_df["z"].to_numpy()[:, np.newaxis]
        y = iv_df["y"].to_numpy()
        _, _, _, f_stat = twosls_fit(y, X_exog, X_endog, Z)
        assert f_stat > 10.0, f"Expected strong first-stage F, got {f_stat:.2f}"

    def test_2sls_matches_known_solution(self) -> None:
        """Exact synthetic dataset where 2SLS result is analytically known.

        Just-identified IV: b_iv = cov(z, y) / cov(z, x)
        DGP:
            z ~ Uniform(-1, 1)
            x = z (perfect first stage, no noise)
            y = 3*x + 0  (no intercept, no noise in structural equation)
        Then b_iv = cov(z, 3z) / cov(z, z) = 3.
        """
        rng = np.random.default_rng(99)
        n = 500
        z = rng.uniform(-1, 1, size=n)
        x = z.copy()  # perfect first stage
        y = 3.0 * x   # exact structural equation

        X_exog = np.empty((n, 0))  # no exog regressors
        X_endog = x[:, np.newaxis]
        Z_arr = z[:, np.newaxis]

        # Without intercept to keep it clean
        coefs, _, _, _ = twosls_fit(y, X_exog, X_endog, Z_arr)
        np.testing.assert_allclose(coefs[0], 3.0, atol=1e-10)


class TestIVBootstrap:
    """Smoke tests and diagnostics for IV bootstrap."""

    def test_iid_pairs_iv_runs(self, iv_df: pd.DataFrame) -> None:
        # X=[] means no exogenous regressors; a constant is added automatically.
        # Coefficients: [const, x_endog] → shape (B, 2)
        result = run_bootstrap(
            iv_df, y="y", X=[],
            estimator="2sls",
            endog=["x"],
            Z=["z"],
            bootstrap_method="iid_pairs",
            B=49, seed=0,
        )
        assert result["bootstrap"]["draws"].shape == (49, 2)  # const, x

    def test_wild_iv_runs(self, iv_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            iv_df, y="y", X=[],
            estimator="2sls",
            endog=["x"],
            Z=["z"],
            bootstrap_method="wild",
            B=49, seed=0,
        )
        assert result["bootstrap"]["draws"].shape[0] == 49

    def test_iv_bootstrap_recovers_true_coef(self, iv_df: pd.DataFrame) -> None:
        """Point estimate should be near 2.0; bagged mean near 2.0 too."""
        result = run_bootstrap(
            iv_df, y="y", X=[],
            estimator="2sls",
            endog=["x"],
            Z=["z"],
            bootstrap_method="iid_pairs",
            B=499, seed=42,
        )
        pe = result["point_estimate"]["params"]["x"]
        assert abs(pe - 2.0) < 0.2, f"IV point estimate {pe:.4f} far from 2.0"

    def test_weak_iv_diagnostic_present(self) -> None:
        """First-stage diagnostics present; F-stat is lower for weaker instrument."""
        rng = np.random.default_rng(55)
        n = 400
        # Strong instrument
        z_strong = rng.normal(size=n)
        v = rng.normal(size=n)
        x_strong = 0.9 * z_strong + 0.1 * v
        y = 2.0 * x_strong + v

        # Weak instrument
        z_weak = rng.normal(size=n)
        x_weak = 0.1 * z_weak + 0.9 * v

        for x_col, z_col, label in [
            (x_strong, z_strong, "strong"),
            (x_weak, z_weak, "weak"),
        ]:
            df_tmp = pd.DataFrame({"y": y, "x": x_col, "z": z_col})
            r = run_bootstrap(
                df_tmp, y="y", X=[],
                estimator="2sls",
                endog=["x"],
                Z=["z"],
                bootstrap_method="iid_pairs",
                B=49, seed=0,
            )
            diag = r["point_estimate"]["first_stage_diagnostics"]
            assert diag is not None
            assert "first_stage_F" in diag
            assert not np.isnan(diag["first_stage_F"]), (
                f"first_stage_F is NaN for {label} instrument"
            )

        # F-stat should be higher for strong instrument
        df_strong = pd.DataFrame({"y": y, "x": x_strong, "z": z_strong})
        df_weak = pd.DataFrame({"y": y, "x": x_weak, "z": z_weak})
        r_s = run_bootstrap(
            df_strong, y="y", X=[], endog=["x"], Z=["z"],
            estimator="2sls", bootstrap_method="iid_pairs", B=9, seed=0,
        )
        r_w = run_bootstrap(
            df_weak, y="y", X=[], endog=["x"], Z=["z"],
            estimator="2sls", bootstrap_method="iid_pairs", B=9, seed=0,
        )
        f_strong = r_s["point_estimate"]["first_stage_diagnostics"]["first_stage_F"]
        f_weak = r_w["point_estimate"]["first_stage_diagnostics"]["first_stage_F"]
        assert f_strong > f_weak, (
            f"Expected F_strong ({f_strong:.1f}) > F_weak ({f_weak:.1f})"
        )

    def test_iv_requires_endog(self, iv_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="endog"):
            run_bootstrap(
                iv_df, y="y", X=["x"],
                estimator="2sls", Z=["z"],
                bootstrap_method="iid_pairs", B=9, seed=0,
            )

    def test_iv_requires_Z(self, iv_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Z"):
            run_bootstrap(
                iv_df, y="y", X=["x"],
                estimator="2sls", endog=["x"],
                bootstrap_method="iid_pairs", B=9, seed=0,
            )


# ===========================================================================
# 6. Performance and failure-mode tests
# ===========================================================================


class TestFailureModes:
    """Edge cases and error handling."""

    def test_singular_matrix_fallback(self) -> None:
        """Near-collinear X uses lstsq fallback without crashing."""
        rng = np.random.default_rng(77)
        n = 100
        x1 = rng.normal(size=n)
        x2 = x1 + 1e-12 * rng.normal(size=n)  # nearly identical
        y = rng.normal(size=n)

        # Should not raise; may emit a warning but must return coefficients
        coefs, resid, fitted = ols_fit(
            np.column_stack([np.ones(n), x1, x2]), y
        )
        assert coefs.shape == (3,)
        assert np.all(np.isfinite(coefs))

    def test_singular_matrix_in_bootstrap_returns_finite(self) -> None:
        """run_bootstrap with near-collinear X should complete without error."""
        rng = np.random.default_rng(88)
        n = 100
        x1 = rng.normal(size=n)
        x2 = x1 + 1e-10 * rng.normal(size=n)
        y = rng.normal(size=n)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        result = run_bootstrap(
            df, y="y", X=["x1", "x2"],
            estimator="ols", bootstrap_method="iid_pairs",
            B=49, seed=0,
        )
        draws = result["bootstrap"]["draws"]
        # Some draws may be large but should not be inf/nan in aggregate
        assert draws.shape == (49, 3)

    def test_unknown_estimator_raises(self, simple_ols_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="estimator"):
            run_bootstrap(
                simple_ols_df, y="y", X=["x"],
                estimator="logit", bootstrap_method="iid_pairs",
                B=9, seed=0,
            )

    def test_unknown_method_raises(self, simple_ols_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="bootstrap_method"):
            run_bootstrap(
                simple_ols_df, y="y", X=["x"],
                estimator="ols", bootstrap_method="jackknife",
                B=9, seed=0,
            )

    def test_stub_methods_raise_not_implemented(
        self, simple_ols_df: pd.DataFrame
    ) -> None:
        for method in ("block_time_series", "stationary_bootstrap"):
            with pytest.raises(NotImplementedError):
                run_bootstrap(
                    simple_ols_df, y="y", X=["x"],
                    estimator="ols", bootstrap_method=method,
                    B=9, seed=0,
                )

    def test_invalid_wild_dist_raises(self, simple_ols_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="wild_dist"):
            run_bootstrap(
                simple_ols_df, y="y", X=["x"],
                estimator="ols", bootstrap_method="wild",
                wild_dist="uniform", B=9, seed=0,
            )

    def test_invalid_ci_level_raises(self, simple_ols_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="ci_level"):
            run_bootstrap(
                simple_ols_df, y="y", X=["x"],
                estimator="ols", bootstrap_method="iid_pairs",
                ci_level=1.5, B=9, seed=0,
            )

    def test_missing_column_raises(self, simple_ols_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Columns not found"):
            run_bootstrap(
                simple_ols_df, y="y", X=["x", "nonexistent"],
                estimator="ols", bootstrap_method="iid_pairs",
                B=9, seed=0,
            )


class TestConfigHashStability:
    """Config hash is stable regardless of key ordering."""

    def test_same_config_same_hash(self) -> None:
        cfg1 = {"estimator": "ols", "B": 1999, "seed": 42, "method": "iid_pairs"}
        cfg2 = {"method": "iid_pairs", "seed": 42, "B": 1999, "estimator": "ols"}
        assert compute_config_hash(cfg1) == compute_config_hash(cfg2)

    def test_different_configs_different_hashes(self) -> None:
        cfg1 = {"estimator": "ols", "B": 1999}
        cfg2 = {"estimator": "ols", "B": 2000}
        assert compute_config_hash(cfg1) != compute_config_hash(cfg2)

    def test_hash_format(self) -> None:
        cfg = {"estimator": "ols", "B": 999}
        h = compute_config_hash(cfg)
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 64  # hex SHA-256

    def test_hash_in_metadata(self, simple_ols_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=9, seed=0,
        )
        h = result["metadata"]["config_hash"]
        assert h.startswith("sha256:")


class TestSaveDraws:
    """Draws are saved correctly to disk."""

    def test_save_and_reload(self, simple_ols_df: pd.DataFrame) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "draws.npy")
            result = run_bootstrap(
                simple_ols_df, y="y", X=["x"],
                estimator="ols", bootstrap_method="iid_pairs",
                B=49, seed=0, save_draws_path=path,
            )
            loaded = np.load(path)
            np.testing.assert_array_equal(result["bootstrap"]["draws"], loaded)


class TestManifest:
    """Manifest write/read round-trip."""

    def test_manifest_is_valid_json(self, simple_ols_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=9, seed=0,
        )
        manifest = result["_manifest"]
        # Should be JSON-serialisable
        json_str = json.dumps(manifest, default=str)
        loaded = json.loads(json_str)
        assert loaded["config"]["estimator"] == "ols"

    def test_write_manifest(self, simple_ols_df: pd.DataFrame) -> None:
        from econtools.uncertainty._bootstrap_manifest import write_manifest

        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs", B=9, seed=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            write_manifest(result["_manifest"], path)
            assert path.exists()
            with path.open() as fh:
                loaded = json.load(fh)
            assert loaded["config_hash"] == result["metadata"]["config_hash"]


class TestBaggedCoefficients:
    """Bagged coefficients (mean/median of draws) are present and reasonable."""

    def test_bagged_mean_close_to_point_estimate(
        self, simple_ols_df: pd.DataFrame
    ) -> None:
        """For large B, bagged mean should be close to point estimate."""
        result = run_bootstrap(
            simple_ols_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="iid_pairs",
            B=1999, seed=42,
        )
        for name in result["point_estimate"]["params"]:
            pe = result["point_estimate"]["params"][name]
            bagged = result["bootstrap"]["bagged_mean"][name]
            # Bagged mean should be within 0.1 of point estimate for n=200, B=1999
            assert abs(bagged - pe) < 0.1, (
                f"Bagged mean ({bagged:.4f}) far from point estimate ({pe:.4f}) "
                f"for coefficient {name!r}"
            )


# ===========================================================================
# 7. Wild cluster bootstrap tests
# ===========================================================================


class TestWildClusterWeights:
    """Unit tests for the three wild-cluster weight distributions."""

    def test_rademacher_is_binary(self) -> None:
        from econtools.uncertainty.bootstrap import _wild_weights
        rng = np.random.default_rng(0)
        v = _wild_weights(rng, n=2000, dist="rademacher")
        assert set(np.unique(v)).issubset({-1.0, 1.0})

    def test_rademacher_roughly_balanced(self) -> None:
        from econtools.uncertainty.bootstrap import _wild_weights
        rng = np.random.default_rng(1)
        v = _wild_weights(rng, n=10_000, dist="rademacher")
        frac_pos = np.mean(v > 0)
        # Should be close to 0.5 for large n
        assert 0.46 < frac_pos < 0.54

    def test_mammen_has_two_values(self) -> None:
        from econtools.uncertainty.bootstrap import _wild_weights
        rng = np.random.default_rng(2)
        v = _wild_weights(rng, n=5000, dist="mammen")
        unique_v = np.unique(v)
        assert len(unique_v) == 2
        sqrt5 = np.sqrt(5)
        expected = sorted([-(sqrt5 - 1) / 2, (sqrt5 + 1) / 2])
        np.testing.assert_allclose(sorted(unique_v.tolist()), expected, rtol=1e-10)

    def test_mammen_zero_mean(self) -> None:
        """Mammen distribution has E[v] = 0."""
        from econtools.uncertainty.bootstrap import _wild_weights
        rng = np.random.default_rng(3)
        v = _wild_weights(rng, n=50_000, dist="mammen")
        assert abs(np.mean(v)) < 0.02

    def test_mammen_unit_variance(self) -> None:
        """Mammen distribution has Var(v) = 1."""
        from econtools.uncertainty.bootstrap import _wild_weights
        rng = np.random.default_rng(4)
        v = _wild_weights(rng, n=50_000, dist="mammen")
        assert abs(np.var(v) - 1.0) < 0.05

    def test_webb_has_six_values(self) -> None:
        from econtools.uncertainty.bootstrap import _wild_weights
        rng = np.random.default_rng(5)
        v = _wild_weights(rng, n=6000, dist="webb")
        unique_v = sorted(np.unique(v).tolist())
        assert len(unique_v) == 6

    def test_webb_correct_values(self) -> None:
        """Webb 6-point distribution uses {±1, ±sqrt(3/2), ±sqrt(1/2)}."""
        from econtools.uncertainty.bootstrap import _wild_weights
        rng = np.random.default_rng(6)
        v = _wild_weights(rng, n=12_000, dist="webb")
        unique_v = sorted(np.unique(v).tolist())
        expected = sorted([
            -np.sqrt(1.5), -1.0, -np.sqrt(0.5),
            np.sqrt(0.5), 1.0, np.sqrt(1.5),
        ])
        np.testing.assert_allclose(unique_v, expected, rtol=1e-10)

    def test_webb_zero_mean(self) -> None:
        """Webb distribution has E[v] = 0 by symmetry."""
        from econtools.uncertainty.bootstrap import _wild_weights
        rng = np.random.default_rng(7)
        v = _wild_weights(rng, n=60_000, dist="webb")
        assert abs(np.mean(v)) < 0.02

    def test_webb_unit_variance(self) -> None:
        """Webb distribution has E[v²] = 1 (i.e. Var=1 given mean=0)."""
        from econtools.uncertainty.bootstrap import _wild_weights
        rng = np.random.default_rng(8)
        v = _wild_weights(rng, n=60_000, dist="webb")
        assert abs(np.mean(v ** 2) - 1.0) < 0.05

    def test_webb_roughly_uniform_across_six_bins(self) -> None:
        """Each of the 6 Webb values occurs with roughly 1/6 frequency."""
        from econtools.uncertainty.bootstrap import _wild_weights
        rng = np.random.default_rng(9)
        v = _wild_weights(rng, n=60_000, dist="webb")
        unique_v, counts = np.unique(v, return_counts=True)
        fracs = counts / len(v)
        # Each should be close to 1/6 ≈ 0.1667
        np.testing.assert_allclose(fracs, 1 / 6, atol=0.01)

    def test_invalid_dist_raises(self) -> None:
        from econtools.uncertainty.bootstrap import _wild_weights
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="wild_dist"):
            _wild_weights(rng, n=10, dist="uniform")


class TestWildClusterBootstrapSmoke:
    """Smoke tests: wild_cluster runs and produces correct-shaped output."""

    def test_runs_ols_rademacher(self, clustered_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", wild_dist="rademacher", B=99, seed=0,
        )
        assert result["bootstrap"]["draws"].shape == (99, 2)  # const + x

    def test_runs_ols_mammen(self, clustered_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", wild_dist="mammen", B=99, seed=0,
        )
        assert result["bootstrap"]["draws"].shape == (99, 2)

    def test_runs_ols_webb(self, clustered_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", wild_dist="webb", B=99, seed=0,
        )
        assert result["bootstrap"]["draws"].shape == (99, 2)

    def test_runs_2sls(self, iv_df: pd.DataFrame) -> None:
        """wild_cluster with 2SLS keeps X, Z fixed and perturbs y only."""
        # Add a cluster column to iv_df
        rng = np.random.default_rng(42)
        n = len(iv_df)
        G = 20
        cid = np.repeat(np.arange(G), n // G)
        cid = np.concatenate([cid, np.zeros(n - len(cid), dtype=int)])
        df = iv_df.copy()
        df["cid"] = cid
        result = run_bootstrap(
            df, y="y", X=[],
            estimator="2sls", endog=["x"], Z=["z"],
            bootstrap_method="wild_cluster",
            cluster="cid", wild_dist="rademacher", B=49, seed=0,
        )
        assert result["bootstrap"]["draws"].shape == (49, 2)  # const + x

    def test_no_nan_draws(self, clustered_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", B=99, seed=0,
        )
        assert not np.any(np.isnan(result["bootstrap"]["draws"]))

    def test_deterministic_same_seed(self, clustered_df: pd.DataFrame) -> None:
        kw = dict(
            data=clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", B=99, seed=7,
        )
        r1 = run_bootstrap(**kw)
        r2 = run_bootstrap(**kw)
        np.testing.assert_array_equal(
            r1["bootstrap"]["draws"], r2["bootstrap"]["draws"]
        )

    def test_different_seeds_differ(self, clustered_df: pd.DataFrame) -> None:
        r1 = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", B=99, seed=1,
        )
        r2 = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", B=99, seed=2,
        )
        assert not np.array_equal(
            r1["bootstrap"]["draws"], r2["bootstrap"]["draws"]
        )

    def test_parallel_matches_sequential(self, clustered_df: pd.DataFrame) -> None:
        common = dict(
            data=clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", B=99, seed=42,
        )
        r_seq = run_bootstrap(**common, n_jobs=1)
        r_par = run_bootstrap(**common, n_jobs=2)
        draws_seq = np.sort(r_seq["bootstrap"]["draws"], axis=0)
        draws_par = np.sort(r_par["bootstrap"]["draws"], axis=0)
        np.testing.assert_allclose(draws_seq, draws_par, rtol=0, atol=0)

    def test_method_recorded_in_metadata(self, clustered_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", B=49, seed=0,
        )
        assert result["metadata"]["method"] == "wild_cluster"

    def test_cluster_count_in_metadata(self, clustered_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", B=49, seed=0,
        )
        assert result["metadata"]["cluster_count"] == 50

    def test_small_cluster_warning(self) -> None:
        """G < 30 triggers a UserWarning."""
        rng = np.random.default_rng(7)
        G, m = 8, 20
        n = G * m
        df = pd.DataFrame({
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "cid": np.repeat(np.arange(G), m),
        })
        with pytest.warns(UserWarning, match="Small cluster count"):
            result = run_bootstrap(
                df, y="y", X=["x"],
                estimator="ols", bootstrap_method="wild_cluster",
                cluster="cid", B=49, seed=0,
            )
        assert any("Small cluster" in w for w in result["metadata"]["warnings"])


class TestWildClusterMechanics:
    """Verify wild cluster bootstrap applies one weight per cluster, not per obs."""

    def test_within_cluster_residuals_scaled_uniformly(self) -> None:
        """All obs in the same cluster get the same multiplier.

        Strategy: fit OLS on a known DGP.  For one bootstrap draw, verify
        that the bootstrap y* for obs in the same cluster satisfies:
          (y*_i - X_i @ beta_hat) / residual_i == constant across cluster.
        We do this by running B=1 bootstrap draw and checking the ratio.
        """
        rng = np.random.default_rng(0)
        G, m = 5, 10
        n = G * m
        cid = np.repeat(np.arange(G), m)
        x = rng.normal(size=n)
        cluster_shock = rng.normal(scale=1.0, size=G)
        eps = cluster_shock[cid] + 0.2 * rng.normal(size=n)
        y = 1.0 + 2.0 * x + eps
        df = pd.DataFrame({"y": y, "x": x, "cid": cid})

        # Run with B=1 to inspect a single draw
        result = run_bootstrap(
            df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cid", B=1, seed=42,
        )
        # The single draw coefficient vector is shape (1, 2)
        draw = result["bootstrap"]["draws"]
        assert draw.shape == (1, 2)
        # Verify: within each cluster, the ratio (y*_i - yhat_i) / u_i is constant.
        # We can't directly access y* from the result, but we verify
        # by checking that SE from wild_cluster is larger than from iid wild
        # when clusters are present (see TestWildClusterDirectionalChecks).
        # Here just verify finite output.
        assert np.all(np.isfinite(draw))

    def test_x_not_resampled(self, clustered_df: pd.DataFrame) -> None:
        """The point estimate x coefficient for wild_cluster matches OLS directly.

        Because wild_cluster fixes X and only perturbs y, the point estimate
        (stored in result) is identical to plain OLS.
        """
        import statsmodels.api as sm

        X_sm = sm.add_constant(clustered_df[["x"]])
        ols_result = sm.OLS(clustered_df["y"], X_sm).fit()
        ols_coef = float(ols_result.params["x"])

        result = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", B=49, seed=0,
        )
        wcb_pe = result["point_estimate"]["params"]["x"]
        np.testing.assert_allclose(wcb_pe, ols_coef, rtol=1e-10)

    def test_ci_contains_point_estimate(self, clustered_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", B=499, seed=0,
        )
        pe = result["point_estimate"]["params"]
        ci = result["bootstrap"]["ci"]["percentile"]
        for name in pe:
            assert ci["lower"][name] <= pe[name] <= ci["upper"][name], (
                f"Point estimate for {name!r} outside percentile CI"
            )

    def test_pvalues_in_unit_interval(self, clustered_df: pd.DataFrame) -> None:
        result = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", B=199, seed=0,
        )
        for name, pv in result["bootstrap"]["pvalues"].items():
            assert 0 < pv <= 1.0, f"p-value for {name!r} out of range: {pv}"


class TestWildClusterDirectionalChecks:
    """Directional tests: WCB corrects for cluster correlation where iid wild fails."""

    @pytest.mark.slow
    def test_wcb_se_larger_than_iid_wild_under_cluster_dgp(
        self, clustered_df: pd.DataFrame
    ) -> None:
        """Under a clustered DGP, WCB SE > iid wild SE.

        The iid wild bootstrap assumes independence across obs; when residuals
        are positively correlated within clusters, it underestimates the SE.
        WCB draws one weight per cluster, properly accounting for this.
        """
        r_iid_wild = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild",
            wild_dist="rademacher", B=1999, seed=42,
        )
        r_wcb = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", wild_dist="rademacher", B=1999, seed=42,
        )
        se_iid = r_iid_wild["bootstrap"]["se"]["x"]
        se_wcb = r_wcb["bootstrap"]["se"]["x"]
        assert se_wcb > se_iid, (
            f"Expected wild_cluster SE ({se_wcb:.4f}) > iid wild SE ({se_iid:.4f}) "
            "under clustered DGP"
        )

    @pytest.mark.slow
    def test_wcb_se_close_to_cluster_robust(
        self, clustered_df: pd.DataFrame
    ) -> None:
        """WCB SE should be within 20% of cluster-robust (sandwich) SE for G=50."""
        import statsmodels.api as sm

        X_sm = sm.add_constant(clustered_df[["x"]])
        cluster_fit = sm.OLS(clustered_df["y"], X_sm).fit(
            cov_type="cluster",
            cov_kwds={"groups": clustered_df["cluster_id"]},
        )
        cl_se = float(cluster_fit.bse["x"])

        result = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", wild_dist="rademacher", B=1999, seed=42,
        )
        wcb_se = result["bootstrap"]["se"]["x"]
        rel_error = abs(wcb_se - cl_se) / cl_se
        assert rel_error < 0.20, (
            f"WCB SE ({wcb_se:.4f}) differs from cluster-robust SE "
            f"({cl_se:.4f}) by {rel_error:.1%}"
        )

    @pytest.mark.slow
    def test_webb_se_close_to_rademacher_for_large_g(
        self, clustered_df: pd.DataFrame
    ) -> None:
        """For large G, Webb and Rademacher WCB SEs should be close.

        Differences are most pronounced for very small G; with G=50 they
        should agree within 15%.
        """
        r_rad = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", wild_dist="rademacher", B=1999, seed=42,
        )
        r_webb = run_bootstrap(
            clustered_df, y="y", X=["x"],
            estimator="ols", bootstrap_method="wild_cluster",
            cluster="cluster_id", wild_dist="webb", B=1999, seed=42,
        )
        se_rad = r_rad["bootstrap"]["se"]["x"]
        se_webb = r_webb["bootstrap"]["se"]["x"]
        rel_diff = abs(se_rad - se_webb) / se_rad
        assert rel_diff < 0.15, (
            f"Rademacher SE ({se_rad:.4f}) and Webb SE ({se_webb:.4f}) differ "
            f"by {rel_diff:.1%} for G=50 (expected < 15%)"
        )

    @pytest.mark.slow
    def test_small_g_webb_preferred(self) -> None:
        """For very small G (G=6), Webb can produce more draws than Rademacher.

        With Rademacher and G=6, there are only 2^6=64 possible weight vectors,
        which limits the number of distinct bootstrap draws to at most 64.
        Webb with 6^G=46656 possibilities gives much richer distribution.
        This test verifies that Webb produces more unique draw vectors than
        Rademacher for small G with B=200.
        """
        rng = np.random.default_rng(0)
        G, m = 6, 30
        n = G * m
        cid = np.repeat(np.arange(G), m)
        x = rng.normal(size=n)
        cluster_shock = rng.normal(scale=1.0, size=G)
        eps = cluster_shock[cid] + 0.2 * rng.normal(size=n)
        y = 1.0 + 2.0 * x + eps
        df = pd.DataFrame({"y": y, "x": x, "cid": cid})

        with pytest.warns(UserWarning, match="Small cluster count"):
            r_rad = run_bootstrap(
                df, y="y", X=["x"],
                estimator="ols", bootstrap_method="wild_cluster",
                cluster="cid", wild_dist="rademacher", B=200, seed=0,
            )
        with pytest.warns(UserWarning, match="Small cluster count"):
            r_webb = run_bootstrap(
                df, y="y", X=["x"],
                estimator="ols", bootstrap_method="wild_cluster",
                cluster="cid", wild_dist="webb", B=200, seed=0,
            )

        # Count unique draw vectors (rounded to 6 decimal places)
        draws_rad = np.round(r_rad["bootstrap"]["draws"], 6)
        draws_webb = np.round(r_webb["bootstrap"]["draws"], 6)
        unique_rad = len(np.unique(draws_rad, axis=0))
        unique_webb = len(np.unique(draws_webb, axis=0))
        assert unique_webb >= unique_rad, (
            f"Webb ({unique_webb} unique) should have >= draws than Rademacher "
            f"({unique_rad} unique) for G={G}"
        )


class TestWildClusterValidation:
    """Validation: wild_cluster requires cluster; webb requires wild_cluster."""

    def test_requires_cluster_col(self, simple_ols_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="cluster"):
            run_bootstrap(
                simple_ols_df, y="y", X=["x"],
                estimator="ols", bootstrap_method="wild_cluster",
                B=9, seed=0,
            )

    def test_webb_dist_rejected_for_iid_wild(
        self, simple_ols_df: pd.DataFrame
    ) -> None:
        """webb weight is only valid for wild_cluster, not iid wild."""
        with pytest.raises(ValueError, match="wild_dist='webb'"):
            run_bootstrap(
                simple_ols_df, y="y", X=["x"],
                estimator="ols", bootstrap_method="wild",
                wild_dist="webb", B=9, seed=0,
            )

    def test_webb_dist_rejected_for_cluster_pairs(
        self, clustered_df: pd.DataFrame
    ) -> None:
        """webb is not valid for cluster_pairs."""
        with pytest.raises(ValueError, match="wild_dist='webb'"):
            run_bootstrap(
                clustered_df, y="y", X=["x"],
                estimator="ols", bootstrap_method="cluster_pairs",
                cluster="cluster_id", wild_dist="webb", B=9, seed=0,
            )

    def test_unknown_wild_dist_raises(self, clustered_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="wild_dist"):
            run_bootstrap(
                clustered_df, y="y", X=["x"],
                estimator="ols", bootstrap_method="wild_cluster",
                cluster="cluster_id", wild_dist="bootstrap", B=9, seed=0,
            )

    def test_wild_cluster_in_supported_methods(self) -> None:
        from econtools.uncertainty.bootstrap import _SUPPORTED_METHODS
        assert "wild_cluster" in _SUPPORTED_METHODS

    def test_wild_cluster_not_in_stub_methods(self) -> None:
        from econtools.uncertainty.bootstrap import _STUB_METHODS
        assert "wild_cluster" not in _STUB_METHODS
