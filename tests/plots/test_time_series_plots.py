from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend — must come before any other matplotlib import

import numpy as np

from econtools.plots.time_series import (
    plot_correlogram,
    plot_distribution,
    plot_series_with_trend,
    plot_residual_diagnostics,
    plot_series_with_trends,
    plot_time_series,
)


def test_plot_time_series_returns_figure() -> None:
    y = np.arange(10, dtype=float)
    fig = plot_time_series(y)
    assert fig is not None


def test_plot_correlogram_returns_figure() -> None:
    y = np.arange(20, dtype=float)
    fig = plot_correlogram(y, lags=5, ci_method="bartlett", ma_q=0)
    assert fig is not None


def test_plot_distribution_returns_figure() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(size=100)
    fig = plot_distribution(y, bins=10)
    assert fig is not None


def test_plot_series_with_trend_returns_figure() -> None:
    y = np.arange(10, dtype=float)
    trend = y + 1.0
    fig = plot_series_with_trend(y, trend)
    assert fig is not None


def test_plot_residual_diagnostics_returns_figs() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(size=50)
    figs = plot_residual_diagnostics(y, lags=5)
    assert set(figs.keys()) == {"series", "correlogram", "distribution"}


def test_plot_series_with_trends_returns_figure() -> None:
    y = np.arange(10, dtype=float)
    trends = {"linear": y + 1.0, "quadratic": y + 2.0}
    fig = plot_series_with_trends(y, trends)
    assert fig is not None
