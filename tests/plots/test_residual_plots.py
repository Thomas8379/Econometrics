"""Tests for econtools.plots.residual_plots."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend for CI

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pytest

from econtools.plots import plot_residuals_vs_fitted, plot_scale_location, plot_qq


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test to avoid resource warnings."""
    yield
    plt.close("all")


def test_plot_residuals_vs_fitted_returns_figure(ols_result) -> None:
    fig = plot_residuals_vs_fitted(ols_result)
    assert isinstance(fig, Figure)


def test_plot_residuals_vs_fitted_has_axes(ols_result) -> None:
    fig = plot_residuals_vs_fitted(ols_result)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Fitted values"
    assert ax.get_ylabel() == "Residuals"


def test_plot_scale_location_returns_figure(ols_result) -> None:
    fig = plot_scale_location(ols_result)
    assert isinstance(fig, Figure)


def test_plot_qq_returns_figure(ols_result) -> None:
    fig = plot_qq(ols_result)
    assert isinstance(fig, Figure)


def test_plot_qq_has_axes(ols_result) -> None:
    fig = plot_qq(ols_result)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert "Q-Q" in ax.get_title()
