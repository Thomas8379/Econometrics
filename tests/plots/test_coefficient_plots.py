"""Tests for econtools.plots.coefficient_plots."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pytest

from econtools.plots import plot_coef_forest


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_coef_forest_returns_figure(ols_result) -> None:
    fig = plot_coef_forest(ols_result)
    assert isinstance(fig, Figure)


def test_plot_coef_forest_excludes_const_by_default(ols_result) -> None:
    fig = plot_coef_forest(ols_result, exclude_const=True)
    ax = fig.axes[0]
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "const" not in tick_labels
    assert "Intercept" not in tick_labels


def test_plot_coef_forest_includes_const_when_requested(ols_result) -> None:
    fig = plot_coef_forest(ols_result, exclude_const=False)
    ax = fig.axes[0]
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "const" in tick_labels
