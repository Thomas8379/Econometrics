"""Figure-returning plot functions — Phase 1+."""

from econtools.plots.residual_plots import (
    plot_residuals_vs_fitted,
    plot_scale_location,
    plot_qq,
)
from econtools.plots.coefficient_plots import plot_coef_forest

__all__ = [
    "plot_residuals_vs_fitted",
    "plot_scale_location",
    "plot_qq",
    "plot_coef_forest",
]
