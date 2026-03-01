"""Heteroskedasticity tests — re-export from diagnostics.heteroskedasticity."""

from econtools.diagnostics.heteroskedasticity import (  # noqa: F401
    breusch_pagan,
    white_test,
)

__all__ = ["breusch_pagan", "white_test"]
