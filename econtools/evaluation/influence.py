"""Influence diagnostics — Phase 3.

TODO(econtools): adapter — implement Cook's D, DFFITS, DFBETAs (Phase 3)
"""

from __future__ import annotations

import pandas as pd

from econtools._core.types import RegressionResult


def cooks_distance(result: RegressionResult) -> pd.Series:
    """Cook's distance for each observation.

    TODO(econtools): adapter — implement Cook's D using hat matrix (Phase 3)
    """
    raise NotImplementedError("Cook's distance is Phase 3 — not yet implemented.")


def dffits(result: RegressionResult) -> pd.Series:
    """DFFITS influence measure for each observation.

    TODO(econtools): adapter — implement DFFITS (Phase 3)
    """
    raise NotImplementedError("DFFITS is Phase 3 — not yet implemented.")


def dfbetas(result: RegressionResult) -> pd.DataFrame:
    """DFBETAs influence measure for each observation and coefficient.

    TODO(econtools): adapter — implement DFBETAs (Phase 3)
    """
    raise NotImplementedError("DFBETAs are Phase 3 — not yet implemented.")
