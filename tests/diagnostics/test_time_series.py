from __future__ import annotations

import numpy as np

from econtools.diagnostics.time_series import (
    granger_causality,
    lead_exogeneity_test,
)


def test_granger_causality_runs() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(size=80)
    x = rng.normal(size=80)
    res = granger_causality(y, x, maxlags=2, ic=None)
    assert res.test_name.startswith("Granger")
    # Value assertions
    assert 0.0 <= res.pvalue <= 1.0
    assert res.statistic >= 0.0


def test_lead_exogeneity_runs() -> None:
    rng = np.random.default_rng(99)
    y = rng.normal(size=80)
    x = rng.normal(size=80)
    res = lead_exogeneity_test(y, x, lead=1, lags=1, add_trend=True)
    assert res.test_name.startswith("Lead exogeneity")
    # Value assertions
    assert 0.0 <= res.pvalue <= 1.0
    assert res.statistic >= 0.0
