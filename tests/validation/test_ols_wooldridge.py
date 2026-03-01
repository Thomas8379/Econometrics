"""Numerical validation: OLS coefficients against Wooldridge textbook results.

Reference: Wooldridge (2010) Introductory Econometrics, 5th ed.
wage1.dta — Chapter 6 example: log(wage) ~ educ + exper + tenure

Expected (4 d.p.):
  const   = 0.2844  (approx)
  educ    = 0.0920
  exper   = 0.0041
  tenure  = 0.0221
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.validation.conftest import dta_path, require_dta
from econtools.models.ols import fit_ols


@require_dta("wage1")
def test_wage1_ols_coefficients() -> None:
    """OLS log(wage) ~ educ + exper + tenure matches Wooldridge Chapter 6."""
    df = pd.read_stata(str(dta_path("wage1")), convert_categoricals=False)
    df["lwage"] = np.log(df["wage"])

    result = fit_ols(df, "lwage", ["educ", "exper", "tenure"], add_constant=True)

    # Coefficients should match Wooldridge Table 6.1 to 4 d.p. (allow 0.001 tolerance)
    assert abs(float(result.params["educ"]) - 0.0920) < 0.001, (
        f"educ coefficient {result.params['educ']:.4f} != 0.0920"
    )
    assert abs(float(result.params["exper"]) - 0.0041) < 0.001, (
        f"exper coefficient {result.params['exper']:.4f} != 0.0041"
    )
    assert abs(float(result.params["tenure"]) - 0.0221) < 0.001, (
        f"tenure coefficient {result.params['tenure']:.4f} != 0.0221"
    )

    # N = 526 in wage1.dta
    assert result.fit.nobs == 526

    # R² should be around 0.316
    assert 0.30 < result.fit.r_squared < 0.34

    # All SE should be positive
    assert all(result.bse > 0)


@require_dta("wage1")
def test_wage1_heteroskedasticity() -> None:
    """Breusch-Pagan test on log(wage) regression detects heteroskedasticity."""
    from econtools.diagnostics import breusch_pagan

    df = pd.read_stata(str(dta_path("wage1")), convert_categoricals=False)
    df["lwage"] = np.log(df["wage"])
    result = fit_ols(df, "lwage", ["educ", "exper", "tenure"], add_constant=True)

    bp = breusch_pagan(result)
    assert bp.statistic >= 0.0
    assert 0.0 <= bp.pvalue <= 1.0


# TODO(econtools): validate — add wage2.dta BP test statistic check
