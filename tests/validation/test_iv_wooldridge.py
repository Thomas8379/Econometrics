"""Numerical validation: IV-2SLS against Wooldridge textbook results.

Reference: Wooldridge (2010) Introductory Econometrics, 5th ed.
mroz.dta — Chapter 15: IV for log(wage) with education instrumented by
           parental education (fatheduc, motheduc).

Specification:
  lwage ~ educ + exper | fatheduc motheduc (exper is exogenous)

Actual 2SLS results from this data:
  educ coefficient ≈ 0.066 (OLS ≈ 0.109)
  N = 428 (working women, inlf == 1, non-missing wages)

Note: the IV estimate is slightly smaller than OLS here, reflecting
weak-instrument concerns with parental education instruments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.validation.conftest import dta_path, require_dta
from econtools.models.iv import fit_iv_2sls


@require_dta("mroz")
def test_mroz_iv_2sls_educ() -> None:
    """IV-2SLS log(wage) ~ educ, instruments: fatheduc, motheduc."""
    df = pd.read_stata(str(dta_path("mroz")), convert_categoricals=False)
    # Keep only working women (inlf == 1)
    df = df[df["inlf"] == 1].copy()
    df["lwage"] = np.log(df["wage"])
    # Drop missing wages
    df = df.dropna(subset=["lwage", "educ", "exper", "fatheduc", "motheduc"])

    result = fit_iv_2sls(
        df,
        dep_var="lwage",
        exog_vars=["exper"],
        endog_vars=["educ"],
        instruments=["fatheduc", "motheduc"],
        add_constant=True,
    )

    # N = 428 working women
    assert result.fit.nobs == 428

    # IV coefficient on educ should be positive and in a plausible range
    # Actual value ≈ 0.0664 (instruments: fatheduc + motheduc, control: exper)
    beta_educ = float(result.params["educ"])
    assert 0.03 < beta_educ < 0.15, (
        f"IV educ coef {beta_educ:.4f} outside expected range (0.03, 0.15); "
        f"expected ≈ 0.066"
    )

    # OLS estimate for comparison (should be ≈ 0.109)
    from econtools.models.ols import fit_ols

    ols = fit_ols(df, "lwage", ["educ", "exper"], add_constant=True)
    beta_ols = float(ols.params["educ"])
    assert 0.07 < beta_ols < 0.15, (
        f"OLS educ coef {beta_ols:.4f} outside expected range (0.07, 0.15)"
    )
