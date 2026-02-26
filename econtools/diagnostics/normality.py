"""Normality test for regression residuals.

Public API
----------
jarque_bera(result) -> TestResult
"""

from __future__ import annotations

from statsmodels.stats.stattools import jarque_bera as _sm_jarque_bera

from econtools.inference.hypothesis import TestResult
from econtools.models._results import RegressionResult


def jarque_bera(result: RegressionResult) -> TestResult:
    """Jarque-Bera test for normality of residuals.

    H₀: residuals are normally distributed (skewness=0, excess kurtosis=0).

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.

    Returns
    -------
    TestResult with JB statistic and p-value (Chi²(2) distribution).
    """
    jb_stat, jb_pval, _skew, _kurtosis = _sm_jarque_bera(result.resid)
    return TestResult(
        test_name="Jarque-Bera",
        statistic=float(jb_stat),
        pvalue=float(jb_pval),
        df=2,
        distribution="Chi2",
        null_hypothesis="Residuals are normally distributed",
        reject=float(jb_pval) < 0.05,
    )
