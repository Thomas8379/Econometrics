"""Heteroskedasticity tests: Breusch-Pagan and White.

Public API
----------
breusch_pagan(result) -> TestResult
white_test(result)    -> TestResult
"""

from __future__ import annotations

import statsmodels.stats.diagnostic as smsd

from econtools.inference.hypothesis import TestResult
from econtools.models._results import RegressionResult


def breusch_pagan(result: RegressionResult) -> TestResult:
    """Breusch-Pagan test for heteroskedasticity (LM version).

    H₀: homoskedasticity (errors have constant variance).

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.

    Returns
    -------
    TestResult with LM statistic and p-value.
    """
    lm_stat, lm_pval, _f_stat, _f_pval = smsd.het_breuschpagan(
        result.resid, result.raw.model.exog
    )
    return TestResult(
        test_name="Breusch-Pagan",
        statistic=float(lm_stat),
        pvalue=float(lm_pval),
        df=None,
        distribution="Chi2",
        null_hypothesis="Homoskedasticity",
        reject=float(lm_pval) < 0.05,
    )


def white_test(result: RegressionResult) -> TestResult:
    """White test for heteroskedasticity (LM version).

    H₀: homoskedasticity.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.

    Returns
    -------
    TestResult with LM statistic and p-value.
    """
    lm_stat, lm_pval, _f_stat, _f_pval = smsd.het_white(
        result.resid, result.raw.model.exog
    )
    return TestResult(
        test_name="White",
        statistic=float(lm_stat),
        pvalue=float(lm_pval),
        df=None,
        distribution="Chi2",
        null_hypothesis="Homoskedasticity",
        reject=float(lm_pval) < 0.05,
    )
