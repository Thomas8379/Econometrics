"""Specification tests for functional form.

Public API
----------
reset_test(result, power=3, use_f=True) -> TestResult
"""

from __future__ import annotations

import numpy as np
import statsmodels.stats.diagnostic as smsd

from econtools.inference.hypothesis import TestResult
from econtools.models._results import RegressionResult


def reset_test(
    result: RegressionResult,
    power: int = 3,
    use_f: bool = True,
) -> TestResult:
    """Ramsey RESET test for functional form misspecification.

    Adds powers of the fitted values (up to ``power``) as additional
    regressors and tests whether their coefficients are jointly zero.

    H₀: functional form is correctly specified.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    power:
        Highest power of fitted values to include (default 3).
    use_f:
        Report an F-statistic if True (default), Chi² otherwise.

    Returns
    -------
    TestResult
    """
    sm_test = smsd.linear_reset(result.raw, power=power, use_f=use_f)
    stat = float(np.squeeze(sm_test.statistic))
    pval = float(np.squeeze(sm_test.pvalue))

    return TestResult(
        test_name="RESET",
        statistic=stat,
        pvalue=pval,
        df=None,
        distribution="F" if use_f else "Chi2",
        null_hypothesis="Functional form is correctly specified",
        reject=pval < 0.05,
    )
