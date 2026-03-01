from __future__ import annotations

import numpy as np
import scipy.stats

from econtools.diagnostics.serial_correlation import (
    box_pierce_from_autocorr,
    ljung_box_from_autocorr,
)


def test_box_pierce_from_autocorr() -> None:
    acf = np.array([0.2, -0.1, 0.05])
    n_obs = 50
    q = n_obs * np.sum(acf**2)
    pval = float(scipy.stats.chi2.sf(q, df=3))

    res = box_pierce_from_autocorr(acf, n_obs)
    assert np.isclose(res.statistic, q)
    assert np.isclose(res.pvalue, pval)
    assert res.df == 3.0


def test_ljung_box_from_autocorr() -> None:
    acf = np.array([0.2, -0.1, 0.05])
    n_obs = 50
    lag_idx = np.arange(1, len(acf) + 1, dtype=float)
    q = n_obs * (n_obs + 2.0) * np.sum(acf**2 / (n_obs - lag_idx))
    pval = float(scipy.stats.chi2.sf(q, df=3))

    res = ljung_box_from_autocorr(acf, n_obs)
    assert np.isclose(res.statistic, q)
    assert np.isclose(res.pvalue, pval)
    assert res.df == 3.0
