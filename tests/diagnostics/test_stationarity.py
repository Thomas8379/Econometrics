from __future__ import annotations

import numpy as np

from econtools.diagnostics.stationarity import adf_test, kpss_test, pp_test


def test_stationarity_tests_return_results() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(size=100)

    res_adf = adf_test(y)
    res_kpss = kpss_test(y)
    res_pp = pp_test(y)

    assert res_adf.test_name == "ADF"
    assert res_kpss.test_name == "KPSS"
    assert res_pp.test_name == "Phillips-Perron"

    # Value assertions: p-values are in [0, 1]
    assert 0.0 <= res_adf.pvalue <= 1.0
    assert 0.0 <= res_kpss.pvalue <= 1.0
    assert 0.0 <= res_pp.pvalue <= 1.0

    # ADF and PP statistics on white noise are typically negative
    assert res_adf.statistic < 0, "ADF statistic on stationary series should be negative"
    assert res_pp.statistic < 0, "PP statistic on stationary series should be negative"

    # For a stationary series, ADF should reject at 5% (p < 0.05)
    assert res_adf.pvalue < 0.1, "ADF should likely reject for white noise"
