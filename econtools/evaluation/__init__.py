"""Evaluation — diagnostics, hypothesis tests, binary metrics.

Phase 1+ content.  This package collects all statistical evaluation
logic: assumption tests, hypothesis tests, influence diagnostics, and
binary-model metrics.  Rendering lives in ``econtools.output``.
"""

from econtools.evaluation.binary_metrics import (  # noqa: F401
    _BinaryMetrics,
    _binary_metrics,
    _marginal_effects,
)
from econtools.evaluation.heteroskedasticity import breusch_pagan, white_test  # noqa: F401
from econtools.evaluation.hypothesis import (  # noqa: F401
    TestResult,
    wald_test,
    f_test,
    t_test_coeff,
    lr_test,
    score_test,
    conf_int,
)
from econtools.evaluation.multicollinearity import compute_vif, condition_number  # noqa: F401
from econtools.evaluation.normality import jarque_bera  # noqa: F401
from econtools.evaluation.serial_correlation import (  # noqa: F401
    autocorr_from_series,
    box_pierce_from_autocorr,
    box_pierce_q,
    ljung_box_from_autocorr,
    ljung_box_q,
)
from econtools.evaluation.specification import reset_test  # noqa: F401
from econtools.evaluation.stationarity import adf_test, kpss_test, pp_test  # noqa: F401
from econtools.evaluation.time_series import (  # noqa: F401
    granger_causality,
    lead_exogeneity_test,
    select_var_lag,
)
from econtools.evaluation.iv_checks import (  # noqa: F401
    basmann_f_test,
    basmann_test,
    run_iv_diagnostics,
    sargan_test,
    weak_instrument_tests,
    wu_hausman_test,
)
from econtools.evaluation.panel_checks import (  # noqa: F401
    lead_test_strict_exogeneity,
    run_panel_diagnostics,
)

__all__ = [
    # binary
    "_BinaryMetrics",
    "_binary_metrics",
    "_marginal_effects",
    # heteroskedasticity
    "breusch_pagan",
    "white_test",
    # hypothesis
    "TestResult",
    "wald_test",
    "f_test",
    "t_test_coeff",
    "lr_test",
    "score_test",
    "conf_int",
    # multicollinearity
    "compute_vif",
    "condition_number",
    # normality
    "jarque_bera",
    # serial correlation
    "autocorr_from_series",
    "box_pierce_from_autocorr",
    "box_pierce_q",
    "ljung_box_from_autocorr",
    "ljung_box_q",
    # specification
    "reset_test",
    # stationarity
    "adf_test",
    "kpss_test",
    "pp_test",
    # time series
    "granger_causality",
    "lead_exogeneity_test",
    "select_var_lag",
    # IV checks
    "basmann_f_test",
    "basmann_test",
    "run_iv_diagnostics",
    "sargan_test",
    "weak_instrument_tests",
    "wu_hausman_test",
    # panel checks
    "lead_test_strict_exogeneity",
    "run_panel_diagnostics",
]
