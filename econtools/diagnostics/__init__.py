"""Assumption tests — re-export shim.

All diagnostics now live in ``econtools.evaluation``.  This module
re-exports everything for backward compatibility.  Import paths of the
form ``from econtools.diagnostics import breusch_pagan`` continue to work.

In a future major version, this shim will be removed.
"""

from econtools.diagnostics.heteroskedasticity import breusch_pagan, white_test  # noqa: F401
from econtools.diagnostics.normality import jarque_bera  # noqa: F401
from econtools.diagnostics.specification import reset_test  # noqa: F401
from econtools.diagnostics.multicollinearity import compute_vif, condition_number  # noqa: F401
from econtools.diagnostics.iv import (  # noqa: F401
    basmann_f_test,
    basmann_test,
    run_iv_diagnostics,
    sargan_test,
    weak_instrument_tests,
    wu_hausman_test,
)
from econtools.diagnostics.panel import lead_test_strict_exogeneity, run_panel_diagnostics  # noqa: F401
from econtools.diagnostics.serial_correlation import (  # noqa: F401
    autocorr_from_series,
    box_pierce_from_autocorr,
    box_pierce_q,
    ljung_box_from_autocorr,
    ljung_box_q,
)
from econtools.diagnostics.stationarity import adf_test, kpss_test, pp_test  # noqa: F401
from econtools.diagnostics.time_series import (  # noqa: F401
    select_var_lag,
    granger_causality,
    lead_exogeneity_test,
)

__all__ = [
    "breusch_pagan",
    "white_test",
    "jarque_bera",
    "reset_test",
    "compute_vif",
    "condition_number",
    "wu_hausman_test",
    "run_iv_diagnostics",
    "sargan_test",
    "basmann_test",
    "basmann_f_test",
    "weak_instrument_tests",
    "lead_test_strict_exogeneity",
    "run_panel_diagnostics",
    "autocorr_from_series",
    "box_pierce_from_autocorr",
    "box_pierce_q",
    "ljung_box_from_autocorr",
    "ljung_box_q",
    "adf_test",
    "kpss_test",
    "pp_test",
    "select_var_lag",
    "granger_causality",
    "lead_exogeneity_test",
]
