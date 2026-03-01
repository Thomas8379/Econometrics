"""Panel diagnostics (lead/lag tests, etc.)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from econtools.inference.hypothesis import TestResult, wald_test
from econtools.models.ols import fit_ols


def lead_test_strict_exogeneity(
    df: pd.DataFrame,
    dep_var: str,
    exog_vars: list[str],
    *,
    entity: str,
    time: str,
    leads: int = 1,
    cov_type: str = "cluster",
) -> TestResult:
    """Lead test for strict exogeneity in a first-difference model.

    Adds leads of the *levels* of regressors to the FD regression and
    tests whether their coefficients are jointly zero.
    """
    if leads < 1:
        raise ValueError("leads must be >= 1.")

    cols = [entity, time, dep_var] + list(exog_vars)
    data = df[cols].dropna().copy()
    data = data.sort_values([entity, time])

    grp = data.groupby(entity, sort=False)
    data["d_y"] = grp[dep_var].diff()
    for x in exog_vars:
        data[f"d_{x}"] = grp[x].diff()
        for k in range(1, leads + 1):
            data[f"lead{k}_{x}"] = grp[x].shift(-k)

    lead_cols = [f"lead{k}_{x}" for k in range(1, leads + 1) for x in exog_vars]
    diff_cols = [f"d_{x}" for x in exog_vars]
    needed = ["d_y"] + diff_cols + lead_cols
    data = data.dropna(subset=needed)

    result = fit_ols(
        data,
        "d_y",
        diff_cols + lead_cols,
        add_constant=False,
        cov_type=cov_type,
        groups=data[entity] if cov_type == "cluster" else None,
    )

    # Wald test: all lead coefficients = 0
    params = list(result.params.index)
    R = np.zeros((len(lead_cols), len(params)))
    for i, name in enumerate(lead_cols):
        if name not in params:
            raise ValueError(f"Lead term '{name}' not in regression.")
        R[i, params.index(name)] = 1.0

    test = wald_test(result, R, use_f=True)
    return TestResult(
        test_name="Lead test (strict exogeneity)",
        statistic=test.statistic,
        pvalue=test.pvalue,
        df=test.df,
        distribution=test.distribution,
        null_hypothesis="Coefficients on leads are jointly zero",
        reject=test.pvalue < 0.05,
    )


def run_panel_diagnostics(
    df: pd.DataFrame,
    dep_var: str,
    exog_vars: list[str],
    *,
    entity: str,
    time: str,
    tests: Iterable[str] | None = None,
    leads: int = 1,
) -> list[TestResult]:
    """Run panel diagnostics; filter by test names if provided."""
    tests = [t.strip().lower() for t in tests] if tests else None
    outputs: list[TestResult] = []

    if tests is None or "lead" in tests or "lead_test" in tests or "lead-test" in tests:
        outputs.append(
            lead_test_strict_exogeneity(
                df,
                dep_var,
                exog_vars,
                entity=entity,
                time=time,
                leads=leads,
            )
        )

    return outputs
