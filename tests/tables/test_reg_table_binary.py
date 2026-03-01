"""Binary model table tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from econtools.models.probit import fit_probit
from econtools.tables import reg_table


@pytest.mark.phase3
def test_reg_table_binary_includes_classification() -> None:
    rng = np.random.default_rng(7)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = -0.2 + 1.0 * x1 - 0.8 * x2 + rng.normal(size=n)
    y = (latent > 0).astype(float)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    res = fit_probit(df, "y", ["x1", "x2"], add_constant=True)
    out = reg_table(res, format="text")
    assert "Classification (c=0.5)" in out
    assert "Pseudo R²" in out or "Pseudo R\u00b2" in out
    assert "Balanced accuracy" in out
