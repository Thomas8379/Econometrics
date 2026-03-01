"""Tests for LR and score tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from econtools.inference.hypothesis import lr_test, score_test
from econtools.models.probit import fit_probit


@pytest.mark.phase3
def test_lr_and_score_tests() -> None:
    rng = np.random.default_rng(123)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    latent = 0.4 + 0.8 * x1 - 0.5 * x2 + 0.2 * x3 + rng.normal(size=n)
    y = (latent > 0).astype(float)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

    res_full = fit_probit(df, "y", ["x1", "x2", "x3"], add_constant=True)
    res_rest = fit_probit(df, "y", ["x1"], add_constant=True)

    lr = lr_test(res_rest, res_full)
    assert lr.statistic >= 0.0
    assert 0.0 <= lr.pvalue <= 1.0
    assert int(lr.df) == 2

    score = score_test(res_rest, df[["x2", "x3"]])
    assert score.statistic >= 0.0
    assert 0.0 <= score.pvalue <= 1.0
    assert int(score.df) == 2
