"""Tests for econtools.diagnostics.multicollinearity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from econtools.diagnostics import compute_vif, condition_number
from econtools.models import fit_ols


def test_compute_vif_returns_series(ols_result) -> None:
    vif = compute_vif(ols_result)
    assert isinstance(vif, pd.Series)
    assert vif.name == "VIF"


def test_compute_vif_orthogonal_near_one(ols_result) -> None:
    """x1, x2 are independent normals — VIFs should be close to 1."""
    vif = compute_vif(ols_result)
    assert "x1" in vif.index
    assert "x2" in vif.index
    assert vif["x1"] < 2.0
    assert vif["x2"] < 2.0


def test_compute_vif_excludes_const(ols_result) -> None:
    vif = compute_vif(ols_result)
    assert "const" not in vif.index
    assert "Intercept" not in vif.index


def test_condition_number_returns_float(ols_result) -> None:
    cn = condition_number(ols_result)
    assert isinstance(cn, float)
    assert cn > 0.0
