"""Tests for econtools.inference.se_types."""

from __future__ import annotations

import numpy as np
import pytest

from econtools.inference.se_types import VALID_COV_TYPES, resolve_cov_args


def test_classical_maps_to_nonrobust() -> None:
    args = resolve_cov_args("classical")
    assert args == {"cov_type": "nonrobust"}


def test_hc3_maps_correctly() -> None:
    args = resolve_cov_args("HC3")
    assert args == {"cov_type": "HC3"}


def test_hc0_maps_correctly() -> None:
    args = resolve_cov_args("HC0")
    assert args == {"cov_type": "HC0"}


def test_hac_with_maxlags() -> None:
    args = resolve_cov_args("HAC", maxlags=4)
    assert args["cov_type"] == "HAC"
    assert args["cov_kwds"]["maxlags"] == 4


def test_newey_west_maps_to_hac() -> None:
    args = resolve_cov_args("newey_west")
    assert args["cov_type"] == "HAC"


def test_cluster_requires_groups() -> None:
    with pytest.raises(ValueError, match="cluster"):
        resolve_cov_args("cluster")


def test_invalid_cov_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown cov_type"):
        resolve_cov_args("HCCM99")


def test_valid_cov_types_is_tuple() -> None:
    assert isinstance(VALID_COV_TYPES, tuple)
    assert "classical" in VALID_COV_TYPES
    assert "HC3" in VALID_COV_TYPES
    assert "cluster" in VALID_COV_TYPES
