"""Tests for econtools.data.inspect."""

import numpy as np
import pandas as pd
import pytest

from econtools.data.inspect import (
    audit_dtypes,
    balance_report,
    cardinality,
    dist_summary,
    missing_report,
    panel_summary,
)


# ---------------------------------------------------------------------------
# missing_report
# ---------------------------------------------------------------------------


def test_missing_report_no_missing(simple_df: pd.DataFrame) -> None:
    report = missing_report(simple_df)
    assert (report["count"] == 0).all()


def test_missing_report_counts(simple_df_missing: pd.DataFrame) -> None:
    report = missing_report(simple_df_missing)
    assert "wage" in report.index
    assert report.loc["wage", "count"] > 0
    assert 0 <= report.loc["wage", "pct"] <= 100


def test_missing_report_threshold(simple_df_missing: pd.DataFrame) -> None:
    # threshold=0 should return all columns with any missing
    report = missing_report(simple_df_missing, threshold=0.0)
    assert all(report["pct"] > 0)


def test_missing_report_columns(simple_df: pd.DataFrame) -> None:
    report = missing_report(simple_df)
    assert set(report.columns) == {"count", "pct", "dtype"}


# ---------------------------------------------------------------------------
# audit_dtypes
# ---------------------------------------------------------------------------


def test_audit_dtypes_flags_numeric_object() -> None:
    df = pd.DataFrame({"a": ["1.0", "2.5", "3.0"], "b": ["x", "y", "z"]})
    result = audit_dtypes(df)
    assert result.loc["a", "parseable_as_float"] == True
    assert result.loc["b", "parseable_as_float"] == False


def test_audit_dtypes_ignores_numeric_cols(simple_df: pd.DataFrame) -> None:
    result = audit_dtypes(simple_df)
    # numeric columns should not be flagged
    for col in ["wage", "educ", "exper"]:
        assert result.loc[col, "parseable_as_float"] == False


# ---------------------------------------------------------------------------
# cardinality
# ---------------------------------------------------------------------------


def test_cardinality_shape(simple_df: pd.DataFrame) -> None:
    result = cardinality(simple_df)
    assert len(result) == len(simple_df.columns)
    assert "n_unique" in result.columns


def test_cardinality_binary(simple_df: pd.DataFrame) -> None:
    result = cardinality(simple_df)
    assert result.loc["female", "n_unique"] == 2


# ---------------------------------------------------------------------------
# balance_report
# ---------------------------------------------------------------------------


def test_balance_report_balanced(panel_df: pd.DataFrame) -> None:
    report = balance_report(panel_df, "id", "year")
    # Every cell should be 1 (one obs per entity×year)
    assert (report.values == 1).all()
    assert report.shape == (10, 5)


def test_balance_report_unbalanced() -> None:
    df = pd.DataFrame(
        {"id": [1, 1, 2], "year": [2000, 2001, 2000], "y": [1, 2, 3]}
    )
    report = balance_report(df, "id", "year")
    assert report.loc[2, 2001] == 0  # entity 2 missing year 2001


# ---------------------------------------------------------------------------
# panel_summary
# ---------------------------------------------------------------------------


def test_panel_summary_balanced(panel_df: pd.DataFrame) -> None:
    summary = panel_summary(panel_df, "id", "year")
    assert summary["n_entities"] == 10
    assert summary["n_periods"] == 5
    assert summary["n_obs"] == 50
    assert summary["balanced"] is True


def test_panel_summary_multiindex(panel_df_multiindex: pd.DataFrame) -> None:
    summary = panel_summary(panel_df_multiindex)
    assert summary["n_entities"] == 10
    assert summary["balanced"] is True


def test_panel_summary_unbalanced() -> None:
    df = pd.DataFrame(
        {"id": [1, 1, 2], "year": [2000, 2001, 2000], "y": [1, 2, 3]}
    )
    summary = panel_summary(df, "id", "year")
    assert summary["balanced"] is False


# ---------------------------------------------------------------------------
# dist_summary
# ---------------------------------------------------------------------------


def test_dist_summary_keys(simple_df: pd.DataFrame) -> None:
    result = dist_summary(simple_df, "wage")
    expected_keys = {"n", "mean", "variance", "std", "min", "max",
                     "skewness", "kurtosis", "p1", "p5", "p25", "p50", "p75", "p95", "p99"}
    assert expected_keys.issubset(result.keys())


def test_dist_summary_values(simple_df: pd.DataFrame) -> None:
    result = dist_summary(simple_df, "wage")
    assert result["min"] <= result["p1"] <= result["p50"] <= result["p99"] <= result["max"]
    assert result["std"] >= 0
    assert result["n"] > 0
