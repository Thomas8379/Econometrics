"""Tests for econtools.data.construct."""

import pandas as pd
import pytest

from econtools.data.construct import (
    is_balanced,
    merge_audit,
    set_panel_index,
    verify_panel_index,
)


# ---------------------------------------------------------------------------
# set_panel_index
# ---------------------------------------------------------------------------


def test_set_panel_index_creates_multiindex(panel_df: pd.DataFrame) -> None:
    result = set_panel_index(panel_df, "id", "year")
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["id", "year"]


def test_set_panel_index_sorts(panel_df: pd.DataFrame) -> None:
    shuffled = panel_df.sample(frac=1, random_state=0)
    result = set_panel_index(shuffled, "id", "year")
    assert result.index.is_monotonic_increasing


def test_set_panel_index_drops_columns(panel_df: pd.DataFrame) -> None:
    result = set_panel_index(panel_df, "id", "year")
    assert "id" not in result.columns
    assert "year" not in result.columns


def test_set_panel_index_no_mutation(panel_df: pd.DataFrame) -> None:
    original_cols = list(panel_df.columns)
    _ = set_panel_index(panel_df, "id", "year")
    assert list(panel_df.columns) == original_cols


# ---------------------------------------------------------------------------
# verify_panel_index
# ---------------------------------------------------------------------------


def test_verify_panel_index_passes(panel_df_multiindex: pd.DataFrame) -> None:
    verify_panel_index(panel_df_multiindex)  # should not raise


def test_verify_panel_index_fails_no_multiindex(simple_df: pd.DataFrame) -> None:
    with pytest.raises(AssertionError, match="MultiIndex"):
        verify_panel_index(simple_df)


# ---------------------------------------------------------------------------
# is_balanced
# ---------------------------------------------------------------------------


def test_is_balanced_true(panel_df_multiindex: pd.DataFrame) -> None:
    assert is_balanced(panel_df_multiindex) is True


def test_is_balanced_false() -> None:
    df = pd.DataFrame(
        {"id": [1, 1, 2], "year": [2000, 2001, 2000], "y": [1, 2, 3]}
    ).set_index(["id", "year"])
    assert is_balanced(df) is False


# ---------------------------------------------------------------------------
# merge_audit
# ---------------------------------------------------------------------------


def test_merge_audit_clean_merge() -> None:
    left = pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
    right = pd.DataFrame({"id": [1, 2, 4], "b": [100, 200, 400]})
    merged = pd.merge(left, right, on="id", how="left")
    report = merge_audit(left, right, merged, "id")

    assert report["n_left"] == 3
    assert report["n_right"] == 3
    assert report["n_merged"] == 3
    assert report["n_matched"] == 2
    assert report["n_left_only"] == 1  # id=3 not in right
    assert report["n_right_only"] == 1  # id=4 not in left
    assert report["n_duplicate_keys_left"] == 0
    assert report["n_duplicate_keys_right"] == 0


def test_merge_audit_row_inflation() -> None:
    left = pd.DataFrame({"id": [1, 2], "a": [10, 20]})
    right = pd.DataFrame({"id": [1, 1], "b": [100, 101]})  # duplicate key in right
    merged = pd.merge(left, right, on="id", how="left")
    report = merge_audit(left, right, merged, "id")
    assert report["row_increase"] > 0
    assert report["n_duplicate_keys_right"] > 0
