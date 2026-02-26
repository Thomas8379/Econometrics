"""Tests for econtools.data.clean."""

import pandas as pd
import pytest

from econtools.data.clean import (
    assert_nonnegative,
    assert_range,
    rename_snake,
    snake_case,
    winsorise,
)


# ---------------------------------------------------------------------------
# snake_case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("YearsOfSchooling", "years_of_schooling"),
        ("logWage2", "log_wage2"),
        ("GDP per capita", "gdp_per_capita"),
        ("already_snake", "already_snake"),
        ("CamelCaseTest", "camel_case_test"),
        ("  spaces  ", "spaces"),
        ("col-name.here", "col_name_here"),
        ("ABCdef", "ab_cdef"),
    ],
)
def test_snake_case(input_str: str, expected: str) -> None:
    assert snake_case(input_str) == expected


def test_rename_snake() -> None:
    df = pd.DataFrame(columns=["YearsOfSchooling", "logWage", "GDP per capita"])
    result = rename_snake(df)
    assert list(result.columns) == ["years_of_schooling", "log_wage", "gdp_per_capita"]


def test_rename_snake_no_mutation(simple_df: pd.DataFrame) -> None:
    original_cols = list(simple_df.columns)
    _ = rename_snake(simple_df)
    assert list(simple_df.columns) == original_cols


# ---------------------------------------------------------------------------
# winsorise
# ---------------------------------------------------------------------------


def test_winsorise_clips_extremes(simple_df: pd.DataFrame) -> None:
    result = winsorise(simple_df, "wage", lower=0.05, upper=0.95)
    import numpy as np
    lo = float(simple_df["wage"].quantile(0.05))
    hi = float(simple_df["wage"].quantile(0.95))
    assert result["wage"].min() >= lo
    assert result["wage"].max() <= hi


def test_winsorise_no_mutation(simple_df: pd.DataFrame) -> None:
    original_min = simple_df["wage"].min()
    _ = winsorise(simple_df, "wage")
    assert simple_df["wage"].min() == original_min


def test_winsorise_invalid_bounds(simple_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="lower"):
        winsorise(simple_df, "wage", lower=0.9, upper=0.1)


def test_winsorise_preserves_other_cols(simple_df: pd.DataFrame) -> None:
    result = winsorise(simple_df, "wage")
    pd.testing.assert_series_equal(result["educ"], simple_df["educ"])


# ---------------------------------------------------------------------------
# assert_nonnegative
# ---------------------------------------------------------------------------


def test_assert_nonnegative_passes(simple_df: pd.DataFrame) -> None:
    assert_nonnegative(simple_df, ["educ", "exper"])  # should not raise


def test_assert_nonnegative_fails() -> None:
    df = pd.DataFrame({"x": [1, -1, 2]})
    with pytest.raises(AssertionError, match="negative"):
        assert_nonnegative(df, "x")


def test_assert_nonnegative_string_col(simple_df: pd.DataFrame) -> None:
    assert_nonnegative(simple_df, "educ")  # single string arg


# ---------------------------------------------------------------------------
# assert_range
# ---------------------------------------------------------------------------


def test_assert_range_passes(simple_df: pd.DataFrame) -> None:
    assert_range(simple_df, "female", 0, 1)  # should not raise


def test_assert_range_fails() -> None:
    df = pd.DataFrame({"p": [0.0, 0.5, 1.1]})
    with pytest.raises(AssertionError, match="outside"):
        assert_range(df, "p", 0.0, 1.0)
