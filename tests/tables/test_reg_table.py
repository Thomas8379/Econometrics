"""Tests for econtools.tables.reg_table."""

from __future__ import annotations

import pytest

from econtools.tables import reg_table


def test_reg_table_text_returns_string(ols_result) -> None:
    out = reg_table(ols_result, format="text")
    assert isinstance(out, str)
    assert len(out) > 0


def test_reg_table_latex_contains_tabular(ols_result) -> None:
    out = reg_table(ols_result, format="latex")
    assert r"\begin{tabular}" in out
    assert r"\end{tabular}" in out


def test_reg_table_html_contains_table_tags(ols_result) -> None:
    out = reg_table(ols_result, format="html")
    assert "<table>" in out
    assert "</table>" in out


def test_reg_table_stars_applied(ols_result) -> None:
    """x1 has true coeff 3 — should have *** stars."""
    out = reg_table(ols_result, format="text", stars=True)
    assert "***" in out


def test_reg_table_no_stars(ols_result) -> None:
    out = reg_table(ols_result, format="text", stars=False)
    assert "***" not in out
    assert "** " not in out


def test_reg_table_digits(ols_result) -> None:
    import re
    out_3 = reg_table(ols_result, format="text", digits=3)
    out_6 = reg_table(ols_result, format="text", digits=6)
    def max_decimal_len(s: str) -> int:
        matches = re.findall(r"\d+\.(\d+)", s)
        return max((len(m) for m in matches), default=0)
    assert max_decimal_len(out_3) == 3
    assert max_decimal_len(out_6) == 6


def test_reg_table_invalid_format_raises(ols_result) -> None:
    with pytest.raises(ValueError, match="Unknown format"):
        reg_table(ols_result, format="markdown")


def test_reg_table_text_contains_dep_var(ols_result) -> None:
    out = reg_table(ols_result, format="text")
    assert "y" in out


def test_reg_table_footer_contains_nobs(ols_result) -> None:
    out = reg_table(ols_result, format="text")
    assert "200" in out  # ols_data has n=200
