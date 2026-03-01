"""Shared fixtures for numerical validation tests.

These tests require .dta files from data_lake/raw/wooldridge_and_oleg/.
They are skipped gracefully if the files are not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_RAW_DIR = Path(__file__).resolve().parents[2] / "data_lake" / "raw" / "wooldridge_and_oleg"


def dta_path(name: str) -> Path:
    return _RAW_DIR / f"{name}.dta"


def require_dta(name: str) -> pytest.MarkDecorator:
    """Return a pytest.mark.skipif decorator that skips if *name*.dta is absent."""
    path = dta_path(name)
    return pytest.mark.skipif(
        not path.exists(),
        reason=f"data_lake/raw/wooldridge_and_oleg/{name}.dta not present",
    )
