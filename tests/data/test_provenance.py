"""Tests for econtools.data.provenance."""

import json
import pathlib

from econtools.data.provenance import log_step, read_log


def test_log_step_creates_file(tmp_path: pathlib.Path) -> None:
    log_path = tmp_path / "prov.json"
    log_step(log_path, "clean.winsorise", {"col": "wage", "lower": 0.01})
    assert log_path.exists()


def test_log_step_appends(tmp_path: pathlib.Path) -> None:
    log_path = tmp_path / "prov.json"
    log_step(log_path, "io.load_raw", {"name": "wage1"})
    log_step(log_path, "clean.winsorise", {"col": "wage"})
    records = read_log(log_path)
    assert len(records) == 2
    assert records[0]["function"] == "io.load_raw"
    assert records[1]["function"] == "clean.winsorise"


def test_log_step_structure(tmp_path: pathlib.Path) -> None:
    log_path = tmp_path / "prov.json"
    log_step(log_path, "transform.log_col", {"col": "wage"}, note="test note")
    records = read_log(log_path)
    r = records[0]
    assert "timestamp" in r
    assert r["function"] == "transform.log_col"
    assert r["kwargs"] == {"col": "wage"}
    assert r["note"] == "test note"


def test_log_step_non_serialisable_kwargs(tmp_path: pathlib.Path) -> None:
    import numpy as np
    log_path = tmp_path / "prov.json"
    # numpy array is not JSON-serialisable — should be coerced to repr
    log_step(log_path, "test.fn", {"arr": np.array([1, 2, 3])})
    records = read_log(log_path)
    assert isinstance(records[0]["kwargs"]["arr"], str)


def test_read_log_missing_file(tmp_path: pathlib.Path) -> None:
    result = read_log(tmp_path / "nonexistent.json")
    assert result == []
