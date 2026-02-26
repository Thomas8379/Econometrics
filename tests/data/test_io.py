"""Tests for econtools.data.io."""

import hashlib
import json
import pathlib

import pandas as pd
import pytest

from econtools.data.io import (
    save_curated,
    verify_hash,
    versioned_path,
)


# ---------------------------------------------------------------------------
# verify_hash
# ---------------------------------------------------------------------------


def test_verify_hash_passes(tmp_path: pathlib.Path) -> None:
    content = b"hello econtools"
    p = tmp_path / "file.bin"
    p.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    verify_hash(p, expected)  # should not raise


def test_verify_hash_fails(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "file.bin"
    p.write_bytes(b"hello econtools")
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify_hash(p, "deadbeef" * 8)


def test_verify_hash_case_insensitive(tmp_path: pathlib.Path) -> None:
    content = b"case test"
    p = tmp_path / "f.bin"
    p.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest().upper()
    verify_hash(p, expected)  # should not raise


# ---------------------------------------------------------------------------
# versioned_path
# ---------------------------------------------------------------------------


def test_versioned_path_string_version(tmp_path: pathlib.Path) -> None:
    parq, meta = versioned_path("wages_panel", "v1", base_dir=tmp_path)
    assert parq.name == "wages_panel_v1.parquet"
    assert meta.name == "wages_panel_v1_meta.json"
    assert parq.parent == tmp_path


def test_versioned_path_int_version(tmp_path: pathlib.Path) -> None:
    parq, meta = versioned_path("test_data", 3, base_dir=tmp_path)
    assert parq.name == "test_data_v3.parquet"
    assert meta.name == "test_data_v3_meta.json"


# ---------------------------------------------------------------------------
# save_curated
# ---------------------------------------------------------------------------


def test_save_curated_creates_files(tmp_path: pathlib.Path, simple_df: pd.DataFrame) -> None:
    meta = {"source_file": "test.dta", "source_type": "dta"}
    parq_path, meta_path = save_curated(simple_df, "test_ds", "v1", meta, base_dir=tmp_path)
    assert parq_path.exists()
    assert meta_path.exists()


def test_save_curated_roundtrip(tmp_path: pathlib.Path, simple_df: pd.DataFrame) -> None:
    meta = {"source_file": "test.dta"}
    parq_path, _ = save_curated(simple_df, "rt_test", 1, meta, base_dir=tmp_path)
    loaded = pd.read_parquet(str(parq_path), engine="pyarrow")
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        simple_df.reset_index(drop=True),
        check_like=True,
    )


def test_save_curated_meta_contents(tmp_path: pathlib.Path, simple_df: pd.DataFrame) -> None:
    meta = {"source_file": "wage1.dta", "source_type": "dta", "data_label": "Test"}
    _, meta_path = save_curated(simple_df, "wage_test", "v2", meta, base_dir=tmp_path)
    with meta_path.open() as fh:
        saved_meta = json.load(fh)
    assert saved_meta["source_file"] == "wage1.dta"
    assert saved_meta["curated_version"] == "v2"
    assert saved_meta["shape"] == list(simple_df.shape)
    assert "saved_at" in saved_meta


def test_save_curated_multiindex_roundtrip(
    tmp_path: pathlib.Path, panel_df_multiindex: pd.DataFrame
) -> None:
    meta = {"source_type": "test"}
    parq_path, _ = save_curated(panel_df_multiindex, "panel_rt", "v1", meta, base_dir=tmp_path)
    loaded = pd.read_parquet(str(parq_path), engine="pyarrow")
    assert isinstance(loaded.index, pd.MultiIndex)
    assert loaded.index.names == ["id", "year"]
