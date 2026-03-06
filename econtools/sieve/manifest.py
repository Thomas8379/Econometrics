"""Run manifest: config hashing, dataset fingerprinting, and provenance.

Every :func:`~econtools.sieve.api.run_sieve` call produces a ``run_manifest.json``
that contains enough information to replicate the run exactly.

Public API
----------
compute_config_hash(config) -> str
compute_dataset_fingerprint(df, used_cols) -> str
build_manifest(run_id, config, df, used_cols, ...) -> dict
write_manifest(manifest, output_dir) -> Path
load_manifest(output_dir) -> dict
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def compute_config_hash(config: dict) -> str:
    """Return a 16-char SHA-256 hash of the sieve configuration dict."""
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_dataset_fingerprint(
    df: pd.DataFrame,
    used_cols: list[str],
    *,
    sample_n: int | None = None,
    seed: int = 0,
) -> str:
    """Return a dataset fingerprint based on shape, dtypes, and a value hash.

    Parameters
    ----------
    df:
        Full DataFrame.
    used_cols:
        Columns actually used in the sieve (only these are hashed).
    sample_n:
        For large datasets (> 50 000 rows), sample this many rows for the
        value hash.  ``None`` uses all rows.  The sampling method is logged.
    seed:
        Seed for sampling (for reproducibility of the fingerprint itself).

    Returns
    -------
    16-char hex string.
    """
    sub = df[[c for c in used_cols if c in df.columns]]
    n = len(sub)
    sample_method = "full"

    if sample_n is not None and n > sample_n:
        sub = sub.sample(n=sample_n, random_state=seed)
        sample_method = f"sampled_{sample_n}"

    # Hash: shape + dtypes + sorted column names + value bytes
    h = hashlib.sha256()
    h.update(json.dumps({"n": n, "cols": sorted(sub.columns.tolist())}).encode())
    for col in sorted(sub.columns):
        try:
            h.update(sub[col].values.tobytes())
        except Exception:
            h.update(str(sub[col].values).encode())

    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Git hash (best-effort)
# ---------------------------------------------------------------------------


def _git_commit_hash() -> str | None:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def build_manifest(
    run_id: str,
    config: dict,
    df: pd.DataFrame,
    used_cols: list[str],
    *,
    n_candidates: int,
    n_selected: int,
    protocol_mode: str,
    exploratory_only: bool,
    split_info: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a run manifest dict.

    Parameters
    ----------
    run_id:
        Unique run identifier (e.g. ``"sieve_20260305_143012"``).
    config:
        Full sieve configuration dict.
    df:
        Full DataFrame.
    used_cols:
        All column names accessed by the sieve.
    n_candidates:
        Total number of candidates evaluated.
    n_selected:
        Number of candidates in the selected shortlist.
    protocol_mode:
        ``"holdout"``, ``"cv"``, or ``"crossfit"``.
    exploratory_only:
        Whether in-sample selection was used.
    split_info:
        Serialisable representation of the train/test split indices.
    extra:
        Any additional metadata to store.

    Returns
    -------
    dict suitable for JSON serialisation.
    """
    fingerprint = compute_dataset_fingerprint(df, used_cols)
    config_hash = compute_config_hash(config)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "config_hash": config_hash,
        "dataset_fingerprint": fingerprint,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "used_cols": sorted(used_cols),
        "n_candidates": n_candidates,
        "n_selected": n_selected,
        "protocol_mode": protocol_mode,
        "exploratory_only": exploratory_only,
        "split_info": split_info or {},
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": _git_commit_hash(),
        },
    }

    if exploratory_only:
        manifest["WARNING"] = (
            "EXPLORATORY ONLY — in-sample selection was used. "
            "Post-selection p-values are not valid. "
            "Do not report these results as confirmatory."
        )

    if extra:
        manifest.update(extra)

    return manifest


def write_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    """Write the manifest to *output_dir/run_manifest.json*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def load_manifest(output_dir: Path) -> dict[str, Any]:
    """Load a previously written manifest from *output_dir/run_manifest.json*."""
    path = Path(output_dir) / "run_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))
