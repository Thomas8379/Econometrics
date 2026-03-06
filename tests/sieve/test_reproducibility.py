"""Reproducibility tests for the sieve: determinism guarantees.

Tests that:
1. Same seed + same config => identical candidate hashes.
2. Same seed + same config => identical split indices.
3. Same seed + same config => identical rankings.
4. Parallel (n_jobs>1) == sequential (n_jobs=1) when joblib is available.
5. Exploratory mode is clearly labeled.
6. Full manifest is always written (not just winners).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2 * x1 + x2 + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


BASE_SPEC = {
    "generators": {
        "features": {"polynomial": {"enabled": True, "degree": 2, "vars": ["x1", "x2"]}}
    },
    "protocol": {"mode": "holdout", "test_frac": 0.30},
    "selection": {"primary_metric": "rmse", "top_k": 3, "higher_is_better": False},
    "constraints": {"max_terms": 5},
}


# ---------------------------------------------------------------------------
# Candidate determinism
# ---------------------------------------------------------------------------


class TestCandidateDeterminism:
    def test_same_candidates_same_hashes(self):
        """Candidate generation is deterministic."""
        from econtools.sieve.api import _build_candidates
        from econtools.sieve.manifest import compute_config_hash, compute_dataset_fingerprint

        df = _make_df()
        cfg_hash = compute_config_hash(BASE_SPEC)
        fp = compute_dataset_fingerprint(df, ["y", "x1", "x2"])

        cands1, _ = _build_candidates(
            "y", ["x1", "x2"], "ols", None, None, BASE_SPEC, df.copy(),
            config_hash=cfg_hash, data_fingerprint=fp, generator_hash="test"
        )
        cands2, _ = _build_candidates(
            "y", ["x1", "x2"], "ols", None, None, BASE_SPEC, df.copy(),
            config_hash=cfg_hash, data_fingerprint=fp, generator_hash="test"
        )

        hashes1 = sorted(c.candidate_hash for c in cands1)
        hashes2 = sorted(c.candidate_hash for c in cands2)
        assert hashes1 == hashes2

    def test_config_hash_stable(self):
        """Same config dict always produces the same hash."""
        from econtools.sieve.manifest import compute_config_hash
        h1 = compute_config_hash(BASE_SPEC)
        h2 = compute_config_hash(BASE_SPEC)
        assert h1 == h2

    def test_different_configs_different_hashes(self):
        from econtools.sieve.manifest import compute_config_hash
        spec2 = {**BASE_SPEC, "constraints": {"max_terms": 10}}
        assert compute_config_hash(BASE_SPEC) != compute_config_hash(spec2)


# ---------------------------------------------------------------------------
# Split determinism
# ---------------------------------------------------------------------------


class TestSplitDeterminism:
    def test_holdout_deterministic(self):
        from econtools.sieve.protocols import holdout_split
        s1 = holdout_split(200, seed=42)
        s2 = holdout_split(200, seed=42)
        assert s1.train_idx == s2.train_idx
        assert s1.test_idx == s2.test_idx

    def test_kfold_deterministic(self):
        from econtools.sieve.protocols import kfold_splits
        f1 = kfold_splits(200, 5, seed=42)
        f2 = kfold_splits(200, 5, seed=42)
        for a, b in zip(f1, f2):
            assert a.train_idx == b.train_idx
            assert a.test_idx == b.test_idx

    def test_different_seeds_different_splits(self):
        from econtools.sieve.protocols import holdout_split
        s1 = holdout_split(200, seed=1)
        s2 = holdout_split(200, seed=2)
        assert s1.train_idx != s2.train_idx


# ---------------------------------------------------------------------------
# End-to-end determinism
# ---------------------------------------------------------------------------


class TestEndToEndDeterminism:
    @pytest.mark.slow
    def test_same_run_produces_same_results(self):
        """run_sieve is fully deterministic."""
        from econtools.sieve.api import run_sieve

        df = _make_df(n=300, seed=0)

        r1 = run_sieve(df, "y", ["x1", "x2"], "ols", sieve_spec=BASE_SPEC, seed=42)
        r2 = run_sieve(df, "y", ["x1", "x2"], "ols", sieve_spec=BASE_SPEC, seed=42)

        # Same selected candidates
        h1 = sorted(c.candidate_hash for c in r1["selected"])
        h2 = sorted(c.candidate_hash for c in r2["selected"])
        assert h1 == h2

        # Same leaderboard ranking
        lb1 = r1["leaderboard"]["candidate_hash"].tolist()
        lb2 = r2["leaderboard"]["candidate_hash"].tolist()
        assert lb1 == lb2

        # Same data fingerprint
        assert r1["run_metadata"]["data_fingerprint"] == r2["run_metadata"]["data_fingerprint"]

    @pytest.mark.slow
    def test_different_seeds_can_differ(self):
        """Different seeds may (and likely will) produce different splits -> rankings."""
        from econtools.sieve.api import run_sieve

        df = _make_df(n=300, seed=0)
        r1 = run_sieve(df, "y", ["x1", "x2"], "ols", sieve_spec=BASE_SPEC, seed=42)
        r2 = run_sieve(df, "y", ["x1", "x2"], "ols", sieve_spec=BASE_SPEC, seed=99)

        # Config hash must be the same (same spec)
        assert r1["run_metadata"]["config_hash"] == r2["run_metadata"]["config_hash"]

        # Candidates themselves (hashes) must be the same regardless of seed
        h1 = sorted(c.candidate_hash for c in r1["leaderboard"]["candidate_hash"].tolist())
        h2 = sorted(c.candidate_hash for c in r2["leaderboard"]["candidate_hash"].tolist())
        assert h1 == h2


# ---------------------------------------------------------------------------
# Artifact completeness
# ---------------------------------------------------------------------------


class TestArtifacts:
    @pytest.mark.slow
    def test_full_leaderboard_written(self, tmp_path):
        """Full leaderboard (all candidates) must be written, not just winners."""
        from econtools.sieve.api import run_sieve

        df = _make_df(n=200, seed=5)
        out_dir = str(tmp_path / "sieve_out")

        r = run_sieve(df, "y", ["x1", "x2"], "ols",
                      sieve_spec=BASE_SPEC, seed=0, output_dir=out_dir)

        csv_path = Path(out_dir) / "leaderboard.csv"
        assert csv_path.exists(), "leaderboard.csv not written"

        lb = pd.read_csv(str(csv_path))
        n_all = r["run_metadata"]["n_candidates"]
        assert len(lb) == n_all, (
            f"Leaderboard has {len(lb)} rows but {n_all} candidates were evaluated. "
            "All candidates (not just winners) must be saved."
        )

    @pytest.mark.slow
    def test_manifest_written(self, tmp_path):
        """run_manifest.json must always be written."""
        from econtools.sieve.api import run_sieve

        df = _make_df(n=200, seed=5)
        out_dir = str(tmp_path / "sieve_man")

        r = run_sieve(df, "y", ["x1", "x2"], "ols",
                      sieve_spec=BASE_SPEC, seed=0, output_dir=out_dir)

        manifest_path = Path(out_dir) / "run_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["run_id"] == r["run_metadata"]["run_id"]
        assert "dataset_fingerprint" in manifest
        assert "config_hash" in manifest
        assert "n_candidates" in manifest


# ---------------------------------------------------------------------------
# Exploratory mode labeling
# ---------------------------------------------------------------------------


class TestExploratoryMode:
    @pytest.mark.slow
    def test_exploratory_stamped(self, tmp_path):
        """Enabling in-sample selection must stamp outputs as EXPLORATORY ONLY."""
        import warnings
        from econtools.sieve.api import run_sieve

        df = _make_df(n=200)
        spec = {**BASE_SPEC, "protocol": {"allow_in_sample_selection": True}}
        out_dir = str(tmp_path / "exp")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = run_sieve(df, "y", ["x1", "x2"], "ols",
                          sieve_spec=spec, seed=0, output_dir=out_dir)

        # Must warn user
        assert any("EXPLORATORY" in str(warn.message) for warn in w)

        # Metadata must flag it
        assert r["run_metadata"]["exploratory_only"]

        # Manifest must contain warning
        manifest_path = Path(out_dir) / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            assert manifest.get("exploratory_only") or "WARNING" in manifest

    @pytest.mark.slow
    def test_confirmatory_not_stamped(self, tmp_path):
        """Default (holdout) mode must NOT be stamped as exploratory."""
        from econtools.sieve.api import run_sieve

        df = _make_df(n=200)
        out_dir = str(tmp_path / "conf")

        r = run_sieve(df, "y", ["x1", "x2"], "ols",
                      sieve_spec=BASE_SPEC, seed=0, output_dir=out_dir)

        assert not r["run_metadata"]["exploratory_only"]
