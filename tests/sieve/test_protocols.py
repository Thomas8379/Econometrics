"""Tests for evaluation protocols — split construction and leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from econtools.sieve.protocols import (
    EvalProtocol,
    SplitSpec,
    aggregate_fold_results,
    crossfit_splits,
    holdout_split,
    kfold_splits,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def n():
    return 100


@pytest.fixture
def groups_100():
    """10 groups of 10 observations each."""
    return np.repeat(np.arange(10), 10)


# ---------------------------------------------------------------------------
# holdout_split
# ---------------------------------------------------------------------------


class TestHoldoutSplit:
    def test_sizes(self, n):
        split = holdout_split(n, test_frac=0.3, seed=42)
        assert split.n_train + split.n_test == n
        assert split.n_test >= 1

    def test_no_overlap(self, n):
        split = holdout_split(n, test_frac=0.3, seed=42)
        train_set = set(split.train_idx)
        test_set = set(split.test_idx)
        assert train_set.isdisjoint(test_set)

    def test_full_coverage(self, n):
        split = holdout_split(n, test_frac=0.3, seed=42)
        all_idx = set(split.train_idx) | set(split.test_idx)
        assert all_idx == set(range(n))

    def test_deterministic(self, n):
        s1 = holdout_split(n, test_frac=0.3, seed=7)
        s2 = holdout_split(n, test_frac=0.3, seed=7)
        assert s1.train_idx == s2.train_idx
        assert s1.test_idx == s2.test_idx

    def test_different_seeds_different_splits(self, n):
        s1 = holdout_split(n, test_frac=0.3, seed=1)
        s2 = holdout_split(n, test_frac=0.3, seed=2)
        # Very unlikely to be identical
        assert s1.train_idx != s2.train_idx

    def test_approximate_fraction(self, n):
        split = holdout_split(n, test_frac=0.3, seed=0)
        assert 0.25 <= split.n_test / n <= 0.35

    def test_grouped_split_no_leakage(self, groups_100):
        n = len(groups_100)
        split = holdout_split(n, test_frac=0.3, seed=0, group_arr=groups_100)
        train_groups = set(groups_100[list(split.train_idx)])
        test_groups = set(groups_100[list(split.test_idx)])
        assert train_groups.isdisjoint(test_groups), (
            "Group leakage: same group appears in both train and test."
        )


# ---------------------------------------------------------------------------
# kfold_splits
# ---------------------------------------------------------------------------


class TestKFoldSplits:
    def test_k_splits_returned(self, n):
        splits = kfold_splits(n, 5, seed=0)
        assert len(splits) == 5

    def test_full_coverage(self, n):
        splits = kfold_splits(n, 5, seed=0)
        all_test = set()
        for split in splits:
            all_test.update(split.test_idx)
        assert all_test == set(range(n))

    def test_no_overlap_between_test_folds(self, n):
        splits = kfold_splits(n, 5, seed=0)
        test_sets = [set(s.test_idx) for s in splits]
        for i in range(len(test_sets)):
            for j in range(i + 1, len(test_sets)):
                assert test_sets[i].isdisjoint(test_sets[j]), (
                    f"Test folds {i} and {j} overlap."
                )

    def test_deterministic(self, n):
        splits1 = kfold_splits(n, 5, seed=99)
        splits2 = kfold_splits(n, 5, seed=99)
        for s1, s2 in zip(splits1, splits2):
            assert s1.train_idx == s2.train_idx
            assert s1.test_idx == s2.test_idx

    def test_grouped_no_leakage(self, groups_100):
        n = len(groups_100)
        splits = kfold_splits(n, 5, seed=0, group_arr=groups_100)
        for fold_i, split in enumerate(splits):
            train_groups = set(groups_100[list(split.train_idx)])
            test_groups = set(groups_100[list(split.test_idx)])
            assert train_groups.isdisjoint(test_groups), (
                f"Group leakage in fold {fold_i}."
            )

    def test_fold_ids(self, n):
        splits = kfold_splits(n, 5, seed=0)
        fold_ids = [s.fold_id for s in splits]
        assert sorted(fold_ids) == list(range(5))


# ---------------------------------------------------------------------------
# crossfit_splits
# ---------------------------------------------------------------------------


class TestCrossfitSplits:
    def test_same_structure_as_kfold(self, n):
        """Cross-fitting uses the same split structure as k-fold."""
        kf = kfold_splits(n, 5, seed=0)
        cf = crossfit_splits(n, 5, seed=0)
        for ks, cs in zip(kf, cf):
            assert ks.train_idx == cs.train_idx
            assert ks.test_idx == cs.test_idx


# ---------------------------------------------------------------------------
# EvalProtocol.from_config
# ---------------------------------------------------------------------------


class TestEvalProtocol:
    def test_holdout_default(self):
        p = EvalProtocol.from_config({})
        assert p.mode == "holdout"
        assert not p.exploratory_mode

    def test_cv_mode(self):
        p = EvalProtocol.from_config({"mode": "cv", "k": 3})
        assert p.mode == "cv"
        assert p.k == 3

    def test_crossfit_mode(self):
        p = EvalProtocol.from_config({"mode": "crossfit"})
        assert p.mode == "crossfit"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown protocol mode"):
            EvalProtocol.from_config({"mode": "magic"})

    def test_exploratory_warns(self):
        with pytest.warns(UserWarning, match="EXPLORATORY ONLY"):
            p = EvalProtocol.from_config({"allow_in_sample_selection": True})
        assert p.exploratory_mode


# ---------------------------------------------------------------------------
# aggregate_fold_results
# ---------------------------------------------------------------------------


class TestAggregateFoldResults:
    def _make_fake_candidate(self, x_terms, estimator="ols"):
        from econtools.sieve.candidates import Candidate
        return Candidate(y="y", X_terms=tuple(x_terms), estimator=estimator)

    def test_basic_aggregation(self):
        c1 = self._make_fake_candidate(["x1"])
        c2 = self._make_fake_candidate(["x1", "x2"])
        raw = [
            {"candidate": c1, "fold_id": 0, "scores": {"rmse": 0.5}, "fit_warnings": [], "n_train": 70, "n_test": 30},
            {"candidate": c1, "fold_id": 1, "scores": {"rmse": 0.6}, "fit_warnings": [], "n_train": 70, "n_test": 30},
            {"candidate": c2, "fold_id": 0, "scores": {"rmse": 0.4}, "fit_warnings": [], "n_train": 70, "n_test": 30},
            {"candidate": c2, "fold_id": 1, "scores": {"rmse": 0.3}, "fit_warnings": [], "n_train": 70, "n_test": 30},
        ]
        lb = aggregate_fold_results(raw, "rmse", higher_is_better=False)
        assert len(lb) == 2
        # c2 should be ranked first (lower RMSE)
        assert lb.iloc[0]["candidate_hash"] == c2.candidate_hash

    def test_empty_scores_handled(self):
        c = self._make_fake_candidate(["x1"])
        raw = [
            {"candidate": c, "fold_id": 0, "scores": {}, "fit_warnings": ["error"], "n_train": 70, "n_test": 30},
        ]
        lb = aggregate_fold_results(raw, "rmse")
        assert len(lb) == 1
        # NaN for missing metric
        import math
        assert math.isnan(lb.iloc[0]["mean_rmse"])
