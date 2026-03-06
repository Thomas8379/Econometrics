"""Evaluation protocols for honest sieve assessment.

Core principle: selection and ranking must be done on data not used for
confirming the selected model.  Three protocols are provided:

1. **Holdout** (default): single 70/30 split; selection on train, report on test.
2. **K-fold cross-validation**: for predictive scoring; grouped splitting available.
3. **Cross-fitting** (recommended for IV): for each fold, feature/instrument
   selection is done on k-1 folds and estimation on the held-out fold.

All protocols return :class:`SplitSpec` objects that carry train/test index
arrays, enabling replication from stored fold indices.

Public API
----------
SplitSpec
holdout_split(n, test_frac, seed, group_arr) -> SplitSpec
kfold_splits(n, k, seed, group_arr) -> list[SplitSpec]
crossfit_splits(n, k, seed, group_arr) -> list[SplitSpec]
EvalProtocol
run_protocol(protocol, candidates, data, scorer_fn, *, seed) -> list[dict]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from econtools.sieve.candidates import Candidate
from econtools.sieve.fitters import FitResult, fit_candidate
from econtools.sieve.scorers import score_iv, score_ols


# ---------------------------------------------------------------------------
# SplitSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitSpec:
    """Train / test index split.

    Parameters
    ----------
    train_idx:
        Integer positions (iloc) for the training set.
    test_idx:
        Integer positions (iloc) for the test/evaluation set.
    fold_id:
        Identifier (0-based for k-fold, 0 for holdout).
    """

    train_idx: tuple[int, ...]
    test_idx: tuple[int, ...]
    fold_id: int | str = 0

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


# ---------------------------------------------------------------------------
# Split constructors
# ---------------------------------------------------------------------------


def holdout_split(
    n: int,
    *,
    test_frac: float = 0.30,
    seed: int = 0,
    group_arr: np.ndarray | None = None,
) -> SplitSpec:
    """Create a single train/test holdout split.

    Parameters
    ----------
    n:
        Total number of observations.
    test_frac:
        Fraction of data reserved for the test (confirm) set.
    seed:
        Random seed for reproducibility.
    group_arr:
        1-D array of group labels (e.g. cluster IDs).  If provided, entire
        groups are kept together (grouped split).

    Returns
    -------
    SplitSpec
    """
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)

    if group_arr is not None:
        groups = np.unique(group_arr)
        rng.shuffle(groups)
        n_test_groups = max(1, int(np.ceil(len(groups) * test_frac)))
        test_groups = set(groups[:n_test_groups])
        test_mask = np.isin(group_arr, list(test_groups))
        test_idx = all_idx[test_mask]
        train_idx = all_idx[~test_mask]
    else:
        perm = rng.permutation(n)
        n_test = max(1, int(np.ceil(n * test_frac)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

    return SplitSpec(
        train_idx=tuple(int(i) for i in sorted(train_idx)),
        test_idx=tuple(int(i) for i in sorted(test_idx)),
        fold_id=0,
    )


def kfold_splits(
    n: int,
    k: int = 5,
    *,
    seed: int = 0,
    group_arr: np.ndarray | None = None,
) -> list[SplitSpec]:
    """Create k-fold cross-validation splits.

    Parameters
    ----------
    n:
        Total number of observations.
    k:
        Number of folds.
    seed:
        Random seed.
    group_arr:
        Group labels for grouped k-fold (no group straddles two folds).

    Returns
    -------
    List of k :class:`SplitSpec` objects.
    """
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)
    splits: list[SplitSpec] = []

    if group_arr is not None:
        groups = np.unique(group_arr)
        rng.shuffle(groups)
        fold_assignments = {g: i % k for i, g in enumerate(groups)}
        for fold in range(k):
            test_mask = np.array([fold_assignments[g] == fold for g in group_arr])
            test_idx = all_idx[test_mask]
            train_idx = all_idx[~test_mask]
            splits.append(SplitSpec(
                train_idx=tuple(int(i) for i in sorted(train_idx)),
                test_idx=tuple(int(i) for i in sorted(test_idx)),
                fold_id=fold,
            ))
    else:
        perm = rng.permutation(n)
        fold_boundaries = np.array_split(perm, k)
        for fold, test_block in enumerate(fold_boundaries):
            test_idx = test_block
            train_idx = np.concatenate([b for i, b in enumerate(fold_boundaries) if i != fold])
            splits.append(SplitSpec(
                train_idx=tuple(int(i) for i in sorted(train_idx)),
                test_idx=tuple(int(i) for i in sorted(test_idx)),
                fold_id=fold,
            ))

    return splits


def crossfit_splits(
    n: int,
    k: int = 5,
    *,
    seed: int = 0,
    group_arr: np.ndarray | None = None,
) -> list[SplitSpec]:
    """Alias for :func:`kfold_splits` with explicit cross-fitting semantics.

    In cross-fitting mode, the caller is responsible for using the training
    folds for feature/instrument selection and the test fold for estimation.
    The split structure is identical to k-fold.
    """
    return kfold_splits(n, k, seed=seed, group_arr=group_arr)


# ---------------------------------------------------------------------------
# EvalProtocol
# ---------------------------------------------------------------------------


@dataclass
class EvalProtocol:
    """Configuration for a sieve evaluation protocol.

    Parameters
    ----------
    mode:
        ``"holdout"`` (default), ``"cv"``, or ``"crossfit"``.
    k:
        Number of folds (for ``"cv"`` and ``"crossfit"``).
    test_frac:
        Hold-out fraction (for ``"holdout"`` only).
    grouped_by:
        Column name to use for grouped splitting (prevents group leakage).
    exploratory_mode:
        If ``True``, in-sample scores are used.  Output is stamped
        ``exploratory_only=True`` in the manifest.
    """

    mode: str = "holdout"
    k: int = 5
    test_frac: float = 0.30
    grouped_by: str | None = None
    exploratory_mode: bool = False

    @classmethod
    def from_config(cls, protocol_cfg: dict) -> "EvalProtocol":
        mode = protocol_cfg.get("mode", "holdout")
        if mode not in ("holdout", "cv", "crossfit"):
            raise ValueError(f"Unknown protocol mode '{mode}'. Choose from: holdout, cv, crossfit.")
        allow_exploratory = protocol_cfg.get("allow_in_sample_selection", False)
        if allow_exploratory:
            import warnings
            warnings.warn(
                "allow_in_sample_selection=true: outputs will be stamped as "
                "EXPLORATORY ONLY and p-values after selection are not valid.",
                UserWarning,
                stacklevel=3,
            )
        return cls(
            mode=mode,
            k=int(protocol_cfg.get("k", 5)),
            test_frac=float(protocol_cfg.get("test_frac", 0.30)),
            grouped_by=protocol_cfg.get("grouped_by"),
            exploratory_mode=allow_exploratory,
        )


def _get_group_arr(data: pd.DataFrame, col: str | None) -> np.ndarray | None:
    if col is None or col not in data.columns:
        return None
    return data[col].values


def run_protocol(
    protocol: EvalProtocol,
    candidates: list[Candidate],
    data: pd.DataFrame,
    *,
    seed: int = 12345,
    score_fn: Callable[[FitResult, pd.DataFrame | None], dict[str, float]] | None = None,
    is_iv: bool = False,
    min_first_stage_f: float = 10.0,
    max_instruments: int | None = None,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Fit and score all candidates according to the evaluation protocol.

    Parameters
    ----------
    protocol:
        :class:`EvalProtocol` configuration.
    candidates:
        List of :class:`~econtools.sieve.candidates.Candidate` objects to evaluate.
    data:
        Full DataFrame.
    seed:
        Master random seed (derived sub-seeds are used per fold).
    score_fn:
        Callable ``(FitResult, eval_data_or_None) -> dict``.  Defaults to
        :func:`~econtools.sieve.scorers.score_ols` or
        :func:`~econtools.sieve.scorers.score_iv`.
    is_iv:
        If ``True``, use IV scorer by default.
    min_first_stage_f / max_instruments:
        Passed to IV scorer.
    n_jobs:
        Number of parallel workers (>1 requires joblib).

    Returns
    -------
    List of result dicts, one per (candidate × fold):
    ``{"candidate", "fold_id", "scores", "fit_warnings", "n_train", "n_test"}``.
    """
    group_arr = _get_group_arr(data, protocol.grouped_by)
    n = len(data)

    if score_fn is None:
        if is_iv:
            score_fn = lambda f, ed: score_iv(
                f, ed, min_first_stage_f=min_first_stage_f, max_instruments=max_instruments
            )
        else:
            score_fn = score_ols

    # Build splits
    if protocol.exploratory_mode:
        # In-sample: no split
        splits = [SplitSpec(
            train_idx=tuple(range(n)),
            test_idx=tuple(range(n)),
            fold_id="in_sample",
        )]
    elif protocol.mode == "holdout":
        splits = [holdout_split(n, test_frac=protocol.test_frac, seed=seed, group_arr=group_arr)]
    elif protocol.mode in ("cv", "crossfit"):
        splits = kfold_splits(n, protocol.k, seed=seed, group_arr=group_arr)
    else:
        raise ValueError(f"Unknown protocol mode: {protocol.mode!r}")

    results: list[dict[str, Any]] = []

    def _eval_one(cand: Candidate, split: SplitSpec) -> dict[str, Any]:
        train_data = data.iloc[list(split.train_idx)]
        eval_data = None if protocol.exploratory_mode else data.iloc[list(split.test_idx)]
        fit = fit_candidate(cand, train_data)
        if fit is None:
            return {
                "candidate": cand,
                "fold_id": split.fold_id,
                "scores": {},
                "fit_warnings": ["fit_candidate returned None"],
                "n_train": split.n_train,
                "n_test": split.n_test,
            }
        scores = score_fn(fit, eval_data)
        return {
            "candidate": cand,
            "fold_id": split.fold_id,
            "scores": scores,
            "fit_warnings": fit.warnings,
            "n_train": split.n_train,
            "n_test": split.n_test,
        }

    if n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(_eval_one)(cand, split)
                for cand in candidates
                for split in splits
            )
        except ImportError:
            # Fall back to sequential
            for cand in candidates:
                for split in splits:
                    results.append(_eval_one(cand, split))
    else:
        for cand in candidates:
            for split in splits:
                results.append(_eval_one(cand, split))

    return results


def aggregate_fold_results(
    raw_results: list[dict[str, Any]],
    primary_metric: str = "rmse",
    higher_is_better: bool = False,
) -> pd.DataFrame:
    """Aggregate per-fold scores into a candidate-level leaderboard.

    Parameters
    ----------
    raw_results:
        Output of :func:`run_protocol`.
    primary_metric:
        Metric to aggregate (mean across folds).
    higher_is_better:
        Determines sort order.

    Returns
    -------
    DataFrame with one row per candidate, sorted by the aggregated primary metric.
    """
    from collections import defaultdict
    import math

    candidate_scores: dict[str, list[float]] = defaultdict(list)
    candidate_meta: dict[str, dict] = {}
    candidate_warnings: dict[str, list[str]] = defaultdict(list)
    all_metric_names: set[str] = set()

    for rec in raw_results:
        cand = rec["candidate"]
        h = cand.candidate_hash
        scores = rec["scores"]
        all_metric_names.update(scores.keys())
        if primary_metric in scores and not math.isnan(scores[primary_metric]):
            candidate_scores[h].append(scores[primary_metric])
        candidate_warnings[h].extend(rec.get("fit_warnings", []))
        if h not in candidate_meta:
            candidate_meta[h] = {
                "candidate_hash": h,
                "estimator": cand.estimator,
                "n_X_terms": len(cand.X_terms),
                "n_Z_terms": len(cand.Z_terms) if cand.Z_terms else 0,
                "cov_type": cand.cov_type,
                "X_terms": "|".join(sorted(cand.X_terms)),
                "Z_terms": "|".join(sorted(cand.Z_terms)) if cand.Z_terms else "",
            }
        # Store all metrics (last fold wins for multi-metric display)
        for k, v in scores.items():
            candidate_meta[h][f"score_{k}"] = v

    rows = []
    for h, meta in candidate_meta.items():
        scores_list = candidate_scores.get(h, [])
        agg = float(sum(scores_list) / len(scores_list)) if scores_list else float("nan")
        rows.append({
            **meta,
            f"mean_{primary_metric}": agg,
            "n_folds": len(scores_list),
            "warnings": "; ".join(set(candidate_warnings[h]))[:200],
        })

    df = pd.DataFrame(rows)
    if f"mean_{primary_metric}" in df.columns:
        df = df.sort_values(
            f"mean_{primary_metric}",
            ascending=not higher_is_better,
            na_position="last",
        )
    return df.reset_index(drop=True)
