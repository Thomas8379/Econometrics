"""Selection rules for sieve candidates.

Applies guardrails, constraint checks, and ranking to produce a curated
shortlist from the full candidate pool.

Public API
----------
SelectionConfig
GuardrailViolation
apply_guardrails(candidates, scores, config) -> (passed, rejected)
select_best(candidates, scores, config) -> SelectionResult
pareto_frontier(candidates, scores, objectives, higher_is_better) -> list[int]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from econtools.sieve.candidates import Candidate


# ---------------------------------------------------------------------------
# Config and violation types
# ---------------------------------------------------------------------------


@dataclass
class SelectionConfig:
    """Configuration for candidate selection.

    Parameters
    ----------
    primary_metric:
        The metric to rank candidates by (e.g. ``"rmse"``, ``"aic"``).
    higher_is_better:
        True for R², first-stage F, etc.; False for RMSE, AIC, BIC.
    top_k:
        Maximum number of candidates to select.
    min_first_stage_f:
        Hard minimum for first-stage F (IV only).  Candidates below this
        threshold are rejected with code ``"weak_iv"``.
    max_terms:
        Maximum number of X terms allowed (complexity guardrail).
    max_instruments:
        Maximum number of instruments allowed.
    allow_in_sample_selection:
        Must be ``True`` to use in-sample metrics; if set, output is stamped
        ``exploratory_only``.
    pareto:
        If ``True``, return the Pareto-optimal set instead of top-k.
    pareto_objectives:
        Metrics to use for Pareto dominance (if ``pareto=True``).
    pareto_higher_is_better:
        Per-objective direction (same order as ``pareto_objectives``).
    sign_constraints:
        Dict of ``{variable_name: "positive" | "negative"}`` enforced on
        the fitted coefficients of the *selected* model.  Candidates
        violating sign constraints are rejected.
    """

    primary_metric: str = "rmse"
    higher_is_better: bool = False
    top_k: int = 5
    min_first_stage_f: float = 10.0
    max_terms: int | None = None
    max_instruments: int | None = None
    allow_in_sample_selection: bool = False
    pareto: bool = False
    pareto_objectives: list[str] = field(default_factory=lambda: ["rmse", "n_terms"])
    pareto_higher_is_better: list[bool] = field(default_factory=lambda: [False, False])
    sign_constraints: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict) -> "SelectionConfig":
        sc = cfg.get("selection", {})
        constraints = cfg.get("constraints", {})
        return cls(
            primary_metric=sc.get("primary_metric", "rmse"),
            higher_is_better=sc.get("higher_is_better", False),
            top_k=int(sc.get("top_k", 5)),
            min_first_stage_f=float(constraints.get("min_first_stage_f", 10.0)),
            max_terms=constraints.get("max_terms"),
            max_instruments=constraints.get("max_instruments"),
            allow_in_sample_selection=cfg.get("protocol", {}).get(
                "allow_in_sample_selection", False
            ),
            pareto=bool(sc.get("pareto", False)),
            pareto_objectives=sc.get("pareto_objectives", ["rmse", "n_terms"]),
            pareto_higher_is_better=sc.get("pareto_higher_is_better", [False, False]),
            sign_constraints=constraints.get("sign_constraints", {}),
        )


@dataclass
class GuardrailViolation:
    """Records why a candidate was rejected.

    Parameters
    ----------
    candidate_hash:
        Hash of the rejected candidate.
    reason_code:
        Short code: ``"weak_iv"``, ``"too_many_terms"``, ``"too_many_instruments"``,
        ``"sign_constraint"``, ``"fit_failed"``, ``"metric_missing"``,
        ``"collinear"``.
    details:
        Human-readable explanation.
    """

    candidate_hash: str
    reason_code: str
    details: str


# ---------------------------------------------------------------------------
# Guardrail application
# ---------------------------------------------------------------------------


def _score_metric(
    scores: dict[str, float],
    metric: str,
) -> float:
    for key in (metric, f"mean_{metric}", f"score_{metric}"):
        if key in scores:
            v = scores[key]
            if v is not None and not math.isnan(v):
                return float(v)
    return float("nan")


def apply_guardrails(
    candidates: list[Candidate],
    score_records: list[dict[str, Any]],
    config: SelectionConfig,
    *,
    fit_results: dict[str, Any] | None = None,
) -> tuple[list[int], list[GuardrailViolation]]:
    """Apply hard constraints and return (passing_indices, violations).

    Parameters
    ----------
    candidates:
        All candidates (indexed).
    score_records:
        Aggregated score records (dict per candidate with scores).
    config:
        Selection configuration.
    fit_results:
        Optional dict of candidate_hash -> FitResult for sign constraint checks.

    Returns
    -------
    (passing_indices, violations)
        ``passing_indices`` are indices into *candidates*; *violations* describe
        every rejected candidate.
    """
    passing: list[int] = []
    violations: list[GuardrailViolation] = []

    # Build score lookup by hash
    score_by_hash: dict[str, dict[str, Any]] = {}
    for rec in score_records:
        h = rec.get("candidate_hash", "")
        if h:
            score_by_hash[h] = rec

    for i, cand in enumerate(candidates):
        h = cand.candidate_hash
        scores = score_by_hash.get(h, {})
        rejected = False

        # Complexity: max_terms
        if config.max_terms is not None and len(cand.X_terms) > config.max_terms:
            violations.append(GuardrailViolation(
                candidate_hash=h,
                reason_code="too_many_terms",
                details=f"{len(cand.X_terms)} terms > max_terms={config.max_terms}",
            ))
            rejected = True

        # IV: first-stage F
        if not rejected and cand.estimator in ("2sls", "fe_2sls", "iv2sls"):
            fs_f = _score_metric(scores, "first_stage_f")
            if not math.isnan(fs_f) and fs_f < config.min_first_stage_f:
                violations.append(GuardrailViolation(
                    candidate_hash=h,
                    reason_code="weak_iv",
                    details=(
                        f"First-stage F={fs_f:.2f} < min_first_stage_f={config.min_first_stage_f}"
                    ),
                ))
                rejected = True

        # IV: max instruments
        if (
            not rejected
            and config.max_instruments is not None
            and cand.Z_terms
            and len(cand.Z_terms) > config.max_instruments
        ):
            violations.append(GuardrailViolation(
                candidate_hash=h,
                reason_code="too_many_instruments",
                details=(
                    f"{len(cand.Z_terms)} instruments > max_instruments={config.max_instruments}"
                ),
            ))
            rejected = True

        # Sign constraints (if fit_results provided)
        if not rejected and config.sign_constraints and fit_results and h in fit_results:
            fr = fit_results[h]
            for var, direction in config.sign_constraints.items():
                if hasattr(fr, "params") and var in fr.params.index:
                    coef = float(fr.params[var])
                    if direction == "positive" and coef < 0:
                        violations.append(GuardrailViolation(
                            candidate_hash=h,
                            reason_code="sign_constraint",
                            details=f"Variable '{var}' coef={coef:.4f} must be positive.",
                        ))
                        rejected = True
                        break
                    elif direction == "negative" and coef > 0:
                        violations.append(GuardrailViolation(
                            candidate_hash=h,
                            reason_code="sign_constraint",
                            details=f"Variable '{var}' coef={coef:.4f} must be negative.",
                        ))
                        rejected = True
                        break

        # Fit failed check
        if not rejected and scores.get("fit_failed"):
            violations.append(GuardrailViolation(
                candidate_hash=h,
                reason_code="fit_failed",
                details="Fitting produced no valid output.",
            ))
            rejected = True

        if not rejected:
            passing.append(i)

    return passing, violations


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


def pareto_frontier(
    score_records: list[dict[str, Any]],
    objectives: list[str],
    higher_is_better: list[bool],
) -> list[int]:
    """Return indices of Pareto-non-dominated score records.

    A record is dominated if another record is at least as good on all
    objectives and strictly better on at least one.

    Parameters
    ----------
    score_records:
        List of score dicts.
    objectives:
        Metric names to consider.
    higher_is_better:
        Whether each objective should be maximised.

    Returns
    -------
    List of indices into *score_records* that are Pareto-non-dominated.
    """
    n = len(score_records)
    dominated = [False] * n

    # Normalise direction: all objectives should be "higher is better"
    def _norm(val: float, hib: bool) -> float:
        return val if hib else -val

    vals: list[list[float]] = []
    for rec in score_records:
        row = [_norm(_score_metric(rec, obj), hib) for obj, hib in zip(objectives, higher_is_better)]
        vals.append(row)

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            # Does j dominate i?
            j_better_all = all(vals[j][k] >= vals[i][k] for k in range(len(objectives)))
            j_better_one = any(vals[j][k] > vals[i][k] for k in range(len(objectives)))
            if j_better_all and j_better_one:
                dominated[i] = True
                break

    return [i for i in range(n) if not dominated[i]]


# ---------------------------------------------------------------------------
# Main selection function
# ---------------------------------------------------------------------------


@dataclass
class SelectionResult:
    """Output of :func:`select_best`.

    Parameters
    ----------
    selected_indices:
        Indices into the original *candidates* list.
    selected_candidates:
        Selected :class:`~econtools.sieve.candidates.Candidate` objects.
    leaderboard:
        Full ranked DataFrame of all candidates (passed + rejected).
    violations:
        List of guardrail violations.
    exploratory_only:
        ``True`` if in-sample selection was used.
    """

    selected_indices: list[int]
    selected_candidates: list[Candidate]
    leaderboard: pd.DataFrame
    violations: list[GuardrailViolation]
    exploratory_only: bool = False


def select_best(
    candidates: list[Candidate],
    leaderboard: pd.DataFrame,
    config: SelectionConfig,
) -> SelectionResult:
    """Select top-k candidates from the leaderboard after applying guardrails.

    Parameters
    ----------
    candidates:
        Full list of candidates (hash must match leaderboard rows).
    leaderboard:
        Output of :func:`~econtools.sieve.protocols.aggregate_fold_results`.
    config:
        Selection configuration.

    Returns
    -------
    :class:`SelectionResult`
    """
    # Add rejection reason column
    leaderboard = leaderboard.copy()
    leaderboard["rejected"] = False
    leaderboard["rejection_reason"] = ""

    passing_indices: list[int] = []
    violations: list[GuardrailViolation] = []

    cand_by_hash = {c.candidate_hash: (i, c) for i, c in enumerate(candidates)}

    primary = f"mean_{config.primary_metric}"
    if primary not in leaderboard.columns:
        primary = config.primary_metric

    for lb_idx, row in leaderboard.iterrows():
        h = row.get("candidate_hash", "")
        if h not in cand_by_hash:
            continue
        cand_idx, cand = cand_by_hash[h]
        rejected = False

        # Complexity
        if config.max_terms and row.get("n_X_terms", 0) > config.max_terms:
            v = GuardrailViolation(h, "too_many_terms",
                                   f"{row['n_X_terms']} > {config.max_terms}")
            violations.append(v)
            leaderboard.at[lb_idx, "rejected"] = True
            leaderboard.at[lb_idx, "rejection_reason"] = "too_many_terms"
            rejected = True

        # IV guardrails
        fs_col = "score_first_stage_f"
        if not rejected and cand.estimator in ("2sls", "fe_2sls"):
            fs_f = float(row.get(fs_col, float("nan")))
            if not math.isnan(fs_f) and fs_f < config.min_first_stage_f:
                v = GuardrailViolation(h, "weak_iv",
                                       f"F={fs_f:.2f} < {config.min_first_stage_f}")
                violations.append(v)
                leaderboard.at[lb_idx, "rejected"] = True
                leaderboard.at[lb_idx, "rejection_reason"] = "weak_iv"
                rejected = True

        if not rejected and config.max_instruments and row.get("n_Z_terms", 0) > config.max_instruments:
            v = GuardrailViolation(h, "too_many_instruments",
                                   f"{row['n_Z_terms']} > {config.max_instruments}")
            violations.append(v)
            leaderboard.at[lb_idx, "rejected"] = True
            leaderboard.at[lb_idx, "rejection_reason"] = "too_many_instruments"
            rejected = True

        if not rejected:
            passing_indices.append(cand_idx)

    # Rank passing candidates
    passed_hashes = {candidates[i].candidate_hash for i in passing_indices}
    passed_lb = leaderboard[leaderboard["candidate_hash"].isin(passed_hashes)]

    if config.pareto and len(passed_lb) > 0:
        passed_records = passed_lb.to_dict("records")
        pareto_idxs = pareto_frontier(
            passed_records,
            config.pareto_objectives,
            config.pareto_higher_is_better,
        )
        selected_hashes = {passed_records[i]["candidate_hash"] for i in pareto_idxs}
    else:
        if primary in passed_lb.columns:
            passed_lb_sorted = passed_lb.sort_values(
                primary, ascending=not config.higher_is_better, na_position="last"
            )
        else:
            passed_lb_sorted = passed_lb
        selected_hashes = set(passed_lb_sorted["candidate_hash"].head(config.top_k).tolist())

    selected_indices = [i for i, c in enumerate(candidates) if c.candidate_hash in selected_hashes]
    selected_candidates = [candidates[i] for i in selected_indices]

    return SelectionResult(
        selected_indices=selected_indices,
        selected_candidates=selected_candidates,
        leaderboard=leaderboard,
        violations=violations,
        exploratory_only=config.allow_in_sample_selection,
    )
