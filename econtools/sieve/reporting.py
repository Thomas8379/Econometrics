"""Sieve result reporting: model cards, comparison tables, and summaries.

Converts :class:`~econtools.sieve.selection.SelectionResult` and per-candidate
fit information into human-readable outputs (JSON model cards, text summaries,
and LaTeX tables via :mod:`econtools.output.tables.pub_latex`).

Public API
----------
model_card(candidate, fit_result, scores) -> dict
write_model_cards(selection, fit_results, scores, output_dir)
leaderboard_summary(leaderboard, n_show) -> str
sieve_latex_table(selection, fit_results, ...) -> str
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from econtools.sieve.candidates import Candidate
from econtools.sieve.fitters import FitResult
from econtools.sieve.selection import SelectionResult


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------


def model_card(
    candidate: Candidate,
    fit_result: FitResult | None,
    scores: dict[str, float],
    *,
    run_id: str = "",
    confirmatory: bool = True,
) -> dict[str, Any]:
    """Build a structured model card for a single selected candidate.

    Parameters
    ----------
    candidate:
        The selected candidate.
    fit_result:
        Full-sample fit result (if available; may be ``None``).
    scores:
        Aggregated evaluation scores.
    run_id:
        Parent run identifier.
    confirmatory:
        If ``False``, the card is stamped as exploratory.

    Returns
    -------
    dict suitable for JSON serialisation.
    """
    card: dict[str, Any] = {
        "run_id": run_id,
        "candidate_hash": candidate.candidate_hash,
        "confirmatory": confirmatory,
        "model": {
            "y": candidate.y,
            "X_terms": list(candidate.X_terms),
            "endog": list(candidate.endog) if candidate.endog else None,
            "Z_terms": list(candidate.Z_terms) if candidate.Z_terms else None,
            "estimator": candidate.estimator,
            "intercept": candidate.intercept,
            "cov_type": candidate.cov_type,
        },
        "scores": {k: (None if math.isnan(v) else v) for k, v in scores.items()},
        "transforms": [t.to_dict() for t in candidate.transforms],
    }

    if fit_result is not None and len(fit_result.params) > 0:
        card["coefficients"] = {
            str(k): {
                "coef": float(fit_result.params[k]),
                "se": float(fit_result.bse[k]) if k in fit_result.bse.index else None,
                "pvalue": float(fit_result.pvalues[k]) if k in fit_result.pvalues.index else None,
            }
            for k in fit_result.params.index
        }
        card["n_obs"] = fit_result.n_obs
        if not math.isnan(fit_result.first_stage_f):
            card["first_stage_f"] = fit_result.first_stage_f
        if not math.isnan(fit_result.overid_pvalue):
            card["overid_pvalue"] = fit_result.overid_pvalue

    if not confirmatory:
        card["WARNING"] = (
            "EXPLORATORY ONLY — in-sample selection. "
            "Post-selection p-values are not valid."
        )

    return card


def write_model_cards(
    selection: SelectionResult,
    fit_results: dict[str, FitResult],
    score_records: list[dict[str, float]],
    output_dir: Path,
    *,
    run_id: str = "",
) -> list[Path]:
    """Write one JSON model card per selected candidate.

    Parameters
    ----------
    selection:
        Output of :func:`~econtools.sieve.selection.select_best`.
    fit_results:
        Dict of candidate_hash -> FitResult (full-sample fits).
    score_records:
        List of score dicts with ``candidate_hash`` keys.
    output_dir:
        Directory to write cards into (``selected_models/`` subfolder created).
    run_id:
        Parent run identifier.

    Returns
    -------
    List of written Path objects.
    """
    out_dir = Path(output_dir) / "selected_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    score_by_hash = {r["candidate_hash"]: r for r in score_records if "candidate_hash" in r}
    written: list[Path] = []

    for cand in selection.selected_candidates:
        h = cand.candidate_hash
        fr = fit_results.get(h)
        scores = score_by_hash.get(h, {})
        card = model_card(
            cand,
            fr,
            {k: v for k, v in scores.items() if isinstance(v, (int, float))},
            run_id=run_id,
            confirmatory=not selection.exploratory_only,
        )
        path = out_dir / f"model_{h}.json"
        path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# Leaderboard text summary
# ---------------------------------------------------------------------------


def leaderboard_summary(
    leaderboard: pd.DataFrame,
    n_show: int = 10,
    *,
    primary_metric: str | None = None,
    exploratory_only: bool = False,
) -> str:
    """Return a human-readable text summary of the leaderboard.

    Parameters
    ----------
    leaderboard:
        Output of :func:`~econtools.sieve.protocols.aggregate_fold_results`.
    n_show:
        Number of rows to show.
    primary_metric:
        Column name to highlight.
    exploratory_only:
        If ``True``, prepend an exploratory warning.

    Returns
    -------
    Multi-line string.
    """
    lines: list[str] = []

    if exploratory_only:
        lines += [
            "=" * 70,
            "WARNING: EXPLORATORY ONLY — in-sample selection was used.",
            "Post-selection p-values are NOT valid for inference.",
            "=" * 70,
            "",
        ]

    n_total = len(leaderboard)
    n_pass = int(leaderboard["rejected"].eq(False).sum()) if "rejected" in leaderboard.columns else n_total
    n_rej = n_total - n_pass

    lines.append(f"Sieve Leaderboard — {n_total} candidates ({n_pass} passed, {n_rej} rejected)")
    lines.append("-" * 70)

    show_cols = ["candidate_hash", "estimator", "n_X_terms"]
    if primary_metric and f"mean_{primary_metric}" in leaderboard.columns:
        show_cols.append(f"mean_{primary_metric}")
    elif primary_metric and primary_metric in leaderboard.columns:
        show_cols.append(primary_metric)

    for col in ["rejected", "rejection_reason"]:
        if col in leaderboard.columns:
            show_cols.append(col)

    sub = leaderboard[show_cols].head(n_show)
    lines.append(sub.to_string(index=False))
    lines.append("-" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX output of sieve results
# ---------------------------------------------------------------------------


def sieve_latex_table(
    selection: SelectionResult,
    fit_results: dict[str, FitResult],
    *,
    caption: str | None = "Sieve-Selected Models",
    label: str | None = "tab:sieve",
    digits: int = 3,
    footer_stats: list[str] | None = None,
    notes: list[str] | None = None,
) -> str:
    """Produce a publication-quality LaTeX table of the selected models.

    Parameters
    ----------
    selection:
        Output of :func:`~econtools.sieve.selection.select_best`.
    fit_results:
        Dict of candidate_hash -> FitResult with full-sample fits.
    caption / label / digits / footer_stats / notes:
        Passed to :class:`~econtools.output.tables.pub_latex.ResultsTable`.

    Returns
    -------
    LaTeX table string.
    """
    from econtools._core.types import Estimate, FitMetrics
    from econtools.output.tables.pub_latex import ResultsTable
    import pandas as pd

    estimates: list[Estimate] = []
    labels: list[str] = []
    estimator_labels: list[str] = []

    for i, cand in enumerate(selection.selected_candidates):
        h = cand.candidate_hash
        fr = fit_results.get(h)
        if fr is None or len(fr.params) == 0:
            continue

        # Reconstruct a minimal Estimate for the table formatter
        n = fr.n_obs
        k = len(fr.params)
        fit_metrics = FitMetrics(
            nobs=n,
            df_model=float(k),
            df_resid=float(max(n - k, 1)),
            r_squared=fr.r_squared,
            f_stat=float("nan"),
        )
        est = Estimate(
            model_type=cand.estimator.upper(),
            dep_var=cand.y,
            params=fr.params,
            bse=fr.bse,
            tvalues=fr.params / fr.bse.replace(0, float("nan")),
            pvalues=fr.pvalues,
            conf_int_lower=pd.Series(dtype=float),
            conf_int_upper=pd.Series(dtype=float),
            resid=fr.resid,
            fitted=fr.fitted,
            cov_params=pd.DataFrame(),
            cov_type=cand.cov_type,
            fit=fit_metrics,
            raw=fr.raw,
        )
        estimates.append(est)
        labels.append(f"({i + 1})")
        estimator_labels.append(cand.estimator.upper())

    if not estimates:
        return "% No sieve-selected models to display."

    all_notes = list(notes or [])
    if selection.exploratory_only:
        all_notes.insert(0, r"\textbf{EXPLORATORY ONLY — in-sample selection.}")

    t = ResultsTable(
        results=estimates,
        labels=labels,
        estimator_labels=estimator_labels,
        footer_stats=footer_stats or ["N", "r_squared"],
        notes=all_notes,
        caption=caption,
        label=label,
        digits=digits,
    )
    return t.to_latex()
