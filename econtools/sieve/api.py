"""Public API for the sieve subsystem.

The primary entry point is :func:`run_sieve`.  It orchestrates candidate
generation, fitting, scoring, selection, and artifact writing.

Public API
----------
run_sieve(data, y, base_X, estimator, ...) -> dict
load_sieve_results(output_dir) -> dict
"""

from __future__ import annotations

import hashlib
import itertools
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from econtools.sieve.candidates import Candidate, TransformSpec, deduplicate_candidates
from econtools.sieve.fitters import FitResult, fit_candidate
from econtools.sieve.generators.features import apply_transforms
from econtools.sieve.generators.instruments import generate_instrument_candidates
from econtools.sieve.manifest import (
    build_manifest,
    compute_config_hash,
    write_manifest,
)
from econtools.sieve.protocols import EvalProtocol, aggregate_fold_results, run_protocol
from econtools.sieve.reporting import leaderboard_summary, write_model_cards
from econtools.sieve.scorers import HIGHER_IS_BETTER
from econtools.sieve.selection import SelectionConfig, select_best


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def _build_candidates(
    y: str,
    base_X: list[str],
    estimator: str,
    endog: list[str] | None,
    base_Z: list[str] | None,
    sieve_spec: dict,
    df: pd.DataFrame,
    *,
    config_hash: str,
    data_fingerprint: str,
    generator_hash: str,
) -> tuple[list[Candidate], pd.DataFrame]:
    """Generate all candidate model specifications.

    Returns ``(candidates, augmented_df)`` where ``augmented_df`` contains
    all generated feature and instrument columns.
    """
    cov_type = sieve_spec.get("cov_type", "HC3")
    cluster_var = sieve_spec.get("cluster_var")
    intercept = bool(sieve_spec.get("intercept", True))

    # ---- Feature generation ------------------------------------------------
    term_sets, df, feat_specs = apply_transforms(base_X, sieve_spec, df)

    # Each candidate is a subset of base_X + some generated features.
    # Strategy: enumerate powerset of term_sets; each item in term_sets
    # is a "block" added atomically.
    max_extra = sieve_spec.get("generators", {}).get("features", {}).get(
        "max_added_blocks", len(term_sets)
    )
    max_terms_total = sieve_spec.get("constraints", {}).get("max_terms_total", None)

    # Always include the baseline (just base_X)
    x_candidate_sets: list[tuple[str, ...]] = [tuple(base_X)]

    # Add each block individually and all combinations up to max_extra
    for r in range(1, min(max_extra, len(term_sets)) + 1):
        for combo in itertools.combinations(range(len(term_sets)), r):
            extra = list(itertools.chain.from_iterable(term_sets[i] for i in combo))
            x_set = tuple(dict.fromkeys(base_X + extra))  # preserve order, deduplicate
            if max_terms_total and len(x_set) > max_terms_total:
                continue
            x_candidate_sets.append(x_set)

    # Deduplicate X sets
    seen_x: set[frozenset[str]] = set()
    unique_x_sets: list[tuple[str, ...]] = []
    for xs in x_candidate_sets:
        fs = frozenset(xs)
        if fs not in seen_x:
            seen_x.add(fs)
            unique_x_sets.append(xs)

    # ---- Instrument generation (IV only) -----------------------------------
    z_candidate_sets: list[tuple[str, ...] | None] = [None]
    all_z_specs: list[TransformSpec] = []

    if estimator in ("2sls", "iv2sls", "fe_2sls") and base_Z:
        z_sets, df, all_z_specs = generate_instrument_candidates(
            base_Z, base_X, sieve_spec, df, y
        )
        max_inst = sieve_spec.get("constraints", {}).get("max_instruments", None)
        z_candidate_sets = [tuple(base_Z)]

        for r in range(1, len(z_sets) + 1):
            for combo in itertools.combinations(range(len(z_sets)), r):
                extra_z = list(itertools.chain.from_iterable(z_sets[i] for i in combo))
                z_set = tuple(dict.fromkeys(base_Z + extra_z))
                if max_inst and len(z_set) > max_inst:
                    continue
                z_candidate_sets.append(z_set)

        seen_z: set[frozenset[str]] = set()
        unique_z_sets: list[tuple[str, ...]] = []
        for zs in z_candidate_sets:
            if zs is None:
                continue
            fs = frozenset(zs)
            if fs not in seen_z:
                seen_z.add(fs)
                unique_z_sets.append(zs)
        z_candidate_sets = unique_z_sets or [tuple(base_Z)]

    # ---- Combine X and Z sets into candidates ------------------------------
    all_transforms = tuple(feat_specs + all_z_specs)
    endog_tuple = tuple(endog) if endog else None
    candidates: list[Candidate] = []

    for x_set in unique_x_sets:
        for z_set in z_candidate_sets:
            cand = Candidate(
                y=y,
                X_terms=x_set,
                endog=endog_tuple,
                Z_terms=z_set if (estimator in ("2sls", "iv2sls", "fe_2sls") and z_set) else None,
                estimator=estimator,
                intercept=intercept,
                cov_type=cov_type,
                cluster_var=cluster_var,
                transforms=all_transforms,
                config_hash=config_hash,
                data_fingerprint=data_fingerprint,
                generator_hash=generator_hash,
            )
            candidates.append(cand)

    return deduplicate_candidates(candidates), df


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def run_sieve(
    data: pd.DataFrame,
    y: str,
    base_X: list[str],
    estimator: str,
    *,
    endog: list[str] | None = None,
    base_Z: list[str] | None = None,
    sieve_spec: dict | None = None,
    seed: int = 12345,
    n_jobs: int = 1,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run a sieve search over candidate model specifications.

    Parameters
    ----------
    data:
        Input :class:`pandas.DataFrame`.
    y:
        Dependent variable name.
    base_X:
        Baseline exogenous regressor names.
    estimator:
        ``"ols"``, ``"2sls"``, ``"fe_ols"``, or ``"fe_2sls"``.
    endog:
        Endogenous variable names (required for ``"2sls"`` / ``"fe_2sls"``).
    base_Z:
        Baseline instrument names (required for IV).
    sieve_spec:
        Parsed sieve configuration dict (from YAML or Python).  If ``None``,
        a minimal default is used (baseline candidate only, holdout protocol).
    seed:
        Master random seed — same seed + same data + same config ⇒ identical results.
    n_jobs:
        Number of parallel worker processes (>1 requires ``joblib``).
    output_dir:
        Directory for artifact output.  If ``None``, nothing is written to disk.

    Returns
    -------
    dict with keys:

    * ``"selected"`` — list of selected :class:`~econtools.sieve.candidates.Candidate` objects
    * ``"leaderboard"`` — :class:`pandas.DataFrame` ranked by primary metric
    * ``"artifacts"`` — dict of written file paths
    * ``"run_metadata"`` — run ID, hashes, dataset fingerprint
    * ``"violations"`` — list of :class:`~econtools.sieve.selection.GuardrailViolation`
    * ``"full_fit_results"`` — dict of candidate_hash -> FitResult (full-sample)
    """
    spec = sieve_spec or {}

    # ---- Validate inputs ---------------------------------------------------
    if y not in data.columns:
        raise ValueError(f"Dependent variable '{y}' not in data.")
    missing_X = [x for x in base_X if x not in data.columns]
    if missing_X:
        raise ValueError(f"base_X columns missing from data: {missing_X}")
    if estimator in ("2sls", "iv2sls", "fe_2sls"):
        if not endog:
            raise ValueError("'endog' is required for IV estimators.")
        if not base_Z:
            raise ValueError("'base_Z' is required for IV estimators.")

    # ---- Run ID & hashing --------------------------------------------------
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"sieve_{ts}_{seed}"
    config_hash = compute_config_hash(spec)

    # Dataset fingerprint (over used columns only)
    used_cols = list(dict.fromkeys([y] + base_X + (endog or []) + (base_Z or [])))
    from econtools.sieve.manifest import compute_dataset_fingerprint
    data_fingerprint = compute_dataset_fingerprint(data, used_cols)

    # Generator hash (hash of the generator config section)
    gen_cfg = spec.get("generators", {})
    generator_hash = hashlib.sha256(
        json.dumps(gen_cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    # ---- Generate candidates -----------------------------------------------
    candidates, augmented_df = _build_candidates(
        y=y,
        base_X=base_X,
        estimator=estimator,
        endog=endog,
        base_Z=base_Z,
        sieve_spec=spec,
        df=data.copy(),
        config_hash=config_hash,
        data_fingerprint=data_fingerprint,
        generator_hash=generator_hash,
    )

    if not candidates:
        raise RuntimeError("No candidates were generated. Check sieve_spec and data.")

    # ---- Evaluation protocol -----------------------------------------------
    protocol_cfg = spec.get("protocol", {})
    protocol = EvalProtocol.from_config(protocol_cfg)

    is_iv = estimator in ("2sls", "iv2sls", "fe_2sls")
    constraints = spec.get("constraints", {})
    min_fs_f = float(constraints.get("min_first_stage_f", 10.0))
    max_inst = constraints.get("max_instruments")

    raw_results = run_protocol(
        protocol,
        candidates,
        augmented_df,
        seed=seed,
        is_iv=is_iv,
        min_first_stage_f=min_fs_f,
        max_instruments=max_inst,
        n_jobs=n_jobs,
    )

    # ---- Aggregate scores --------------------------------------------------
    selection_cfg = SelectionConfig.from_config(spec)
    primary_metric = selection_cfg.primary_metric
    higher_is_better = primary_metric in HIGHER_IS_BETTER or selection_cfg.higher_is_better

    leaderboard = aggregate_fold_results(raw_results, primary_metric, higher_is_better)

    # ---- Selection ---------------------------------------------------------
    selection = select_best(candidates, leaderboard, selection_cfg)

    # ---- Full-sample fits for selected candidates --------------------------
    full_fit_results: dict[str, FitResult] = {}
    for cand in selection.selected_candidates:
        fr = fit_candidate(cand, augmented_df)
        if fr is not None:
            full_fit_results[cand.candidate_hash] = fr

    # ---- Artifacts ---------------------------------------------------------
    artifacts: dict[str, Any] = {}

    if output_dir:
        out_path = Path(output_dir)

        # Manifest
        manifest = build_manifest(
            run_id=run_id,
            config=spec,
            df=data,
            used_cols=used_cols,
            n_candidates=len(candidates),
            n_selected=len(selection.selected_candidates),
            protocol_mode=protocol.mode,
            exploratory_only=selection.exploratory_only,
        )
        manifest_path = write_manifest(manifest, out_path)
        artifacts["manifest"] = str(manifest_path)

        # Leaderboard (parquet + CSV)
        try:
            leaderboard_path = out_path / "leaderboard.parquet"
            selection.leaderboard.to_parquet(str(leaderboard_path), index=False)
            artifacts["leaderboard_parquet"] = str(leaderboard_path)
        except Exception:
            pass
        try:
            csv_path = out_path / "leaderboard.csv"
            selection.leaderboard.to_csv(str(csv_path), index=False)
            artifacts["leaderboard_csv"] = str(csv_path)
        except Exception:
            pass

        # Model cards
        score_records = [
            {**r["scores"], "candidate_hash": r["candidate"].candidate_hash}
            for r in raw_results
        ]
        card_paths = write_model_cards(
            selection,
            full_fit_results,
            score_records,
            out_path,
            run_id=run_id,
        )
        artifacts["model_cards"] = [str(p) for p in card_paths]

    return {
        "selected": selection.selected_candidates,
        "leaderboard": selection.leaderboard,
        "artifacts": artifacts,
        "run_metadata": {
            "run_id": run_id,
            "config_hash": config_hash,
            "data_fingerprint": data_fingerprint,
            "n_candidates": len(candidates),
            "n_selected": len(selection.selected_candidates),
            "protocol": protocol.mode,
            "exploratory_only": selection.exploratory_only,
            "seed": seed,
        },
        "violations": selection.violations,
        "full_fit_results": full_fit_results,
    }


def load_sieve_results(output_dir: str | Path) -> dict[str, Any]:
    """Load previously saved sieve artifacts from *output_dir*.

    Parameters
    ----------
    output_dir:
        Directory containing ``run_manifest.json`` and ``leaderboard.parquet``/``.csv``.

    Returns
    -------
    dict with ``"manifest"`` and ``"leaderboard"`` keys.
    """
    from econtools.sieve.manifest import load_manifest
    out_path = Path(output_dir)
    result: dict[str, Any] = {}

    try:
        result["manifest"] = load_manifest(out_path)
    except FileNotFoundError:
        result["manifest"] = None

    for fname in ("leaderboard.parquet", "leaderboard.csv"):
        fpath = out_path / fname
        if fpath.exists():
            try:
                if fname.endswith(".parquet"):
                    result["leaderboard"] = pd.read_parquet(str(fpath))
                else:
                    result["leaderboard"] = pd.read_csv(str(fpath))
                break
            except Exception:
                pass

    card_dir = out_path / "selected_models"
    cards = []
    if card_dir.exists():
        for f in sorted(card_dir.glob("model_*.json")):
            try:
                cards.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
    result["model_cards"] = cards

    return result
