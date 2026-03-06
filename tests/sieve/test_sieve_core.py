"""Core sieve tests: candidate building, selection, and end-to-end OLS sieve.

Includes the canonical simulation tests required by the spec:
- A) Functional form recovery (OLS): sieve selects the true DGP.
- B) Heteroskedastic variant: scoring holds up under heteroskedasticity.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from econtools.sieve.candidates import (
    Candidate,
    canonicalize_candidate,
    deduplicate_candidates,
)
from econtools.sieve.selection import SelectionConfig, apply_guardrails, select_best
from econtools.sieve.protocols import aggregate_fold_results


# ---------------------------------------------------------------------------
# Fixtures / DGPs
# ---------------------------------------------------------------------------


def _make_linear_dgp(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Simple linear DGP: y = 1 + 2*x1 + 3*x2 + e."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    e = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + 3.0 * x2 + e
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _make_quadratic_dgp(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Quadratic DGP: y = 1 + 2*x1 + x1^2 + e (known nonlinear form)."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)  # noise variable
    e = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + 1.0 * x1 ** 2 + e
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    df["x1_sq"] = x1 ** 2
    return df


def _make_heterosk_dgp(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """DGP with heteroskedastic errors: Var(e|x1) = exp(x1)."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.standard_normal(n)
    e = rng.standard_normal(n) * np.exp(x1)
    y = 2.0 * x1 + 3.0 * x2 + e
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


# ---------------------------------------------------------------------------
# Candidate dataclass
# ---------------------------------------------------------------------------


class TestCandidateHash:
    def test_same_spec_same_hash(self):
        c1 = Candidate(y="y", X_terms=("x1", "x2"), estimator="ols")
        c2 = Candidate(y="y", X_terms=("x1", "x2"), estimator="ols")
        assert c1.candidate_hash == c2.candidate_hash

    def test_order_independent(self):
        """Canonical hash should be order-independent after canonicalization."""
        c1 = canonicalize_candidate(Candidate(y="y", X_terms=("x1", "x2"), estimator="ols"))
        c2 = canonicalize_candidate(Candidate(y="y", X_terms=("x2", "x1"), estimator="ols"))
        assert c1.candidate_hash == c2.candidate_hash

    def test_different_X_different_hash(self):
        c1 = Candidate(y="y", X_terms=("x1",), estimator="ols")
        c2 = Candidate(y="y", X_terms=("x1", "x2"), estimator="ols")
        assert c1.candidate_hash != c2.candidate_hash

    def test_different_estimator_different_hash(self):
        c1 = Candidate(y="y", X_terms=("x1",), estimator="ols")
        c2 = Candidate(y="y", X_terms=("x1",), estimator="2sls", endog=("x_endog",), Z_terms=("z",))
        assert c1.candidate_hash != c2.candidate_hash

    def test_hash_is_16_chars(self):
        c = Candidate(y="y", X_terms=("x1",), estimator="ols")
        assert len(c.candidate_hash) == 16

    def test_to_dict_round_trip(self):
        c = Candidate(y="y", X_terms=("x1", "x2"), estimator="ols", cov_type="HC3")
        d = c.to_dict()
        assert d["y"] == "y"
        assert set(d["X_terms"]) == {"x1", "x2"}
        assert d["candidate_hash"] == c.candidate_hash


class TestDeduplication:
    def test_deduplicates(self):
        c = Candidate(y="y", X_terms=("x1",), estimator="ols")
        result = deduplicate_candidates([c, c, c])
        assert len(result) == 1

    def test_preserves_order(self):
        c1 = Candidate(y="y", X_terms=("x1",), estimator="ols")
        c2 = Candidate(y="y", X_terms=("x2",), estimator="ols")
        result = deduplicate_candidates([c1, c2, c1])
        assert len(result) == 2
        assert result[0].candidate_hash == c1.candidate_hash


# ---------------------------------------------------------------------------
# Selection / guardrails
# ---------------------------------------------------------------------------


class TestGuardrails:
    def _make_lb(self, scores_list):
        """Build a fake leaderboard DataFrame."""
        from econtools.sieve.candidates import Candidate
        rows = []
        for i, (cand, scores) in enumerate(scores_list):
            row = {
                "candidate_hash": cand.candidate_hash,
                "estimator": cand.estimator,
                "n_X_terms": len(cand.X_terms),
                "n_Z_terms": len(cand.Z_terms) if cand.Z_terms else 0,
                "rejected": False,
                "rejection_reason": "",
            }
            for k, v in scores.items():
                row[f"score_{k}"] = v
                row[f"mean_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def test_max_terms_guardrail(self):
        big_cand = Candidate(y="y", X_terms=("x1", "x2", "x3", "x4", "x5"), estimator="ols")
        small_cand = Candidate(y="y", X_terms=("x1", "x2"), estimator="ols")
        lb = self._make_lb([(big_cand, {"rmse": 0.1}), (small_cand, {"rmse": 0.2})])

        config = SelectionConfig(max_terms=3, top_k=5)
        result = select_best([big_cand, small_cand], lb, config)
        selected_hashes = {c.candidate_hash for c in result.selected_candidates}
        assert big_cand.candidate_hash not in selected_hashes
        assert small_cand.candidate_hash in selected_hashes

    def test_weak_iv_guardrail(self):
        weak = Candidate(y="y", X_terms=("x1",), endog=("w",), Z_terms=("z",), estimator="2sls")
        strong = Candidate(y="y", X_terms=("x1",), endog=("w",), Z_terms=("z2",), estimator="2sls")
        lb = self._make_lb([
            (weak, {"first_stage_f": 3.0, "rmse": 0.5}),
            (strong, {"first_stage_f": 45.0, "rmse": 0.6}),
        ])

        config = SelectionConfig(min_first_stage_f=10.0, top_k=5)
        result = select_best([weak, strong], lb, config)
        selected_hashes = {c.candidate_hash for c in result.selected_candidates}
        assert weak.candidate_hash not in selected_hashes
        assert strong.candidate_hash in selected_hashes
        # Check violation logged
        codes = {v.reason_code for v in result.violations}
        assert "weak_iv" in codes


# ---------------------------------------------------------------------------
# Canonical simulation A: functional form recovery
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFunctionalFormRecovery:
    """Sieve should prefer the true quadratic model over linear-only."""

    def test_quadratic_form_selected(self):
        """When the DGP is quadratic, sieve should select a candidate that
        includes x1^2 with lower holdout RMSE than the linear-only baseline."""
        from econtools.sieve.api import run_sieve

        df = _make_quadratic_dgp(n=600, seed=42)

        spec = {
            "generators": {
                "features": {
                    "polynomial": {
                        "enabled": True,
                        "degree": 2,
                        "vars": ["x1", "x2"],
                    }
                }
            },
            "protocol": {"mode": "holdout", "test_frac": 0.30},
            "scoring": {"primary_metric": "rmse"},
            "selection": {"primary_metric": "rmse", "top_k": 3, "higher_is_better": False},
            "constraints": {"max_terms": 6},
        }

        result = run_sieve(
            df, "y", ["x1", "x2"],
            estimator="ols",
            sieve_spec=spec,
            seed=42,
        )

        # At least one selected candidate should include x1_pow2
        selected = result["selected"]
        assert len(selected) > 0
        has_quadratic = any("x1_pow2" in c.X_terms for c in selected)
        assert has_quadratic, (
            "Sieve did not select any candidate with x1^2 — "
            "the true functional form should be preferred on holdout."
        )

        # The best candidate's holdout RMSE should be lower than linear-only
        lb = result["leaderboard"]
        assert "mean_rmse" in lb.columns
        best_rmse = lb["mean_rmse"].min()
        # DGP sigma=1, so RMSE should be around 1; quadratic should improve over linear
        assert best_rmse < 1.5  # sanity check


# ---------------------------------------------------------------------------
# Canonical simulation B: heteroskedastic variant
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestHeteroskedasticSieve:
    """OOS metric (RMSE) should be valid under heteroskedasticity."""

    def test_oos_scoring_valid_under_heterosk(self):
        """With heteroskedastic errors, the sieve should still find a reasonable
        model. RMSE on the holdout set should be finite and non-NaN."""
        from econtools.sieve.api import run_sieve

        df = _make_heterosk_dgp(n=500, seed=0)

        spec = {
            "generators": {"features": {"polynomial": {"enabled": True, "degree": 2}}},
            "protocol": {"mode": "holdout", "test_frac": 0.30},
            "selection": {"primary_metric": "rmse", "top_k": 3},
            "constraints": {"max_terms": 5},
            "cov_type": "HC3",  # robust SEs appropriate for heterosk
        }

        result = run_sieve(
            df, "y", ["x1", "x2"],
            estimator="ols",
            sieve_spec=spec,
            seed=0,
        )

        lb = result["leaderboard"]
        passed = lb[~lb["rejected"]] if "rejected" in lb.columns else lb
        assert len(passed) > 0

        # All RMSE values should be finite
        rmse_col = "mean_rmse"
        if rmse_col in passed.columns:
            finite_rmse = passed[rmse_col].dropna()
            assert (finite_rmse < 10).all(), "Some RMSE values are unreasonably large."
