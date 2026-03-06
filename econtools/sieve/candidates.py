"""Candidate dataclass, hashing, and canonicalization.

A :class:`Candidate` is a fully-specified model configuration: which
variables are used, what transforms have been applied, which estimator and
covariance type are requested, and (for IV) which instruments are included.

Every candidate carries a stable :attr:`candidate_hash` — a 16-character
hexadecimal string derived deterministically from its content.  The same
spec always produces the same hash regardless of evaluation order or
parallelism.

Public API
----------
TransformSpec
Candidate
canonicalize_candidate(candidate) -> Candidate
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# TransformSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformSpec:
    """Description of a single feature transform.

    Parameters
    ----------
    kind:
        Transform kind: ``"identity"``, ``"log"``, ``"log1p"``, ``"square"``,
        ``"inverse"``, ``"poly"``, ``"interaction"``, ``"spline"``,
        ``"lag"``, ``"group_mean"``, ``"loo_group_mean"``.
    base_vars:
        Input variable names (tuple for hashability).
    output_vars:
        Output column names produced by this transform.
    params:
        Keyword parameters (degree, n_knots, shift, lag_k, group_col, …).
        Must be JSON-serialisable.
    """

    kind: str
    base_vars: tuple[str, ...]
    output_vars: tuple[str, ...]
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure params is a plain dict (not a subclass)
        object.__setattr__(self, "params", dict(self.params))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "base_vars": list(self.base_vars),
            "output_vars": list(self.output_vars),
            "params": self.params,
        }


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """Fully-specified model candidate.

    Parameters
    ----------
    y:
        Dependent variable name.
    X_terms:
        Final exogenous regressor column names (after transforms).
    endog:
        Endogenous variable names (IV only).  ``None`` for OLS/panel.
    Z_terms:
        Instrument column names (IV only).  ``None`` for OLS.
    estimator:
        ``"ols"``, ``"2sls"``, ``"fe_ols"``, ``"fe_2sls"``.
    intercept:
        Whether to include an intercept.
    fe_spec:
        Fixed effects specification dict (e.g. ``{"entity_effects": True}``).
    transforms:
        Transforms applied to produce ``X_terms`` / ``Z_terms`` from raw data.
    cov_type:
        Covariance type (``"classical"``, ``"HC3"``, ``"cluster"``, etc.).
    cluster_var:
        Clustering variable name (required when ``cov_type="cluster"``).
    constraints:
        Human-readable constraint descriptions that this candidate passed.
    generator_hash:
        Hash of the generator configuration that produced this candidate.
    config_hash:
        Hash of the full sieve configuration.
    data_fingerprint:
        Dataset fingerprint at generation time.
    """

    y: str
    X_terms: tuple[str, ...]
    endog: tuple[str, ...] | None = None
    Z_terms: tuple[str, ...] | None = None
    estimator: str = "ols"
    intercept: bool = True
    fe_spec: dict | None = None
    transforms: tuple[TransformSpec, ...] = field(default_factory=tuple)
    cov_type: str = "HC3"
    cluster_var: str | None = None
    constraints: tuple[str, ...] = field(default_factory=tuple)
    generator_hash: str = ""
    config_hash: str = ""
    data_fingerprint: str = ""

    @property
    def candidate_hash(self) -> str:
        """16-char stable hash derived from the model specification."""
        payload = {
            "y": self.y,
            "X_terms": sorted(self.X_terms),
            "endog": sorted(self.endog) if self.endog else None,
            "Z_terms": sorted(self.Z_terms) if self.Z_terms else None,
            "estimator": self.estimator,
            "intercept": self.intercept,
            "fe_spec": self.fe_spec,
            "cov_type": self.cov_type,
            "cluster_var": self.cluster_var,
            "transforms": [t.to_dict() for t in self.transforms],
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def n_terms(self) -> int:
        """Total number of model terms (X + Z if IV)."""
        n = len(self.X_terms)
        if self.Z_terms:
            n += len(self.Z_terms)
        return n

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (for manifest storage)."""
        return {
            "candidate_hash": self.candidate_hash,
            "y": self.y,
            "X_terms": list(self.X_terms),
            "endog": list(self.endog) if self.endog else None,
            "Z_terms": list(self.Z_terms) if self.Z_terms else None,
            "estimator": self.estimator,
            "intercept": self.intercept,
            "fe_spec": self.fe_spec,
            "transforms": [t.to_dict() for t in self.transforms],
            "cov_type": self.cov_type,
            "cluster_var": self.cluster_var,
            "constraints": list(self.constraints),
            "generator_hash": self.generator_hash,
            "config_hash": self.config_hash,
            "data_fingerprint": self.data_fingerprint,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def canonicalize_candidate(candidate: Candidate) -> Candidate:
    """Return a new :class:`Candidate` with sorted X/Z terms for deduplication."""
    return Candidate(
        y=candidate.y,
        X_terms=tuple(sorted(candidate.X_terms)),
        endog=tuple(sorted(candidate.endog)) if candidate.endog else None,
        Z_terms=tuple(sorted(candidate.Z_terms)) if candidate.Z_terms else None,
        estimator=candidate.estimator,
        intercept=candidate.intercept,
        fe_spec=candidate.fe_spec,
        transforms=candidate.transforms,
        cov_type=candidate.cov_type,
        cluster_var=candidate.cluster_var,
        constraints=candidate.constraints,
        generator_hash=candidate.generator_hash,
        config_hash=candidate.config_hash,
        data_fingerprint=candidate.data_fingerprint,
    )


def deduplicate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Remove duplicate candidates (same canonical hash)."""
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in candidates:
        h = canonicalize_candidate(c).candidate_hash
        if h not in seen:
            seen.add(h)
            out.append(c)
    return out
