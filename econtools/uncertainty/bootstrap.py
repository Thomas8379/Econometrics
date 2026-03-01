"""Bootstrap standard errors — Phase 3.

TODO(econtools): adapter — implement pairs bootstrap for OLS (Phase 3)
TODO(econtools): adapter — implement wild bootstrap for clustered SEs (Phase 3)
TODO(econtools): adapter — implement cluster bootstrap (Phase 3)
"""

from __future__ import annotations

import pandas as pd

from econtools._core.types import Estimate
from econtools.model.spec import ModelSpec


def pairs_bootstrap(
    spec: ModelSpec,
    df: pd.DataFrame,
    n_bootstrap: int = 999,
    seed: int | None = None,
) -> Estimate:
    """Pairs bootstrap standard errors.

    TODO(econtools): adapter — implement pairs bootstrap (Phase 3)
    """
    raise NotImplementedError("Pairs bootstrap is Phase 3 — not yet implemented.")


def wild_bootstrap(
    spec: ModelSpec,
    df: pd.DataFrame,
    n_bootstrap: int = 999,
    seed: int | None = None,
) -> Estimate:
    """Wild bootstrap standard errors for clustered inference.

    TODO(econtools): adapter — implement wild bootstrap (Phase 3)
    """
    raise NotImplementedError("Wild bootstrap is Phase 3 — not yet implemented.")
