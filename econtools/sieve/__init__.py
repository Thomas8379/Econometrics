"""Sieve infrastructure: systematic candidate generation, fitting, scoring, and selection.

Public API
----------
run_sieve(data, y, base_X, estimator, ...)  -> dict
load_sieve_results(output_dir)              -> dict
"""

from econtools.sieve.api import load_sieve_results, run_sieve

__all__ = ["run_sieve", "load_sieve_results"]
