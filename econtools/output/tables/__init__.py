"""Table renderers — re-export from econtools.tables."""

from econtools.tables.compare_table import compare_table  # noqa: F401
from econtools.tables.diagnostic_table import diagnostic_table  # noqa: F401
from econtools.tables.reg_table import reg_table  # noqa: F401

__all__ = ["compare_table", "diagnostic_table", "reg_table"]
