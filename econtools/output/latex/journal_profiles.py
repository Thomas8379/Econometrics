"""Journal-specific LaTeX profiles for typesetting regression output.

Public API
----------
JournalProfile
ECONOMETRICA
AER
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JournalProfile:
    """Parameters controlling LaTeX output style for a specific journal.

    Parameters
    ----------
    name:
        Journal name (human-readable).
    document_class:
        LaTeX document class (e.g. ``'article'``, ``'econsocart'``).
    class_options:
        Options passed to the document class.
    table_numbering:
        ``'roman'`` (TABLE I) or ``'arabic'`` (Table 1).
    caption_position:
        ``'above'`` or ``'below'`` the table body.
    notes_command:
        LaTeX command for table notes
        (``'legend'`` for Econometrica, ``'footnotesize'`` for others).
    float_specifier:
        Float placement specifier (e.g. ``'[H]'``); ``None`` omits it.
    star_symbols:
        Ordered list of star symbols from least to most significant.
    significance_levels:
        Ordered p-value thresholds matching ``star_symbols``.
    bst_file:
        BibTeX style file.
    """

    name: str
    document_class: str = "article"
    class_options: list[str] = field(default_factory=list)
    table_numbering: str = "arabic"   # "roman" | "arabic"
    caption_position: str = "above"   # "above" | "below"
    notes_command: str = "footnotesize"
    float_specifier: str | None = "[H]"
    star_symbols: list[str] = field(default_factory=lambda: ["*", "**", "***"])
    significance_levels: list[float] = field(default_factory=lambda: [0.1, 0.05, 0.01])
    bst_file: str = "plain"

    # TODO(econtools): profile — validate significance_levels / star_symbols lengths match


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

ECONOMETRICA = JournalProfile(
    name="Econometrica",
    document_class="econsocart",
    class_options=["ecta", "nameyear"],
    table_numbering="roman",
    caption_position="above",
    notes_command="legend",
    float_specifier=None,         # Econometrica disallows [H]
    star_symbols=["*", "**", "***"],
    significance_levels=[0.1, 0.05, 0.01],
    bst_file="ecta",
)

AER = JournalProfile(
    name="American Economic Review",
    document_class="aea",
    class_options=[],
    table_numbering="arabic",
    caption_position="above",
    notes_command="footnotesize",
    float_specifier="[H]",
    star_symbols=["*", "**", "***"],
    significance_levels=[0.1, 0.05, 0.01],
    bst_file="aea",
)

# TODO(econtools): profile — add QJE, RES, JPE profiles
