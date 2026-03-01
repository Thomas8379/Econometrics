"""Full LaTeX document assembly from fragments.

TODO(econtools): render — implement full .tex document assembly using JournalProfile
"""

from __future__ import annotations

from pathlib import Path

from econtools.output.latex.journal_profiles import JournalProfile


def assemble_document(
    fragments: list[str],
    profile: JournalProfile,
    *,
    title: str | None = None,
    author: str | None = None,
    abstract: str | None = None,
) -> str:
    """Assemble a full LaTeX document from table/figure fragments.

    Parameters
    ----------
    fragments:
        List of LaTeX fragment strings (tables, figures).
    profile:
        :class:`JournalProfile` controlling document style.
    title:
        Optional document title.
    author:
        Optional author string.
    abstract:
        Optional abstract text.

    Returns
    -------
    str
        Complete LaTeX document source.

    TODO(econtools): render — implement per-profile document assembly
    """
    opts = "[" + ",".join(profile.class_options) + "]" if profile.class_options else ""
    lines: list[str] = [
        rf"\documentclass{opts}{{{profile.document_class}}}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{graphicx}",
    ]
    if profile.float_specifier:
        lines.append(r"\usepackage{float}")
    lines.append(r"\begin{document}")
    if title:
        lines.append(rf"\title{{{title}}}")
    if author:
        lines.append(rf"\author{{{author}}}")
    if title or author:
        lines.append(r"\maketitle")
    if abstract:
        lines.append(r"\begin{abstract}")
        lines.append(abstract)
        lines.append(r"\end{abstract}")
    for fragment in fragments:
        lines.append("")
        lines.append(fragment)
    lines.append(r"\end{document}")
    return "\n".join(lines)


def write_document(
    fragments: list[str],
    out_path: Path,
    profile: JournalProfile,
    **kwargs,
) -> Path:
    """Write a complete LaTeX document to *out_path*."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = assemble_document(fragments, profile, **kwargs)
    out_path.write_text(source, encoding="utf-8")
    return out_path
