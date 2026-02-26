"""Single-model regression table formatter.

Supports text (fixed-width), LaTeX, and HTML output.

Public API
----------
reg_table(result, stars, se_in_parens, digits, format) -> str
"""

from __future__ import annotations

from econtools.models._results import RegressionResult


def reg_table(
    result: RegressionResult,
    stars: bool = True,
    se_in_parens: bool = True,
    digits: int = 3,
    format: str = "text",
) -> str:
    """Format a regression result as a printable table.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    stars:
        Append significance stars (``*`` p<0.1, ``**`` p<0.05,
        ``***`` p<0.01).
    se_in_parens:
        Print the standard error in parentheses on the row below each
        coefficient.
    digits:
        Number of decimal places for all numeric values.
    format:
        ``'text'`` (fixed-width), ``'latex'`` (tabular env), or
        ``'html'`` (table tags).

    Returns
    -------
    Formatted table string.

    Raises
    ------
    ValueError
        If ``format`` is not one of the supported values.
    """
    if format == "text":
        return _text_table(result, stars, se_in_parens, digits)
    if format == "latex":
        return _latex_table(result, stars, se_in_parens, digits)
    if format == "html":
        return _html_table(result, stars, se_in_parens, digits)
    raise ValueError(
        f"Unknown format '{format}'. Choose from 'text', 'latex', 'html'."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _star(pval: float) -> str:
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.1:
        return "*"
    return ""


def _fmt(x: float, digits: int) -> str:
    return f"{x:.{digits}f}"


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------

_COL_W = 12  # numeric column width


def _text_table(
    result: RegressionResult,
    stars: bool,
    se_in_parens: bool,
    digits: int,
) -> str:
    var_w = max(20, max(len(n) for n in result.params.index) + 2)
    sep = "-" * (var_w + _COL_W * 4)

    lines: list[str] = [
        sep,
        f"{'Dep. variable:':<{var_w}} {result.dep_var}",
        f"{'Estimator:':<{var_w}} {result.model_type}",
        f"{'SE type:':<{var_w}} {result.cov_type}",
        sep,
        (
            f"{'Variable':<{var_w}}"
            f"{'Coef':>{_COL_W}}"
            f"{'Std Err':>{_COL_W}}"
            f"{'t-stat':>{_COL_W}}"
            f"{'p-value':>{_COL_W}}"
        ),
        sep,
    ]

    for name in result.params.index:
        coef = float(result.params[name])
        se = float(result.bse[name])
        tval = float(result.tvalues[name])
        pval = float(result.pvalues[name])
        s = _star(pval) if stars else ""

        coef_str = _fmt(coef, digits) + s
        lines.append(
            f"{name:<{var_w}}"
            f"{coef_str:>{_COL_W}}"
            f"{_fmt(se, digits):>{_COL_W}}"
            f"{_fmt(tval, digits):>{_COL_W}}"
            f"{_fmt(pval, digits):>{_COL_W}}"
        )
        if se_in_parens:
            se_str = f"({_fmt(se, digits)})"
            lines.append(f"{'':<{var_w}}{se_str:>{_COL_W}}")

    fit = result.fit
    lines += [
        sep,
        f"{'N':<{var_w}}{fit.nobs:>{_COL_W}}",
        f"{'R\u00b2':<{var_w}}{_fmt(fit.r_squared, digits):>{_COL_W}}",
        f"{'Adj. R\u00b2':<{var_w}}{_fmt(fit.r_squared_adj, digits):>{_COL_W}}",
        f"{'F-stat':<{var_w}}{_fmt(fit.f_stat, digits):>{_COL_W}}",
        f"{'RMSE':<{var_w}}{_fmt(fit.rmse, digits):>{_COL_W}}",
        sep,
    ]

    if stars:
        lines.append("* p<0.1  ** p<0.05  *** p<0.01")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX renderer
# ---------------------------------------------------------------------------


def _latex_star(s: str) -> str:
    """Convert star string to LaTeX superscript."""
    if not s:
        return ""
    count = s.count("*")
    return r"$^{" + r"\ast" * count + r"}$"


def _latex_table(
    result: RegressionResult,
    stars: bool,
    se_in_parens: bool,
    digits: int,
) -> str:
    lines: list[str] = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Variable & Coef & Std Err & $t$ & $p$ \\",
        r"\hline",
    ]

    for name in result.params.index:
        coef = float(result.params[name])
        se = float(result.bse[name])
        tval = float(result.tvalues[name])
        pval = float(result.pvalues[name])
        s = _star(pval) if stars else ""

        coef_str = _fmt(coef, digits) + _latex_star(s)
        lines.append(
            f"{name} & {coef_str} & {_fmt(se, digits)} & "
            f"{_fmt(tval, digits)} & {_fmt(pval, digits)} \\\\"
        )
        if se_in_parens:
            lines.append(f" & $({_fmt(se, digits)})$ & & & \\\\")

    fit = result.fit
    lines += [
        r"\hline",
        f"$N$ & {fit.nobs} & & & \\\\",
        f"$R^2$ & {_fmt(fit.r_squared, digits)} & & & \\\\",
        f"Adj.\\ $R^2$ & {_fmt(fit.r_squared_adj, digits)} & & & \\\\",
        f"$F$-stat & {_fmt(fit.f_stat, digits)} & & & \\\\",
        f"RMSE & {_fmt(fit.rmse, digits)} & & & \\\\",
        r"\hline",
        r"\end{tabular}",
    ]

    if stars:
        lines.append(
            r"\noindent\footnotesize{* $p<0.1$, ** $p<0.05$, *** $p<0.01$}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------


def _html_table(
    result: RegressionResult,
    stars: bool,
    se_in_parens: bool,
    digits: int,
) -> str:
    lines: list[str] = [
        "<table>",
        "<tr>"
        "<th>Variable</th><th>Coef</th>"
        "<th>Std Err</th><th><i>t</i></th><th><i>p</i></th>"
        "</tr>",
    ]

    for name in result.params.index:
        coef = float(result.params[name])
        se = float(result.bse[name])
        tval = float(result.tvalues[name])
        pval = float(result.pvalues[name])
        s = _star(pval) if stars else ""

        coef_str = (
            f"{_fmt(coef, digits)}<sup>{s}</sup>" if s else _fmt(coef, digits)
        )
        lines.append(
            f"<tr><td>{name}</td><td>{coef_str}</td>"
            f"<td>{_fmt(se, digits)}</td>"
            f"<td>{_fmt(tval, digits)}</td>"
            f"<td>{_fmt(pval, digits)}</td></tr>"
        )
        if se_in_parens:
            lines.append(
                f"<tr><td></td><td>({_fmt(se, digits)})</td>"
                f"<td></td><td></td><td></td></tr>"
            )

    fit = result.fit
    lines += [
        "<tr><td colspan='5'><hr></td></tr>",
        f"<tr><td>N</td><td>{fit.nobs}</td>"
        f"<td></td><td></td><td></td></tr>",
        f"<tr><td>R&sup2;</td><td>{_fmt(fit.r_squared, digits)}</td>"
        f"<td></td><td></td><td></td></tr>",
        f"<tr><td>Adj. R&sup2;</td><td>{_fmt(fit.r_squared_adj, digits)}</td>"
        f"<td></td><td></td><td></td></tr>",
        f"<tr><td>F-stat</td><td>{_fmt(fit.f_stat, digits)}</td>"
        f"<td></td><td></td><td></td></tr>",
        f"<tr><td>RMSE</td><td>{_fmt(fit.rmse, digits)}</td>"
        f"<td></td><td></td><td></td></tr>",
        "</table>",
    ]

    if stars:
        lines.append(
            "<p><small>* p&lt;0.1, ** p&lt;0.05, *** p&lt;0.01</small></p>"
        )

    return "\n".join(lines)
