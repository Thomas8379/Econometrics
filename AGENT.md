# AGENT.md

## Summary of Work
- Added `reg` CLI command (OLS) with export options matching `des`/`summ`; supports robust/cluster/HAC SEs.
- Added IV/2SLS support to `reg` (`--endog`, `--instruments`), with OLS vs IV comparison table (`--compare-ols`).
- Added panel first-difference estimator (`--panel fd` + `--entity`/`--time`) via `linearmodels`.
- Added diagnostics system and tables:
  - IV diagnostics: Wu–Hausman, Sargan, Basmann, Basmann F, weak-instrument first-stage F.
  - Panel lead/feedback test for strict exogeneity (FD lead test).
- Added multi-model comparison table and diagnostics table renderers.
- Added LaTeX utilities: combine multiple `.tex` into a single wrapper, compile to PDF, and open (`econ texpdf`).
- Improved LaTeX output: table spacing, left-aligned footnotes, section headers when combining, and no page-splitting tables (`table[H]`).
- Added label and session support for exports (`--label`, `--session`); filenames use safe label slugs.
- Updated `econ-shell.bat` to include `reg`.

## Current Commands (CMD shell)
From `econ-shell.bat`:
- `project <name>`: set active project (stored in `.econ_project`)
- `project`: show active project
- `des <dataset> --export [latex|html|text] [--pdf]`
- `summ <dataset> --export [latex|html|text] [--pdf]`
- `reg <dataset> <dep> <x1> ... [--endog ... --instruments ...] [--compare-ols] [--panel fd --entity ... --time ...]`
- `curate <dataset>`
- `findcols col1,col2,...`

Example:
- `project supervisions`
- `des crime4 --export latex --pdf`
- `summ crime4 --export latex --pdf`
- `reg crime4 crmrte polpc west central urban --export latex --pdf`

## Export Behavior
- If `--export` is used, output is written to:
  `projects/<project>/output/` unless `--out` is specified.
- `--session <name>` writes to `projects/<project>/output/<name>/`.
- `--label <text>` adds a title to exports and is used in filenames (slugged).
- `--export latex --pdf` runs LaTeX compilation. Uses `pdflatex` first, falls back if needed.
- LaTeX tables are escaped to avoid `_` and other special character errors.

## Files Added / Modified
- Added: `econtools/models/iv.py`, `econtools/models/panel.py`
- Added: `econtools/tables/compare_table.py`, `econtools/tables/diagnostic_table.py`, `econtools/tables/latex_utils.py`
- Added: `econtools/diagnostics/iv.py`, `econtools/diagnostics/panel.py`
- Modified: `econtools/cli.py` (reg, IV, panel FD, diagnostics, texpdf, labels/sessions)
- Modified: `econtools/tables/reg_table.py`
- Modified: `econtools/tables/__init__.py`
- Modified: `econtools/diagnostics/__init__.py`
- Modified: `econ-shell.bat`

## Known Issues / Notes
- MiKTeX requires running `pdflatex` outside the sandbox (or configure user dir); `econ texpdf` uses MiKTeX when given full path.
- For combined PDFs, large tables are forced to stay on one page (`table[H]`), which can cause overfull vbox warnings if too tall.
- If PDF fails, check `projects/<project>/output/*_wrapper.log` or `combined_report*.log`.

## Next Potential Improvements
- Add `--open` to auto-open PDF after export.
- Optionally default to exporting when a project is active.
- Add time effects / lag options for panel lead tests.

## Documentation
- Binary model output extensions (Logit/Probit): `docs/binary_model_output.md`
