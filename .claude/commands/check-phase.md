# check-phase

Check the implementation status of a given phase against the spec.

Usage: `/check-phase <phase_number>`

Example: `/check-phase 0`

$ARGUMENTS

For Phase $ARGUMENTS:
1. Read the relevant sections of `rough spec plan.txt`
2. List every function/capability that should be implemented
3. Check which source files exist in the corresponding `econtools/` subdirectory
4. For each expected function, grep the source to confirm it's implemented
5. Run the corresponding tests: `python -m pytest tests/ -v -k "phase$ARGUMENTS or data"`
6. Report: implemented ✓, missing ✗, test missing ⚠
