# new-module

Scaffold a new module file in the econtools package following project conventions.

Usage: `/new-module <phase>/<module_name>`

Example: `/new-module models/ols`

$ARGUMENTS

Create the file `econtools/$ARGUMENTS.py` with:
1. Module docstring describing what it implements and which spec section it covers
2. `__all__` list
3. Type-annotated function stubs for each public function (with `raise NotImplementedError`)
4. A corresponding test file at `tests/$ARGUMENTS_test.py` with:
   - One `test_<function>_smoke` test per function (calls function, checks output type)
   - Fixtures imported from `conftest.py`

Follow the naming conventions in CLAUDE.md. Consult `rough spec plan.txt` for the
exact function signatures and return types before writing stubs.
