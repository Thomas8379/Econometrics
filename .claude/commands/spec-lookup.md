# spec-lookup

Look up the implementation spec for a given section or capability.

The ground-truth spec is at `C:\Econometrics\rough spec plan.txt`.

Usage examples:
- `/spec-lookup OLS` — look up the OLS estimator spec (§1.1)
- `/spec-lookup panel SE` — look up panel standard errors (§2.2)
- `/spec-lookup diagnostics heteroskedasticity` — look up BP/White tests (§4.1)
- `/spec-lookup plots residual` — look up residual diagnostic plots (§7.1)

$ARGUMENTS

Search the spec file for the section matching the arguments provided.
Summarise: the exact library call, result object type, key output attributes,
and any implementation gaps (marked "custom" in the spec).
