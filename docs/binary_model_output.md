# Binary Model Output Extensions (Logit/Probit)

This note documents the extended output reported for binary-response models in `econtools` (Logit/Probit) and how each statistic is computed.

## Where this is implemented

- Table renderer: `econtools/tables/reg_table.py`
- Binary estimator wrapper: `econtools/models/probit.py`

The logic is triggered when the model type includes `probit` or `logit`, or when the underlying statsmodels result exposes `prsquared`.

## Fit and diagnostic metrics

**Likelihood-based**
- Log-likelihood: `llf`
- LR statistic / p-value: `llr`, `llr_pvalue`
- McFadden pseudo-R²: `1 - llf / llnull` (reported explicitly as McFadden)
- AIC / BIC

**Prediction quality**
- Corr²(y, p̂): squared correlation between observed `y` and predicted probabilities `p̂`.
- Efron R²: `1 - sum((y - p̂)^2) / sum((y - ȳ)^2)`.
- Brier score: mean squared error of predicted probabilities, `mean((y - p̂)^2)`.
- AUC: ROC AUC computed via the rank statistic (Mann–Whitney formulation) to avoid extra dependencies.

## Classification summaries

The following are reported at two thresholds:

**c = 0.5**
- Percent correct (accuracy)
- Balanced accuracy `(TPR + TNR) / 2`
- Confusion counts (TP, TN, FP, FN)
- Rates: TPR, TNR, PPV, NPV

**c* (match ȳ)**
- `c*` chosen so predicted prevalence matches the observed prevalence `ȳ` (quantile threshold).
- Percent correct
- Confusion counts (TP, TN, FP, FN)
- Rates: TPR, TNR, PPV, NPV

## Marginal effects (interpretation)

For each regressor (excluding the constant), the table reports:
- AME (average marginal effect): `get_margeff(at='overall', dummy=True)`
- MEM (marginal effect at mean): `get_margeff(at='mean', dummy=True)`
- Standard errors and p-values for both AME and MEM

`dummy=True` ensures discrete changes are used for binary regressors (recommended for logit/probit).

## Notes / caveats

- Accuracy at a fixed threshold can be misleading for imbalanced outcomes. This is why c* and balanced accuracy are included.
- Corr²(y, p̂) is not a likelihood-based R²; it is reported separately from McFadden and Efron.
- AUC is threshold-free and is often preferred for ranking performance.

## How to regenerate

Example (Probit):
```
C:\Econometrics\econ.bat probit fatkids_db692 obesec ageyrs female white black hisp tvyest povrat --curated --export latex --session 26feb_question_6.3
```
