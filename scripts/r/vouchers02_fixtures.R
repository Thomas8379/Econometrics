# vouchers02_fixtures.R — R translation of table2_final.sas and table3_final.sas
#
# Source: Angrist, Bettinger, Bloom, King & Kremer (2002)
#         "Vouchers for Private Schooling in Colombia" AER.
#         doi:10.7910/DVN/K57TOZ
#
# Translates: table2_final.sas / table3_final.sas
#   proc reg; model outcome = vouch0 <controls> /acov;
#
# /acov in SAS proc reg = HC0 (White) heteroskedasticity-consistent SEs.
# R equivalent: lm() + sandwich::vcovHC(type="HC0").
#
# Note: HC0 in SAS = sandwich estimator WITHOUT small-sample correction.
# Use type="HC0" in sandwich to match SAS /acov exactly.

suppressPackageStartupMessages({
  library(sandwich)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(args, flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default)
  args[idx + 1]
}

data_path <- parse_arg(args, "--data", "data_lake/raw/angrist_replication/aerdat4.tab")
out_path  <- parse_arg(args, "--out",  "tests/validation/fixtures/angrist/vouchers02.json")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
cat("Loading", data_path, "...\n", file = stderr())
df <- read.table(data_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
cat(sprintf("Loaded %d rows x %d cols\n", nrow(df), ncol(df)), file = stderr())

# ---------------------------------------------------------------------------
# Helper: OLS with HC0 SEs (matching SAS proc reg /acov)
# ---------------------------------------------------------------------------
ols_hc0 <- function(data, outcome, predictors) {
  fml <- as.formula(paste(outcome, "~", paste(predictors, collapse = " + ")))
  data_clean <- data[complete.cases(data[, c(outcome, predictors)]), ]
  if (nrow(data_clean) == 0) return(NULL)
  fit <- lm(fml, data = data_clean)
  vcv <- vcovHC(fit, type = "HC0")
  se  <- sqrt(diag(vcv))
  list(
    coef = round(coef(fit)[["vouch0"]], 8),
    se   = round(se[["vouch0"]], 8),
    nobs = as.integer(nobs(fit))
  )
}

# Combined survey sample: bog95smp | bog97smp | jam93smp
svy_mask <- (
  (!is.na(df$bog95smp) & df$bog95smp == 1) |
  (!is.na(df$bog97smp) & df$bog97smp == 1) |
  (!is.na(df$jam93smp) & df$jam93smp == 1)
)
df_svy <- df[svy_mask, ]
cat(sprintf("Survey sample N = %d\n", nrow(df_svy)), file = stderr())

# Controls for Panel C (Table 3 of the paper):
# vouch0 + svy + hsvisit + djamundi + dbogota + d1993 + d1995 + d1997
#         + dmonth1-12 + darea1-19
area_cols  <- grep("^darea",  names(df_svy), value = TRUE)
month_cols <- grep("^dmonth", names(df_svy), value = TRUE)
panel_c_controls <- c(
  "vouch0", "svy", "hsvisit", "djamundi", "dbogota",
  "d1993", "d1995", "d1997",
  month_cols, area_cols
)

# ---------------------------------------------------------------------------
# Panel C regressions (Table 3, combined survey sample)
# proc reg; model outcome = vouch0 svy hsvisit djamundi dbogota d1993 d1995 d1997
#                           dmonth1-12 darea1-19 /acov;
# ---------------------------------------------------------------------------
outcomes_panelC <- c("phone", "scyfnsh", "prscha_1", "prscha_2", "totscyrs",
                     "inschl", "usngsch")
panelC_results <- list()
for (y in outcomes_panelC) {
  if (!(y %in% names(df_svy))) next
  r <- ols_hc0(df_svy, y, panel_c_controls)
  if (!is.null(r)) {
    panelC_results[[y]] <- r
    cat(sprintf("  Panel C %-12s: vouch0=%.5f  se=%.5f  N=%d\n",
                y, r$coef, r$se, r$nobs), file = stderr())
  }
}

# ---------------------------------------------------------------------------
# Table 2: Application-combined sample (dbogota | djamundi)
# Balance on phone, age2, sex_name
# proc reg; model phone = vouch0 dbogota djamundi d1993 d1995 d1997 /acov;
# ---------------------------------------------------------------------------
app_mask <- (!is.na(df$dbogota) & df$dbogota == 1) |
            (!is.na(df$djamundi) & df$djamundi == 1)
app_controls <- c("vouch0", "dbogota", "djamundi", "d1993", "d1995", "d1997")
df_app <- df[app_mask & df$age2 >= 9 & df$age2 <= 25 & !is.na(df$age2), ]
cat(sprintf("Application sample (age2 9-25) N = %d\n", nrow(df_app)), file = stderr())

app_results <- list()
for (y in c("phone", "age2", "sex_name")) {
  if (!(y %in% names(df_app))) next
  r <- ols_hc0(df_app, y, app_controls)
  if (!is.null(r)) {
    app_results[[y]] <- r
    cat(sprintf("  App %-12s: vouch0=%.5f  se=%.5f  N=%d\n",
                y, r$coef, r$se, r$nobs), file = stderr())
  }
}

# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------
results <- list(
  paper       = "Angrist et al. (2002)",
  source_doi  = "doi:10.7910/DVN/K57TOZ",
  source_code = "table2_final.sas + table3_final.sas",
  generator   = "scripts/r/vouchers02_fixtures.R",
  r_version   = paste(R.version$major, R.version$minor, sep = "."),
  se_type     = "HC0 (matches SAS proc reg /acov)",
  survey_sample_n = as.integer(nrow(df_svy)),
  panel_c = panelC_results,
  application_sample = app_results
)

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write(toJSON(results, pretty = TRUE, auto_unbox = TRUE, digits = 10), out_path)
cat("Fixture written to", out_path, "\n", file = stderr())
