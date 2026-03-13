# ak91_fixtures.R — R translation of ak91.sas
#
# Source: Angrist & Krueger (1991) "Does Compulsory School Attendance Affect
#         Schooling and Earnings?" QJE.  doi:10.7910/DVN/ENLGZX
#
# Translates: ak91.sas  (proc syslin OLS + proc syslin 2sls)
#
# Outputs JSON to stdout with the following estimates:
#   ols_educ_coef, ols_educ_se, ols_nobs, ols_rsq
#   iv30_educ_coef, iv30_educ_se, iv30_nobs        (30-instrument 2SLS)
#   iv3_educ_coef,  iv3_educ_se,  iv3_nobs         (3 QOB instruments)
#
# Usage:
#   Rscript scripts/r/ak91_fixtures.R \
#           --data data_lake/raw/angrist_replication/asciiqob.tab \
#           --out  tests/validation/fixtures/angrist/ak91.json

suppressPackageStartupMessages({
  library(AER)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(args, flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default)
  args[idx + 1]
}

data_path <- parse_arg(args, "--data", "data_lake/raw/angrist_replication/asciiqob.tab")
out_path  <- parse_arg(args, "--out",  "tests/validation/fixtures/angrist/ak91.json")

# ---------------------------------------------------------------------------
# Load data (matches ak91.sas: infile indat; input lwklywge educ yob qob pob)
# ---------------------------------------------------------------------------
cat("Loading", data_path, "...\n", file = stderr())
df <- read.table(data_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
stopifnot(nrow(df) == 329509)

# Column name aliases to match SAS variable names
names(df)[names(df) == "log_weekly_wage"]  <- "lwklywge"
names(df)[names(df) == "education"]        <- "educ"
names(df)[names(df) == "year_of_birth"]    <- "yob"
names(df)[names(df) == "quarter_of_birth"] <- "qob"
names(df)[names(df) == "place_of_birth"]   <- "pob"

# ---------------------------------------------------------------------------
# Create dummies  (exactly as in ak91.sas)
# Year dummies: YR20 = (yob==30), ..., YR29 = (yob==39); drop YR20 as reference
# The SAS model uses yr21-yr29 (9 dummies), so YR20 (yob==30) is the reference
# ---------------------------------------------------------------------------
for (yr in 21:29) {
  df[[paste0("yr", yr)]] <- as.integer(df$yob == (yr - 20 + 29))
}
# yr21 -> yob==31, yr22 -> yob==32, ..., yr29 -> yob==39
# Check: SAS uses YR21=((YOB=31) OR (YOB=41)) but data only has 30-39, so OR is irrelevant
for (yr in 21:29) {
  df[[paste0("yr", yr)]] <- as.integer(df$yob == (yr - 21 + 31))
}

# Quarter dummies
df$qtr1 <- as.integer(df$qob == 1)
df$qtr2 <- as.integer(df$qob == 2)
df$qtr3 <- as.integer(df$qob == 3)
df$qtr4 <- as.integer(df$qob == 4)

# QTR x YR interactions (30 instruments in ak91.sas: QTR220..QTR429 minus QTR1xx)
# SAS macro QTRSYR = QTR2-4 x YR20-29 = 3*10 = 30 instruments
# But note SAS drops QTR120 (Q1*YR20) from the macro — it uses QTR2,3,4 x all 10 years
qtr_vars <- c()
for (q in 2:4) {
  for (yr in 20:29) {
    vname <- paste0("qtr", q, "yr", yr)
    df[[vname]] <- as.integer(df$qob == q) * as.integer(df$yob == (yr + 10))
    qtr_vars <- c(qtr_vars, vname)
  }
}
# 30 QTR*YR instruments

yr_dummies <- paste0("yr", 21:29)

# ---------------------------------------------------------------------------
# OLS: lwklywge ~ yr21-yr29 + educ  (proc syslin, model lwklywge = yr21-yr29 educ)
# ---------------------------------------------------------------------------
cat("Running OLS...\n", file = stderr())
ols_formula <- as.formula(paste("lwklywge ~", paste(c(yr_dummies, "educ"), collapse = " + ")))
ols_fit <- lm(ols_formula, data = df)
ols_sum <- summary(ols_fit)

ols_coef <- coef(ols_fit)[["educ"]]
ols_se   <- sqrt(diag(vcov(ols_fit)))[["educ"]]
ols_nobs <- nobs(ols_fit)
ols_rsq  <- ols_sum$r.squared

cat(sprintf("OLS: educ=%.6f  se=%.6f  N=%d  R2=%.6f\n",
            ols_coef, ols_se, ols_nobs, ols_rsq), file = stderr())

# ---------------------------------------------------------------------------
# 2SLS-30: 30-instrument case  (proc syslin 2sls, instruments yr21-yr29 %qtrsyr)
# Endogenous: educ.  Exog included: yr21-yr29.  Excluded instruments: qtr_vars (30).
# ---------------------------------------------------------------------------
cat("Running 2SLS-30...\n", file = stderr())
iv30_formula <- as.formula(paste(
  "lwklywge ~", paste(c(yr_dummies, "educ"), collapse = " + "),
  "|",
  paste(c(yr_dummies, qtr_vars), collapse = " + ")
))
iv30_fit  <- ivreg(iv30_formula, data = df)
iv30_coef <- coef(iv30_fit)[["educ"]]
iv30_se   <- sqrt(diag(vcov(iv30_fit)))[["educ"]]
iv30_nobs <- nobs(iv30_fit)

cat(sprintf("2SLS-30: educ=%.6f  se=%.6f  N=%d\n",
            iv30_coef, iv30_se, iv30_nobs), file = stderr())

# ---------------------------------------------------------------------------
# 2SLS-3: 3 QOB instruments (standard AK91 Table IV-A col 2 spec)
# Instruments: qtr1, qtr2, qtr3 (dropping qtr4 as reference via AER convention)
# ---------------------------------------------------------------------------
cat("Running 2SLS-3...\n", file = stderr())
iv3_formula <- as.formula(paste(
  "lwklywge ~", paste(c(yr_dummies, "educ"), collapse = " + "),
  "|",
  paste(c(yr_dummies, "qtr1", "qtr2", "qtr3"), collapse = " + ")
))
iv3_fit  <- ivreg(iv3_formula, data = df)
iv3_coef <- coef(iv3_fit)[["educ"]]
iv3_se   <- sqrt(diag(vcov(iv3_fit)))[["educ"]]
iv3_nobs <- nobs(iv3_fit)

cat(sprintf("2SLS-3:  educ=%.6f  se=%.6f  N=%d\n",
            iv3_coef, iv3_se, iv3_nobs), file = stderr())

# ---------------------------------------------------------------------------
# Write JSON fixture
# ---------------------------------------------------------------------------
results <- list(
  paper        = "Angrist & Krueger (1991)",
  source_doi   = "doi:10.7910/DVN/ENLGZX",
  source_code  = "ak91.sas",
  generator    = "scripts/r/ak91_fixtures.R",
  r_version    = paste(R.version$major, R.version$minor, sep = "."),
  aer_version  = as.character(packageVersion("AER")),
  ols = list(
    spec        = "lwklywge ~ yr21-yr29 + educ",
    educ_coef   = round(ols_coef, 8),
    educ_se     = round(ols_se,   8),
    nobs        = as.integer(ols_nobs),
    r_squared   = round(ols_rsq, 8)
  ),
  iv30 = list(
    spec        = "lwklywge ~ yr21-yr29 + educ | yr21-yr29 + QTR{2,3,4}xYR{20-29} (30 instruments)",
    educ_coef   = round(iv30_coef, 8),
    educ_se     = round(iv30_se,   8),
    nobs        = as.integer(iv30_nobs)
  ),
  iv3 = list(
    spec        = "lwklywge ~ yr21-yr29 + educ | yr21-yr29 + qtr1 + qtr2 + qtr3 (3 instruments)",
    educ_coef   = round(iv3_coef, 8),
    educ_se     = round(iv3_se,   8),
    nobs        = as.integer(iv3_nobs)
  )
)

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write(toJSON(results, pretty = TRUE, auto_unbox = TRUE, digits = 10), out_path)
cat("Fixture written to", out_path, "\n", file = stderr())
