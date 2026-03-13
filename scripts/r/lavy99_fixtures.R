# lavy99_fixtures.R — R translation of AngristLavy_Table4.do
#
# Source: Angrist & Lavy (1999) "Using Maimonides' Rule to Estimate the Effect
#         of Class Size on Scholastic Achievement" QJE.  doi:10.7910/DVN/XRSUJU
#
# Translates: AngristLavy_Table4.do
#   mmoulton dvar (classize=func1) tipuach [c_size], clu(schlcode) 2sls
#
# mmoulton with 2sls = IV regression; clu(schlcode) = cluster-robust SEs.
# R equivalent: AER::ivreg() + sandwich::vcovCL(cluster=~schlcode).
#
# Outputs JSON with IV estimates for grade 5 and grade 4, verbal and math.

suppressPackageStartupMessages({
  library(AER)
  library(sandwich)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(args, flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default)
  args[idx + 1]
}

data_dir <- parse_arg(args, "--data-dir", "data_lake/raw/angrist_replication")
out_path <- parse_arg(args, "--out", "tests/validation/fixtures/angrist/lavy99.json")

# ---------------------------------------------------------------------------
# Helper: load and apply AL99 sample restrictions
# ---------------------------------------------------------------------------
load_lavy <- function(grade) {
  fname <- file.path(data_dir, paste0("final", grade, ".tab"))
  cat("Loading", fname, "...\n", file = stderr())
  # quote="" required: townname column contains Hebrew strings with apostrophes
  # that confuse read.table's default quoting logic
  df <- read.table(fname, header = TRUE, sep = "\t", stringsAsFactors = FALSE,
                   quote = "", fileEncoding = "latin1")

  # Correct values above 100 (coding artefact in original data)
  df$avgverb <- ifelse(df$avgverb > 100, df$avgverb - 100, df$avgverb)
  df$avgmath <- ifelse(df$avgmath > 100, df$avgmath - 100, df$avgmath)

  # Maimonides rule instrument
  df$func1 <- df$c_size / (floor((df$c_size - 1) / 40) + 1)

  # Set missing outcomes when size == 0
  df$avgverb[df$verbsize == 0] <- NA
  df$avgmath[df$mathsize == 0] <- NA
  df$passverb[df$verbsize == 0] <- NA
  df$passmath[df$mathsize == 0] <- NA

  # Sample restrictions (identical to Table4.do)
  df <- df[df$classize > 1 & df$classize < 45 & df$c_size > 5, ]
  df <- df[df$c_leom == 1 & df$c_pik < 3, ]

  # Table 4 additionally drops rows with missing avgverb
  df <- df[!is.na(df$avgverb), ]

  df
}

# ---------------------------------------------------------------------------
# Fit IV with cluster-robust SEs and return coef + se
# Matches: mmoulton dvar (classize=func1) [controls], clu(schlcode) 2sls
# ---------------------------------------------------------------------------
iv_clustered <- function(df, outcome, controls = character(0)) {
  # Build formula: outcome ~ controls + classize | controls + func1
  rhs_endog <- paste(c(controls, "classize"), collapse = " + ")
  rhs_excl  <- paste(c(controls, "func1"),    collapse = " + ")
  fml <- as.formula(paste(outcome, "~", rhs_endog, "|", rhs_excl))

  fit <- ivreg(fml, data = df)
  vcv <- vcovCL(fit, cluster = ~ schlcode, data = df)
  se  <- sqrt(diag(vcv))

  list(
    coef = round(coef(fit)[["classize"]], 8),
    se   = round(se[["classize"]], 8),
    nobs = as.integer(nobs(fit))
  )
}

# ---------------------------------------------------------------------------
# OLS with cluster-robust SEs
# Matches: mmoulton dvar func1 [controls], clu(schlcode)   (OLS — no 2sls flag)
# ---------------------------------------------------------------------------
ols_clustered <- function(df, outcome, controls = character(0)) {
  rhs <- paste(c(controls, "classize"), collapse = " + ")
  fml <- as.formula(paste(outcome, "~", rhs))
  fit <- lm(fml, data = df)
  vcv <- vcovCL(fit, cluster = ~ schlcode, data = df)
  se  <- sqrt(diag(vcv))
  list(
    coef = round(coef(fit)[["classize"]], 8),
    se   = round(se[["classize"]], 8),
    nobs = as.integer(nobs(fit))
  )
}

results <- list(
  paper       = "Angrist & Lavy (1999)",
  source_doi  = "doi:10.7910/DVN/XRSUJU",
  source_code = "AngristLavy_Table4.do",
  generator   = "scripts/r/lavy99_fixtures.R",
  r_version   = paste(R.version$major, R.version$minor, sep = "."),
  aer_version = as.character(packageVersion("AER"))
)

for (grade in c(5, 4)) {
  df <- load_lavy(grade)
  gkey <- paste0("grade", grade)
  cat(sprintf("Grade %d: N_full=%d\n", grade, nrow(df)), file = stderr())

  # --- Table IV specs (IV) ---
  # Col 1/4: avgverb/avgmath ~ classize | func1 + tipuach
  iv_verb_tp   <- iv_clustered(df[!is.na(df$avgverb), ], "avgverb",  c("tipuach"))
  iv_math_tp   <- iv_clustered(df[!is.na(df$avgmath), ], "avgmath",  c("tipuach"))
  # Col 2/5: avgverb/avgmath ~ classize | func1 + tipuach + c_size
  df_v <- df[!is.na(df$avgverb), ]
  df_m <- df[!is.na(df$avgmath), ]
  iv_verb_tpc  <- iv_clustered(df_v, "avgverb", c("tipuach", "c_size"))
  iv_math_tpc  <- iv_clustered(df_m, "avgmath", c("tipuach", "c_size"))

  # --- Table II specs (OLS, for cross-check) ---
  ols_verb_none <- ols_clustered(df_v, "avgverb")
  ols_verb_tp   <- ols_clustered(df_v, "avgverb", "tipuach")
  ols_verb_tpc  <- ols_clustered(df_v, "avgverb", c("tipuach", "c_size"))

  cat(sprintf("  IV verbal (tipuach):           classize=%.5f  se=%.5f  N=%d\n",
              iv_verb_tp$coef, iv_verb_tp$se, iv_verb_tp$nobs), file = stderr())
  cat(sprintf("  IV verbal (tipuach+c_size):    classize=%.5f  se=%.5f  N=%d\n",
              iv_verb_tpc$coef, iv_verb_tpc$se, iv_verb_tpc$nobs), file = stderr())
  cat(sprintf("  IV math  (tipuach+c_size):     classize=%.5f  se=%.5f  N=%d\n",
              iv_math_tpc$coef, iv_math_tpc$se, iv_math_tpc$nobs), file = stderr())

  results[[gkey]] <- list(
    nobs_verbal = as.integer(sum(!is.na(df$avgverb))),
    nobs_math   = as.integer(sum(!is.na(df$avgmath))),
    ols = list(
      verbal_no_controls = ols_verb_none,
      verbal_tipuach     = ols_verb_tp,
      verbal_tipuach_csize = ols_verb_tpc
    ),
    iv = list(
      verbal_tipuach       = iv_verb_tp,
      verbal_tipuach_csize = iv_verb_tpc,
      math_tipuach         = iv_math_tp,
      math_tipuach_csize   = iv_math_tpc
    )
  )
}

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write(toJSON(results, pretty = TRUE, auto_unbox = TRUE, digits = 10), out_path)
cat("Fixture written to", out_path, "\n", file = stderr())
