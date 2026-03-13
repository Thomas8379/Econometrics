"""
NIST Statistical Reference Datasets (StRD) — Linear Least Squares Regression.

Source: Extracted from the statsmodels fixture tree at
    C:/Econometrics/tests/validation/fixtures/statsmodels_tests/

Specifically, the certified numerical values come from two sources within
that tree:

1. statsmodels/sandbox/regression/data/Longley.dat
   An ASCII copy of the official NIST StRD file for the Longley dataset.
   The certified coefficients, SEs, R-squared, and F-statistic are embedded
   verbatim in the header of that file.

2. statsmodels/regression/tests/results/results_regression.py
   Python class ``Longley`` whose ``__init__`` carries the same certified
   values, explicitly attributed to NIST:
       "The results for the Longley dataset were obtained from NIST
        http://www.itl.nist.gov/div898/strd/general/dataarchive.html"

See the official NIST StRD Linear Regression page:
    https://www.itl.nist.gov/div898/strd/lls/lls.shtml

----

DATASETS not found in this fixture tree:

The following NIST StRD linear-regression datasets were searched for but are
NOT embedded anywhere in these fixture files:
    Norris, Pontius, NoInt1, NoInt2, Filip, Wampler1, Wampler2,
    Wampler3, Wampler4, Wampler5

To obtain certified values for those datasets, visit:
    https://www.itl.nist.gov/div898/strd/lls/lls.shtml

----

NIST ANOVA datasets found but not included here:

The following NIST StRD one-way ANOVA .dat files ARE present under
    statsmodels/sandbox/regression/data/
but are not linear-regression datasets so are out of scope for this module:
    SiRstv.dat, AtmWtAg.dat, SmLs01.dat ... SmLs09.dat

----

STRUCTURE OF EACH ENTRY IN ``DATASETS``
---------------------------------------
Each value is a dict with the following keys:

    "n"                : int  — number of observations
    "p"                : int  — number of predictor variables (excl. intercept)
    "y"                : list of float — response variable, in data order
    "X"                : list of list of float — predictor columns in data
                         order (each inner list is one predictor; does NOT
                         include the intercept column).  Row i corresponds to
                         y[i].
    "certified_coefs"  : list of float — NIST certified regression coefficients
                         in parameter order [B0 (intercept), B1, B2, ...].
    "certified_ses"    : list of float — NIST certified standard deviations of
                         the parameter estimates, same order as certified_coefs.
    "certified_r2"     : float — NIST certified R-squared.
    "certified_f"      : float — NIST certified F-statistic (regression vs
                         residual; df_num = p, df_denom = n - p - 1).
    "df_regression"    : int  — degrees of freedom for regression (= p).
    "df_residual"      : int  — degrees of freedom for residual (= n - p - 1).
    "ssr_regression"   : float — regression sum of squares.
    "ssr_residual"     : float — residual sum of squares.
    "mse_regression"   : float — regression mean square.
    "mse_residual"     : float — residual mean square (= scale).
    "residual_sd"      : float — NIST certified residual standard deviation
                                 (= sqrt(mse_residual)).
    "difficulty"       : str  — NIST difficulty level ("Higher", "Average",
                                 "Lower", or "Very High").
    "notes"            : str  — provenance note.
"""

# ---------------------------------------------------------------------------
# Longley dataset
# ---------------------------------------------------------------------------
# NIST StRD reference: Longley (Longley.dat)
# Reference: Longley, J. W. (1967). An Appraisal of Least Squares Programs
#   for the Electronic Computer from the Viewpoint of the User.
#   Journal of the American Statistical Association, 62, pp. 819-841.
#
# Model: y = B0 + B1*x1 + B2*x2 + B3*x3 + B4*x4 + B5*x5 + B6*x6 + e
#   y   = Total Employment (TOTEMP)
#   x1  = GNP Deflator (GNPDEFL)
#   x2  = Gross National Product (GNP)
#   x3  = Unemployment (UNEMP)
#   x4  = Size of Armed Forces (ARMED)
#   x5  = Non-institutional Population >= 14 (POP)
#   x6  = Year (YEAR)
#
# 16 observations, 6 predictors, Higher Level of Difficulty.
#
# Raw data sourced from:
#   statsmodels/sandbox/regression/data/Longley.dat
# Certified values also verified against:
#   statsmodels/regression/tests/results/results_regression.py  (class Longley)

LONGLEY = {
    "n": 16,
    "p": 6,

    # Response variable y (TOTEMP) — 16 observations
    "y": [
        60323.0, 61122.0, 60171.0, 61187.0, 63221.0, 63639.0,
        64989.0, 63761.0, 66019.0, 67857.0, 68169.0, 66513.0,
        68655.0, 69564.0, 69331.0, 70551.0,
    ],

    # Predictor variables — each inner list is one predictor column.
    # X[0] = x1 (GNPDEFL), X[1] = x2 (GNP), X[2] = x3 (UNEMP),
    # X[3] = x4 (ARMED),   X[4] = x5 (POP), X[5] = x6 (YEAR).
    # Row i: [x1_i, x2_i, x3_i, x4_i, x5_i, x6_i] — in observation order.
    "X": [
        [83.0,  234289.0, 2356.0, 1590.0, 107608.0, 1947.0],
        [88.5,  259426.0, 2325.0, 1456.0, 108632.0, 1948.0],
        [88.2,  258054.0, 3682.0, 1616.0, 109773.0, 1949.0],
        [89.5,  284599.0, 3351.0, 1650.0, 110929.0, 1950.0],
        [96.2,  328975.0, 2099.0, 3099.0, 112075.0, 1951.0],
        [98.1,  346999.0, 1932.0, 3594.0, 113270.0, 1952.0],
        [99.0,  365385.0, 1870.0, 3547.0, 115094.0, 1953.0],
        [100.0, 363112.0, 3578.0, 3350.0, 116219.0, 1954.0],
        [101.2, 397469.0, 2904.0, 3048.0, 117388.0, 1955.0],
        [104.6, 419180.0, 2822.0, 2857.0, 118734.0, 1956.0],
        [108.4, 442769.0, 2936.0, 2798.0, 120445.0, 1957.0],
        [110.8, 444546.0, 4681.0, 2637.0, 121950.0, 1958.0],
        [112.6, 482704.0, 3813.0, 2552.0, 123366.0, 1959.0],
        [114.2, 502601.0, 3931.0, 2514.0, 125368.0, 1960.0],
        [115.7, 518173.0, 4806.0, 2572.0, 127852.0, 1961.0],
        [116.9, 554894.0, 4007.0, 2827.0, 130081.0, 1962.0],
    ],

    # NIST certified regression coefficients: [B0, B1, B2, B3, B4, B5, B6]
    # B0 = intercept, B1..B6 correspond to x1..x6 above.
    "certified_coefs": [
        -3482258.63459582,          # B0 (intercept)
         15.0618722713733,          # B1 (GNPDEFL)
        -0.0358191792925910,        # B2 (GNP)       [= -0.358191792925910E-01]
        -2.02022980381683,          # B3 (UNEMP)
        -1.03322686717359,          # B4 (ARMED)
        -0.0511041056535807,        # B5 (POP)        [= -0.511041056535807E-01]
         1829.15146461355,          # B6 (YEAR)
    ],

    # NIST certified standard deviations of parameter estimates (same order)
    "certified_ses": [
        890420.383607373,           # SE(B0)
        84.9149257747669,           # SE(B1)
         0.0334910077722432,        # SE(B2)  [= 0.334910077722432E-01]
         0.488399681651699,         # SE(B3)
         0.214274163161675,         # SE(B4)
         0.226073200069370,         # SE(B5)
        455.478499142212,           # SE(B6)
    ],

    # NIST certified R-squared
    "certified_r2": 0.995479004577296,

    # NIST certified F-statistic (df_num=6, df_denom=9)
    "certified_f": 330.285339234588,

    # Degrees of freedom
    "df_regression": 6,
    "df_residual":   9,

    # NIST certified sums of squares
    "ssr_regression": 184172401.944494,
    "ssr_residual":   836424.055505915,

    # NIST certified mean squares
    "mse_regression": 30695400.3240823,
    "mse_residual":   92936.0061673238,   # = scale in statsmodels sense

    # NIST certified residual standard deviation = sqrt(mse_residual)
    "residual_sd": 304.854073561965,

    "difficulty": "Higher",

    "notes": (
        "Certified values extracted verbatim from "
        "statsmodels/sandbox/regression/data/Longley.dat "
        "(ASCII copy of NIST StRD file) and cross-checked against "
        "statsmodels/regression/tests/results/results_regression.py "
        "(class Longley, attributed to NIST StRD). "
        "Raw data rows read from lines 61-76 of Longley.dat. "
        "Note: statsmodels results_regression.py stores params in order "
        "[B1..B6, B0] (no-intercept-first convention used in that test); "
        "certified_coefs here follow the canonical NIST order [B0, B1..B6]."
    ),
}

# ---------------------------------------------------------------------------
# Top-level registry
# ---------------------------------------------------------------------------

DATASETS = {
    "Longley": LONGLEY,
}
