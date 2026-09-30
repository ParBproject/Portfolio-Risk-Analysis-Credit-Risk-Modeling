"""Quantities the loan file does not identify.

These are scenario choices, not estimates. Replace them before quoting
expected loss or VaR as a property of the book.
"""

# Constant loss given default applied to every loan. Not estimated:
# the file has no recovery, collateral, or charge-off column.
ASSUMED_LGD = 0.45

# One-factor Gaussian asset correlation used for the illustrative
# credit-loss distribution. Not a Basel risk weight.
ASSET_CORRELATION = 0.15

# Draws for the simulated loss distribution. The seed makes the
# quantiles repeatable for a pinned NumPy version.
VAR_SIMULATIONS = 20_000
VAR_SEED = 42
VAR_CHUNK = 2_000

# Conditional credit stress: systematic factor at this percentile
# (low factor is the adverse direction in the copula below).
STRESS_FACTOR_QUANTILE = 0.05

HIGH_PD = 0.20
OP_RISK_FLAG = 60.0

# Holdout used only for discrimination. It does not replace the
# published in-sample PD column on the book.
N_FOLDS = 5
MODEL_SEED = 42

# Earnings shocks from the original memo. They move borrower revenue
# and expenses. They are not a credit-loss scenario.
REVENUE_SHOCK = "0.20"
EXPENSE_SHOCK = "0.10"
