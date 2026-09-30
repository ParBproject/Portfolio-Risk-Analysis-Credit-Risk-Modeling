"""Seeded synthetic book. This does not recreate ``portfolio_data.csv``.

The checked-in file has no saved seed. PD on that file is an in-sample
fit. Here the scorecard coefficients are fixed constants: default is
drawn from PD, and PD is not re-estimated from the draw.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Hand-set scorecard. These are not fitted to the synthetic defaults.
SCORECARD = {
    "intercept": 6.5,
    "Credit_Score": -0.012,
    "Loan_Amount": 0.0,
    "Operational_Risk_Score": 0.004,
}

DEFAULT_SEED = 42
DEFAULT_N = 1_000


def scorecard_pd(frame: pd.DataFrame) -> np.ndarray:
    logit = (
        SCORECARD["intercept"]
        + SCORECARD["Credit_Score"] * frame["Credit_Score"].to_numpy(dtype=float)
        + SCORECARD["Loan_Amount"] * frame["Loan_Amount"].to_numpy(dtype=float)
        + SCORECARD["Operational_Risk_Score"]
        * frame["Operational_Risk_Score"].to_numpy(dtype=float)
    )
    return 1.0 / (1.0 + np.exp(-logit))


def generate_portfolio(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Draw one book. The same seed returns the same rows."""
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    credit_score = np.clip(np.rint(rng.normal(680, 70, n)), 300, 850).astype(int)
    loan_amount = np.clip(
        -160_000 + 380 * credit_score + rng.normal(0, 12_000, n),
        10_000,
        None,
    )
    loan_amount = np.round(loan_amount, 2)
    operational_risk = np.clip(np.round(rng.normal(40, 15, n), 1), 0, 100)
    revenue = np.clip(np.round(rng.normal(500_000, 140_000, n), 2), 100_000, None)
    expense_ratio = rng.uniform(0.60, 0.90, n)
    expenses = np.round(revenue * expense_ratio, 2)
    net_income = np.round(revenue - expenses, 2)
    frame = pd.DataFrame(
        {
            "Customer_ID": [f"CUST_{i:04d}" for i in range(n)],
            "Credit_Score": credit_score,
            "Loan_Amount": loan_amount,
            "Operational_Risk_Score": operational_risk,
            "Revenue": revenue,
            "Expenses": expenses,
            "Net_Income": net_income,
        }
    )
    pd_score = scorecard_pd(frame)
    default = (rng.random(n) < pd_score).astype(int)
    frame["Default"] = default
    frame["PD_Score"] = pd_score
    return frame


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write a seeded synthetic loan book")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_portfolio.csv"),
    )
    args = parser.parse_args(argv)
    frame = generate_portfolio(n=args.n, seed=args.seed)
    frame.to_csv(args.output, index=False)
    print(f"wrote {len(frame)} loans to {args.output} (seed={args.seed})")


if __name__ == "__main__":
    main()
