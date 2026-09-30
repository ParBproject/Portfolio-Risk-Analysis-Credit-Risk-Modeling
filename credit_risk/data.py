"""Load the checked-in loan book."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "portfolio_data.csv"

REQUIRED_COLUMNS = (
    "Customer_ID",
    "Credit_Score",
    "Loan_Amount",
    "Operational_Risk_Score",
    "Revenue",
    "Expenses",
    "Net_Income",
    "Default",
    "PD_Score",
)

# Columns that must not enter a PD fit. PD_Score is the published
# in-sample score. Default is the label. Earnings are contemporaneous
# borrower results, not application features in this case study.
LEAKAGE_COLUMNS = (
    "Default",
    "PD_Score",
    "Revenue",
    "Expenses",
    "Net_Income",
)

PD_FEATURES = (
    "Credit_Score",
    "Loan_Amount",
    "Operational_Risk_Score",
)

# Name fragments that would mark an origination, snapshot, or default
# date. This file has none, so an out-of-time split is not identified.
_TIME_TOKENS = ("date", "time", "origination", "vintage", "asof", "as_of")

MONEY_COLUMNS = ("Loan_Amount", "Revenue", "Expenses", "Net_Income")


def money_sums(path: Path | None = None) -> dict[str, Decimal]:
    """Sum money columns in decimal cents, matching the CSV text."""
    frame = pd.read_csv(path or DATA_PATH, dtype={c: str for c in MONEY_COLUMNS})
    totals = {}
    for column in MONEY_COLUMNS:
        total = Decimal("0")
        for raw in frame[column]:
            total += Decimal(raw)
        totals[column] = total
    return totals


def has_time_column(frame: pd.DataFrame) -> bool:
    """True when a column name looks like an origination or default date."""
    for column in frame.columns:
        name = str(column).lower().replace(" ", "").replace("-", "")
        if any(token in name for token in _TIME_TOKENS):
            return True
    return False


def load_portfolio(path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or DATA_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"portfolio file is missing columns: {missing}")
    if frame["Customer_ID"].duplicated().any():
        raise ValueError("Customer_ID values are not unique")
    if not frame["Default"].isin([0, 1]).all():
        raise ValueError("Default must be 0 or 1")
    if ((frame["PD_Score"] <= 0) | (frame["PD_Score"] >= 1)).any():
        raise ValueError("PD_Score must be strictly between 0 and 1")
    if (frame["Loan_Amount"] <= 0).any():
        raise ValueError("Loan_Amount must be positive")
    return frame
