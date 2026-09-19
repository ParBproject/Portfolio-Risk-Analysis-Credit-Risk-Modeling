"""Credit-risk model monitoring, portfolio risk, and stress analytics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


REQUIRED_COLUMNS = {
    "Customer_ID",
    "Credit_Score",
    "Loan_Amount",
    "Operational_Risk_Score",
    "Revenue",
    "Expenses",
    "Net_Income",
    "Default",
    "PD_Score",
}


@dataclass(frozen=True)
class ModelDiagnostics:
    """Discrimination and calibration diagnostics for PD estimates."""

    observations: int
    default_rate: float
    average_pd: float
    roc_auc: float
    ks_statistic: float
    brier_score: float
    log_loss: float


@dataclass(frozen=True)
class ConcentrationSummary:
    """Exposure concentration diagnostics."""

    total_exposure: float
    largest_exposure_share: float
    top_10_exposure_share: float
    hhi: float
    effective_borrower_count: float


@dataclass(frozen=True)
class ExpectedLossSummary:
    """Portfolio expected-loss result."""

    total_exposure: float
    expected_loss: float
    expected_loss_ratio: float
    average_pd: float
    lgd: float


def validate_portfolio(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize borrower-level credit-risk data."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    clean = frame.copy()
    numeric = [
        "Credit_Score",
        "Loan_Amount",
        "Operational_Risk_Score",
        "Revenue",
        "Expenses",
        "Net_Income",
        "Default",
        "PD_Score",
    ]
    for column in numeric:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    if clean["Customer_ID"].isna().any():
        raise ValueError("Customer_ID contains missing values")
    if clean["Customer_ID"].duplicated().any():
        raise ValueError("Customer_ID must be unique")
    if clean[numeric].isna().any(axis=None):
        raise ValueError("numeric risk fields contain missing or invalid values")
    if not clean["Default"].isin([0, 1]).all():
        raise ValueError("Default must be binary")
    if not clean["PD_Score"].between(0.0, 1.0).all():
        raise ValueError("PD_Score must be between 0 and 1")
    if (clean["Loan_Amount"] < 0).any():
        raise ValueError("Loan_Amount must be non-negative")
    return clean


def model_diagnostics(frame: pd.DataFrame) -> ModelDiagnostics:
    """Calculate PD discrimination and calibration diagnostics."""
    clean = validate_portfolio(frame)
    y_true = clean["Default"].astype(int).to_numpy()
    pd_score = clean["PD_Score"].to_numpy(dtype=float)

    if np.unique(y_true).size < 2:
        raise ValueError("both default classes are required for discrimination metrics")

    fpr, tpr, _ = roc_curve(y_true, pd_score)
    ks = float(np.max(tpr - fpr))
    clipped = np.clip(pd_score, 1e-12, 1.0 - 1e-12)

    return ModelDiagnostics(
        observations=len(clean),
        default_rate=float(y_true.mean()),
        average_pd=float(pd_score.mean()),
        roc_auc=float(roc_auc_score(y_true, pd_score)),
        ks_statistic=ks,
        brier_score=float(brier_score_loss(y_true, pd_score)),
        log_loss=float(log_loss(y_true, clipped, labels=[0, 1])),
    )


def calibration_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate predicted vs observed default rate by PD risk band."""
    clean = validate_portfolio(frame)
    bins = [-1e-12, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0]
    labels = [
        "0–5%",
        "5–10%",
        "10–20%",
        "20–40%",
        "40–60%",
        "60–80%",
        "80–100%",
    ]
    band = pd.cut(
        clean["PD_Score"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    table = (
        clean.assign(PD_Band=band)
        .groupby("PD_Band", observed=True)
        .agg(
            Borrowers=("Customer_ID", "size"),
            Exposure=("Loan_Amount", "sum"),
            Average_PD=("PD_Score", "mean"),
            Observed_Default_Rate=("Default", "mean"),
        )
        .reset_index()
    )
    table["Calibration_Gap"] = (
        table["Observed_Default_Rate"] - table["Average_PD"]
    )
    return table


def concentration_summary(frame: pd.DataFrame) -> ConcentrationSummary:
    """Measure borrower exposure concentration using HHI and top shares."""
    clean = validate_portfolio(frame)
    exposure = clean["Loan_Amount"].to_numpy(dtype=float)
    total = float(exposure.sum())
    if total <= 0:
        raise ValueError("total exposure must be positive")
    shares = exposure / total
    ordered = np.sort(shares)[::-1]
    hhi = float(np.sum(shares**2))
    return ConcentrationSummary(
        total_exposure=total,
        largest_exposure_share=float(ordered[0]),
        top_10_exposure_share=float(ordered[: min(10, len(ordered))].sum()),
        hhi=hhi,
        effective_borrower_count=float(1.0 / hhi),
    )


def expected_loss_summary(
    frame: pd.DataFrame,
    *,
    lgd: float = 0.45,
    pd_multiplier: float = 1.0,
    lgd_multiplier: float = 1.0,
) -> ExpectedLossSummary:
    """Calculate exposure-weighted PD × LGD × EAD expected loss."""
    clean = validate_portfolio(frame)
    for name, value in {
        "lgd": lgd,
        "pd_multiplier": pd_multiplier,
        "lgd_multiplier": lgd_multiplier,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    stressed_pd = np.minimum(clean["PD_Score"].to_numpy() * pd_multiplier, 1.0)
    stressed_lgd = min(float(lgd) * lgd_multiplier, 1.0)
    exposure = clean["Loan_Amount"].to_numpy(dtype=float)
    total_exposure = float(exposure.sum())
    expected_loss = float(np.sum(exposure * stressed_pd * stressed_lgd))

    return ExpectedLossSummary(
        total_exposure=total_exposure,
        expected_loss=expected_loss,
        expected_loss_ratio=expected_loss / total_exposure if total_exposure else 0.0,
        average_pd=float(np.mean(stressed_pd)),
        lgd=stressed_lgd,
    )


def stress_grid(
    frame: pd.DataFrame,
    *,
    lgd: float = 0.45,
    pd_multipliers: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0),
    lgd_multipliers: tuple[float, ...] = (1.0, 1.25, 1.5),
) -> pd.DataFrame:
    """Build a transparent PD/LGD expected-loss stress grid."""
    rows = []
    for pd_multiplier in pd_multipliers:
        for lgd_multiplier in lgd_multipliers:
            result = expected_loss_summary(
                frame,
                lgd=lgd,
                pd_multiplier=pd_multiplier,
                lgd_multiplier=lgd_multiplier,
            )
            rows.append(
                {
                    "PD Multiplier": pd_multiplier,
                    "LGD Multiplier": lgd_multiplier,
                    "Expected Loss": result.expected_loss,
                    "Expected Loss Ratio": result.expected_loss_ratio,
                    "Stressed Average PD": result.average_pd,
                    "Stressed LGD": result.lgd,
                }
            )
    return pd.DataFrame(rows)


def profitability_stress(
    frame: pd.DataFrame,
    *,
    revenue_multiplier: float = 0.80,
    expense_multiplier: float = 1.10,
) -> dict[str, float]:
    """Apply deterministic revenue/expense shocks to borrower operating results."""
    clean = validate_portfolio(frame)
    if revenue_multiplier < 0 or expense_multiplier < 0:
        raise ValueError("stress multipliers must be non-negative")

    base_income = float(clean["Net_Income"].sum())
    stressed_income = float(
        (
            clean["Revenue"] * revenue_multiplier
            - clean["Expenses"] * expense_multiplier
        ).sum()
    )
    return {
        "baseline_net_income": base_income,
        "stressed_net_income": stressed_income,
        "net_income_change": stressed_income - base_income,
        "loss_making_share": float(
            (
                clean["Revenue"] * revenue_multiplier
                - clean["Expenses"] * expense_multiplier
                < 0
            ).mean()
        ),
    }


def risk_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Create transparent PD-based portfolio segments for monitoring."""
    clean = validate_portfolio(frame)
    bins = [-1e-12, 0.10, 0.20, 0.40, 0.60, 1.0]
    labels = ["Low", "Moderate", "Elevated", "High", "Severe"]
    segments = pd.cut(
        clean["PD_Score"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    result = (
        clean.assign(Risk_Segment=segments)
        .groupby("Risk_Segment", observed=True)
        .agg(
            Borrowers=("Customer_ID", "size"),
            Exposure=("Loan_Amount", "sum"),
            Average_PD=("PD_Score", "mean"),
            Observed_Default_Rate=("Default", "mean"),
            Average_Credit_Score=("Credit_Score", "mean"),
        )
        .reset_index()
    )
    result["Exposure_Share"] = result["Exposure"] / result["Exposure"].sum()
    return result


def top_risk_accounts(frame: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Prioritize high-PD and high-exposure accounts for review."""
    clean = validate_portfolio(frame)
    if limit < 1:
        raise ValueError("limit must be at least 1")
    exposure_share = clean["Loan_Amount"] / clean["Loan_Amount"].sum()
    clean["Risk_Priority_Score"] = clean["PD_Score"] * np.sqrt(exposure_share)
    return (
        clean.sort_values("Risk_Priority_Score", ascending=False)
        .head(limit)
        [
            [
                "Customer_ID",
                "Credit_Score",
                "Loan_Amount",
                "Operational_Risk_Score",
                "PD_Score",
                "Default",
                "Risk_Priority_Score",
            ]
        ]
        .reset_index(drop=True)
    )
