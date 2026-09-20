"""Credit-risk model monitoring, portfolio risk, and stress analytics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


BASE_REQUIRED_COLUMNS = {
    "Customer_ID",
    "Credit_Score",
    "Loan_Amount",
    "Operational_Risk_Score",
    "Revenue",
    "Expenses",
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

    @property
    def discrimination_available(self) -> bool:
        """Whether ROC-AUC and KS are defined for the observed sample."""
        return bool(np.isfinite(self.roc_auc) and np.isfinite(self.ks_statistic))


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


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        "Credit_Score",
        "Loan_Amount",
        "Operational_Risk_Score",
        "Revenue",
        "Expenses",
        "PD_Score",
    ]
    columns.extend(
        column for column in ("Net_Income", "Default") if column in frame.columns
    )
    return columns


def validate_portfolio(
    frame: pd.DataFrame,
    *,
    require_default: bool = False,
) -> pd.DataFrame:
    """Validate and normalize borrower-level credit-risk data.

    Default and Net_Income are optional so current unlabeled portfolios can still
    be used for exposure, expected-loss, stress, segmentation, and review analytics.
    Model-validation functions explicitly require realized defaults.
    """
    if frame is None or frame.empty:
        raise ValueError("portfolio must contain at least one borrower")

    missing = BASE_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if require_default and "Default" not in frame.columns:
        raise ValueError("Default is required for model-validation metrics")

    clean = frame.copy()
    numeric = _numeric_columns(clean)
    for column in numeric:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    customer_id = clean["Customer_ID"].astype("string")
    if customer_id.isna().any() or customer_id.str.strip().eq("").any():
        raise ValueError("Customer_ID contains missing or blank values")
    clean["Customer_ID"] = customer_id.str.strip()
    if clean["Customer_ID"].duplicated().any():
        raise ValueError("Customer_ID must be unique")

    if clean[numeric].isna().any(axis=None):
        raise ValueError("numeric risk fields contain missing or invalid values")
    if not np.isfinite(clean[numeric].to_numpy(dtype=float)).all():
        raise ValueError("numeric risk fields must contain only finite values")

    if "Default" in clean.columns and not clean["Default"].isin([0, 1]).all():
        raise ValueError("Default must be binary when supplied")
    if not clean["PD_Score"].between(0.0, 1.0).all():
        raise ValueError("PD_Score must be between 0 and 1")
    if (clean["Loan_Amount"] < 0).any():
        raise ValueError("Loan_Amount must be non-negative")
    if float(clean["Loan_Amount"].sum()) <= 0:
        raise ValueError("portfolio total exposure must be positive")
    if (clean["Revenue"] < 0).any() or (clean["Expenses"] < 0).any():
        raise ValueError("Revenue and Expenses must be non-negative")
    if (clean["Operational_Risk_Score"] < 0).any():
        raise ValueError("Operational_Risk_Score must be non-negative")
    if (clean["Credit_Score"] <= 0).any():
        raise ValueError("Credit_Score must be positive")

    return clean


def model_diagnostics(frame: pd.DataFrame) -> ModelDiagnostics:
    """Calculate PD discrimination and calibration diagnostics.

    ROC-AUC and KS are returned as NaN when the sample contains only one observed
    outcome class. Calibration and probability-loss metrics remain available.
    """
    clean = validate_portfolio(frame, require_default=True)
    y_true = clean["Default"].astype(int).to_numpy()
    pd_score = clean["PD_Score"].to_numpy(dtype=float)

    if np.unique(y_true).size >= 2:
        fpr, tpr, _ = roc_curve(y_true, pd_score)
        ks = float(np.max(tpr - fpr))
        roc_auc = float(roc_auc_score(y_true, pd_score))
    else:
        ks = float("nan")
        roc_auc = float("nan")

    clipped = np.clip(pd_score, 1e-12, 1.0 - 1e-12)
    return ModelDiagnostics(
        observations=len(clean),
        default_rate=float(y_true.mean()),
        average_pd=float(pd_score.mean()),
        roc_auc=roc_auc,
        ks_statistic=ks,
        brier_score=float(brier_score_loss(y_true, pd_score)),
        log_loss=float(log_loss(y_true, clipped, labels=[0, 1])),
    )


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    z2 = z**2
    denominator = 1.0 + z2 / total
    centre = (proportion + z2 / (2.0 * total)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, float(centre - margin)), min(1.0, float(centre + margin))


def calibration_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate predicted vs observed default rate by PD risk band."""
    clean = validate_portfolio(frame, require_default=True)
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
            Defaults=("Default", "sum"),
            Exposure=("Loan_Amount", "sum"),
            Average_PD=("PD_Score", "mean"),
            Observed_Default_Rate=("Default", "mean"),
        )
        .reset_index()
    )
    intervals = [
        _wilson_interval(int(row.Defaults), int(row.Borrowers))
        for row in table.itertuples(index=False)
    ]
    table["Observed_DR_Lower_95"] = [lower for lower, _ in intervals]
    table["Observed_DR_Upper_95"] = [upper for _, upper in intervals]
    table["Calibration_Gap"] = (
        table["Observed_Default_Rate"] - table["Average_PD"]
    )
    return table


def concentration_summary(frame: pd.DataFrame) -> ConcentrationSummary:
    """Measure borrower exposure concentration using HHI and top shares."""
    clean = validate_portfolio(frame)
    exposure = clean["Loan_Amount"].to_numpy(dtype=float)
    total = float(exposure.sum())
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
    if not np.isfinite(lgd) or not 0.0 <= lgd <= 1.0:
        raise ValueError("lgd must be finite and between 0 and 1")
    for name, value in {
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
        expected_loss_ratio=expected_loss / total_exposure,
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
    """Apply deterministic revenue/expense shocks on a consistent operating basis."""
    clean = validate_portfolio(frame)
    for name, value in {
        "revenue_multiplier": revenue_multiplier,
        "expense_multiplier": expense_multiplier,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    baseline_borrower_income = clean["Revenue"] - clean["Expenses"]
    stressed_borrower_income = (
        clean["Revenue"] * revenue_multiplier
        - clean["Expenses"] * expense_multiplier
    )
    baseline_income = float(baseline_borrower_income.sum())
    stressed_income = float(stressed_borrower_income.sum())

    result = {
        "baseline_net_income": baseline_income,
        "stressed_net_income": stressed_income,
        "net_income_change": stressed_income - baseline_income,
        "loss_making_share": float((stressed_borrower_income < 0).mean()),
    }
    if "Net_Income" in clean.columns:
        reported_income = float(clean["Net_Income"].sum())
        result["reported_net_income"] = reported_income
        result["net_income_reconciliation_gap"] = reported_income - baseline_income
    return result


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

    aggregations = {
        "Borrowers": ("Customer_ID", "size"),
        "Exposure": ("Loan_Amount", "sum"),
        "Average_PD": ("PD_Score", "mean"),
        "Average_Credit_Score": ("Credit_Score", "mean"),
    }
    if "Default" in clean.columns:
        aggregations["Observed_Default_Rate"] = ("Default", "mean")

    result = (
        clean.assign(Risk_Segment=segments)
        .groupby("Risk_Segment", observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    result["Exposure_Share"] = result["Exposure"] / result["Exposure"].sum()
    if "Observed_Default_Rate" not in result.columns:
        result["Observed_Default_Rate"] = np.nan
    return result


def top_risk_accounts(
    frame: pd.DataFrame,
    limit: int = 20,
    *,
    lgd: float = 0.45,
) -> pd.DataFrame:
    """Prioritize accounts by expected-loss contribution."""
    clean = validate_portfolio(frame)
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not np.isfinite(lgd) or not 0.0 <= lgd <= 1.0:
        raise ValueError("lgd must be finite and between 0 and 1")

    exposure_share = clean["Loan_Amount"] / clean["Loan_Amount"].sum()
    clean["Expected_Loss_Contribution"] = (
        clean["Loan_Amount"] * clean["PD_Score"] * lgd
    )
    total_expected_loss = float(clean["Expected_Loss_Contribution"].sum())
    clean["Portfolio_EL_Share"] = (
        clean["Expected_Loss_Contribution"] / total_expected_loss
        if total_expected_loss > 0
        else 0.0
    )
    clean["Risk_Priority_Score"] = clean["PD_Score"] * np.sqrt(exposure_share)

    columns = [
        "Customer_ID",
        "Credit_Score",
        "Loan_Amount",
        "Operational_Risk_Score",
        "PD_Score",
        "Expected_Loss_Contribution",
        "Portfolio_EL_Share",
        "Risk_Priority_Score",
    ]
    if "Default" in clean.columns:
        columns.insert(5, "Default")

    return (
        clean.sort_values(
            ["Expected_Loss_Contribution", "PD_Score"],
            ascending=False,
        )
        .head(int(limit))[columns]
        .reset_index(drop=True)
    )
