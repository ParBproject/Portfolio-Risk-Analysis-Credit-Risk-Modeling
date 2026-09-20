from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.risk_analytics import (
    calibration_table,
    concentration_summary,
    decile_lift_table,
    expected_loss_summary,
    model_diagnostics,
    profitability_stress,
    risk_segments,
    stress_grid,
    top_risk_accounts,
    validate_portfolio,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def portfolio():
    return pd.read_csv(ROOT / "portfolio_data.csv")


def test_portfolio_schema_is_valid(portfolio):
    clean = validate_portfolio(portfolio)
    assert len(clean) == len(portfolio)
    assert clean["PD_Score"].between(0, 1).all()


def test_model_diagnostics_are_bounded(portfolio):
    metrics = model_diagnostics(portfolio)
    assert 0 <= metrics.roc_auc <= 1
    assert -1 <= metrics.gini_coefficient <= 1
    assert metrics.gini_coefficient == pytest.approx(2 * metrics.roc_auc - 1)
    assert 0 <= metrics.ks_statistic <= 1
    assert metrics.brier_score >= 0
    assert metrics.log_loss >= 0
    assert metrics.discrimination_available


def test_single_class_defaults_keep_calibration_metrics_available(portfolio):
    one_class = portfolio.copy()
    one_class["Default"] = 0
    metrics = model_diagnostics(one_class)
    assert np.isnan(metrics.roc_auc)
    assert np.isnan(metrics.ks_statistic)
    assert metrics.brier_score >= 0
    assert metrics.log_loss >= 0
    assert not metrics.discrimination_available


def test_decile_lift_table_reconciles_population_and_defaults(portfolio):
    table = decile_lift_table(portfolio, n_buckets=10)
    assert table["Borrowers"].sum() == len(portfolio)
    assert table["Defaults"].sum() == portfolio["Default"].sum()
    assert table["Risk_Decile"].is_monotonic_increasing
    if portfolio["Default"].sum() > 0:
        assert table["Cumulative_Default_Capture"].iloc[-1] == pytest.approx(1.0)
    assert table["Cumulative_Borrower_Share"].iloc[-1] == pytest.approx(1.0)


def test_decile_lift_rejects_invalid_bucket_count(portfolio):
    with pytest.raises(ValueError, match="positive integer"):
        decile_lift_table(portfolio, n_buckets=0)


def test_calibration_table_reconciles_borrower_count(portfolio):
    table = calibration_table(portfolio)
    assert table["Borrowers"].sum() == len(portfolio)
    assert (table["Observed_DR_Lower_95"] >= 0).all()
    assert (table["Observed_DR_Upper_95"] <= 1).all()
    assert (
        table["Observed_DR_Lower_95"]
        <= table["Observed_Default_Rate"]
    ).all()
    assert (
        table["Observed_DR_Upper_95"]
        >= table["Observed_Default_Rate"]
    ).all()


def test_concentration_reconciles_exposure(portfolio):
    summary = concentration_summary(portfolio)
    assert summary.total_exposure == pytest.approx(portfolio["Loan_Amount"].sum())
    assert 0 < summary.hhi <= 1
    assert summary.top_10_exposure_share >= summary.largest_exposure_share


def test_expected_loss_matches_hand_calculation(portfolio):
    toy = portfolio.head(2).copy()
    toy["Customer_ID"] = ["A", "B"]
    toy["Loan_Amount"] = [100.0, 200.0]
    toy["PD_Score"] = [0.10, 0.20]
    result = expected_loss_summary(toy, lgd=0.50)
    assert result.total_exposure == pytest.approx(300.0)
    assert result.expected_loss == pytest.approx(25.0)
    assert result.expected_loss_ratio == pytest.approx(25.0 / 300.0)


def test_concentration_matches_hand_calculation(portfolio):
    toy = portfolio.head(2).copy()
    toy["Customer_ID"] = ["A", "B"]
    toy["Loan_Amount"] = [25.0, 75.0]
    result = concentration_summary(toy)
    expected_hhi = 0.25**2 + 0.75**2
    assert result.largest_exposure_share == pytest.approx(0.75)
    assert result.top_10_exposure_share == pytest.approx(1.0)
    assert result.hhi == pytest.approx(expected_hhi)
    assert result.effective_borrower_count == pytest.approx(1 / expected_hhi)


def test_account_expected_loss_contributions_reconcile(portfolio):
    accounts = top_risk_accounts(portfolio, limit=len(portfolio), lgd=0.45)
    summary = expected_loss_summary(portfolio, lgd=0.45)
    assert accounts["Expected_Loss_Contribution"].sum() == pytest.approx(
        summary.expected_loss
    )
    assert accounts["Portfolio_EL_Share"].sum() == pytest.approx(1.0)


def test_expected_loss_increases_under_severe_stress(portfolio):
    baseline = expected_loss_summary(portfolio)
    stressed = expected_loss_summary(
        portfolio,
        pd_multiplier=1.5,
        lgd_multiplier=1.25,
    )
    assert stressed.expected_loss >= baseline.expected_loss
    assert stressed.expected_loss_ratio >= baseline.expected_loss_ratio


def test_invalid_baseline_lgd_is_rejected(portfolio):
    with pytest.raises(ValueError, match="between 0 and 1"):
        expected_loss_summary(portfolio, lgd=1.20)


def test_stress_grid_contains_baseline(portfolio):
    grid = stress_grid(
        portfolio,
        pd_multipliers=(1.0, 1.5),
        lgd_multipliers=(1.0, 1.25),
    )
    assert len(grid) == 4
    assert (grid["Expected Loss"] >= 0).all()


def test_profitability_stress_uses_consistent_operating_income_basis(portfolio):
    modified = portfolio.copy()
    modified["Net_Income"] = modified["Net_Income"] + 1000.0
    result = profitability_stress(
        modified,
        revenue_multiplier=0.80,
        expense_multiplier=1.10,
    )
    expected_baseline = float((modified["Revenue"] - modified["Expenses"]).sum())
    assert result["baseline_operating_income"] == pytest.approx(expected_baseline)
    assert result["stressed_operating_income"] < result["baseline_operating_income"]
    assert result["net_income_reconciliation_gap"] == pytest.approx(
        float(modified["Net_Income"].sum()) - expected_baseline
    )
    assert 0 <= result["loss_making_share"] <= 1


def test_empty_portfolio_is_rejected():
    with pytest.raises(ValueError, match="at least one borrower"):
        validate_portfolio(pd.DataFrame())


def test_infinite_numeric_value_is_rejected(portfolio):
    invalid = portfolio.head(3).copy()
    invalid.loc[invalid.index[0], "Loan_Amount"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_portfolio(invalid)


def test_zero_total_exposure_is_rejected(portfolio):
    invalid = portfolio.head(3).copy()
    invalid["Loan_Amount"] = 0.0
    with pytest.raises(ValueError, match="total exposure"):
        validate_portfolio(invalid)


def test_expected_loss_exposure_weighted_pd_matches_formula(portfolio):
    toy = portfolio.head(2).copy()
    toy["Customer_ID"] = ["A", "B"]
    toy["Loan_Amount"] = [100.0, 900.0]
    toy["PD_Score"] = [0.10, 0.30]
    result = expected_loss_summary(toy, lgd=0.50)
    expected_weighted_pd = (100.0 * 0.10 + 900.0 * 0.30) / 1000.0
    assert result.exposure_weighted_pd == pytest.approx(expected_weighted_pd)
    assert result.borrower_average_pd == pytest.approx(0.20)
    assert result.expected_loss_ratio == pytest.approx(expected_weighted_pd * 0.50)


def test_segment_expected_loss_reconciles_to_portfolio(portfolio):
    segments = risk_segments(portfolio, lgd=0.45)
    summary = expected_loss_summary(portfolio, lgd=0.45)
    assert segments["Exposure"].sum() == pytest.approx(summary.total_exposure)
    assert segments["Expected_Loss"].sum() == pytest.approx(summary.expected_loss)
    assert segments["Expected_Loss_Share"].sum() == pytest.approx(1.0)


def test_unlabeled_current_portfolio_supports_risk_analytics(portfolio):
    current = portfolio.drop(columns=["Default", "Net_Income"]).head(20)
    clean = validate_portfolio(current)
    expected_loss = expected_loss_summary(clean)
    segments = risk_segments(clean)
    accounts = top_risk_accounts(clean, limit=5)

    assert expected_loss.expected_loss > 0
    assert segments["Observed_Default_Rate"].isna().all()
    assert "Default" not in accounts.columns
    assert "Expected_Loss_Contribution" in accounts.columns
    assert accounts["Expected_Loss_Contribution"].is_monotonic_decreasing


def test_model_validation_requires_realized_default_column(portfolio):
    current = portfolio.drop(columns=["Default"]).head(20)
    with pytest.raises(ValueError, match="Default is required"):
        model_diagnostics(current)


def test_top_risk_accounts_rejects_non_integer_limit(portfolio):
    with pytest.raises(ValueError, match="positive integer"):
        top_risk_accounts(portfolio, limit=2.5)
