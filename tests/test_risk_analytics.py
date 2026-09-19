from pathlib import Path

import pandas as pd
import pytest

from src.risk_analytics import (
    calibration_table,
    concentration_summary,
    expected_loss_summary,
    model_diagnostics,
    profitability_stress,
    stress_grid,
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
    assert 0 <= metrics.ks_statistic <= 1
    assert metrics.brier_score >= 0
    assert metrics.log_loss >= 0


def test_calibration_table_reconciles_borrower_count(portfolio):
    table = calibration_table(portfolio)
    assert table["Borrowers"].sum() == len(portfolio)


def test_concentration_reconciles_exposure(portfolio):
    summary = concentration_summary(portfolio)
    assert summary.total_exposure == pytest.approx(portfolio["Loan_Amount"].sum())
    assert 0 < summary.hhi <= 1
    assert summary.top_10_exposure_share >= summary.largest_exposure_share


def test_expected_loss_increases_under_severe_stress(portfolio):
    baseline = expected_loss_summary(portfolio)
    stressed = expected_loss_summary(
        portfolio,
        pd_multiplier=1.5,
        lgd_multiplier=1.25,
    )
    assert stressed.expected_loss >= baseline.expected_loss
    assert stressed.expected_loss_ratio >= baseline.expected_loss_ratio


def test_stress_grid_contains_baseline(portfolio):
    grid = stress_grid(
        portfolio,
        pd_multipliers=(1.0, 1.5),
        lgd_multipliers=(1.0, 1.25),
    )
    assert len(grid) == 4
    assert (grid["Expected Loss"] >= 0).all()


def test_profitability_stress_reduces_income(portfolio):
    result = profitability_stress(
        portfolio,
        revenue_multiplier=0.80,
        expense_multiplier=1.10,
    )
    assert result["stressed_net_income"] < result["baseline_net_income"]
    assert 0 <= result["loss_making_share"] <= 1
