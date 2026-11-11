"""Closed-form conditional PD, tail-mean ES, and Monte Carlo error."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from credit_risk.metrics import (
    _tail_mean,
    borrower_income_percentiles,
    conditional_pd,
    earnings_scenarios,
    loss_given_default,
    loss_quantiles,
    monte_carlo_intervals,
    round_cents,
    scorecard_fit,
    simulate_portfolio_loss,
)
from credit_risk.report import usd


def test_conditional_pd_is_the_vasicek_formula_with_the_adverse_tail_low():
    probability = np.array([0.30])
    stressed = conditional_pd(probability, factor_quantile=0.05, rho=0.15)
    factor = float(norm.ppf(0.05))
    expected = float(
        norm.cdf(
            (norm.ppf(0.30) - np.sqrt(0.15) * factor) / np.sqrt(1.0 - 0.15)
        )
    )
    assert stressed[0] == pytest.approx(expected)
    assert stressed[0] == pytest.approx(0.54862356, abs=1e-8)
    # M = 0 is not the unconditional PD: the sqrt(1-rho) scale remains.
    median = conditional_pd(np.array([0.20]), 0.50, 0.15)[0]
    assert median == pytest.approx(0.18065641, abs=1e-6)
    assert median < 0.20
    assert conditional_pd(np.array([0.20]), 0.05, 0.15)[0] > 0.20
    assert conditional_pd(np.array([0.20]), 0.95, 0.15)[0] < median
    # No systematic factor: the conditional PD is the input PD.
    unchanged = conditional_pd(np.array([0.2, 0.5, 0.8]), 0.05, 0.0)
    assert unchanged == pytest.approx([0.2, 0.5, 0.8])
    clipped = conditional_pd(np.array([0.0, 1.0]), 0.5, 0.0)
    assert clipped == pytest.approx([1e-12, 1.0 - 1e-12])
    draws = np.random.default_rng(0).standard_normal(200_000)
    threshold = float(norm.ppf(0.20))
    rho = 0.15
    averaged = norm.cdf((threshold - np.sqrt(rho) * draws) / np.sqrt(1.0 - rho))
    assert averaged.mean() == pytest.approx(0.20, abs=0.002)


def test_expected_shortfall_is_the_mean_of_a_fixed_worst_count():
    losses = np.arange(20_000, dtype=float)
    quantiles = loss_quantiles(losses)
    assert quantiles["es_95_count"] == 1000
    assert quantiles["es_99_count"] == 200
    assert quantiles["es_95"] == pytest.approx(np.arange(19_000, 20_000).mean())
    assert quantiles["es_99"] == pytest.approx(np.arange(19_800, 20_000).mean())
    assert quantiles["es_99"] > quantiles["es_95"] > quantiles["var_95"]
    flat, count = _tail_mean(np.array([1.0] * 10 + [2.0] * 10), 0.75)
    assert count == 5
    assert flat == 2.0


def test_simulation_error_is_wider_than_one_cent_and_contains_the_point(portfolio):
    _, results = portfolio
    error = results["simulation_error"]
    assert error["n_boot"] == 1000
    assert error["seed"] == 42
    assert usd(error["var_95_low"]) == "$9,436,973.61"
    assert usd(error["var_95_high"]) == "$9,622,340.47"
    assert usd(error["var_99_low"]) == "$11,768,211.11"
    assert usd(error["var_99_high"]) == "$12,085,866.62"
    assert usd(error["es_95_low"]) == "$10,862,816.41"
    assert usd(error["es_95_high"]) == "$11,107,677.87"
    assert usd(error["es_99_low"]) == "$12,907,830.34"
    assert usd(error["es_99_high"]) == "$13,362,475.13"
    assert error["var_95_low"] < results["credit_loss_var_95"] < error["var_95_high"]
    assert error["var_99_low"] < results["credit_loss_var_99"] < error["var_99_high"]
    assert error["var_95_high"] - error["var_95_low"] > 100_000
    assert error["es_95_low"] < results["credit_loss_es_95"] < error["es_95_high"]
    assert abs(results["mean_simulated_loss"] - results["expected_loss"]) < 50_000
    assert results["credit_loss_es_95_count"] == 1000
    assert results["credit_loss_es_99_count"] == 200


def test_flat_losses_have_equal_var_and_expected_shortfall():
    quantiles = loss_quantiles(np.full(100, 5.0))
    assert quantiles["var_95"] == quantiles["var_99"] == 5.0
    assert quantiles["es_95"] == quantiles["es_99"] == 5.0
    # One percent of 20 observations rounds to an empty tail.
    with pytest.raises(ValueError):
        loss_quantiles(np.arange(20, dtype=float))


def test_loss_given_default_rounds_each_loan_to_the_cent():
    amounts = pd.Series([100.0, 10.01, 2.22, 0.03, 0.01, 0.0])
    # 45.00; 1001 cents * 0.45 = 450.45 → 4.50; 222 * 0.45 = 99.9 → 1.00;
    # 3 * 0.45 = 1.35 → 0.01; 1 * 0.45 = 0.45 → 0; zero exposure stays 0.
    assert loss_given_default(amounts, 0.45).tolist() == [45.0, 4.5, 1.0, 0.01, 0.0, 0.0]
    # Half a cent rounds half up: 0.5 → 1 cent, 1.5 → 2 cents, 500.5 → 501 cents.
    assert loss_given_default(pd.Series([0.01, 0.03, 10.01]), 0.5).tolist() == [
        0.01,
        0.02,
        5.01,
    ]
    assert loss_given_default(pd.Series([50.0, 0.0]), 0.0).tolist() == [0.0, 0.0]
    assert round_cents(Decimal("1.005")) == Decimal("1.01")
    assert round_cents(Decimal("1.004")) == Decimal("1.00")
    assert round_cents(Decimal("-1.005")) == Decimal("-1.01")


def test_earnings_shocks_are_twenty_percent_and_ten_percent():
    scenarios = earnings_scenarios(Decimal("100"), Decimal("40"))
    assert scenarios["base"] == Decimal("60")
    assert scenarios["revenue_down_20"] == Decimal("40.00")
    assert scenarios["expenses_up_10"] == Decimal("56.00")
    assert scenarios["combined"] == Decimal("36.00")


def test_income_percentiles_are_higher_order_statistics():
    # 100 incomes, 0 through 99, stored in reverse. Index ceil((n - 1) * q):
    # q = 0.01 → ceil(0.99) = 1 → 1; q = 0.05 → ceil(4.95) = 5 → 5.
    income = pd.Series(np.arange(99, -1, -1, dtype=float))
    tails = borrower_income_percentiles(income)
    assert tails["p01"] == 1.0
    assert tails["p05"] == 5.0
    assert borrower_income_percentiles(pd.Series([12.5])) == {"p05": 12.5, "p01": 12.5}


def test_scorecard_fit_recovers_a_full_rank_logit():
    credit = np.array([500.0, 600.0, 700.0, 800.0])
    loan = np.array([10_000.0, 80_000.0, 20_000.0, 50_000.0])
    operational = np.array([10.0, 70.0, 20.0, 40.0])
    intercept, credit_coef, loan_coef, operational_coef = -1.5, 0.002, -1e-5, 0.01
    logit = (
        intercept
        + credit_coef * credit
        + loan_coef * loan
        + operational_coef * operational
    )
    frame = pd.DataFrame(
        {
            "Credit_Score": credit,
            "Loan_Amount": loan,
            "Operational_Risk_Score": operational,
            "PD_Score": 1.0 / (1.0 + np.exp(-logit)),
        }
    )
    fit = scorecard_fit(frame)
    assert fit["intercept"] == pytest.approx(intercept)
    assert fit["Credit_Score"] == pytest.approx(credit_coef)
    assert fit["Loan_Amount"] == pytest.approx(loan_coef)
    assert fit["Operational_Risk_Score"] == pytest.approx(operational_coef)
    assert fit["max_abs_pd_gap"] < 1e-8


def test_portfolio_loss_on_a_known_normal_draw(monkeypatch):
    # rho = 0.36. One factor draw of -1 and idiosyncratic draws (1, -1).
    # Asset = 0.6 * M + 0.8 * Z, so the loans are 0.2 and -1.4.
    # At PD 0.5 the threshold is 0: only the second loan defaults.
    class FixedDraws:
        def standard_normal(self, size):
            if isinstance(size, tuple):
                return np.array([[1.0, -1.0]])
            return np.array([-1.0])

    monkeypatch.setattr(
        "credit_risk.metrics.np.random.default_rng", lambda _seed: FixedDraws()
    )
    losses = simulate_portfolio_loss(
        np.array([0.5, 0.5]),
        np.array([10.0, 25.50]),
        rho=0.36,
        n_sims=1,
        seed=0,
    )
    assert losses.tolist() == [25.50]


def test_zero_loss_given_default_produces_zero_portfolio_loss():
    losses = simulate_portfolio_loss(
        np.array([0.2, 0.9]),
        np.array([0.0, 0.0]),
        n_sims=4,
        seed=1,
    )
    assert losses.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_asset_correlation_must_be_in_the_unit_interval():
    with pytest.raises(ValueError):
        simulate_portfolio_loss(np.array([0.2]), np.array([1.0]), rho=1.0, n_sims=1)
    with pytest.raises(ValueError):
        simulate_portfolio_loss(np.array([0.2]), np.array([1.0]), rho=-0.1, n_sims=1)


def test_a_constant_loss_sample_has_no_simulation_width():
    losses = np.full(2_000, 10.0)
    band = monte_carlo_intervals(losses, n_boot=40, seed=1)
    assert band["var_95_low"] == band["var_95_high"] == 10.0
    assert band["es_99_low"] == band["es_99_high"] == 10.0
