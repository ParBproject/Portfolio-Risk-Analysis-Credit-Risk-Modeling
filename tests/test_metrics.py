"""Pins for the checked-in book. Literals are the corrected figures."""

from decimal import Decimal

from credit_risk.report import pct, usd


def test_book_size_and_exposure(portfolio):
    _, results = portfolio
    assert results["n_loans"] == 1000
    assert results["n_defaults"] == 298
    assert results["default_rate"] == 0.298
    assert results["total_ead"] == Decimal("68121079.07")
    assert usd(results["total_ead"]) == "$68,121,079.07"


def test_pd_is_count_weighted_unless_labeled(portfolio):
    _, results = portfolio
    assert results["mean_pd"] == pytest_approx_rate(0.298, abs=1e-6)
    assert pct(results["mean_pd"]) == "29.80%"
    assert pct(results["ead_weighted_pd"]) == "17.23%"
    assert results["ead_weighted_pd"] < results["mean_pd"]


def test_high_pd_concentration_is_not_the_ead_share(portfolio):
    _, results = portfolio
    assert results["high_pd_count"] == 504
    assert results["high_pd_count_share"] == 0.504
    assert pct(results["high_pd_count_share"], 1) == "50.4%"
    assert pct(results["high_pd_ead_share"]) == "29.72%"
    assert results["high_pd_ead"] == pytest_approx_rate(20_248_304.07, abs=0.01)
    assert usd(results["high_pd_ead"]) == "$20,248,304.07"
    assert results["high_pd_ead_share"] < results["high_pd_count_share"]


def test_expected_loss_units(portfolio):
    _, results = portfolio
    assert results["assumed_lgd"] == 0.45
    # Sum of PD x EAD, before LGD. Not the draft's $12.6 million figure.
    assert usd(results["expected_defaulted_exposure"]) == "$11,734,442.82"
    assert usd(results["expected_loss"]) == "$5,280,499.32"
    assert results["expected_loss"] == pytest_approx_rate(
        results["assumed_lgd"] * results["expected_defaulted_exposure"],
        abs=2.0,
    )


def test_earnings_stress_is_borrower_net_income(portfolio):
    _, results = portfolio
    earnings = results["earnings"]
    assert earnings["base"] == Decimal("124207710.82")
    assert earnings["revenue_down_20"] == Decimal("24749229.5520")
    assert earnings["expenses_up_10"] == Decimal("86899241.2680")
    assert earnings["combined"] == Decimal("-12559240.0000")
    assert results["net_income_column"] == Decimal("124207710.74")
    assert earnings["base"] - results["net_income_column"] == Decimal("0.08")
    assert usd(earnings["combined"]) == "-$12,559,240.00"
    assert usd(earnings["revenue_down_20"]) == "$24,749,229.55"
    assert usd(earnings["expenses_up_10"]) == "$86,899,241.27"
    assert usd(results["expected_loss"]) != "$12,559,240.00"


def test_income_percentiles_are_not_loss_var(portfolio):
    _, results = portfolio
    assert results["income_percentile_05"] == 43146.55
    assert results["income_percentile_01"] == 25890.7
    assert results["income_percentile_01"] < results["income_percentile_05"]
    assert usd(results["income_percentile_05"]) == "$43,146.55"
    assert usd(results["income_percentile_01"]) == "$25,890.70"
    assert results["credit_loss_var_99"] > results["credit_loss_var_95"]
    assert results["credit_loss_var_95"] > results["expected_loss"]


def test_illustrative_credit_loss_var(portfolio):
    _, results = portfolio
    assert results["var_simulations"] == 20_000
    assert results["var_seed"] == 42
    assert results["asset_correlation"] == 0.15
    assert usd(results["credit_loss_var_95"]) == "$9,526,628.91"
    assert usd(results["credit_loss_var_99"]) == "$11,936,201.30"
    assert usd(results["credit_loss_es_95"]) == "$10,989,179.23"
    assert usd(results["credit_loss_es_99"]) == "$13,140,300.09"
    assert usd(results["conditional_expected_loss"]) == "$9,457,484.79"
    assert results["credit_loss_es_99"] > results["credit_loss_es_95"] > results["credit_loss_var_95"]


def test_operational_risk_default_rates(portfolio):
    _, results = portfolio
    assert results["op_risk_high_n"] == 91
    assert results["op_risk_high_defaults"] == 27
    assert results["op_risk_rest_n"] == 909
    assert results["op_risk_rest_defaults"] == 271
    assert pct(27 / 91) == "29.67%"
    assert pct(271 / 909) == "29.81%"
    assert abs(results["corr_op_risk_default"]) < 0.05


def pytest_approx_rate(value, abs=1e-12):
    import pytest

    return pytest.approx(value, abs=abs)
