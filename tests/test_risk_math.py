"""Closed-form conditional PD, tail-mean ES, and Monte Carlo error."""

import numpy as np
import pytest
from scipy.stats import norm

from credit_risk.metrics import (
    _tail_mean,
    conditional_pd,
    loss_quantiles,
    monte_carlo_intervals,
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


def test_a_constant_loss_sample_has_no_simulation_width():
    losses = np.full(2_000, 10.0)
    band = monte_carlo_intervals(losses, n_boot=40, seed=1)
    assert band["var_95_low"] == band["var_95_high"] == 10.0
    assert band["es_99_low"] == band["es_99_high"] == 10.0
