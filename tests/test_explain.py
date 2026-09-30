"""Exact Shapley values of the linear logit, on both scales."""

import numpy as np
import pytest

from credit_risk.explain import (
    attribute_scorecard,
    logit_contributions,
    probability_shapley,
)


def test_probability_shapley_matches_a_two_feature_enumeration():
    phi, predicted, reference = probability_shapley(
        0.0,
        np.array([1.0, 0.0]),
        np.array([[1.0, 5.0]]),
        np.array([0.0, 0.0]),
    )
    gap = 1.0 / (1.0 + np.exp(-1.0)) - 0.5
    assert phi[0, 0] == pytest.approx(gap)
    assert phi[0, 1] == pytest.approx(0.0)
    assert predicted[0] - reference[0] == pytest.approx(phi[0].sum())


def test_a_zero_coefficient_has_no_probability_attribution():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, 3))
    baseline = np.zeros(3)
    phi, predicted, reference = probability_shapley(
        -0.2,
        np.array([0.0, 0.4, -0.7]),
        features,
        baseline,
    )
    assert np.max(np.abs(phi[:, 0])) == 0.0
    assert np.max(np.abs(phi.sum(axis=1) - (predicted - reference))) < 1e-12


def test_logit_contributions_sum_to_the_log_odds_gap():
    phi, reference_logit = logit_contributions(
        0.5,
        np.array([2.0, -1.0]),
        np.array([[1.0, 3.0]]),
        np.array([0.0, 1.0]),
    )
    assert phi[0, 0] == pytest.approx(2.0)
    assert phi[0, 1] == pytest.approx(-2.0)
    assert reference_logit + phi[0].sum() == pytest.approx(-0.5)


def test_book_attributions_separate_the_partial_effect_from_the_average(portfolio):
    frame, results = portfolio
    explanation = results["explanation"]
    assert explanation["reference_pd"] == pytest.approx(0.20607083, rel=0, abs=1e-6)
    assert explanation["probability_efficiency_gap"] < 1e-9
    assert explanation["logit_efficiency_gap"] < 1e-9
    assert explanation["published_pd_gap"] < 1e-8
    assert explanation["mean_abs_probability"]["Credit_Score"] == pytest.approx(0.239523, abs=1e-4)
    assert explanation["mean_abs_probability"]["Loan_Amount"] == pytest.approx(0.023154, abs=1e-4)
    assert explanation["mean_abs_probability"]["Operational_Risk_Score"] == pytest.approx(0.015327, abs=1e-4)
    assert explanation["credit_score_largest_share"] == pytest.approx(0.95)
    assert explanation["reference_pd"] < results["mean_pd"] - 0.05
    direct = attribute_scorecard(frame, results["scorecard"])
    assert direct["loan_amount_positive_share"] == pytest.approx(0.478)
    # Probability-scale values are not the logit contributions pushed through a sigmoid.
    assert explanation["mean_abs_logit"]["Credit_Score"] > 1.0
    assert explanation["mean_abs_probability"]["Credit_Score"] < 0.5
