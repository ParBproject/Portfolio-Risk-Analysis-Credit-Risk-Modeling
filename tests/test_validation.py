"""Pins for discrimination, calibration, and the fold-wise scaler."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from credit_risk.data import PD_FEATURES, has_time_column
from credit_risk.validation import (
    delong_difference,
    delong_roc,
    ks_statistic,
    logistic_insample,
    logistic_oof,
)


def test_delong_auc_matches_the_mann_whitney_form():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.7, 0.4, 0.4, 0.2])
    result = delong_roc(labels, scores)
    assert result["auc"] == pytest.approx(roc_auc_score(labels, scores))
    assert result["auc"] == pytest.approx(0.875)


def test_delong_difference_is_zero_for_identical_scores():
    labels = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.6, 0.4, 0.2, 0.8, 0.1])
    result = delong_difference(labels, scores, scores)
    assert result["diff"] == 0.0
    assert result["se"] == 0.0
    assert result["ci_low"] == result["ci_high"] == 0.0


def test_holdout_intervals_and_the_credit_score_only_model(portfolio):
    _, results = portfolio
    validation = results["validation"]
    assert validation["oof_auc"] == pytest.approx(0.852430, abs=1e-4)
    assert validation["oof_auc_ci_low"] == pytest.approx(0.827866, abs=1e-4)
    assert validation["oof_auc_ci_high"] == pytest.approx(0.876995, abs=1e-4)
    assert validation["oof_gini"] == pytest.approx(2 * validation["oof_auc"] - 1)
    assert validation["oof_ks"] == pytest.approx(0.545756, abs=1e-4)
    assert validation["oof_ks_ci_low"] == pytest.approx(0.504877, abs=1e-4)
    assert validation["oof_ks_ci_high"] == pytest.approx(0.612186, abs=1e-4)
    assert validation["oof_ks_ci_low"] < validation["oof_ks"] < validation["oof_ks_ci_high"]
    assert validation["oof_calibration_intercept"] == pytest.approx(-0.02163, abs=1e-3)
    assert validation["oof_calibration_slope"] == pytest.approx(0.9735, abs=1e-3)
    assert validation["oof_brier_skill"] == pytest.approx(0.3387, abs=1e-3)
    assert validation["oof_mean_pd"] == pytest.approx(results["default_rate"], abs=0.01)
    assert validation["credit_score_oof_auc"] == pytest.approx(0.852951, abs=1e-4)
    assert validation["auc_minus_credit_score_ci_low"] < 0 < validation["auc_minus_credit_score_ci_high"]
    assert validation["has_time_column"] == 0
    assert has_time_column(portfolio[0]) is False


def test_published_scorecard_family_reproduces_pd(portfolio):
    """The fold model is the unpenalized logit, not an L2 fit on another scale."""
    frame, _ = portfolio
    fitted = logistic_insample(frame.loc[:, PD_FEATURES].to_numpy(float), frame["Default"].to_numpy())
    assert np.max(np.abs(fitted - frame["PD_Score"].to_numpy(float))) < 1e-4


def test_earnings_are_not_a_measured_leak_when_the_scaler_is_honest(portfolio):
    _, results = portfolio
    validation = results["validation"]
    assert validation["earnings_scaled_oof_auc"] == pytest.approx(0.852167, abs=1e-4)
    assert validation["earnings_scaled_oof_auc"] <= validation["oof_auc"] + 1e-4
    assert validation["earnings_unscaled_oof_auc"] == pytest.approx(0.838831, abs=1e-4)
    assert validation["earnings_unscaled_insample_auc"] < validation["insample_auc"] - 0.02


def test_standardizer_is_fit_on_the_training_fold_only(portfolio, monkeypatch):
    frame, _ = portfolio
    seen = []
    real_fit = StandardScaler.fit

    def wrapped(self, X, y=None, **kwargs):
        seen.append(len(X))
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", wrapped)
    columns = list(PD_FEATURES) + ["Revenue", "Expenses", "Net_Income"]
    logistic_oof(
        frame.loc[:, columns].to_numpy(float),
        frame["Default"].to_numpy(int),
        scale=True,
    )
    assert seen == [800, 800, 800, 800, 800]


def test_ks_is_the_distance_between_class_score_cdfs():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert ks_statistic(labels, scores) == pytest.approx(1.0)
