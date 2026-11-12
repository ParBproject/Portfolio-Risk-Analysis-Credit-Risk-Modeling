"""Pins for discrimination, calibration, and the fold-wise scaler."""

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from credit_risk.data import PD_FEATURES, has_time_column
from credit_risk.validation import (
    auc,
    brier_skill,
    calibration_intercept_slope,
    delong_difference,
    delong_roc,
    ks_bootstrap,
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
    # Placements are (1, 0.75) and (0.75, 1). Each sample variance is
    # 0.03125, so the DeLong variance is 0.03125/2 + 0.03125/2.
    assert result["variance"] == pytest.approx(0.03125)
    assert result["se"] == pytest.approx(0.03125 ** 0.5)
    z = float(norm.ppf(0.975))
    assert result["ci_low"] == pytest.approx(0.875 - z * result["se"])
    assert result["ci_high"] == pytest.approx(0.875 + z * result["se"])
    with pytest.raises(ValueError):
        delong_roc(np.array([1, 0]), np.array([0.9, 0.1]))


def test_delong_difference_is_zero_for_identical_scores():
    labels = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.6, 0.4, 0.2, 0.8, 0.1])
    result = delong_difference(labels, scores, scores)
    assert result["diff"] == 0.0
    assert result["se"] == 0.0
    assert result["ci_low"] == result["ci_high"] == 0.0


def test_delong_difference_variance_includes_the_covariance():
    labels = np.array([1, 1, 0, 0])
    # AUC 0.75. Positive placements (1, 0.5), negative placements (0.5, 1).
    scores_a = np.array([0.8, 0.3, 0.5, 0.1])
    # AUC 0.875, the tied case above. Covariance of the placements is 0.0625.
    scores_b = np.array([0.7, 0.4, 0.4, 0.2])
    result = delong_difference(labels, scores_a, scores_b)
    assert result["auc_a"] == pytest.approx(0.75)
    assert result["auc_b"] == pytest.approx(0.875)
    assert result["diff"] == pytest.approx(-0.125)
    # 0.125 + 0.03125 - 2 * 0.0625 = 0.03125.
    assert result["se"] == pytest.approx(0.03125 ** 0.5)
    z = float(norm.ppf(0.975))
    assert result["ci_low"] == pytest.approx(result["diff"] - z * result["se"])
    assert result["ci_high"] == pytest.approx(result["diff"] + z * result["se"])
    # One default. The variance divisor is zero, so this used to report a
    # standard error of 0 around an AUC difference of 2/3.
    with pytest.raises(ValueError):
        delong_difference(
            np.array([1, 0, 0, 0]),
            np.array([0.9, 0.1, 0.2, 0.3]),
            np.array([0.2, 0.8, 0.4, 0.1]),
        )


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
    with pytest.raises(ValueError):
        ks_statistic(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.9]))


def test_ks_ties_count_as_one_cdf_step():
    # One default and one non-default at the same score. Both CDFs jump
    # together, so the distance is 0 either way the rows are ordered.
    assert ks_statistic(np.array([0, 1]), np.array([0.5, 0.5])) == 0.0
    assert ks_statistic(np.array([1, 0]), np.array([0.5, 0.5])) == 0.0
    assert ks_statistic(np.array([0, 1, 0, 1]), np.full(4, 0.2)) == 0.0
    # After 0.1: gap 1/2. After both loans at 0.5: gap 1/2. After 0.9: gap 0.
    forward = ks_statistic(np.array([0, 0, 1, 1]), np.array([0.1, 0.5, 0.5, 0.9]))
    swapped = ks_statistic(np.array([0, 1, 0, 1]), np.array([0.1, 0.5, 0.5, 0.9]))
    assert forward == pytest.approx(0.5)
    assert swapped == pytest.approx(0.5)
    # Score 2 holds two non-defaults and one default. n0 = n1 = 3,
    # so the gap there is |1/3 - 1| = 2/3.
    grouped = ks_statistic(
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([1, 2, 2, 2, 3, 4], dtype=float),
    )
    assert grouped == pytest.approx(2 / 3)


def test_ks_bootstrap_collapses_when_every_draw_has_the_same_distance():
    labels = np.array([0, 0, 1, 1])
    tied = ks_bootstrap(labels, np.full(4, 0.4), n_boot=25, seed=0)
    assert tied["ks"] == 0.0
    assert tied["ci_low"] == tied["ci_high"] == 0.0
    separated = ks_bootstrap(labels, np.array([0.1, 0.2, 0.8, 0.9]), n_boot=25, seed=0)
    assert separated["ks"] == 1.0
    assert separated["ci_low"] == separated["ci_high"] == 1.0


def test_brier_skill_is_one_minus_brier_over_the_base_rate():
    labels = np.array([0, 0, 1, 1])
    assert brier_skill(labels, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(1.0)
    assert brier_skill(labels, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)
    # Squared error averages 0.3125. The base-rate Brier is 0.25, so skill is -0.25.
    assert brier_skill(labels, np.full(4, 0.25)) == pytest.approx(-0.25)
    with pytest.raises(ValueError):
        brier_skill(np.ones(4, dtype=int), np.full(4, 0.2))


def test_calibration_slope_matches_the_saturated_two_point_fit():
    # Four loans at logit 0, one default (rate 1/4). Four at logit 1, three
    # defaults (rate 3/4). The fit is logit(1/4) and logit(3/4) - logit(1/4).
    high = 1.0 / (1.0 + np.exp(-1.0))
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.5, high, high, high, high])
    intercept, slope = calibration_intercept_slope(labels, scores)
    logit_low = np.log(0.25 / 0.75)
    logit_high = np.log(0.75 / 0.25)
    # Default L2 (C=1) lands near slope 0.67. This tolerance only passes
    # the unpenalized MLE.
    assert intercept == pytest.approx(logit_low, abs=1e-8)
    assert slope == pytest.approx(logit_high - logit_low, abs=1e-8)
    # A constant score leaves the slope unidentified, but the probability
    # at that score is still the default rate.
    intercept, slope = calibration_intercept_slope(
        np.array([0, 0, 0, 1]), np.full(4, 0.4)
    )
    eta = intercept + slope * np.log(0.4 / 0.6)
    assert 1.0 / (1.0 + np.exp(-eta)) == pytest.approx(0.25, abs=1e-4)


def test_auc_counts_a_tie_as_one_half():
    assert auc(np.array([1, 0]), np.array([0.9, 0.1])) == 1.0
    assert auc(np.array([1, 0]), np.array([0.1, 0.9])) == 0.0
    assert auc(np.array([1, 0]), np.array([0.4, 0.4])) == 0.5


def test_constant_design_predicts_the_training_default_rate():
    labels = np.array([0, 0, 0, 1])
    fitted = logistic_insample(np.zeros((4, 1)), labels)
    # Unpenalized logistic stops at the solver tolerance, short of the exact 1/4.
    assert fitted == pytest.approx([0.25, 0.25, 0.25, 0.25], abs=1e-3)
    # Each training fold of a 2-fold stratified split still has rate 1/4.
    holdout, _iterations = logistic_oof(
        np.zeros((8, 1)),
        np.array([0, 0, 0, 1, 0, 0, 0, 1]),
        n_folds=2,
        seed=0,
    )
    assert holdout == pytest.approx(np.full(8, 0.25), abs=1e-3)
