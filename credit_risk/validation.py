"""Discrimination, calibration, and fold checks for the published scorecard.

The out-of-fold fit is an unpenalized logistic regression, the same
estimator as the in-sample scorecard. Intervals below are for the
scores that fit produced. They do not add a second layer of noise for
the refit itself.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from credit_risk.assumptions import KS_BOOTSTRAP, MODEL_SEED, N_FOLDS

_Z_95 = float(norm.ppf(0.975))


def logistic_oof(
    design: np.ndarray,
    labels: np.ndarray,
    *,
    scale: bool = False,
    max_iter: int = 2000,
    n_folds: int = N_FOLDS,
    seed: int = MODEL_SEED,
) -> tuple[np.ndarray, int]:
    """Out-of-fold probabilities and the worst fold iteration count.

    When ``scale`` is true, ``StandardScaler`` is fit on the training
    rows of each fold and applied to that fold's test rows. It is not
    fit on the full sample.
    """
    design = np.asarray(design, dtype=float)
    labels = np.asarray(labels, dtype=int)
    holdout = np.zeros(len(labels), dtype=float)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    worst_iter = 0
    for train_index, test_index in splitter.split(design, labels):
        train_x, test_x = _fold_matrices(
            design[train_index], design[test_index], scale=scale
        )
        model = LogisticRegression(solver="lbfgs", max_iter=max_iter, C=np.inf)
        model.fit(train_x, labels[train_index])
        holdout[test_index] = model.predict_proba(test_x)[:, 1]
        worst_iter = max(worst_iter, int(np.max(model.n_iter_)))
    return holdout, worst_iter


def _fold_matrices(
    train_x: np.ndarray, test_x: np.ndarray, *, scale: bool
) -> tuple[np.ndarray, np.ndarray]:
    if not scale:
        return train_x, test_x
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    return scaler.fit_transform(train_x), scaler.transform(test_x)


def delong_roc(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """AUC and the DeLong (1988) variance of that AUC.

    Ties in ``scores`` count as one half, matching the Mann-Whitney
    form of the area under the ROC curve.
    """
    auc, v_pos, v_neg = _structural_components(labels, scores)
    n1 = v_pos.size
    n0 = v_neg.size
    if n1 < 2 or n0 < 2:
        raise ValueError("DeLong variance needs at least two loans in each class")
    s10 = float(np.sum((v_pos - auc) ** 2) / (n1 - 1))
    s01 = float(np.sum((v_neg - auc) ** 2) / (n0 - 1))
    variance = max(0.0, s10 / n1 + s01 / n0)
    se = math.sqrt(variance)
    return {
        "auc": auc,
        "variance": variance,
        "se": se,
        "ci_low": auc - _Z_95 * se,
        "ci_high": auc + _Z_95 * se,
    }


def delong_difference(
    labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray
) -> dict[str, float]:
    """DeLong test of AUC(scores_a) minus AUC(scores_b) on the same loans."""
    auc_a, pos_a, neg_a = _structural_components(labels, scores_a)
    auc_b, pos_b, neg_b = _structural_components(labels, scores_b)
    if pos_a.shape != pos_b.shape or neg_a.shape != neg_b.shape:
        raise ValueError("score vectors imply different class counts")
    n1 = pos_a.size
    n0 = neg_a.size
    s10_a = float(np.sum((pos_a - auc_a) ** 2) / (n1 - 1))
    s01_a = float(np.sum((neg_a - auc_a) ** 2) / (n0 - 1))
    s10_b = float(np.sum((pos_b - auc_b) ** 2) / (n1 - 1))
    s01_b = float(np.sum((neg_b - auc_b) ** 2) / (n0 - 1))
    s10_ab = float(np.sum((pos_a - auc_a) * (pos_b - auc_b)) / (n1 - 1))
    s01_ab = float(np.sum((neg_a - auc_a) * (neg_b - auc_b)) / (n0 - 1))
    var_a = s10_a / n1 + s01_a / n0
    var_b = s10_b / n1 + s01_b / n0
    cov = s10_ab / n1 + s01_ab / n0
    variance = max(0.0, var_a + var_b - 2.0 * cov)
    se = math.sqrt(variance)
    diff = auc_a - auc_b
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "diff": diff,
        "se": se,
        "ci_low": diff - _Z_95 * se,
        "ci_high": diff + _Z_95 * se,
    }


def _structural_components(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both classes are required")
    diff = positive[:, None] - negative[None, :]
    psi = np.ones(diff.shape, dtype=float)
    psi[diff < 0.0] = 0.0
    psi[diff == 0.0] = 0.5
    v_pos = psi.mean(axis=1)
    v_neg = psi.mean(axis=0)
    return float(psi.mean()), v_pos, v_neg


def ks_statistic(labels: np.ndarray, scores: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between default and non-default scores."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    ordered = labels[order]
    n1 = int(ordered.sum())
    n0 = int(ordered.size - n1)
    if n1 == 0 or n0 == 0:
        raise ValueError("both classes are required")
    cdf_default = np.cumsum(ordered) / n1
    cdf_other = np.cumsum(1 - ordered) / n0
    return float(np.max(np.abs(cdf_default - cdf_other)))


def ks_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    n_boot: int = KS_BOOTSTRAP,
    seed: int = MODEL_SEED,
) -> dict[str, float]:
    """Percentile interval for KS, resampling the fixed scores."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    point = ks_statistic(labels, scores)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    n = labels.size
    filled = 0
    # A single-class resample is discarded. At this default rate that
    # draw is vanishingly rare; the loop bound keeps the seed finite.
    attempts = 0
    while filled < n_boot:
        attempts += 1
        if attempts > n_boot * 5:
            raise RuntimeError("bootstrap could not draw both classes")
        index = rng.integers(0, n, n)
        sample = labels[index]
        if sample.min() == sample.max():
            continue
        draws[filled] = ks_statistic(sample, scores[index])
        filled += 1
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "ks": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_boot": float(n_boot),
    }


def calibration_intercept_slope(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Intercept and slope of default on the logit of the score.

    Slope 1 and intercept 0 is the recalibration target. This is a
    logistic fit, not a bin chart.
    """
    labels = np.asarray(labels).astype(int)
    probability = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", max_iter=2000, C=np.inf)
    model.fit(logit, labels)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def brier_skill(labels: np.ndarray, scores: np.ndarray) -> float:
    """Skill versus a constant forecast at the sample default rate.

    Skill is 1 minus model Brier over the base-rate Brier. Zero means
    the score is no better than quoting the default rate. Class weights
    are not used: they would move average PD off that rate.
    """
    labels = np.asarray(labels).astype(int)
    brier = float(brier_score_loss(labels, scores))
    base_rate = float(labels.mean())
    reference = float(np.mean((labels - base_rate) ** 2))
    if reference <= 0.0:
        raise ValueError("Brier skill is undefined when every label is the same")
    return 1.0 - brier / reference


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def logistic_insample(
    design: np.ndarray, labels: np.ndarray, *, max_iter: int = 2000
) -> np.ndarray:
    """Full-sample unpenalized probabilities. Used to spot a failed fit."""
    design = np.asarray(design, dtype=float)
    labels = np.asarray(labels, dtype=int)
    model = LogisticRegression(solver="lbfgs", max_iter=max_iter, C=np.inf)
    model.fit(design, labels)
    return model.predict_proba(design)[:, 1]
