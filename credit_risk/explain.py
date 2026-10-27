"""Exact Shapley values for the published three-feature logit.

On the logit scale, feature ``j`` contributes ``beta_j`` times its gap
from the book-average feature vector. On the probability scale the
sigmoid is not linear, so each value is the average marginal
contribution over the coalitions of that sigmoid. A feature left out
of a coalition is set to the average, which ignores the correlation
between credit score and loan amount. The loan-amount value follows
the partial coefficient, including its positive sign.
"""

from __future__ import annotations

import math

import numpy as np

from credit_risk.data import PD_FEATURES


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    logit = np.asarray(logit, dtype=float)
    out = np.empty(logit.shape, dtype=float)
    positive = logit >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exp_z = np.exp(logit[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def logit_contributions(
    intercept: float,
    coefficients: np.ndarray,
    features: np.ndarray,
    baseline: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Shapley values of the linear log-odds, and the reference log-odds."""
    coefficients = np.asarray(coefficients, dtype=float)
    features = np.asarray(features, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    phi = (features - baseline) * coefficients
    reference_logit = float(intercept + baseline @ coefficients)
    return phi, reference_logit


def probability_shapley(
    intercept: float,
    coefficients: np.ndarray,
    features: np.ndarray,
    baseline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Shapley values of ``sigmoid(intercept + x @ beta)``.

    Returns ``(phi, predicted_pd, reference_pd)``. More than 12 features
    is refused, because the coalitions are enumerated.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    features = np.asarray(features, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    n_rows, n_features = features.shape
    if coefficients.shape != (n_features,):
        raise ValueError("coefficients do not match the feature columns")
    if not 1 <= n_features <= 12:
        raise ValueError("exact Shapley enumeration supports 1 to 12 features")
    contrib = (features - baseline) * coefficients
    reference_logit = float(intercept + baseline @ coefficients)
    n_masks = 1 << n_features
    masks = np.arange(n_masks)
    bits = ((masks[:, None] >> np.arange(n_features)[None, :]) & 1).astype(float)
    subset_pd = _sigmoid(reference_logit + bits @ contrib.T)
    factorial = [math.factorial(i) for i in range(n_features + 1)]
    denom = float(factorial[n_features])
    phi = np.zeros((n_rows, n_features), dtype=float)
    for feature in range(n_features):
        others = [index for index in range(n_features) if index != feature]
        total = np.zeros(n_rows, dtype=float)
        n_others = len(others)
        for mask in range(1 << n_others):
            coalition = 0
            size = 0
            for bit, index in enumerate(others):
                if mask & (1 << bit):
                    coalition |= 1 << index
                    size += 1
            weight = factorial[size] * factorial[n_features - 1 - size] / denom
            with_feature = subset_pd[coalition | (1 << feature)]
            without = subset_pd[coalition]
            total += weight * (with_feature - without)
        phi[:, feature] = total
    full = subset_pd[(1 << n_features) - 1]
    reference = subset_pd[0]
    return phi, full, reference


def attribute_scorecard(frame, fit: dict[str, float]) -> dict[str, float | dict[str, float]]:
    """Summaries of an in-sample scorecard. Row-level values are not stored.

    ``fit`` is the coefficient map from :func:`credit_risk.metrics.scorecard_fit`.
    """
    coefficients = np.array([fit[name] for name in PD_FEATURES], dtype=float)
    features = frame.loc[:, PD_FEATURES].to_numpy(dtype=float)
    baseline = features.mean(axis=0)
    logit_phi, reference_logit = logit_contributions(
        fit["intercept"], coefficients, features, baseline
    )
    probability_phi, predicted, reference = probability_shapley(
        fit["intercept"], coefficients, features, baseline
    )
    published = frame["PD_Score"].to_numpy(dtype=float)
    logit = np.log(published / (1.0 - published))
    mean_abs_probability = np.abs(probability_phi).mean(axis=0)
    share = mean_abs_probability / mean_abs_probability.sum()
    largest = np.argmax(np.abs(probability_phi), axis=1)
    credit_index = PD_FEATURES.index("Credit_Score")
    loan_index = PD_FEATURES.index("Loan_Amount")
    return {
        "reference_pd": float(reference[0]),
        "reference_logit": reference_logit,
        "probability_efficiency_gap": float(
            np.max(np.abs(probability_phi.sum(axis=1) - (predicted - reference)))
        ),
        "logit_efficiency_gap": float(
            np.max(np.abs(reference_logit + logit_phi.sum(axis=1) - logit))
        ),
        "published_pd_gap": float(np.max(np.abs(predicted - published))),
        "mean_abs_probability": {
            name: float(value) for name, value in zip(PD_FEATURES, mean_abs_probability)
        },
        "probability_share": {
            name: float(value) for name, value in zip(PD_FEATURES, share)
        },
        "mean_abs_logit": {
            name: float(value)
            for name, value in zip(PD_FEATURES, np.abs(logit_phi).mean(axis=0))
        },
        "credit_score_largest_share": float(np.mean(largest == credit_index)),
        "loan_amount_positive_share": float(np.mean(probability_phi[:, loan_index] > 0.0)),
    }
