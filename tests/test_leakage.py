from credit_risk.data import LEAKAGE_COLUMNS, PD_FEATURES


def test_published_pd_matches_full_sample_logistic(portfolio):
    _, results = portfolio
    assert results["scorecard"]["max_abs_pd_gap"] < 1e-8
    assert abs(results["mean_pd"] - results["default_rate"]) < 1e-9


def test_pd_features_exclude_label_score_and_earnings():
    assert set(PD_FEATURES).isdisjoint(LEAKAGE_COLUMNS)
    assert PD_FEATURES == (
        "Credit_Score",
        "Loan_Amount",
        "Operational_Risk_Score",
    )


def test_holdout_discrimination_is_pinned(portfolio):
    _, results = portfolio
    validation = results["validation"]
    assert validation["oof_auc"] == pytest_approx(0.852430, abs=1e-4)
    assert validation["oof_brier"] == pytest_approx(0.138343, abs=1e-4)
    assert validation["insample_auc"] == pytest_approx(0.855035, abs=1e-4)
    assert validation["insample_brier"] == pytest_approx(0.136963, abs=1e-4)
    assert validation["oof_auc"] < validation["insample_auc"]


def test_loan_amount_partial_effect_is_documented(portfolio):
    """Positive loan coefficient beside a negative raw association. Spec unchanged."""
    frame, results = portfolio
    assert results["corr_loan_credit_score"] > 0.9
    assert frame["Loan_Amount"].corr(frame["Default"]) < 0
    assert results["scorecard"]["Loan_Amount"] > 0


def pytest_approx(value, abs):
    import pytest

    return pytest.approx(value, abs=abs)
