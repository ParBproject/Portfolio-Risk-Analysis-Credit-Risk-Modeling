import numpy as np
import pandas as pd

from credit_risk.generate import generate_portfolio, scorecard_pd


def test_generator_is_repeatable():
    first = generate_portfolio(n=1000, seed=42)
    second = generate_portfolio(n=1000, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_generator_seed_changes_the_book():
    first = generate_portfolio(n=200, seed=1)
    second = generate_portfolio(n=200, seed=2)
    assert not first["Credit_Score"].equals(second["Credit_Score"])


def test_generated_pd_is_the_scorecard_not_a_fit_to_default():
    frame = generate_portfolio(n=1000, seed=42)
    assert np.allclose(scorecard_pd(frame), frame["PD_Score"])
    flipped = frame.copy()
    flipped["Default"] = 1 - flipped["Default"]
    assert np.allclose(scorecard_pd(flipped), frame["PD_Score"])
    # An intercept logistic fit would force these together. This draw does not.
    assert abs(frame["PD_Score"].mean() - frame["Default"].mean()) > 0.005


def test_seed_42_book_is_pinned():
    frame = generate_portfolio(n=1000, seed=42)
    row = frame.iloc[0]
    assert row["Customer_ID"] == "CUST_0000"
    assert int(row["Credit_Score"]) == 701
    assert float(row["Loan_Amount"]) == np.float64(105668.61) or abs(float(row["Loan_Amount"]) - 105668.61) < 1e-9
    assert float(row["Operational_Risk_Score"]) == 33.2
    assert int(row["Default"]) == 0
    assert abs(float(row["PD_Score"]) - 0.14440194599458206) < 1e-12
    assert int(frame["Default"].sum()) == 224
    assert abs(float(frame["Loan_Amount"].sum()) - 96_602_539.32) < 0.01
