# Data note

This note supersedes `archive/Risk_Assessment_Report.docx`. The loan file in this repository is the source data. Expected loss and borrower concentration are calculated in [ParBproject/Advanced-Financial-Models](https://github.com/ParBproject/Advanced-Financial-Models) from a committed copy of `portfolio_data.csv`.

## What the file supports

Checked directly from `portfolio_data.csv`:

- 1,000 loans and $68,121,079.07 of `Loan_Amount`.
- Average `PD_Score` 29.80%. 504 loans have `PD_Score` above 20%.
- `Default` is 1 on 298 loans (29.8% overall).
- Combined stress, defined as `0.8 × Revenue − 1.1 × Expenses`, sums to −$12,559,240.00.

`PD_Score` is a column on the file. This repository does not contain the code that produced it.

## Three retired claims

The Word memo is not current. These three statements are incorrect:

1. **No logistic regression here.** The memo says probability of default was fit with scikit-learn. This repository has no Python modules and no model artifact.
2. **The dollar "VaR" figures are net-income percentiles.** $43,146.55 and $25,890.70 match NumPy's higher 5th and 1st percentiles of per-customer `Net_Income`. They are positive income levels for one customer. They are not a 95% or 99% portfolio loss VaR.
3. **Operational-risk score above 60 does not show a 2.5× default rate.** Loans with `Operational_Risk_Score` above 60 default at 29.67% (27 of 91). Loans at or below 60 default at 29.81% (271 of 909). The rates sit on top of each other.

## Where the analysis runs

Advanced Financial Models maps `Customer_ID` to the loan id and the borrower, `Loan_Amount` to exposure, and `Credit_Score` to the score used by the tested probability-of-default bands. It does not treat `PD_Score` as the probability in expected loss.
