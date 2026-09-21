# Portfolio Risk Analysis & Credit-Risk Modeling

This repository is the source loan file for the credit case. The expected-loss and concentration analysis lives in [ParBproject/Advanced-Financial-Models](https://github.com/ParBproject/Advanced-Financial-Models). That dashboard defaults to this book.

## Source data

[portfolio_data.csv](portfolio_data.csv) is the data. Recomputed from the file:

| Check | Value |
|---|---:|
| Loans | 1,000 |
| Total exposure (`Loan_Amount`) | $68,121,079.07 |
| Average reported `PD_Score` | 29.80% |
| Loans with `PD_Score` above 20% | 504 |
| Combined stress net income (revenue × 0.8 − expenses × 1.1) | −$12,559,240.00 |

There is no model script in this repository. Do not treat the charts below as the calculation.

## Retired memo

[archive/Risk_Assessment_Report.docx](archive/Risk_Assessment_Report.docx) is retired. [DATA_NOTE.md](DATA_NOTE.md) supersedes it. Do not use the Word file as the analysis. Three claims in it are wrong:

1. It describes a scikit-learn logistic regression. This repository has no Python and no logistic-regression code.
2. It calls $43,146.55 a 95% VaR and $25,890.70 a 99% VaR. Those figures are the higher 5th and 1st percentiles of per-customer net income. They are positive income levels, not a portfolio loss VaR.
3. It says operational-risk scores above 60 have a 2.5× default rate. On this file the default rate is 29.67% (27 of 91) above 60 and 29.81% (271 of 909) at or below 60.

## Charts

The images in [Screenshot/](Screenshot/) are historical charts from the retired write-up. They are not a substitute for the tested expected-loss and concentration functions in Advanced Financial Models.

## Data note

The portfolio is synthetic and for demonstration. Results are not lending advice. A production model would need validation, monitoring, and governance that this file does not provide.
