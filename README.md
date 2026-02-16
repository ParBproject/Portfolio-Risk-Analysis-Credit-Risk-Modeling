# Portfolio Risk Analysis & Credit Risk Modeling

**Advanced credit risk assessment and portfolio stress testing using logistic regression, VaR, and scenario analysis**

![Project Banner / Dashboard Screenshot](https://via.placeholder.com/1200x400/1e3a8a/ffffff?text=Portfolio+Risk+Analysis+Dashboard)
<!-- Replace the line above with your actual screenshot: e.g. ![Dashboard](screenshots/dashboard-overview.png) -->

## 📖 Project Overview

This repository contains a complete **credit risk and portfolio risk analysis** project for a simulated loan portfolio of **1,000 customers**.

The analysis includes:

- Probability of Default (**PD**) modeling using logistic regression
- Value at Risk (**VaR**) calculation
- Stress testing under adverse economic scenarios
- Operational risk scoring integration
- Expected Loss (**EL**) estimation
- Actionable business recommendations

The project demonstrates end-to-end risk analytics workflow: data understanding → modeling → stress testing → visualization → reporting.

**Key business question answered**:  
How resilient is the $68M loan portfolio to defaults and adverse scenarios — and what mitigation strategies should be implemented?

## 🎯 Key Results

- **Portfolio size**: $68,121,079.07 (1,000 loans)
- **Average PD**: **29.8%**
- **Actual default rate**: **29.8%** (very well calibrated model)
- **High-risk loans** (PD > 20%): **504** (50.4% of portfolio)
- **VaR (95%)**: ≈ **$43,147** per customer tail risk
- **Combined stress scenario** (–20% revenue +10% expenses): **–$12.56M** net income
- **Strongest PD driver**: Credit Score (correlation –0.93)

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **Core libraries**:
  - pandas · numpy
  - scikit-learn (logistic regression)
  - matplotlib · seaborn (visualizations)
- **Other**: OpenPyXL (report handling), statsmodels

## 📊 Data

- **File**: [`portfolio_data.csv`](portfolio_data.csv) (1,000 rows × 9 columns)
- **Columns**:

| Column                  | Description                              | Type    |
|-------------------------|------------------------------------------|---------|
| Customer_ID             | Unique identifier                        | string  |
| Credit_Score            | FICO-like score (300–850)                | int     |
| Loan_Amount             | Approved loan amount ($)                 | float   |
| Operational_Risk_Score  | Operational risk (0–100)                 | float   |
| Revenue                 | Customer annual revenue ($)              | float   |
| Expenses                | Customer annual expenses ($)             | float   |
| Net_Income              | Revenue – Expenses ($)                   | float   |
| Default                 | Actual default (0 = paid, 1 = default)   | int     |
| PD_Score                | Predicted probability of default (0–1)   | float   |

Sample rows:
## 📈 Visualizations


<!-- Screenshot area 1 -->
![PD Distribution](screenshots/01_pd_distribution.png)  
*Histogram of predicted PD scores with actual defaults overlay*

<!-- or even better – descriptive name -->
![PD Score Distribution](screenshots/pd-distribution.png)

![Pasted image 2](screenshots/Pasted%20image%20(2).png)

<!-- or with alt text -->
![Screenshot showing PD distribution](screenshots/Pasted%20image%20(2).png)

<!-- Screenshot area 2 -->
![Credit Score vs PD](screenshots/02_credit_vs_pd_scatter.png)  
*Very strong negative relationship (correlation ≈ -0.93)*

<!-- Screenshot area 3 -->
![Stress Test Comparison](screenshots/03_stress_test_results.png)  
*Base case vs adverse scenarios*

<!-- Screenshot area 4 -->
![High-Risk Segment](screenshots/04_high_risk_loans.png)  
*Concentration of loans with PD > 20%*

```text
Customer_ID  Credit_Score  Loan_Amount  Operational_Risk_Score  Revenue   Expenses  Net_Income  Default  PD_Score
CUST_0000           720     105858.9                    29.9   213829   159411     54418.3        0   0.0932
CUST_0001           669      73338.9                    37.8   370942   232069    138873.0        0   0.2651
...


