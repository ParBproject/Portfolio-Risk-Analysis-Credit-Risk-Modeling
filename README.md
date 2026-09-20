# Credit Risk Analytics & Model Monitoring Lab

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](requirements.txt)
[![Risk](https://img.shields.io/badge/Risk-PD%20%7C%20LGD%20%7C%20EAD-0F766E)](src/risk_analytics.py)
[![Validation](https://img.shields.io/badge/Validation-AUC%20%7C%20KS%20%7C%20Brier-2563EB)](MODEL_VALIDATION.md)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](app.py)
[![CI](https://github.com/ParBproject/Portfolio-Risk-Analysis-Credit-Risk-Modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/ParBproject/Portfolio-Risk-Analysis-Credit-Risk-Modeling/actions/workflows/ci.yml)

A reproducible credit-risk portfolio project focused on **PD model monitoring, calibration, expected loss, exposure concentration, deterministic stress testing, profitability stress, and account prioritization**.

The bundled portfolio is **synthetic** and designed for methodology demonstration. The project does not present synthetic data as a live lending portfolio.

## Employer snapshot

| Capability | Evidence |
|---|---|
| Credit-risk analytics | Exposure-weighted PD and PD × LGD × EAD expected loss |
| Model discrimination | ROC-AUC, Gini, KS, decile lift and default capture |
| Probability calibration | Brier score, log loss, calibration bands + Wilson intervals |
| Concentration risk | Largest exposure, top-10 share, HHI, effective borrower count |
| Stress testing | Independent PD and LGD multipliers with bounded probabilities |
| Business stress | Revenue/expense shocks and loss-making borrower share |
| Portfolio segmentation | Transparent PD risk bands and exposure shares |
| Review prioritization | Expected-loss contribution + PD/exposure comparison score |
| Data quality | Empty-input, finite numeric, ID, optional-label, PD-range and exposure checks |
| Reporting | Professional Streamlit model-monitoring dashboard |
| Governance | Dedicated validation note with production limitations |
| Engineering | Modular Python, tests, CI on Python 3.10/3.12 |

## Executive portfolio snapshot

The bundled illustrative portfolio contains approximately:

- **1,000 borrower records**
- **$68M of exposure**
- borrower-level observed default labels
- borrower-level probability-of-default scores
- credit score, operating-risk, revenue, expense and income fields

The original report remains available as **[Risk_Assessment_Report.docx](Risk_Assessment_Report.docx)**.

## Analytical workflow

### Credit-risk architecture

```mermaid
flowchart LR
    A[Borrower Portfolio] --> B[Schema & Quality Validation]
    B --> C[PD Discrimination]
    B --> D[PD Calibration]
    B --> E[Exposure Concentration]
    B --> F[Expected Loss]
    F --> G[PD x LGD Stress]
    B --> H[Profitability Stress]
    C --> I[Credit Risk Dashboard]
    D --> I
    E --> I
    G --> I
    H --> I
```


```text
Borrower-level portfolio data
        ↓
Schema & quality validation
        ↓
PD discrimination
(AUC / KS)
        ↓
Probability calibration
(Brier / log loss / calibration bands)
        ↓
Exposure concentration
(HHI / top shares)
        ↓
Expected loss
(EAD × PD × LGD)
        ↓
PD × LGD stress grid
        ↓
Revenue / expense operating stress
        ↓
Risk segmentation + account review queue
        ↓
Interactive monitoring dashboard
```

## Model validation

A probability model should not be evaluated with one metric alone.

The dashboard separates **discrimination** from **calibration**.

### ROC-AUC

Measures whether higher PD estimates tend to rank observed defaults above non-defaults.

A high AUC does not guarantee accurate probability levels.

### KS statistic

Measures the maximum separation between cumulative default and non-default score distributions across thresholds.

### Risk deciles and lift

Borrowers are ranked from highest to lowest predicted PD and split into approximately equal-count risk buckets. The dashboard reports observed default rate, lift, and cumulative default capture so ranking quality is visible beyond a single AUC number.

### Brier score


```text
Brier = mean((PD - Default)^2)
```

Evaluates probability accuracy.

### Log loss

Penalizes confident probability errors more heavily.

### Calibration bands

Predicted probabilities are grouped into transparent PD bands and compared with observed default rates.

The dashboard reports:

- borrower count;
- exposure;
- average predicted PD;
- observed default rate;
- calibration gap;
- approximate 95% Wilson interval for observed default rate.

See **[MODEL_VALIDATION.md](MODEL_VALIDATION.md)** for model-governance context and production limitations.

### Important validation provenance note

The repository evaluates a supplied `PD_Score`, but it does not contain the original versioned PD-model development pipeline establishing the training sample or whether the bundled records are truly out-of-time.

Accordingly, the bundled AUC, KS, Brier, log-loss, and calibration results should be described as **sample diagnostics**, not independent out-of-sample validation, unless model provenance is established externally.

### Exposure-weighted portfolio PD

The application reports both borrower-average PD and exposure-weighted PD. Portfolio expected loss is tied to exposure-weighted risk so larger exposures receive proportionate economic weight.

## Expected loss


The portfolio expected-loss engine uses:

```text
Expected Loss = EAD × PD × LGD
```

where:

- **EAD** is represented by `Loan_Amount`;
- **PD** is represented by `PD_Score`;
- **LGD** is an explicit user assumption.

Expected loss is reported both in dollars and as a percentage of portfolio exposure.

## PD × LGD stress testing

The dashboard lets a reviewer independently shock default probability and loss severity.

```text
PD_stressed  = min(PD × PD multiplier, 100%)
LGD_stressed = min(LGD × LGD multiplier, 100%)
```

A two-dimensional stress grid makes the expected-loss sensitivity visible instead of hiding scenario assumptions in prose.

These are **deterministic sensitivity tests**, not estimates of scenario probability.

## Exposure concentration

The portfolio layer reports:

- largest-borrower exposure share;
- top-10 exposure share;
- Herfindahl-Hirschman Index;
- effective borrower count.

These metrics describe exposure concentration. They do not estimate correlated default behavior or economic capital.

## Profitability stress

Borrower operating performance can be shocked through explicit revenue and expense multipliers:

```text
Stressed Net Income
= Revenue × Revenue Multiplier
- Expenses × Expense Multiplier
```

Baseline and stressed operating income use the same `Revenue - Expenses` definition. If an uploaded `Net_Income` field differs, the dashboard reports the reconciliation gap.

Outputs include:

- baseline operating income;
- stressed operating income;
- income change;
- share of borrowers becoming loss-making.

## Professional dashboard

Run locally:

```bash
git clone https://github.com/ParBproject/Portfolio-Risk-Analysis-Credit-Risk-Modeling.git
cd Portfolio-Risk-Analysis-Credit-Risk-Modeling

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

streamlit run app.py
```

The dashboard contains five workbenches:

**Model Validation**  
ROC curve, AUC, KS, Brier score, log loss, calibration chart and calibration table.

**Portfolio Risk**  
Exposure concentration, risk segments and credit-score/PD diagnostics.

**Stress Testing**  
Expected-loss sensitivity grid plus borrower profitability stress.

**Account Review**  
Transparent PD/exposure priority queue.

**Methodology**  
Metric definitions, assumptions, limitations and governance boundaries.

## Account prioritization

The review queue is ranked by expected-loss contribution:

```text
Expected Loss Contribution = EAD × PD × LGD
```

The earlier `PD × sqrt(Exposure Share)` score remains visible as a secondary comparison field.

This is a transparent triage mechanism. It is **not** an underwriting rule, credit cutoff, adverse-action model, or lending decision.

## Original visual evidence

### Default-probability distribution

![Distribution of predicted default probability](Screenshot/1.png)

### Credit score vs predicted risk

![Credit score versus default probability](Screenshot/2.png)

### Stress-test comparison

![Portfolio stress-test comparison](Screenshot/3.png)

### High-risk concentration

![High-risk loan concentration](Screenshot/4.png)

The current branch adds a fully interactive monitoring dashboard; screenshots should be regenerated after deployment to reflect the new interface.

## Repository structure

```text
Portfolio-Risk-Analysis-Credit-Risk-Modeling/
├── app.py
├── portfolio_data.csv
├── Risk_Assessment_Report.docx
├── src/
│   └── risk_analytics.py
├── tests/
│   └── test_risk_analytics.py
├── Screenshot/
├── .streamlit/config.toml
├── .github/workflows/ci.yml
├── MODEL_VALIDATION.md
├── requirements.txt
└── README.md
```

## Quality and reproducibility

The automated test suite verifies:

- portfolio schema, finite values, PD ranges, and positive total exposure;
- one-class outcome samples do not crash probability monitoring;
- calibration-band borrower reconciliation and confidence bounds;
- exposure concentration reconciliation;
- stressed expected loss does not decrease under more severe PD/LGD assumptions;
- invalid baseline LGD is rejected;
- profitability stress uses a consistent operating-income basis;
- unlabeled current portfolios retain risk/stress/account-review analytics;
- account limits are type-checked;
- the Streamlit dashboard starts without a runtime exception.

GitHub Actions runs linting, source compilation, tests and import checks on Python **3.10 and 3.12**.

## Skills demonstrated

**Quantitative risk:** expected loss, PD/LGD sensitivity, calibration, ROC-AUC, KS, Brier score, concentration risk.

**Data analysis:** pandas, NumPy, segmentation, aggregation, validation, stress comparison, portfolio reporting.

**Model monitoring:** discrimination vs calibration, calibration gaps, bounded probability checks, governance limitations.

**Visualization:** Streamlit, Plotly, ROC curves, calibration plots, stress heatmaps, risk segmentation.

**Engineering:** modular Python, automated tests, CI/CD, uploadable data schema, defensive validation.

## Dataset limitation

The bundled portfolio is synthetic.

That is intentional and explicitly disclosed. This project demonstrates **credit-risk methodology and model-monitoring design**, while other repositories in the portfolio demonstrate work with real historical financial-market data.

A production credit model would require independent validation, out-of-time testing, stability monitoring, fairness/bias review where applicable, controlled data lineage, change governance, and regulatory/policy review.

## Responsible use

Educational credit-risk analytics project only. Nothing in this repository should be used to make real lending, underwriting, investment, or regulatory decisions.
