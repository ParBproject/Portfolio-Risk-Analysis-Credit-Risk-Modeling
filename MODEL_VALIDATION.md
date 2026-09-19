# Credit Risk Model Validation & Governance Note

## Scope

This repository evaluates an existing probability-of-default score (`PD_Score`) on a synthetic 1,000-borrower portfolio.

The bundled PD score is **not retrained** by the dashboard. The project demonstrates the surrounding quantitative risk workflow:

- portfolio data validation;
- PD discrimination;
- probability calibration;
- exposure concentration;
- expected loss;
- deterministic stress testing;
- profitability stress;
- account prioritization;
- monitoring-oriented reporting.

## Dataset status

The bundled dataset is synthetic and designed for methodology demonstration.

It must not be described as a live bank, lender, customer, or regulatory portfolio.

The application also accepts an uploaded CSV with the same schema so the analytical workflow can be reused with appropriately governed data.

## Probability-of-default validation

### ROC-AUC

ROC-AUC measures whether higher PD scores tend to rank defaults above non-defaults.

It is a **discrimination** metric and does not prove that the probabilities are calibrated.

### KS statistic

The Kolmogorov–Smirnov statistic is calculated as the maximum difference between the true-positive and false-positive rates across score thresholds.

It measures separation between default and non-default score distributions.

### Brier score

```text
Brier = mean((PD - Default)^2)
```

The Brier score evaluates probability accuracy.

Lower values are preferred, but interpretation depends on event prevalence and the benchmark used.

### Log loss

Log loss penalizes confident probability errors more strongly than the Brier score.

Scores are clipped away from exact 0 and 1 for numerical stability.

## Calibration monitoring

Borrowers are grouped into transparent PD bands.

For each band the dashboard compares:

- borrower count;
- exposure;
- average predicted PD;
- observed default rate;
- calibration gap.

A model can have strong ROC-AUC and still be poorly calibrated, so discrimination and calibration are shown separately.

## Expected loss

The project uses the standard expected-loss identity:

```text
Expected Loss = EAD × PD × LGD
```

where:

- EAD is represented by `Loan_Amount`;
- PD is represented by `PD_Score`;
- LGD is a user-controlled assumption.

Portfolio expected loss is reported both in dollars and as a percentage of total exposure.

## Stress testing

PD and LGD multipliers are applied independently:

```text
PD_stressed = min(PD × PD multiplier, 100%)
LGD_stressed = min(LGD × LGD multiplier, 100%)
```

The dashboard shows a PD × LGD expected-loss stress grid.

These are deterministic sensitivities, not probabilities of a macroeconomic scenario.

## Profitability stress

The operating stress layer applies explicit revenue and expense multipliers:

```text
Stressed Net Income
= Revenue × Revenue Multiplier
- Expenses × Expense Multiplier
```

It reports:

- total stressed net income;
- change from baseline net income;
- share of borrowers becoming loss-making under the scenario.

This is a borrower operating-stress proxy, not a complete credit migration model.

## Concentration risk

The dashboard reports:

- largest exposure share;
- top-10 exposure share;
- Herfindahl-Hirschman Index (HHI);
- effective borrower count (1 / HHI).

These metrics describe **exposure concentration** only. They do not estimate joint-default correlation or economic capital.

## Account review queue

The account-priority heuristic combines:

```text
PD × sqrt(exposure share)
```

This surfaces accounts that are both risky and economically material.

It is a transparent review queue—not an underwriting cutoff, credit decision, or adverse-action model.

## Data-quality controls

The pipeline validates:

- required columns;
- unique customer IDs;
- numeric risk fields;
- binary observed defaults;
- PD values inside [0,1];
- non-negative exposure.

## Production requirements not implemented

A regulated production model would normally require substantially more governance, including:

- development/validation sample separation;
- out-of-time validation;
- challenger benchmarks;
- calibration recalibration procedures;
- population stability monitoring;
- characteristic analysis;
- bias/fairness testing where applicable;
- explainability standards;
- data lineage and source controls;
- approval, override, and change-management processes;
- monitoring thresholds and escalation;
- independent model validation;
- regulatory and policy review.

## Responsible use

This repository demonstrates credit-risk analytics methodology.

The bundled portfolio is synthetic, and no output should be used to make real lending, underwriting, investment, or regulatory decisions.
