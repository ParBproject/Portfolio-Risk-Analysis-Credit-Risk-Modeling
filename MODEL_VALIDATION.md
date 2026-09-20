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

The application also accepts appropriately governed uploaded CSVs. Realized `Default` outcomes and reported `Net_Income` are optional for current-portfolio risk analysis; realized defaults are required only for retrospective model-validation metrics.

## PD score provenance and validation interpretation

The repository treats `PD_Score` as a supplied probability estimate.

The codebase does **not** contain a versioned development pipeline establishing how the bundled `PD_Score` values were originally fitted, what training sample was used, or whether the bundled observations are truly out-of-time relative to model development.

Therefore:

- AUC, KS, Brier score, log loss, and calibration results on the bundled data should be described as **sample diagnostics**;
- they should not be presented as independent out-of-sample validation unless score provenance and sample separation are established externally;
- a production validation would require a documented development sample, independent validation sample, and preferably out-of-time performance evidence.

This limitation is intentional and explicit so the project does not overstate the strength of the model evidence.

## Probability-of-default validation

### ROC-AUC

ROC-AUC measures whether higher PD scores tend to rank defaults above non-defaults.

It is a **discrimination** metric and does not prove that the probabilities are calibrated.

### Gini coefficient

The Gini coefficient is derived directly from ROC-AUC:

```text
Gini = 2 × ROC-AUC - 1
```

It provides a familiar rank-ordering summary for credit-risk review.

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
- calibration gap;
- approximate 95% Wilson confidence interval for the observed default rate.

A model can have strong ROC-AUC and still be poorly calibrated, so discrimination and calibration are shown separately.

## Risk deciles, lift, and default capture

Borrowers are ranked from highest to lowest predicted PD and divided into approximately equal-count risk buckets.

For each bucket the dashboard reports:

- borrower count;
- exposure;
- average predicted PD;
- observed default rate;
- lift relative to the portfolio default rate;
- cumulative borrower share;
- cumulative share of observed defaults captured.

The cumulative-default-capture chart compares the PD ranking with a random-ranking diagonal. This is a ranking diagnostic; it should still be interpreted together with calibration and probability-loss metrics.

## Exposure-weighted PD

For portfolio risk, a simple borrower-average PD and an exposure-weighted PD answer different questions.

The dashboard therefore distinguishes:

```text
Borrower Average PD
= mean(PD)

Exposure-Weighted PD
= sum(EAD × PD) / sum(EAD)
```

Expected-loss ratios are driven by the exposure-weighted quantity, not by treating every borrower as economically equal.

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

Baseline and stressed operating income are calculated on the same accounting basis:

```text
Operating Income = Revenue - Expenses
```

If a supplied `Net_Income` field does not reconcile to that definition, the dashboard reports the reconciliation gap rather than mixing accounting bases.

It reports:

- baseline operating income;
- stressed operating income;
- change from baseline;
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

Accounts are ranked primarily by expected-loss contribution:

```text
Expected Loss Contribution = EAD × PD × LGD
```

The dashboard also retains the earlier `PD × sqrt(exposure share)` heuristic as a secondary comparison field.

Expected-loss contribution is more directly tied to economic materiality, while neither metric should be interpreted as an underwriting cutoff, credit decision, or adverse-action model.

## Data-quality controls

The pipeline validates:

- non-empty portfolio input;
- required base columns;
- unique, non-blank customer IDs;
- numeric fields with no NaN or infinite values;
- binary observed defaults when supplied;
- PD values inside [0,1];
- non-negative borrower exposure and positive total exposure;
- non-negative revenue, expenses, and operational-risk scores.

A current portfolio may omit realized `Default` labels. In that case, model-validation panels are disabled while expected-loss, concentration, stress, segmentation, and account-review analytics remain available.

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
