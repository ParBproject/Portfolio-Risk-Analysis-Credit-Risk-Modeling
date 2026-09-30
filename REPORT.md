# Portfolio risk note

Figures in this note are produced by `python -m credit_risk` from `portfolio_data.csv`. They replace the February 2023 draft where that draft used a different definition.

## Definitions

- **PD** is an account probability of default. On this file it is the `PD_Score` column: the in-sample logistic fit on credit score, loan amount, and operational risk score. The largest gap between that column and the refit logit is 9.59e-14. The unweighted mean PD is 29.80%, equal to the observed default rate (298 of 1000), which is the intercept logistic fit on the same rows.
- **EAD** is outstanding `Loan_Amount`, in dollars. Total EAD is $68,121,079.07. There is no undrawn commitment in the file.
- **LGD** is not in the file. There is no recovery or collateral column. Illustrative dollar losses below use a constant LGD of 45%. That is an assumption, not an estimate.
- **Expected loss** is the sum of PD × LGD × EAD, in dollars. With the assumed LGD it is $5,280,499.32. Before any LGD, PD-weighted exposure (sum of PD × EAD) is $11,734,442.82.
- **Credit-loss VaR** is a quantile of simulated portfolio loss under a one-factor Gaussian copula with asset correlation 0.15, the same constant LGD, 20,000 draws, and seed 42. Correlation is an assumption, not a Basel weight. The 95% loss quantile is $9,526,628.91 and the 99% loss quantile is $11,936,201.30. Expected shortfall is $10,989,179.23 at 95% and $13,140,300.09 at 99%. The 95% quantile sits $4,246,129.59 above expected loss.

## What the draft called VaR

The draft reported a 95% VaR of $43,146.55 and a 99% VaR of $25,890.70. Those are the 5th and 1st percentiles of **per-account borrower net income** (NumPy percentile, method `higher`). They are not a portfolio loss. A loss VaR cannot fall when the confidence level rises; these income percentiles do, because they are a lower tail of income.

## Portfolio PD and concentration

Unweighted mean PD is 29.80%. EAD-weighted PD is 17.23%. The weighted figure is lower because higher-PD loans are smaller.

504 loans have PD above 20%. That is 50.4% of the loan **count** and 29.72% of **EAD** ($20,248,304.07 of exposure).

Credit score and PD move together (correlation -0.94). Observed credit scores run from 421 to 850 (mean 681.0).

## Borrower earnings stress is not credit loss

The revenue and expense shocks change aggregate borrower net income. They do not change PD, LGD, or EAD, so they do not change expected loss under the scorecard in this file.

| Scenario | Aggregate borrower net income |
|---|---:|
| Base (revenue − expenses) | $124,207,710.82 |
| Revenue down 20% | $24,749,229.55 |
| Expenses up 10% | $86,899,241.27 |
| Combined | -$12,559,240.00 |

The `Net_Income` column sums to $124,207,710.74, $0.08 below revenue minus expenses, from cent rounding on the rows. The scenario table uses revenue and expenses so the shocks add up.

A 20% revenue decline cuts aggregate borrower net income by 80.07%. A 10% expense increase cuts it by 30.04%. The file has no product-segment column. Combined borrower net income is -$12,559,240.00. That number is not expected credit loss and not an annual loss on the loan book.

## Credit-loss stress

Conditional on a systematic factor at its 5% (asset correlation 0.15), illustrative expected loss is $9,457,484.79. For a fine-grained one-factor book this sits near the 95% loss quantile; the gap is idiosyncratic risk across 1,000 loans.

## Operational risk

Loans with operational risk score above 60: 27 defaults out of 91 (29.67%). The rest of the book: 271 defaults out of 909 (29.81%). Correlation of operational risk with default is 0.02. The draft claim of a 2.5x default-rate multiple is not what this file shows.

## Holdout check

The published PD is fit on all 1,000 rows, and the file stores that PD beside `Default`. Discrimination below uses five stratified folds. The fold model sees credit score, loan amount, and operational risk only. It does not see `Default`, `PD_Score`, revenue, expenses, or net income.

- Out-of-fold AUC 0.852, Brier 0.138.
- In-sample AUC 0.855, Brier 0.137.

Loan amount and credit score are highly collinear (correlation 0.92). The in-sample loan-amount coefficient is 5.429e-06. The positive sign is a partial effect next to credit score, not a finding that larger loans default more. The three-feature specification is unchanged.

## Draft figures that this note corrects

| Quantity | Draft | This note |
|---|---|---|
| Portfolio PD | 29.80% as the portfolio average | Unweighted 29.80%; EAD-weighted 17.23% |
| 95% / 99% VaR | $43,146.55 / $25,890.70 | Those are borrower-income percentiles per account. Illustrative loss VaR is $9,526,628.91 / $11,936,201.30 |
| High PD share | 50.4% of the portfolio | 50.4% of count; 29.72% of EAD |
| Combined stress | $12,559,240 annual loss | Borrower net income -$12,559,240.00. Illustrative EL $5,280,499.32 |
| Operational risk above 60 | 2.5x default rate | 29.67% versus 29.81% |

No measured effect is available for the draft's claim that the recommendations would cut PD or improve the stress result by a stated amount. Those rates were not estimated.
