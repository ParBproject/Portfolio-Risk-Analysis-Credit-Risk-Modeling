# Portfolio risk note

Figures in this note are produced by `python -m credit_risk` from `portfolio_data.csv`. They replace the February 2023 draft where that draft used a different definition.

## Definitions

- **PD** is an account probability of default. On this file it is the `PD_Score` column: the in-sample logistic fit on credit score, loan amount, and operational risk score. The largest gap between that column and the refit logit is 9.59e-14. The unweighted mean PD is 29.80%, equal to the observed default rate (298 of 1000), which is the intercept logistic fit on the same rows.
- **EAD** is outstanding `Loan_Amount`, in dollars. Total EAD is $68,121,079.07. There is no undrawn commitment in the file.
- **LGD** is not in the file. There is no recovery or collateral column. Illustrative dollar losses below use a constant LGD of 45%. That is an assumption, not an estimate.
- **Expected loss** is the sum of PD × LGD × EAD, in dollars. With the assumed LGD it is $5,280,499.32. Before any LGD, PD-weighted exposure (sum of PD × EAD) is $11,734,442.82.
- **Credit-loss VaR** is a quantile of simulated portfolio loss under a one-factor Gaussian copula with asset correlation 0.15, the same constant LGD, 20,000 draws, and seed 42. Correlation is an assumption, not a Basel weight. The 95% loss quantile is $9,526,628.91 and the 99% loss quantile is $11,936,201.30. Expected shortfall is the mean of the worst 1,000 draws ($10,989,179.23) and the worst 200 draws ($13,140,300.09). The 95% quantile sits $4,246,129.59 above expected loss. The cents are the seeded point estimate. Simulation error around them is in the credit-loss section.

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

Resampling the 20,000 simulated losses (1,000 bootstrap samples, seed 42) gives a 95% interval for the Monte Carlo error of $9,436,973.61 to $9,622,340.47 around the 95% VaR, and $11,768,211.11 to $12,085,866.62 around the 99% VaR. The same interval for expected shortfall is $10,862,816.41 to $11,107,677.87 at 95% and $12,907,830.34 to $13,362,475.13 at 99%. That band is simulation error under this model. It is not uncertainty about LGD or correlation.

The copula is Gaussian and has one factor. Defaults are correlated only through that factor, and a Gaussian copula has no tail dependence. Loss given default stays at the assumed 45% in every draw, including the tail, so the simulation has no downturn LGD and no wrong-way recovery. Both choices make the far tail thinner than a book in which defaults cluster more heavily and recoveries fall in a crisis.

## Operational risk

Loans with operational risk score above 60: 27 defaults out of 91 (29.67%). The rest of the book: 271 defaults out of 909 (29.81%). Correlation of operational risk with default is 0.02. The draft claim of a 2.5x default-rate multiple is not what this file shows.

## Holdout check

The published PD is fit on all 1,000 rows, and the file stores that PD beside `Default`. The file has no origination date and no default date, so an out-of-time split is not identified. Discrimination uses five stratified random folds (seed 42). That is not a test of stability through time. The fold model is an unpenalized logistic regression, the same estimator as the published scorecard, on credit score, loan amount, and operational risk only. It does not see `Default`, `PD_Score`, revenue, expenses, or net income. No scaler and no weight-of-evidence binning is fit for this scorecard. The model is not class-weighted. The book default rate is 29.80%, so this is not a rare-event sample, and reweighting the classes would move average PD off the default rate and bias expected loss.

- Out-of-fold AUC 0.852 (DeLong 95% CI 0.828 to 0.877). Gini 0.705 (0.656 to 0.754). KS 0.546 (bootstrap 95% CI 0.505 to 0.612, 2,000 resamples of the out-of-fold scores, seed 42). Brier 0.138.
- In-sample AUC 0.855, Brier 0.137.
- Out-of-fold calibration intercept -0.022, slope 0.974. Brier skill versus a constant forecast at the default rate is 0.339. Mean out-of-fold PD is 29.89%.
- A credit-score-only logit has out-of-fold AUC 0.853. The three-feature AUC minus that figure is -0.0005 (DeLong 95% CI -0.0036 to 0.0026), which includes zero. The extra two features do not improve discrimination on this file.
- Standardizing inside each training fold, then adding revenue, expenses, and net income, gives out-of-fold AUC 0.8522 against 0.8524 for the three-feature scorecard. Those columns do not improve discrimination, so they are not a measured leak. They stay out of the scorecard because they are not application features here. The same columns without scaling give out-of-fold AUC 0.8388 and in-sample AUC 0.8170, which is worse than the three-feature in-sample AUC. An unpenalized logit that has reached the maximum likelihood cannot lose in-sample discrimination by adding columns, so that drop is numerical and is not evidence of a reversed earnings effect.

The DeLong interval treats the out-of-fold scores as fixed. It does not add a further allowance for refitting. The KS interval resamples those same scores.

Loan amount and credit score are highly collinear (correlation 0.92). The in-sample loan-amount coefficient is 5.429e-06. The positive sign is a partial effect next to credit score, not a finding that larger loans default more. The three-feature specification is unchanged.

## What the scorecard attributes

Attributions are exact Shapley values of the published in-sample logit. The reference point is the book-average feature vector. They explain that in-sample score, not the out-of-fold model. On the probability scale the three values sum to account PD minus PD at the average features, which is 20.61%. That reference is not the 29.80% average PD. The sigmoid of the average logit is not the average of the sigmoid.

Mean absolute probability-scale Shapley values: credit score 0.240, loan amount 0.023, operational risk 0.015. Credit score is the largest absolute attribution on 95.0% of loans.

These are interventional values. A feature left out of a coalition is set to its average, and the correlation between credit score and loan amount is ignored. Loan amount's coefficient is positive, so a larger-than-average loan is attributed a higher PD on 47.8% of loans, even though loan amount and default move in opposite directions. That attribution is not a finding that larger loans are riskier on their own. Logit-scale contributions equal the coefficient times the gap from the average feature. They are not the probability-scale values, because the sigmoid is not linear.

## Draft figures that this note corrects

| Quantity | Draft | This note |
|---|---|---|
| Portfolio PD | 29.80% as the portfolio average | Unweighted 29.80%; EAD-weighted 17.23% |
| 95% / 99% VaR | $43,146.55 / $25,890.70 | Those are borrower-income percentiles per account. Illustrative loss VaR is $9,526,628.91 / $11,936,201.30 |
| High PD share | 50.4% of the portfolio | 50.4% of count; 29.72% of EAD |
| Combined stress | $12,559,240 annual loss | Borrower net income -$12,559,240.00. Illustrative EL $5,280,499.32 |
| Operational risk above 60 | 2.5x default rate | 29.67% versus 29.81% |

No measured effect is available for the draft's claim that the recommendations would cut PD or improve the stress result by a stated amount. Those rates were not estimated.
