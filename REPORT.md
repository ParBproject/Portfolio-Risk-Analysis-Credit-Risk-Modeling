# Portfolio risk note

Figures come from `python -m credit_risk` on `portfolio_data.csv`. Where the February 2023 draft used a different definition, this note says so.

## Definitions

- **PD** is an account probability of default. On this file it is `PD_Score`: the in-sample logistic fit on credit score, loan amount, and operational risk. The largest gap versus a refit of that logit is 9.59e-14. Unweighted mean PD is 29.80%, the same as the default rate (298 of 1000).
- **EAD** is outstanding `Loan_Amount`. Total EAD is $68,121,079.07. The file has no undrawn commitment.
- **LGD** is not in the file. Dollar losses use a constant LGD of 45%. That is an assumption.
- **Expected loss** is the sum of PD × LGD × EAD: $5,280,499.32. Before LGD, the sum of PD × EAD is $11,734,442.82.
- **Credit-loss VaR** is a quantile of simulated portfolio loss. The simulation is a one-factor Gaussian copula with asset correlation 0.15, the same LGD, 20,000 draws, and seed 42. Correlation is an assumption, not a Basel weight. The 95% loss quantile is $9,526,628.91 and the 99% quantile is $11,936,201.30. Expected shortfall is the mean of the worst 1,000 draws ($10,989,179.23) and the worst 200 draws ($13,140,300.09). The 95% quantile is $4,246,129.59 above expected loss. The cents are the seeded point estimate.

## Income percentiles the draft called VaR

The draft reported a 95% VaR of $43,146.55 and a 99% VaR of $25,890.70. Those are the 5th and 1st percentiles of per-account borrower net income (NumPy percentile, method `higher`). They are not a portfolio loss. A loss quantile does not fall when the confidence level rises.

## Portfolio

Unweighted mean PD is 29.80%. EAD-weighted PD is 17.23%, lower because higher-PD loans are smaller.

504 loans have PD above 20%: 50.4% of the count and 29.72% of EAD ($20,248,304.07).

Credit score and PD move together (correlation -0.94). Scores run from 421 to 850 (mean 681.0). Loan amount and credit score are collinear (correlation 0.92). The in-sample loan-amount coefficient is 5.429e-06. The positive sign is the partial effect beside credit score. It is not evidence that larger loans default more.

## Borrower earnings

The revenue and expense shocks change aggregate borrower net income. They do not change PD, LGD, or EAD, so they do not change expected loss.

| Scenario | Aggregate borrower net income |
|---|---:|
| Base (revenue − expenses) | $124,207,710.82 |
| Revenue down 20% | $24,749,229.55 |
| Expenses up 10% | $86,899,241.27 |
| Combined | -$12,559,240.00 |

The `Net_Income` column sums to $124,207,710.74, $0.08 below revenue minus expenses, from cent rounding on the rows.

A 20% revenue decline cuts aggregate borrower net income by 80.07%. A 10% expense increase cuts it by 30.04%. The file has no product-segment column. Combined borrower net income is -$12,559,240.00. That is not expected credit loss, and it is not an annual loss on the loan book.

## Credit-loss stress

With the systematic factor at its 5% and asset correlation 0.15, expected loss is $9,457,484.79. On a one-factor book that conditional loss sits near the 95% loss quantile. The gap is idiosyncratic risk across the loans.

Resampling the 20,000 simulated losses (1,000 bootstrap samples, seed 42) gives a 95% interval for the Monte Carlo error of $9,436,973.61 to $9,622,340.47 around the 95% VaR, and $11,768,211.11 to $12,085,866.62 around the 99% VaR. The same interval for expected shortfall is $10,862,816.41 to $11,107,677.87 at 95% and $12,907,830.34 to $13,362,475.13 at 99%. That band is simulation error. It is not a range for LGD or correlation.

The copula is Gaussian and has one factor, so it has no tail dependence. LGD stays at the assumed 45% in every draw, including the tail.

## Operational risk

Loans with operational risk above 60: 27 defaults out of 91 (29.67%). The rest of the book: 271 defaults out of 909 (29.81%). Correlation with default is 0.02. The draft's 2.5x default-rate multiple is not in this file.

## Holdout

The published PD is the fit on all 1,000 rows. The file stores that PD beside `Default`. The file has no origination date and no default date, so an out-of-time split is not identified. Discrimination below uses five stratified random folds (seed 42). It is not a test of stability through time. The fold model is an unpenalized logistic regression on credit score, loan amount, and operational risk. It does not see `Default`, `PD_Score`, revenue, expenses, or net income, and it is not class-weighted. The default rate is 29.80%. Reweighting the classes would move average PD off that rate.

- Out-of-fold AUC 0.852 (DeLong 95% CI 0.828 to 0.877). Gini 0.705 (0.656 to 0.754). KS 0.546 (bootstrap 95% CI 0.505 to 0.612, 2,000 resamples of the out-of-fold scores, seed 42). Brier 0.138.
- In-sample AUC 0.855, Brier 0.137.
- Out-of-fold calibration intercept -0.022, slope 0.974. Brier skill versus a constant forecast at the default rate is 0.339. Mean out-of-fold PD is 29.89%.
- A credit-score-only logit has out-of-fold AUC 0.853. The three-feature AUC minus that figure is -0.0005 (DeLong 95% CI -0.0036 to 0.0026), which includes zero. Loan amount and operational risk do not improve discrimination here.
- Adding revenue, expenses, and net income, after standardizing inside each training fold, gives out-of-fold AUC 0.8522 against 0.8524 for the three-feature scorecard. Those columns are not a measured leak. They stay out of the scorecard because they are not application features here. Without scaling, out-of-fold AUC is 0.8388 and in-sample AUC is 0.8170, below the three-feature in-sample AUC. An unpenalized logit at the maximum likelihood does not lose in-sample fit by adding columns, so that drop is numerical.

The DeLong interval treats the out-of-fold scores as fixed. The KS interval resamples those same scores.

## Scorecard attributions

Attributions are exact Shapley values of the in-sample logit, not of the out-of-fold model. The reference is the average feature vector, where PD is 20.61%. That is not the 29.80% average PD: the sigmoid of the average logit is not the average of the sigmoid.

Mean absolute probability-scale values: credit score 0.240, loan amount 0.023, operational risk 0.015. Credit score is the largest absolute attribution on 95.0% of loans.

A feature left out of a coalition is set to its average, so the correlation between credit score and loan amount is ignored. Loan amount's coefficient is positive, so a larger-than-average loan is attributed a higher PD on 47.8% of loans. That is the partial coefficient, not a finding that larger loans default more on their own.

## Draft figures this note corrects

| Quantity | Draft | This note |
|---|---|---|
| Portfolio PD | 29.80% as the portfolio average | Unweighted 29.80%; EAD-weighted 17.23% |
| 95% / 99% VaR | $43,146.55 / $25,890.70 | Borrower-income percentiles per account. Loss VaR is $9,526,628.91 / $11,936,201.30 |
| High PD share | 50.4% of the portfolio | 50.4% of count; 29.72% of EAD |
| Combined stress | $12,559,240 annual loss | Borrower net income -$12,559,240.00. Expected loss $5,280,499.32 |
| Operational risk above 60 | 2.5x default rate | 29.67% versus 29.81% |

The draft assigned its recommendations a PD reduction and a stress improvement. Those effects were not estimated.
