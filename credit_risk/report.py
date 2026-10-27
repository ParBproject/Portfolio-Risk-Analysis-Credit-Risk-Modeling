"""Text of the numerical report. Figures of record are computed, not typed."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

_CENTS = Decimal("0.01")


def _cents(amount: float | Decimal) -> Decimal:
    value = amount if isinstance(amount, Decimal) else Decimal(str(amount))
    return value.quantize(_CENTS, rounding=ROUND_HALF_UP)


def usd(amount: float | Decimal) -> str:
    rounded = _cents(amount)
    sign = "-" if rounded < 0 else ""
    return f"{sign}${abs(rounded):,.2f}"


def pct(share: float, digits: int = 2) -> str:
    return f"{share * 100:.{digits}f}%"


def render_report(results: dict) -> str:
    earnings = results["earnings"]
    gap = earnings["base"] - results["net_income_column"]
    base = earnings["base"]
    revenue_drop = (base - earnings["revenue_down_20"]) / base
    expense_drop = (base - earnings["expenses_up_10"]) / base
    high_defaults = results["op_risk_high_defaults"]
    high_n = results["op_risk_high_n"]
    rest_defaults = results["op_risk_rest_defaults"]
    rest_n = results["op_risk_rest_n"]
    validation = results["validation"]
    scorecard = results["scorecard"]
    error = results["simulation_error"]
    explanation = results["explanation"]
    unexpected_95 = _cents(results["credit_loss_var_95"]) - _cents(results["expected_loss"])
    abs_probability = explanation["mean_abs_probability"]
    if validation["has_time_column"] == 0:
        time_sentence = (
            "The file has no origination date and no default date, so an out-of-time split is not identified. "
        )
    else:
        time_sentence = (
            "A date-like column is present. The split below is still a random fold, not an out-of-time cut. "
        )
    if (
        validation["auc_minus_credit_score_ci_low"]
        < 0
        < validation["auc_minus_credit_score_ci_high"]
    ):
        auc_gap_sentence = "includes zero. Loan amount and operational risk do not improve discrimination here."
    else:
        auc_gap_sentence = "does not include zero."
    lines = [
        "# Portfolio risk note",
        "",
        "Figures come from `python -m credit_risk` on `portfolio_data.csv`. "
        "Where the February 2023 draft used a different definition, this note says so.",
        "",
        "## Definitions",
        "",
        "- **PD** is an account probability of default. On this file it is `PD_Score`: "
        "the in-sample logistic fit on credit score, loan amount, and operational risk. "
        f"The largest gap versus a refit of that logit is {scorecard['max_abs_pd_gap']:.2e}. "
        f"Unweighted mean PD is {pct(results['mean_pd'])}, the same as the default rate "
        f"({results['n_defaults']} of {results['n_loans']}).",
        "- **EAD** is outstanding `Loan_Amount`. Total EAD is "
        f"{usd(results['total_ead'])}. The file has no undrawn commitment.",
        "- **LGD** is not in the file. Dollar losses use a constant LGD of "
        f"{pct(results['assumed_lgd'], 0)}. That is an assumption.",
        "- **Expected loss** is the sum of PD × LGD × EAD: "
        f"{usd(results['expected_loss'])}. "
        f"Before LGD, the sum of PD × EAD is {usd(results['expected_defaulted_exposure'])}.",
        "- **Credit-loss VaR** is a quantile of simulated portfolio loss. The simulation is a "
        f"one-factor Gaussian copula with asset correlation {results['asset_correlation']}, "
        f"the same LGD, {results['var_simulations']:,} draws, and seed {results['var_seed']}. "
        "Correlation is an assumption, not a Basel weight. "
        f"The 95% loss quantile is {usd(results['credit_loss_var_95'])} and the 99% quantile is "
        f"{usd(results['credit_loss_var_99'])}. "
        f"Expected shortfall is the mean of the worst {results['credit_loss_es_95_count']:,} draws "
        f"({usd(results['credit_loss_es_95'])}) and the worst {results['credit_loss_es_99_count']:,} draws "
        f"({usd(results['credit_loss_es_99'])}). "
        f"The 95% quantile is {usd(unexpected_95)} above expected loss. "
        "The cents are the seeded point estimate.",
        "",
        "## Income percentiles the draft called VaR",
        "",
        f"The draft reported a 95% VaR of {usd(results['income_percentile_05'])} and a 99% VaR of "
        f"{usd(results['income_percentile_01'])}. Those are the 5th and 1st percentiles of per-account "
        "borrower net income (NumPy percentile, method `higher`). They are not a portfolio loss. "
        "A loss quantile does not fall when the confidence level rises.",
        "",
        "## Portfolio",
        "",
        f"Unweighted mean PD is {pct(results['mean_pd'])}. "
        f"EAD-weighted PD is {pct(results['ead_weighted_pd'])}, "
        "lower because higher-PD loans are smaller.",
        "",
        f"{results['high_pd_count']} loans have PD above 20%: "
        f"{pct(results['high_pd_count_share'], 1)} of the count and "
        f"{pct(results['high_pd_ead_share'])} of EAD "
        f"({usd(results['high_pd_ead'])}).",
        "",
        f"Credit score and PD move together (correlation {results['corr_credit_score_pd']:.2f}). "
        f"Scores run from {results['credit_score_min']} to {results['credit_score_max']} "
        f"(mean {results['credit_score_mean']:.1f}). "
        f"Loan amount and credit score are collinear (correlation {results['corr_loan_credit_score']:.2f}). "
        f"The in-sample loan-amount coefficient is {scorecard['Loan_Amount']:.3e}. "
        "The positive sign is the partial effect beside credit score. It is not evidence that larger loans default more.",
        "",
        "## Borrower earnings",
        "",
        "The revenue and expense shocks change aggregate borrower net income. "
        "They do not change PD, LGD, or EAD, so they do not change expected loss.",
        "",
        "| Scenario | Aggregate borrower net income |",
        "|---|---:|",
        f"| Base (revenue − expenses) | {usd(earnings['base'])} |",
        f"| Revenue down 20% | {usd(earnings['revenue_down_20'])} |",
        f"| Expenses up 10% | {usd(earnings['expenses_up_10'])} |",
        f"| Combined | {usd(earnings['combined'])} |",
        "",
        f"The `Net_Income` column sums to {usd(results['net_income_column'])}, "
        f"{usd(gap)} below revenue minus expenses, from cent rounding on the rows.",
        "",
        f"A 20% revenue decline cuts aggregate borrower net income by {pct(float(revenue_drop))}. "
        f"A 10% expense increase cuts it by {pct(float(expense_drop))}. "
        "The file has no product-segment column. "
        f"Combined borrower net income is {usd(earnings['combined'])}. "
        "That is not expected credit loss, and it is not an annual loss on the loan book.",
        "",
        "## Credit-loss stress",
        "",
        f"With the systematic factor at its {pct(results['stress_factor_quantile'], 0)} "
        f"and asset correlation {results['asset_correlation']}, expected loss is "
        f"{usd(results['conditional_expected_loss'])}. "
        "On a one-factor book that conditional loss sits near the 95% loss quantile. "
        "The gap is idiosyncratic risk across the loans.",
        "",
        f"Resampling the {results['var_simulations']:,} simulated losses "
        f"({int(error['n_boot']):,} bootstrap samples, seed {int(error['seed'])}) "
        "gives a 95% interval for the Monte Carlo error of "
        f"{usd(error['var_95_low'])} to {usd(error['var_95_high'])} around the 95% VaR, and "
        f"{usd(error['var_99_low'])} to {usd(error['var_99_high'])} around the 99% VaR. "
        f"The same interval for expected shortfall is {usd(error['es_95_low'])} to {usd(error['es_95_high'])} "
        f"at 95% and {usd(error['es_99_low'])} to {usd(error['es_99_high'])} at 99%. "
        "That band is simulation error. It is not a range for LGD or correlation.",
        "",
        "The copula is Gaussian and has one factor, so it has no tail dependence. "
        "LGD stays at the assumed 45% in every draw, including the tail.",
        "",
        "## Operational risk",
        "",
        f"Loans with operational risk above 60: {high_defaults} defaults out of {high_n} "
        f"({pct(high_defaults / high_n)}). "
        f"The rest of the book: {rest_defaults} defaults out of {rest_n} "
        f"({pct(rest_defaults / rest_n)}). "
        f"Correlation with default is {results['corr_op_risk_default']:.2f}. "
        "The draft's 2.5x default-rate multiple is not in this file.",
        "",
        "## Holdout",
        "",
        "The published PD is the fit on all 1,000 rows. The file stores that PD beside `Default`. "
        + time_sentence
        + "Discrimination below uses five stratified random folds (seed 42). "
        "It is not a test of stability through time. "
        "The fold model is an unpenalized logistic regression on credit score, loan amount, and operational risk. "
        "It does not see `Default`, `PD_Score`, revenue, expenses, or net income, and it is not class-weighted. "
        f"The default rate is {pct(results['default_rate'])}. "
        "Reweighting the classes would move average PD off that rate.",
        "",
        f"- Out-of-fold AUC {validation['oof_auc']:.3f} "
        f"(DeLong 95% CI {validation['oof_auc_ci_low']:.3f} to {validation['oof_auc_ci_high']:.3f}). "
        f"Gini {validation['oof_gini']:.3f} "
        f"({validation['oof_gini_ci_low']:.3f} to {validation['oof_gini_ci_high']:.3f}). "
        f"KS {validation['oof_ks']:.3f} "
        f"(bootstrap 95% CI {validation['oof_ks_ci_low']:.3f} to {validation['oof_ks_ci_high']:.3f}, "
        f"{int(validation['oof_ks_boot']):,} resamples of the out-of-fold scores, seed 42). "
        f"Brier {validation['oof_brier']:.3f}.",
        f"- In-sample AUC {validation['insample_auc']:.3f}, Brier {validation['insample_brier']:.3f}.",
        f"- Out-of-fold calibration intercept {validation['oof_calibration_intercept']:.3f}, "
        f"slope {validation['oof_calibration_slope']:.3f}. "
        f"Brier skill versus a constant forecast at the default rate is {validation['oof_brier_skill']:.3f}. "
        f"Mean out-of-fold PD is {pct(validation['oof_mean_pd'])}.",
        f"- A credit-score-only logit has out-of-fold AUC {validation['credit_score_oof_auc']:.3f}. "
        f"The three-feature AUC minus that figure is {validation['auc_minus_credit_score']:.4f} "
        f"(DeLong 95% CI {validation['auc_minus_credit_score_ci_low']:.4f} to "
        f"{validation['auc_minus_credit_score_ci_high']:.4f}), which "
        + auc_gap_sentence,
        f"- Adding revenue, expenses, and net income, after standardizing inside each training fold, "
        f"gives out-of-fold AUC {validation['earnings_scaled_oof_auc']:.4f} "
        f"against {validation['oof_auc']:.4f} for the three-feature scorecard. "
        "Those columns are not a measured leak. They stay out of the scorecard because they are not application features here. "
        f"Without scaling, out-of-fold AUC is {validation['earnings_unscaled_oof_auc']:.4f} "
        f"and in-sample AUC is {validation['earnings_unscaled_insample_auc']:.4f}, "
        "below the three-feature in-sample AUC. "
        "An unpenalized logit at the maximum likelihood does not lose in-sample fit by adding columns, "
        "so that drop is numerical.",
        "",
        "The DeLong interval treats the out-of-fold scores as fixed. "
        "The KS interval resamples those same scores.",
        "",
        "## Scorecard attributions",
        "",
        "Attributions are exact Shapley values of the in-sample logit, "
        "not of the out-of-fold model. The reference is the average feature vector, "
        f"where PD is {pct(explanation['reference_pd'])}. "
        f"That is not the {pct(results['mean_pd'])} average PD: "
        "the sigmoid of the average logit is not the average of the sigmoid.",
        "",
        "Mean absolute probability-scale values: "
        f"credit score {abs_probability['Credit_Score']:.3f}, "
        f"loan amount {abs_probability['Loan_Amount']:.3f}, "
        f"operational risk {abs_probability['Operational_Risk_Score']:.3f}. "
        f"Credit score is the largest absolute attribution on "
        f"{pct(explanation['credit_score_largest_share'], 1)} of loans.",
        "",
        "A feature left out of a coalition is set to its average, so the correlation "
        "between credit score and loan amount is ignored. "
        "Loan amount's coefficient is positive, so a larger-than-average loan is attributed a higher PD "
        f"on {pct(explanation['loan_amount_positive_share'], 1)} of loans. "
        "That is the partial coefficient, not a finding that larger loans default more on their own.",
        "",
        "## Draft figures this note corrects",
        "",
        "| Quantity | Draft | This note |",
        "|---|---|---|",
        "| Portfolio PD | 29.80% as the portfolio average | "
        f"Unweighted {pct(results['mean_pd'])}; EAD-weighted {pct(results['ead_weighted_pd'])} |",
        "| 95% / 99% VaR | "
        f"{usd(results['income_percentile_05'])} / {usd(results['income_percentile_01'])} | "
        "Borrower-income percentiles per account. "
        f"Loss VaR is {usd(results['credit_loss_var_95'])} / {usd(results['credit_loss_var_99'])} |",
        "| High PD share | 50.4% of the portfolio | "
        f"{pct(results['high_pd_count_share'], 1)} of count; {pct(results['high_pd_ead_share'])} of EAD |",
        "| Combined stress | $12,559,240 annual loss | "
        f"Borrower net income {usd(earnings['combined'])}. "
        f"Expected loss {usd(results['expected_loss'])} |",
        "| Operational risk above 60 | 2.5x default rate | "
        f"{pct(high_defaults / high_n)} versus {pct(rest_defaults / rest_n)} |",
        "",
        "The draft assigned its recommendations a PD reduction and a stress improvement. "
        "Those effects were not estimated.",
        "",
    ]
    return "\n".join(lines)
