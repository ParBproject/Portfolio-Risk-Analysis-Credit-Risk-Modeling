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
    unexpected_95 = _cents(results["credit_loss_var_95"]) - _cents(results["expected_loss"])
    lines = [
        "# Portfolio risk note",
        "",
        "Figures in this note are produced by `python -m credit_risk` from `portfolio_data.csv`. "
        "They replace the February 2023 draft where that draft used a different definition.",
        "",
        "## Definitions",
        "",
        "- **PD** is an account probability of default. On this file it is the `PD_Score` column: "
        "the in-sample logistic fit on credit score, loan amount, and operational risk score. "
        f"The largest gap between that column and the refit logit is {scorecard['max_abs_pd_gap']:.2e}. "
        f"The unweighted mean PD is {pct(results['mean_pd'])}, equal to the observed default rate "
        f"({results['n_defaults']} of {results['n_loans']}), which is the intercept logistic fit on the same rows.",
        "- **EAD** is outstanding `Loan_Amount`, in dollars. Total EAD is "
        f"{usd(results['total_ead'])}. There is no undrawn commitment in the file.",
        "- **LGD** is not in the file. There is no recovery or collateral column. "
        f"Illustrative dollar losses below use a constant LGD of {pct(results['assumed_lgd'], 0)}. "
        "That is an assumption, not an estimate.",
        "- **Expected loss** is the sum of PD × LGD × EAD, in dollars. "
        f"With the assumed LGD it is {usd(results['expected_loss'])}. "
        f"Before any LGD, PD-weighted exposure (sum of PD × EAD) is {usd(results['expected_defaulted_exposure'])}.",
        "- **Credit-loss VaR** is a quantile of simulated portfolio loss under a one-factor Gaussian copula "
        f"with asset correlation {results['asset_correlation']}, the same constant LGD, "
        f"{results['var_simulations']:,} draws, and seed {results['var_seed']}. "
        "Correlation is an assumption, not a Basel weight. "
        f"The 95% loss quantile is {usd(results['credit_loss_var_95'])} and the 99% loss quantile is "
        f"{usd(results['credit_loss_var_99'])}. "
        f"Expected shortfall is {usd(results['credit_loss_es_95'])} at 95% and "
        f"{usd(results['credit_loss_es_99'])} at 99%. "
        f"The 95% quantile sits {usd(unexpected_95)} above expected loss.",
        "",
        "## What the draft called VaR",
        "",
        f"The draft reported a 95% VaR of {usd(results['income_percentile_05'])} and a 99% VaR of "
        f"{usd(results['income_percentile_01'])}. Those are the 5th and 1st percentiles of **per-account borrower net income** "
        "(NumPy percentile, method `higher`). They are not a portfolio loss. A loss VaR cannot fall when the "
        "confidence level rises; these income percentiles do, because they are a lower tail of income.",
        "",
        "## Portfolio PD and concentration",
        "",
        f"Unweighted mean PD is {pct(results['mean_pd'])}. "
        f"EAD-weighted PD is {pct(results['ead_weighted_pd'])}. "
        "The weighted figure is lower because higher-PD loans are smaller.",
        "",
        f"{results['high_pd_count']} loans have PD above 20%. "
        f"That is {pct(results['high_pd_count_share'], 1)} of the loan **count** and "
        f"{pct(results['high_pd_ead_share'])} of **EAD** "
        f"({usd(results['high_pd_ead'])} of exposure).",
        "",
        f"Credit score and PD move together (correlation {results['corr_credit_score_pd']:.2f}). "
        f"Observed credit scores run from {results['credit_score_min']} to {results['credit_score_max']} "
        f"(mean {results['credit_score_mean']:.1f}).",
        "",
        "## Borrower earnings stress is not credit loss",
        "",
        "The revenue and expense shocks change aggregate borrower net income. They do not change PD, LGD, or EAD, "
        "so they do not change expected loss under the scorecard in this file.",
        "",
        "| Scenario | Aggregate borrower net income |",
        "|---|---:|",
        f"| Base (revenue − expenses) | {usd(earnings['base'])} |",
        f"| Revenue down 20% | {usd(earnings['revenue_down_20'])} |",
        f"| Expenses up 10% | {usd(earnings['expenses_up_10'])} |",
        f"| Combined | {usd(earnings['combined'])} |",
        "",
        f"The `Net_Income` column sums to {usd(results['net_income_column'])}, "
        f"{usd(gap)} below revenue minus expenses, from cent rounding on the rows. "
        "The scenario table uses revenue and expenses so the shocks add up.",
        "",
        f"A 20% revenue decline cuts aggregate borrower net income by {pct(float(revenue_drop))}. "
        f"A 10% expense increase cuts it by {pct(float(expense_drop))}. "
        "The file has no product-segment column. "
        f"Combined borrower net income is {usd(earnings['combined'])}. "
        "That number is not expected credit loss and not an annual loss on the loan book.",
        "",
        "## Credit-loss stress",
        "",
        f"Conditional on a systematic factor at its {pct(results['stress_factor_quantile'], 0)} "
        f"(asset correlation {results['asset_correlation']}), illustrative expected loss is "
        f"{usd(results['conditional_expected_loss'])}. "
        "For a fine-grained one-factor book this sits near the 95% loss quantile; "
        "the gap is idiosyncratic risk across 1,000 loans.",
        "",
        "## Operational risk",
        "",
        f"Loans with operational risk score above 60: {high_defaults} defaults out of {high_n} "
        f"({pct(high_defaults / high_n)}). "
        f"The rest of the book: {rest_defaults} defaults out of {rest_n} "
        f"({pct(rest_defaults / rest_n)}). "
        f"Correlation of operational risk with default is {results['corr_op_risk_default']:.2f}. "
        "The draft claim of a 2.5x default-rate multiple is not what this file shows.",
        "",
        "## Holdout check",
        "",
        "The published PD is fit on all 1,000 rows, and the file stores that PD beside `Default`. "
        "Discrimination below uses five stratified folds. The fold model sees credit score, loan amount, "
        "and operational risk only. It does not see `Default`, `PD_Score`, revenue, expenses, or net income.",
        "",
        f"- Out-of-fold AUC {validation['oof_auc']:.3f}, Brier {validation['oof_brier']:.3f}.",
        f"- In-sample AUC {validation['insample_auc']:.3f}, Brier {validation['insample_brier']:.3f}.",
        "",
        f"Loan amount and credit score are highly collinear (correlation {results['corr_loan_credit_score']:.2f}). "
        f"The in-sample loan-amount coefficient is {scorecard['Loan_Amount']:.3e}. "
        "The positive sign is a partial effect next to credit score, not a finding that larger loans default more. "
        "The three-feature specification is unchanged.",
        "",
        "## Draft figures that this note corrects",
        "",
        "| Quantity | Draft | This note |",
        "|---|---|---|",
        "| Portfolio PD | 29.80% as the portfolio average | "
        f"Unweighted {pct(results['mean_pd'])}; EAD-weighted {pct(results['ead_weighted_pd'])} |",
        "| 95% / 99% VaR | "
        f"{usd(results['income_percentile_05'])} / {usd(results['income_percentile_01'])} | "
        "Those are borrower-income percentiles per account. "
        f"Illustrative loss VaR is {usd(results['credit_loss_var_95'])} / {usd(results['credit_loss_var_99'])} |",
        "| High PD share | 50.4% of the portfolio | "
        f"{pct(results['high_pd_count_share'], 1)} of count; {pct(results['high_pd_ead_share'])} of EAD |",
        "| Combined stress | $12,559,240 annual loss | "
        f"Borrower net income {usd(earnings['combined'])}. "
        f"Illustrative EL {usd(results['expected_loss'])} |",
        "| Operational risk above 60 | 2.5x default rate | "
        f"{pct(high_defaults / high_n)} versus {pct(rest_defaults / rest_n)} |",
        "",
        "No measured effect is available for the draft's claim that the recommendations would "
        "cut PD or improve the stress result by a stated amount. Those rates were not estimated.",
        "",
    ]
    return "\n".join(lines)
