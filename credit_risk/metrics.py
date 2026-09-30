"""PD, LGD, EAD, expected loss, earnings stress, and credit-loss VaR.

Definitions used here
---------------------
PD
    Account-level probability of default. On the checked-in book this
    is the ``PD_Score`` column. That column is the in-sample logistic
    fit on credit score, loan amount, and operational risk. It is not
    a holdout prediction.
LGD
    Loss given default, as a fraction of EAD. It is not in the file.
    Dollar figures that need it use :data:`ASSUMED_LGD` and are
    labeled as illustrative.
EAD
    Exposure at default, in dollars. Taken as the outstanding
    ``Loan_Amount``. There is no undrawn commitment in the file.
EL
    Expected loss in dollars: sum of PD x LGD x EAD. The same LGD x EAD
    amounts, in cents, are the loss given default in the simulation,
    so the population mean of simulated loss is this EL.
Credit-loss VaR
    A quantile of the simulated portfolio loss distribution (a loss
    amount). The 99% quantile is at least the 95% quantile.
    This is not the lower tail of borrower net income.
Expected shortfall
    The mean of the worst ``(1 - alpha) * n`` simulated losses, which
    is an integer count here (1,000 of 20,000 at 95%, 200 at 99%).
    The cent on a quantile is a seeded point estimate. Monte Carlo
    error is reported beside it.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, roc_auc_score

from credit_risk.assumptions import (
    ASSUMED_LGD,
    ASSET_CORRELATION,
    EXPENSE_SHOCK,
    HIGH_PD,
    MODEL_SEED,
    N_FOLDS,
    OP_RISK_FLAG,
    REVENUE_SHOCK,
    STRESS_FACTOR_QUANTILE,
    VAR_BOOTSTRAP,
    VAR_BOOTSTRAP_SEED,
    VAR_CHUNK,
    VAR_SEED,
    VAR_SIMULATIONS,
)
from credit_risk.data import LEAKAGE_COLUMNS, PD_FEATURES, has_time_column, money_sums
from credit_risk.explain import attribute_scorecard
from credit_risk.validation import (
    auc,
    brier_skill,
    calibration_intercept_slope,
    delong_difference,
    delong_roc,
    ks_bootstrap,
    logistic_insample,
    logistic_oof,
)

_CENTS = Decimal("0.01")
_ONE_CENT = Decimal("1")


def round_cents(amount: Decimal) -> Decimal:
    return amount.quantize(_CENTS, rounding=ROUND_HALF_UP)


def _loss_given_default_cents(ead: Decimal, lgd: Decimal) -> int:
    ead_cents = (ead * 100).quantize(_ONE_CENT, rounding=ROUND_HALF_UP)
    return int((ead_cents * lgd).quantize(_ONE_CENT, rounding=ROUND_HALF_UP))


def loss_given_default(loan_amounts: pd.Series, lgd: float = ASSUMED_LGD) -> np.ndarray:
    """Per-loan loss if that loan defaults, in dollars, rounded to the cent."""
    lgd_dec = Decimal(str(lgd))
    cents = [
        _loss_given_default_cents(Decimal(str(amount)), lgd_dec)
        for amount in loan_amounts
    ]
    return np.asarray(cents, dtype=np.int64) / 100.0


def earnings_scenarios(revenue: Decimal, expenses: Decimal) -> dict[str, Decimal]:
    """Aggregate borrower net income under the original earnings shocks.

    Base is revenue minus expenses, not the rounded ``Net_Income`` column.
    The shocks do not enter PD, LGD, or EAD.
    """
    revenue_keep = Decimal("1") - Decimal(REVENUE_SHOCK)
    expense_factor = Decimal("1") + Decimal(EXPENSE_SHOCK)
    return {
        "base": revenue - expenses,
        "revenue_down_20": revenue * revenue_keep - expenses,
        "expenses_up_10": revenue - expenses * expense_factor,
        "combined": revenue * revenue_keep - expenses * expense_factor,
    }


def borrower_income_percentiles(net_income: pd.Series) -> dict[str, float]:
    """Lower tail of per-account borrower net income.

    The draft memo labeled these as 95% and 99% VaR. They are the
    5th and 1st percentiles (NumPy ``higher`` order statistic) of
    borrower net income, in dollars per account.
    """
    values = net_income.to_numpy(dtype=float)
    return {
        "p05": float(np.percentile(values, 5, method="higher")),
        "p01": float(np.percentile(values, 1, method="higher")),
    }


def _clip_pd(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(probabilities.astype(float), 1e-12, 1 - 1e-12)


def conditional_pd(
    probabilities: np.ndarray,
    factor_quantile: float = STRESS_FACTOR_QUANTILE,
    rho: float = ASSET_CORRELATION,
) -> np.ndarray:
    """Gaussian-copula PD conditional on a systematic-factor percentile.

    Low factor values are adverse: asset value is
    ``sqrt(rho) * M + sqrt(1 - rho) * Z``. The factor at its median is
    not the unconditional PD. The idiosyncratic scale ``sqrt(1 - rho)``
    remains in the denominator, and the unconditional PD is the average
    of this function over M, not its value at M = 0.
    """
    factor = float(norm.ppf(factor_quantile))
    threshold = norm.ppf(_clip_pd(probabilities))
    scale = np.sqrt(1.0 - rho)
    return norm.cdf((threshold - np.sqrt(rho) * factor) / scale)


def simulate_portfolio_loss(
    probabilities: np.ndarray,
    loss_if_default: np.ndarray,
    rho: float = ASSET_CORRELATION,
    n_sims: int = VAR_SIMULATIONS,
    seed: int = VAR_SEED,
) -> np.ndarray:
    """One-factor Gaussian copula portfolio loss, in dollars.

    Marginal default probabilities are the supplied PDs. Loss given
    default is deterministic. The draws are reproducible for a fixed
    NumPy seed and version.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("asset correlation must be in [0, 1)")
    threshold = norm.ppf(_clip_pd(probabilities))
    # Cents keep each scenario total exact. A float dot product is not
    # associative, so the quantile could move by a cent across CPUs.
    loss_cents = np.rint(np.asarray(loss_if_default, dtype=float) * 100.0).astype(np.int64)
    rng = np.random.default_rng(seed)
    losses = np.empty(n_sims, dtype=float)
    weight_m = np.sqrt(rho)
    weight_z = np.sqrt(1.0 - rho)
    start = 0
    while start < n_sims:
        size = min(VAR_CHUNK, n_sims - start)
        factor = rng.standard_normal(size)
        idiosyncratic = rng.standard_normal((size, threshold.size))
        asset = weight_m * factor[:, None] + weight_z * idiosyncratic
        defaulted = asset < threshold
        for row in range(size):
            losses[start + row] = int(loss_cents[defaulted[row]].sum()) / 100.0
        start += size
    return losses


def _tail_count(n: int, alpha: float) -> int:
    count = int(round((1.0 - alpha) * n))
    if count < 1 or count >= n:
        raise ValueError("tail count must leave observations on both sides")
    return count


def _tail_mean(losses: np.ndarray, alpha: float) -> tuple[float, int]:
    """Mean of the worst ``(1 - alpha) * n`` losses, rounded to an integer count."""
    n = int(losses.shape[0])
    count = _tail_count(n, alpha)
    worst = np.partition(losses, n - count)[n - count :]
    return float(worst.mean()), count


def loss_quantiles(losses: np.ndarray) -> dict[str, float]:
    """Loss VaR and expected shortfall. 99% VaR is at least 95% VaR.

    VaR is the linear (type 7) sample quantile. Expected shortfall is
    the mean of the upper tail of a fixed count, not the mean of every
    draw that happens to sit above the interpolated quantile. On this
    book those two sets are the same; the count is the definition.
    """
    losses = np.asarray(losses, dtype=float)
    var_95 = float(np.quantile(losses, 0.95, method="linear"))
    var_99 = float(np.quantile(losses, 0.99, method="linear"))
    if var_99 + 1e-9 < var_95:
        raise RuntimeError("loss VaR decreased as the confidence level increased")
    es_95, n_95 = _tail_mean(losses, 0.95)
    es_99, n_99 = _tail_mean(losses, 0.99)
    if es_95 + 1e-6 < var_95 or es_99 + 1e-6 < var_99:
        raise RuntimeError("expected shortfall is below the loss quantile")
    return {
        "var_95": var_95,
        "var_99": var_99,
        "es_95": es_95,
        "es_99": es_99,
        "es_95_count": float(n_95),
        "es_99_count": float(n_99),
        "mean": float(losses.mean()),
    }


def monte_carlo_intervals(
    losses: np.ndarray,
    *,
    n_boot: int = VAR_BOOTSTRAP,
    seed: int = VAR_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Percentile intervals for the Monte Carlo error of VaR and ES.

    Each replicate resamples the simulated losses with replacement.
    The interval is simulation error around this model, not a
    confidence interval for a different copula or a different LGD.
    """
    losses = np.asarray(losses, dtype=float)
    n = int(losses.shape[0])
    specs = ((0.95, "95"), (0.99, "99"))
    counts = {label: _tail_count(n, alpha) for alpha, label in specs}
    collected = {
        label: {"var": np.empty(n_boot), "es": np.empty(n_boot)} for _, label in specs
    }
    rng = np.random.default_rng(seed)
    start = 0
    chunk = 100
    while start < n_boot:
        size = min(chunk, n_boot - start)
        sample = losses[rng.integers(0, n, size=(size, n))]
        for alpha, label in specs:
            collected[label]["var"][start : start + size] = np.quantile(
                sample, alpha, axis=1, method="linear"
            )
            count = counts[label]
            tail = np.partition(sample, n - count, axis=1)[:, n - count :]
            collected[label]["es"][start : start + size] = tail.mean(axis=1)
        start += size
    intervals: dict[str, float] = {"n_boot": float(n_boot), "seed": float(seed)}
    for _, label in specs:
        var_low, var_high = np.quantile(collected[label]["var"], [0.025, 0.975])
        es_low, es_high = np.quantile(collected[label]["es"], [0.025, 0.975])
        intervals[f"var_{label}_low"] = float(var_low)
        intervals[f"var_{label}_high"] = float(var_high)
        intervals[f"es_{label}_low"] = float(es_low)
        intervals[f"es_{label}_high"] = float(es_high)
    return intervals


def scorecard_fit(frame: pd.DataFrame) -> dict[str, float]:
    """In-sample logistic scorecard implied by ``PD_Score``.

    Returns coefficients of ``logit(PD)`` on an intercept and the three
    published features, plus the largest absolute gap versus ``PD_Score``.
    """
    probability = frame["PD_Score"].to_numpy(dtype=float)
    logit = np.log(probability / (1.0 - probability))
    design = np.column_stack(
        [np.ones(len(frame)), frame.loc[:, PD_FEATURES].to_numpy(dtype=float)]
    )
    coefficients, *_ = np.linalg.lstsq(design, logit, rcond=None)
    fitted = 1.0 / (1.0 + np.exp(-(design @ coefficients)))
    names = ("intercept",) + PD_FEATURES
    result = {name: float(value) for name, value in zip(names, coefficients)}
    result["max_abs_pd_gap"] = float(np.max(np.abs(fitted - probability)))
    return result


def out_of_fold_pd(frame: pd.DataFrame) -> dict[str, float]:
    """Stratified K-fold PD scores. Features exclude the label and PD_Score.

    The fold estimator is unpenalized (``C=inf``), matching the
    published scorecard. There is no class weight: expected loss needs
    a probability, and balancing the classes would move mean PD off
    the default rate. There is no origination date on this file, so
    the split is not out of time.
    """
    leaked = [column for column in PD_FEATURES if column in LEAKAGE_COLUMNS]
    if leaked:
        raise ValueError(f"PD features include leakage columns: {leaked}")
    design = frame.loc[:, PD_FEATURES].to_numpy(dtype=float)
    labels = frame["Default"].to_numpy(dtype=int)
    holdout, _ = logistic_oof(design, labels, scale=False, max_iter=2000)
    credit_only, _ = logistic_oof(
        frame[["Credit_Score"]].to_numpy(dtype=float), labels, scale=False
    )
    earnings = frame.loc[
        :, list(PD_FEATURES) + ["Revenue", "Expenses", "Net_Income"]
    ].to_numpy(dtype=float)
    earnings_scaled, _ = logistic_oof(earnings, labels, scale=True, max_iter=2000)
    earnings_raw, raw_iters = logistic_oof(
        earnings, labels, scale=False, max_iter=5000
    )
    earnings_raw_insample = logistic_insample(earnings, labels, max_iter=5000)
    published = frame["PD_Score"].to_numpy(dtype=float)
    roc = delong_roc(labels, holdout)
    ks = ks_bootstrap(labels, holdout, seed=MODEL_SEED)
    gap = delong_difference(labels, holdout, credit_only)
    intercept, slope = calibration_intercept_slope(labels, holdout)
    gini = 2.0 * roc["auc"] - 1.0
    return {
        "oof_auc": float(roc_auc_score(labels, holdout)),
        "oof_auc_se": roc["se"],
        "oof_auc_ci_low": roc["ci_low"],
        "oof_auc_ci_high": roc["ci_high"],
        "oof_gini": gini,
        "oof_gini_ci_low": 2.0 * roc["ci_low"] - 1.0,
        "oof_gini_ci_high": 2.0 * roc["ci_high"] - 1.0,
        "oof_ks": ks["ks"],
        "oof_ks_ci_low": ks["ci_low"],
        "oof_ks_ci_high": ks["ci_high"],
        "oof_ks_boot": ks["n_boot"],
        "oof_brier": float(brier_score_loss(labels, holdout)),
        "oof_brier_skill": brier_skill(labels, holdout),
        "oof_mean_pd": float(holdout.mean()),
        "oof_calibration_intercept": intercept,
        "oof_calibration_slope": slope,
        "insample_auc": float(roc_auc_score(labels, published)),
        "insample_brier": float(brier_score_loss(labels, published)),
        "credit_score_oof_auc": auc(labels, credit_only),
        "auc_minus_credit_score": gap["diff"],
        "auc_minus_credit_score_se": gap["se"],
        "auc_minus_credit_score_ci_low": gap["ci_low"],
        "auc_minus_credit_score_ci_high": gap["ci_high"],
        "earnings_scaled_oof_auc": auc(labels, earnings_scaled),
        "earnings_unscaled_oof_auc": auc(labels, earnings_raw),
        "earnings_unscaled_insample_auc": auc(labels, earnings_raw_insample),
        "earnings_unscaled_max_iter": float(raw_iters),
        "has_time_column": float(has_time_column(frame)),
        "n_folds": float(N_FOLDS),
        "model_seed": float(MODEL_SEED),
    }


def _pearson(frame: pd.DataFrame, left: str, right: str) -> float:
    return float(frame[left].corr(frame[right]))


def analyze(frame: pd.DataFrame, path=None) -> dict:
    """Compute the case-study figures from a portfolio frame.

    ``path`` is the CSV used for exact money sums. It defaults to the
    checked-in book.
    """
    totals = money_sums(path)
    probability = frame["PD_Score"].to_numpy(dtype=float)
    ead = frame["Loan_Amount"].to_numpy(dtype=float)
    high = probability > HIGH_PD
    loss_if_default = loss_given_default(frame["Loan_Amount"], ASSUMED_LGD)
    # Population EL in dollars. The Python loop keeps the sum ordered.
    expected_loss = 0.0
    for prob, loss in zip(probability, loss_if_default):
        expected_loss += float(prob) * float(loss)
    expected_defaulted_exposure = 0.0
    for prob, exposure in zip(probability, ead):
        expected_defaulted_exposure += float(prob) * float(exposure)

    high_op = frame["Operational_Risk_Score"] > OP_RISK_FLAG
    scenarios = earnings_scenarios(totals["Revenue"], totals["Expenses"])
    income_tail = borrower_income_percentiles(frame["Net_Income"])
    stressed_pd = conditional_pd(probability)
    conditional_el = 0.0
    for prob, loss in zip(stressed_pd, loss_if_default):
        conditional_el += float(prob) * float(loss)
    simulated = simulate_portfolio_loss(probability, loss_if_default)
    quantiles = loss_quantiles(simulated)
    simulation_error = monte_carlo_intervals(simulated)
    scorecard = scorecard_fit(frame)
    validation = out_of_fold_pd(frame)
    explanation = attribute_scorecard(frame, scorecard)
    ead_total = float(totals["Loan_Amount"])
    return {
        "n_loans": int(len(frame)),
        "n_defaults": int(frame["Default"].sum()),
        "default_rate": float(frame["Default"].mean()),
        "total_ead": totals["Loan_Amount"],
        "revenue": totals["Revenue"],
        "expenses": totals["Expenses"],
        "net_income_column": totals["Net_Income"],
        "mean_pd": float(probability.mean()),
        "ead_weighted_pd": float(np.average(probability, weights=ead)),
        "high_pd_count": int(high.sum()),
        "high_pd_count_share": float(high.mean()),
        "high_pd_ead": float(ead[high].sum()),
        "high_pd_ead_share": float(ead[high].sum() / ead_total),
        "credit_score_min": int(frame["Credit_Score"].min()),
        "credit_score_max": int(frame["Credit_Score"].max()),
        "credit_score_mean": float(frame["Credit_Score"].mean()),
        "expected_defaulted_exposure": expected_defaulted_exposure,
        "assumed_lgd": ASSUMED_LGD,
        "expected_loss": expected_loss,
        "asset_correlation": ASSET_CORRELATION,
        "stress_factor_quantile": STRESS_FACTOR_QUANTILE,
        "conditional_expected_loss": conditional_el,
        "var_simulations": VAR_SIMULATIONS,
        "var_seed": VAR_SEED,
        "credit_loss_var_95": quantiles["var_95"],
        "credit_loss_var_99": quantiles["var_99"],
        "credit_loss_es_95": quantiles["es_95"],
        "credit_loss_es_99": quantiles["es_99"],
        "credit_loss_es_95_count": int(quantiles["es_95_count"]),
        "credit_loss_es_99_count": int(quantiles["es_99_count"]),
        "mean_simulated_loss": quantiles["mean"],
        "simulation_error": simulation_error,
        "explanation": explanation,
        "income_percentile_05": income_tail["p05"],
        "income_percentile_01": income_tail["p01"],
        "earnings": scenarios,
        "op_risk_flag": OP_RISK_FLAG,
        "op_risk_high_n": int(high_op.sum()),
        "op_risk_high_defaults": int(frame.loc[high_op, "Default"].sum()),
        "op_risk_rest_n": int((~high_op).sum()),
        "op_risk_rest_defaults": int(frame.loc[~high_op, "Default"].sum()),
        "corr_credit_score_pd": _pearson(frame, "Credit_Score", "PD_Score"),
        "corr_loan_credit_score": _pearson(frame, "Loan_Amount", "Credit_Score"),
        "corr_op_risk_default": _pearson(frame, "Operational_Risk_Score", "Default"),
        "scorecard": scorecard,
        "validation": validation,
        "features": PD_FEATURES,
    }
