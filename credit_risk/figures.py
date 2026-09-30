"""Write the four case-study charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from credit_risk.assumptions import HIGH_PD

_BLUE = "#1f4e79"
_RUST = "#b85c38"
_GRID = "#d9e2ec"


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#94a3b8",
            "axes.labelcolor": "#1e293b",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0f172a",
            "grid.color": _GRID,
            "font.size": 11,
        }
    )


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_figures(frame: pd.DataFrame, results: dict, out_dir: Path) -> list[Path]:
    """Regenerate Screenshot/1.png through Screenshot/4.png."""
    _style()
    out_dir = Path(out_dir)
    paths = [out_dir / f"{index}.png" for index in range(1, 5)]
    _pd_distribution(frame, paths[0])
    _score_versus_pd(frame, paths[1])
    _stress(results, paths[2])
    _concentration(frame, results, paths[3])
    return paths


def _pd_distribution(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.histplot(
        frame["PD_Score"],
        bins=25,
        color=_BLUE,
        edgecolor="white",
        ax=ax,
    )
    ax.axvline(HIGH_PD, color=_RUST, linestyle="--", linewidth=1.5, label="PD = 20%")
    ax.set_title("Distribution of in-sample probability of default")
    ax.set_xlabel("Account PD (in-sample scorecard)")
    ax.set_ylabel("Loans")
    ax.legend(frameon=False)
    _finish(fig, path)


def _score_versus_pd(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    plotted = frame.copy()
    plotted["Outcome"] = plotted["Default"].map(
        {0: "Did not default", 1: "Defaulted"}
    )
    sns.scatterplot(
        data=plotted,
        x="Credit_Score",
        y="PD_Score",
        hue="Outcome",
        hue_order=["Did not default", "Defaulted"],
        palette={"Did not default": _BLUE, "Defaulted": _RUST},
        alpha=0.75,
        s=28,
        ax=ax,
        edgecolor="none",
    )
    ax.set_title("Credit score versus in-sample PD")
    ax.set_xlabel("Credit score")
    ax.set_ylabel("Account PD")
    ax.legend(frameon=False)
    _finish(fig, path)


def _stress(results: dict, path: Path) -> None:
    earnings = results["earnings"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    labels = ["Base", "Revenue −20%", "Expenses +10%", "Combined"]
    keys = ["base", "revenue_down_20", "expenses_up_10", "combined"]
    values = [float(earnings[key]) / 1e6 for key in keys]
    colors = [_BLUE, _BLUE, _BLUE, _RUST]
    axes[0].bar(labels, values, color=colors)
    axes[0].axhline(0, color="#64748b", linewidth=0.8)
    axes[0].set_title("Borrower net income")
    axes[0].set_ylabel("Aggregate net income ($ millions)")
    axes[0].tick_params(axis="x", labelrotation=15)

    credit_labels = ["EL\n(LGD 45% assumed)", "Conditional EL\n(factor at 5th pct)"]
    credit_values = [
        results["expected_loss"] / 1e6,
        results["conditional_expected_loss"] / 1e6,
    ]
    axes[1].bar(credit_labels, credit_values, color=[_BLUE, _RUST])
    axes[1].set_title("Illustrative credit loss")
    axes[1].set_ylabel("Expected loss ($ millions)")
    fig.suptitle(
        "Earnings stress and credit-loss stress are different quantities",
        fontsize=13,
    )
    _finish(fig, path)


def _concentration(frame: pd.DataFrame, results: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    axes[0].bar(
        ["Share of loans", "Share of EAD"],
        [results["high_pd_count_share"] * 100, results["high_pd_ead_share"] * 100],
        color=[_RUST, _BLUE],
    )
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Loans with PD above 20%")

    high_n = results["op_risk_high_n"]
    rest_n = results["op_risk_rest_n"]
    high_rate = 100 * results["op_risk_high_defaults"] / high_n
    rest_rate = 100 * results["op_risk_rest_defaults"] / rest_n
    axes[1].bar(
        [f"Op. risk > 60\n(n={high_n})", f"Op. risk ≤ 60\n(n={rest_n})"],
        [high_rate, rest_rate],
        color=[_RUST, _BLUE],
    )
    axes[1].axhline(
        results["default_rate"] * 100,
        color="#64748b",
        linestyle="--",
        linewidth=1,
        label="Book default rate",
    )
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Observed default rate (%)")
    axes[1].set_title("Operational risk does not separate defaults")
    axes[1].legend(frameon=False)
    fig.suptitle("Count concentration is not exposure concentration", fontsize=13)
    _finish(fig, path)
