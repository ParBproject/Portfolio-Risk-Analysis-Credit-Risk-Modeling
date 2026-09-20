"""Interactive credit-risk model monitoring and portfolio stress dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from src.risk_analytics import (
    calibration_table,
    concentration_summary,
    decile_lift_table,
    expected_loss_summary,
    model_diagnostics,
    profitability_stress,
    risk_segments,
    stress_grid,
    top_risk_accounts,
    validate_portfolio,
)


ROOT = Path(__file__).resolve().parent
NAVY = "#0F172A"
TEAL = "#0F766E"
CYAN = "#38BDF8"
ROSE = "#E11D48"
AMBER = "#D97706"
GRID = "#E2E8F0"


st.set_page_config(
    page_title="Credit Risk Analytics Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #F8FAFC; }
    .block-container {
        max-width: 1450px;
        padding-top: 1.35rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background:
            radial-gradient(circle at 88% 12%, rgba(56,189,248,.21), transparent 30%),
            linear-gradient(135deg, #0F172A 0%, #312E81 56%, #0F766E 130%);
        color: white;
        box-shadow: 0 18px 48px rgba(15,23,42,.14);
        margin-bottom: 1.1rem;
    }
    .hero small {
        color: #99F6E4;
        text-transform: uppercase;
        letter-spacing: .15em;
        font-weight: 750;
    }
    .hero h1 {
        margin: .45rem 0 0;
        font-size: 2.25rem;
        letter-spacing: -.03em;
    }
    .hero p {
        margin: .75rem 0 0;
        max-width: 940px;
        color: #DCE7F4;
        line-height: 1.62;
    }
    .signal-card {
        padding: 1rem 1.05rem;
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        min-height: 104px;
        box-shadow: 0 5px 18px rgba(15,23,42,.04);
    }
    .signal-label {
        color: #64748B;
        font-size: .75rem;
        text-transform: uppercase;
        letter-spacing: .09em;
        font-weight: 750;
    }
    .signal-value {
        color: #0F172A;
        font-size: 1.45rem;
        font-weight: 760;
        margin-top: .28rem;
    }
    .signal-note {
        color: #64748B;
        font-size: .79rem;
        margin-top: .18rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 15px;
        padding: .85rem 1rem;
        box-shadow: 0 4px 14px rgba(15,23,42,.035);
    }
    section[data-testid="stSidebar"] {
        background: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }
    .method-box {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 17px;
        padding: 1.05rem 1.2rem;
        line-height: 1.55;
        margin-bottom: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def signal_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="signal-card">
          <div class="signal-label">{label}</div>
          <div class="signal-value">{value}</div>
          <div class="signal-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_bundled() -> pd.DataFrame:
    return pd.read_csv(ROOT / "portfolio_data.csv")


with st.sidebar:
    st.markdown("### Portfolio input")
    uploaded = st.file_uploader(
        "Optional portfolio CSV",
        type=["csv"],
        help="If omitted, the bundled illustrative 1,000-loan portfolio is used.",
    )

    st.divider()
    st.markdown("#### Expected-loss assumptions")
    lgd = st.slider("Baseline LGD", 0.0, 1.0, 0.45, 0.05)
    pd_multiplier = st.slider("PD stress multiplier", 0.5, 3.0, 1.50, 0.05)
    lgd_multiplier = st.slider("LGD stress multiplier", 0.5, 2.0, 1.25, 0.05)

    st.divider()
    st.markdown("#### Profitability stress")
    revenue_multiplier = st.slider(
        "Revenue multiplier",
        0.40,
        1.20,
        0.80,
        0.05,
    )
    expense_multiplier = st.slider(
        "Expense multiplier",
        0.70,
        1.60,
        1.10,
        0.05,
    )

    st.divider()
    top_n = st.slider("Accounts in review queue", 5, 50, 20, 5)

if uploaded is None:
    raw = load_bundled()
    source_label = "Bundled illustrative portfolio"
else:
    try:
        raw = pd.read_csv(uploaded)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        st.stop()
    source_label = "Uploaded portfolio"

try:
    portfolio = validate_portfolio(raw)
    has_realized_defaults = "Default" in portfolio.columns
    diagnostics = model_diagnostics(portfolio) if has_realized_defaults else None
    calibration = calibration_table(portfolio) if has_realized_defaults else None
    deciles = decile_lift_table(portfolio) if has_realized_defaults else None
    concentration = concentration_summary(portfolio)
    baseline_el = expected_loss_summary(portfolio, lgd=lgd)
    stressed_el = expected_loss_summary(
        portfolio,
        lgd=lgd,
        pd_multiplier=pd_multiplier,
        lgd_multiplier=lgd_multiplier,
    )
    segments = risk_segments(portfolio)
    profitability = profitability_stress(
        portfolio,
        revenue_multiplier=revenue_multiplier,
        expense_multiplier=expense_multiplier,
    )
except ValueError as exc:
    st.error(f"Portfolio analytics failed: {exc}")
    st.stop()


st.markdown(
    """
    <div class="hero">
      <small>Credit Risk & Model Monitoring</small>
      <h1>Credit Risk Analytics Lab</h1>
      <p>
        Probability-of-default diagnostics, calibration monitoring, exposure
        concentration, PD × LGD × EAD expected loss, deterministic stress testing,
        profitability shocks, and prioritized account review in one reproducible
        portfolio-risk workflow.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    signal_card(
        "Portfolio exposure",
        "$" + f"{concentration.total_exposure / 1_000_000:.1f}M",
        f"{len(portfolio):,} borrower records",
    )
with c2:
    if diagnostics is None:
        signal_card(
            "ROC-AUC",
            "N/A",
            "Upload realized Default labels for validation",
        )
    elif diagnostics.discrimination_available:
        signal_card(
            "ROC-AUC",
            f"{diagnostics.roc_auc:.3f}",
            f"KS {diagnostics.ks_statistic:.3f}",
        )
    else:
        signal_card(
            "ROC-AUC",
            "N/A",
            "Only one observed default class in sample",
        )
with c3:
    signal_card(
        "Baseline expected loss",
        "$" + f"{baseline_el.expected_loss / 1_000_000:.2f}M",
        f"{baseline_el.expected_loss_ratio:.2%} of exposure",
    )
with c4:
    signal_card(
        "Stressed expected loss",
        "$" + f"{stressed_el.expected_loss / 1_000_000:.2f}M",
        f"{stressed_el.expected_loss_ratio:.2%} of exposure",
    )

st.caption(
    f"Data source: {source_label}. The bundled dataset is synthetic and designed "
    "for methodology demonstration, not underwriting or lending decisions."
)
st.write("")

validation_tab, portfolio_tab, stress_tab, accounts_tab, methodology_tab = st.tabs(
    [
        "Model Validation",
        "Portfolio Risk",
        "Stress Testing",
        "Account Review",
        "Methodology",
    ]
)


with validation_tab:
    st.markdown("### PD discrimination and calibration")

    if diagnostics is None or calibration is None:
        st.info(
            "This portfolio does not include realized Default outcomes. "
            "Exposure, expected-loss, stress, segmentation, and account-review "
            "analytics remain available, but model-validation metrics require labels."
        )
    else:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(
            "ROC-AUC",
            f"{diagnostics.roc_auc:.3f}"
            if diagnostics.discrimination_available
            else "N/A",
        )
        m2.metric(
            "Gini",
            f"{diagnostics.gini_coefficient:.3f}"
            if diagnostics.discrimination_available
            else "N/A",
        )
        m3.metric(
            "KS statistic",
            f"{diagnostics.ks_statistic:.3f}"
            if diagnostics.discrimination_available
            else "N/A",
        )
        m4.metric("Brier score", f"{diagnostics.brier_score:.4f}")
        m5.metric("Log loss", f"{diagnostics.log_loss:.4f}")
        m6.metric(
            "Observed vs average PD",
            f"{diagnostics.default_rate:.1%}",
            delta=f"{diagnostics.default_rate - diagnostics.average_pd:+.1%}",
            delta_color="off",
        )

        left, right = st.columns(2)
        with left:
            if diagnostics.discrimination_available:
                y_true = portfolio["Default"].astype(int).to_numpy()
                scores = portfolio["PD_Score"].to_numpy()
                fpr, tpr, _ = roc_curve(y_true, scores)
                roc_fig = go.Figure()
                roc_fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name="PD model",
                        line={"color": TEAL, "width": 3},
                    )
                )
                roc_fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random",
                        line={"color": "#94A3B8", "dash": "dash"},
                    )
                )
                roc_fig.update_layout(
                    title={"text": "ROC Curve", "x": 0.02},
                    height=430,
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    xaxis={"title": "False Positive Rate", "gridcolor": GRID},
                    yaxis={"title": "True Positive Rate", "gridcolor": GRID},
                    legend={"orientation": "h", "y": -0.18},
                    margin={"l": 55, "r": 25, "t": 65, "b": 75},
                )
                st.plotly_chart(roc_fig, use_container_width=True)
            else:
                st.warning(
                    "ROC-AUC and KS are undefined because the current sample "
                    "contains only one observed Default class."
                )

        with right:
            lower_error = (
                calibration["Observed_Default_Rate"]
                - calibration["Observed_DR_Lower_95"]
            )
            upper_error = (
                calibration["Observed_DR_Upper_95"]
                - calibration["Observed_Default_Rate"]
            )
            calibration_fig = go.Figure()
            calibration_fig.add_trace(
                go.Scatter(
                    x=calibration["Average_PD"],
                    y=calibration["Observed_Default_Rate"],
                    mode="lines+markers",
                    name="Observed",
                    line={"color": TEAL, "width": 3},
                    marker={"size": 9},
                    error_y={
                        "type": "data",
                        "array": upper_error,
                        "arrayminus": lower_error,
                        "visible": True,
                        "color": "#64748B",
                    },
                )
            )
            calibration_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Perfect calibration",
                    line={"color": "#94A3B8", "dash": "dash"},
                )
            )
            calibration_fig.update_layout(
                title={"text": "PD Calibration by Risk Band", "x": 0.02},
                height=430,
                paper_bgcolor="white",
                plot_bgcolor="white",
                xaxis={
                    "title": "Average predicted PD",
                    "tickformat": ".0%",
                    "gridcolor": GRID,
                    "range": [0, 1],
                },
                yaxis={
                    "title": "Observed default rate",
                    "tickformat": ".0%",
                    "gridcolor": GRID,
                    "range": [0, 1],
                },
                legend={"orientation": "h", "y": -0.18},
                margin={"l": 55, "r": 25, "t": 65, "b": 75},
            )
            st.plotly_chart(calibration_fig, use_container_width=True)

        st.markdown("#### Calibration table")
        st.dataframe(
            calibration.style.format(
                {
                    "Exposure": "$ {:,.0f}",
                    "Average_PD": "{:.1%}",
                    "Observed_Default_Rate": "{:.1%}",
                    "Observed_DR_Lower_95": "{:.1%}",
                    "Observed_DR_Upper_95": "{:.1%}",
                    "Calibration_Gap": "{:+.1%}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Observed default-rate error bars use approximate 95% Wilson intervals. "
            "Wide intervals indicate sparse risk bands and should temper conclusions."
        )

        if deciles is not None:
            st.markdown("#### Risk deciles and default capture")
            if deciles["Cumulative_Default_Capture"].notna().any():
                gains = go.Figure()
                gains.add_trace(
                    go.Scatter(
                        x=deciles["Cumulative_Borrower_Share"],
                        y=deciles["Cumulative_Default_Capture"],
                        mode="lines+markers",
                        name="PD ranking",
                        line={"color": TEAL, "width": 3},
                    )
                )
                gains.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random ranking",
                        line={"color": "#94A3B8", "dash": "dash"},
                    )
                )
                gains.update_layout(
                    title={"text": "Cumulative Default Capture", "x": 0.02},
                    height=410,
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    xaxis={
                        "title": "Cumulative borrower share",
                        "tickformat": ".0%",
                        "gridcolor": GRID,
                        "range": [0, 1],
                    },
                    yaxis={
                        "title": "Cumulative defaults captured",
                        "tickformat": ".0%",
                        "gridcolor": GRID,
                        "range": [0, 1],
                    },
                    legend={"orientation": "h", "y": -0.20},
                    margin={"l": 55, "r": 25, "t": 65, "b": 80},
                )
                st.plotly_chart(gains, use_container_width=True)
            else:
                st.info(
                    "Default capture is unavailable because this labeled sample "
                    "contains no observed defaults."
                )

            st.dataframe(
                deciles.style.format(
                    {
                        "Exposure": "$ {:,.0f}",
                        "Average_PD": "{:.1%}",
                        "Observed_Default_Rate": "{:.1%}",
                        "Lift": "{:.2f}",
                        "Cumulative_Default_Capture": "{:.1%}",
                        "Cumulative_Borrower_Share": "{:.1%}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )


with portfolio_tab:
    st.markdown("### Exposure concentration and risk segments")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric(
        "Largest borrower share",
        f"{concentration.largest_exposure_share:.2%}",
    )
    p2.metric(
        "Top-10 exposure share",
        f"{concentration.top_10_exposure_share:.2%}",
    )
    p3.metric("Exposure HHI", f"{concentration.hhi:.4f}")
    p4.metric(
        "Effective borrower count",
        f"{concentration.effective_borrower_count:.1f}",
    )

    left, right = st.columns([3, 2])
    with left:
        segment_fig = go.Figure(
            go.Bar(
                x=segments["Risk_Segment"].astype(str),
                y=segments["Exposure"],
                marker_color=[TEAL, CYAN, AMBER, ROSE, NAVY][: len(segments)],
                text=[f"{value / 1_000_000:.1f}M" for value in segments["Exposure"]],
                textposition="outside",
            )
        )
        segment_fig.update_layout(
            title={"text": "Exposure by PD Risk Segment", "x": 0.02},
            height=430,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis={"title": ""},
            yaxis={"title": "Exposure ($)", "gridcolor": GRID},
            margin={"l": 55, "r": 25, "t": 65, "b": 55},
        )
        st.plotly_chart(segment_fig, use_container_width=True)

    with right:
        segment_display = segments.copy()
        segment_formats = {
            "Exposure": "$ {:,.0f}",
            "Average_PD": "{:.1%}",
            "Average_Credit_Score": "{:.0f}",
            "Exposure_Share": "{:.1%}",
        }
        if segment_display["Observed_Default_Rate"].notna().any():
            segment_formats["Observed_Default_Rate"] = "{:.1%}"
        else:
            segment_display = segment_display.drop(
                columns=["Observed_Default_Rate"]
            )
        st.dataframe(
            segment_display.style.format(segment_formats),
            hide_index=True,
            use_container_width=True,
        )

    scatter = go.Figure(
        go.Scatter(
            x=portfolio["Credit_Score"],
            y=portfolio["PD_Score"],
            mode="markers",
            marker={
                "size": 7,
                "color": portfolio["Loan_Amount"],
                "colorscale": "Tealgrn",
                "opacity": 0.60,
                "colorbar": {"title": "Exposure"},
            },
            text=portfolio["Customer_ID"],
            hovertemplate=(
                "%{text}<br>Credit score: %{x:.0f}"
                "<br>PD: %{y:.1%}<extra></extra>"
            ),
        )
    )
    scatter.update_layout(
        title={"text": "Credit Score vs Predicted Default Risk", "x": 0.02},
        height=450,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis={"title": "Credit Score", "gridcolor": GRID},
        yaxis={"title": "Predicted PD", "tickformat": ".0%", "gridcolor": GRID},
        margin={"l": 55, "r": 25, "t": 65, "b": 55},
    )
    st.plotly_chart(scatter, use_container_width=True)


with stress_tab:
    st.markdown("### Expected-loss and operating stress")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "Baseline EL",
        "$" + f"{baseline_el.expected_loss:,.0f}",
    )
    s2.metric(
        "Stressed EL",
        "$" + f"{stressed_el.expected_loss:,.0f}",
        delta="$" + f"{stressed_el.expected_loss - baseline_el.expected_loss:,.0f}",
        delta_color="inverse",
    )
    s3.metric(
        "Stressed operating income",
        "$" + f"{profitability['stressed_net_income']:,.0f}",
        delta="$" + f"{profitability['net_income_change']:,.0f}",
        delta_color="normal",
    )
    s4.metric(
        "Loss-making borrowers",
        f"{profitability['loss_making_share']:.1%}",
    )

    grid = stress_grid(portfolio, lgd=lgd)
    pivot = grid.pivot(
        index="LGD Multiplier",
        columns="PD Multiplier",
        values="Expected Loss Ratio",
    )
    heat = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0.0, "#ECFDF5"],
                [0.55, "#FDE68A"],
                [1.0, "#BE123C"],
            ],
            text=pivot.values,
            texttemplate="%{text:.2%}",
            colorbar={"title": "EL ratio", "tickformat": ".1%"},
            hovertemplate=(
                "PD multiplier: %{x:.2f}×"
                "<br>LGD multiplier: %{y:.2f}×"
                "<br>Expected-loss ratio: %{z:.2%}<extra></extra>"
            ),
        )
    )
    heat.update_layout(
        title={"text": "PD × LGD Expected-Loss Stress Grid", "x": 0.02},
        height=480,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis={"title": "PD multiplier"},
        yaxis={"title": "LGD multiplier"},
        margin={"l": 65, "r": 35, "t": 65, "b": 55},
    )
    st.plotly_chart(heat, use_container_width=True)

    st.info(
        "Stress multipliers are deterministic sensitivity assumptions. "
        "They do not estimate the probability that a macroeconomic scenario will occur."
    )
    if "net_income_reconciliation_gap" in profitability:
        gap = profitability["net_income_reconciliation_gap"]
        if abs(gap) > 1e-6:
            st.caption(
                "Reported Net_Income does not exactly reconcile to Revenue − Expenses. "
                "Stress deltas therefore use a consistent operating-income basis; "
                "reported-vs-calculated reconciliation gap: $" + f"{gap:,.0f}."
            )


with accounts_tab:
    st.markdown("### Prioritized account review")
    accounts = top_risk_accounts(portfolio, limit=top_n, lgd=lgd)
    st.dataframe(
        accounts.style.format(
            {
                "Loan_Amount": "$ {:,.0f}",
                "Operational_Risk_Score": "{:.1f}",
                "PD_Score": "{:.1%}",
                "Expected_Loss_Contribution": "$ {:,.0f}",
                "Portfolio_EL_Share": "{:.2%}",
                "Risk_Priority_Score": "{:.5f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Accounts are ranked by expected-loss contribution (EAD × PD × LGD). "
        "The legacy PD/exposure heuristic remains visible for comparison. "
        "Neither metric is a lending decision rule."
    )


with methodology_tab:
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="method-box">
            <b>Model monitoring</b><br><br>
            ROC-AUC and KS measure discrimination. Brier score and log loss
            measure probability accuracy. Calibration bands compare average
            predicted PD with realized default rates.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="method-box">
            <b>Expected loss</b><br><br>
            Portfolio expected loss uses <b>EAD × PD × LGD</b>. PD and LGD are
            shocked independently and bounded at 100% under stress.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="method-box">
            <b>Concentration risk</b><br><br>
            Largest-borrower share, top-10 exposure share, HHI, and effective
            borrower count describe exposure concentration. They do not model
            default correlation.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="method-box">
            <b>Governance boundary</b><br><br>
            The bundled portfolio is synthetic. A production PD model would
            require independent validation, stability monitoring, bias/fairness
            review, data lineage, approval controls, and ongoing performance monitoring.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Important limitations")
    st.markdown(
        """
        - The bundled data is synthetic and should not be presented as a live lending portfolio.
        - PD scores are evaluated, not re-trained, by this dashboard.
        - Stress scenarios are deterministic sensitivities, not macroeconomic probabilities.
        - Expected loss is not regulatory capital or unexpected loss.
        - Account priority scores are triage aids, not underwriting decisions.
        """
    )

st.caption(
    "Educational credit-risk portfolio project. The bundled data is synthetic "
    "and results do not constitute lending, investment, or regulatory advice."
)
