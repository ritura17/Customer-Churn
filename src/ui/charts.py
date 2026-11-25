"""Interactive Plotly chart builders with consistent SaaS styling."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.ui.styles import PALETTE

BASE_LAYOUT = dict(
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color=PALETTE["slate_900"], size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=24, r=24, t=56, b=72),
    title_x=0.0,
    title_xanchor="left",
    title_y=0.96,
    title_yanchor="top",
    title_font=dict(size=15),
    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
)


def make_churn_donut(df: pd.DataFrame) -> go.Figure:
    """Create an interactive donut chart of Stayed vs Churned customers."""
    counts = df["Churn Status"].value_counts().reset_index()
    counts.columns = ["Status", "Customers"]
    fig = px.pie(
        counts,
        names="Status",
        values="Customers",
        hole=0.58,
        color="Status",
        color_discrete_map={"Stayed": PALETTE["retain"], "Churned": PALETTE["churn"]},
        title="Customer Retention vs. Churn",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(**BASE_LAYOUT)
    return fig


def make_churn_by_category(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    """Create a grouped bar chart showing customer counts and churn rate by category."""
    grouped = (
        df.groupby([col, "Churn Status"], observed=False)
        .size()
        .reset_index(name="Customers")
    )
    fig = px.bar(
        grouped,
        x=col,
        y="Customers",
        color="Churn Status",
        barmode="group",
        color_discrete_map={"Stayed": PALETTE["retain"], "Churned": PALETTE["churn"]},
        title=title,
    )
    fig.update_layout(**BASE_LAYOUT, xaxis_title=col, yaxis_title="Customers")
    return fig


def make_churn_rate_by_group(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    """Create a bar chart showing churn rate (%) across segments."""
    summary = (
        df.groupby(col, observed=False)["Churn Label"]
        .agg(["mean", "count"])
        .reset_index()
    )
    summary["Churn Rate (%)"] = (summary["mean"] * 100).round(2)
    fig = px.bar(
        summary,
        x=col,
        y="Churn Rate (%)",
        text="Churn Rate (%)",
        hover_data=["count"],
        color="Churn Rate (%)",
        color_continuous_scale=["#93C5FD", "#2563EB", "#DC2626"],
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(**BASE_LAYOUT, coloraxis_showscale=False, yaxis_range=[0, max(summary["Churn Rate (%)"].max() * 1.18, 10)])
    return fig


def make_tenure_histogram(df: pd.DataFrame) -> go.Figure:
    """Create overlapping histogram of Tenure Months by Churn Status."""
    fig = px.histogram(
        df,
        x="Tenure Months",
        color="Churn Status",
        nbins=24,
        barmode="overlay",
        opacity=0.72,
        color_discrete_map={"Stayed": PALETTE["retain"], "Churned": PALETTE["churn"]},
        title="Tenure Distribution by Churn Status",
    )
    fig.update_layout(**BASE_LAYOUT, xaxis_title="Tenure (Months)", yaxis_title="Customers")
    return fig


def make_monthly_charges_box(df: pd.DataFrame) -> go.Figure:
    """Create box plot comparing Monthly Charges for Stayed vs Churned."""
    fig = px.box(
        df,
        x="Churn Status",
        y="Monthly Charges",
        color="Churn Status",
        color_discrete_map={"Stayed": PALETTE["retain"], "Churned": PALETTE["churn"]},
        title="Monthly Charges ($) vs. Churn Status",
    )
    fig.update_layout(**BASE_LAYOUT, showlegend=False, xaxis_title="", yaxis_title="Monthly Charges ($)")
    return fig


def make_probability_gauge(probability: float, threshold: float) -> go.Figure:
    """Create a clean Plotly gauge chart showing churn probability and decision threshold."""
    pct = probability * 100
    bar_color = PALETTE["churn"] if probability >= threshold else PALETTE["retain"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 34, "color": PALETTE["slate_900"]}},
            title={"text": "Estimated Churn Probability", "font": {"size": 14, "color": PALETTE["slate_500"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": PALETTE["slate_500"]},
                "bar": {"color": bar_color, "thickness": 0.32},
                "bgcolor": "#F1F5F9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, max(threshold * 100 - 10, 0)], "color": "#ECFDF5"},
                    {"range": [max(threshold * 100 - 10, 0), threshold * 100], "color": "#FFFBEB"},
                    {"range": [threshold * 100, 100], "color": "#FEF2F2"},
                ],
                "threshold": {
                    "line": {"color": PALETTE["slate_900"], "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=32, r=36, t=42, b=16),
        height=250,
    )
    return fig


def make_shap_bar(drivers: List[Dict[str, Any]]) -> go.Figure:
    """Create horizontal bar chart of local SHAP model-influence drivers."""
    df = pd.DataFrame(drivers).iloc[::-1]
    colors = [PALETTE["churn"] if v > 0 else PALETTE["retain"] for v in df["shap_impact"]]
    labels = [f"{r['feature']} ({r['value']})" for _, r in df.iterrows()]

    fig = go.Figure(
        go.Bar(
            x=df["shap_impact"],
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>SHAP Impact: %{x:+.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=PALETTE["slate_500"])
    fig.update_layout(
        **BASE_LAYOUT,
        title="Local Model Drivers (SHAP Log-Odds Impact)",
        xaxis_title="SHAP Impact on Log-Odds (← Stay | Churn →)",
        yaxis_title="",
        height=310,
    )
    return fig


def make_model_comparison_chart(comp_df: pd.DataFrame) -> go.Figure:
    """Create interactive grouped bar chart of 5-fold CV ROC-AUC and F1 Score."""
    melted = comp_df.melt(
        id_vars="Model",
        value_vars=["CV ROC-AUC", "CV F1 Score", "CV Recall", "CV Precision"],
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_sequence=["#2563EB", "#059669", "#D97706", "#64748B"],
        title="5-Fold Stratified CV Performance Comparison",
    )
    fig.update_layout(**BASE_LAYOUT, yaxis_range=[0, 1.0])
    return fig


def make_threshold_tradeoff_chart(thr_df: pd.DataFrame, selected_thr: float) -> go.Figure:
    """Create interactive line chart of Precision, Recall, and F1 across thresholds."""
    fig = go.Figure()
    for col, name, color in [
        ("precision", "Precision", "#2563EB"),
        ("recall", "Recall", "#059669"),
        ("f1", "F1 Score", "#D97706"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=thr_df["threshold"],
                y=thr_df[col],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2.5),
            )
        )
    fig.add_vline(
        x=selected_thr,
        line_dash="dash",
        line_color=PALETTE["churn"],
        annotation_text=f"Selected ({selected_thr:.2f})",
        annotation_position="bottom right",
    )
    fig.update_layout(
        **BASE_LAYOUT,
        title="CV Threshold Trade-Off: Precision vs. Recall vs. F1",
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        yaxis_range=[0, 1.0],
    )
    return fig


def make_confusion_matrix_chart(cm: np.ndarray, title: str) -> go.Figure:
    """Create an annotated Plotly heatmap for the confusion matrix."""
    labels = ["No Churn (0)", "Churn (1)"]
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title=title,
    )
    fig.update_layout(
        **BASE_LAYOUT,
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        coloraxis_showscale=False,
    )
    return fig


def make_risk_distribution_chart(risk_df: pd.DataFrame) -> go.Figure:
    """Create a bar chart of model-scored customer risk tiers on the test cohort."""
    order = ["Low Risk", "Medium Risk", "High Risk"]
    counts = (
        risk_df["Risk Tier"]
        .value_counts()
        .reindex(order, fill_value=0)
        .reset_index()
    )
    counts.columns = ["Risk Tier", "Customers"]
    fig = px.bar(
        counts,
        x="Risk Tier",
        y="Customers",
        text="Customers",
        color="Risk Tier",
        color_discrete_map={
            "Low Risk": PALETTE["retain"],
            "Medium Risk": PALETTE["medium"],
            "High Risk": PALETTE["churn"],
        },
        title="Model Risk Tier Distribution (Test Cohort, n = 1,409)",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**BASE_LAYOUT, showlegend=False, yaxis_range=[0, max(counts["Customers"].max() * 1.18, 10)])
    return fig
