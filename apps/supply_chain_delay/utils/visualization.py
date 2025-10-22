"""
Reusable Visualization Components
Interactive charts and maps for the Streamlit app
"""

from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from utils.theme_adaptive import get_adaptive_colors, configure_plotly_figure
from utils.constants import load_runtime_thresholds

# Load runtime thresholds & bands (from final_metadata.json if present)
_THRESH = load_runtime_thresholds()
_OPT_THR_PCT = _THRESH["THRESHOLD_PCT"]  # e.g., 19.66
_LOW_MAX = _THRESH["LOW_MAX"]            # e.g., 12
_MED_MAX = _THRESH["MED_MAX"]            # e.g., 30


# ----------------------------- Risk Gauge ---------------------------------- #

def create_risk_gauge(risk_score: float, risk_level: str):
    """
    Risk gauge aligned with current notebook cut-points:
      LOW:    0–(_LOW_MAX-1)%
      MEDIUM: _LOW_MAX–(_MED_MAX-1)%
      HIGH:   _MED_MAX%+
    """
    colors = get_adaptive_colors()
    bar_color = (
        colors["low_risk"] if risk_level == "LOW"
        else colors["medium_risk"] if risk_level == "MEDIUM"
        else colors["high_risk"]
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(risk_score),
        domain={"x": [0, 1], "y": [0, 1]},
        title={
            "text": (
                f"<b>Late Delivery Risk</b><br>"
                f"<span style='font-size:0.8em;color:{bar_color}'>{risk_level} RISK</span>"
            ),
            "font": {"size": 20},
        },
        number={"font": {"size": 44}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 2,
                "tickcolor": "gray",
                "tickmode": "array",
                "tickvals": [0, _LOW_MAX, _MED_MAX, 50, 75, 100],
                "ticktext": ["0", str(_LOW_MAX), str(_MED_MAX), "50", "75", "100"],
            },
            "bar": {"color": bar_color, "thickness": 0.75},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, _LOW_MAX], "color": "rgba(46, 204, 113, 0.18)"},
                {"range": [_LOW_MAX, _MED_MAX], "color": "rgba(243, 156, 18, 0.18)"},
                {"range": [_MED_MAX, 100], "color": "rgba(231, 76, 60, 0.18)"},
            ],
            "threshold": {
                "line": {"color": "#000", "width": 3},
                "thickness": 0.8,
                "value": _OPT_THR_PCT,
            },
        },
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=30, r=30, t=100, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ------------------------- SHAP Waterfall (optional) ----------------------- #

def create_shap_waterfall(feature_contributions: dict, base_value: float, prediction_value: float):
    colors = get_adaptive_colors()
    top = sorted(feature_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    names = [k for k, _ in top]
    contribs = [v for _, v in top]
    total_delta = prediction_value - base_value

    fig = go.Figure(go.Waterfall(
        name="Feature Contributions",
        orientation="v",
        measure=["relative"] * len(names) + ["total"],
        x=names + ["Final<br>Prediction"],
        textposition="outside",
        text=[f"{c:+.3f}" for c in contribs] + [f"{prediction_value:.3f}"],
        y=contribs + [total_delta],
        connector={"line": {"color": "rgba(99,99,99,0.6)"}},
        increasing={"marker": {"color": colors["high_risk"]}},
        decreasing={"marker": {"color": colors["low_risk"]}},
        totals={"marker": {"color": colors["primary"]}},
    ))

    fig.update_layout(
        title="🔍 Feature Contribution to Prediction (SHAP Waterfall)",
        title_font_size=16,
        showlegend=False,
        height=520,
        xaxis_title="Features",
        yaxis_title="SHAP Value (Impact on Prediction)",
        margin=dict(l=50, r=50, t=90, b=90),
    )
    fig.update_xaxes(tickangle=-45)
    return configure_plotly_figure(fig)


# ------------------------ Correlation Heatmap ------------------------------ #

def create_correlation_heatmap(features_df: pd.DataFrame, feature_names_mapping: dict | None = None):
    corr = features_df.corr(numeric_only=True)
    if feature_names_mapping:
        corr.index = [feature_names_mapping.get(c, c) for c in corr.index]
        corr.columns = [feature_names_mapping.get(c, c) for c in corr.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmid=0,
        text=corr.values, texttemplate="%{text:.2f}", textfont={"size": 8},
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(
        title="📊 Feature Correlation Heatmap",
        title_font_size=16,
        height=720,
        margin=dict(l=200, r=50, t=90, b=150),
    )
    fig.update_xaxes(tickangle=-45)
    return configure_plotly_figure(fig)


# ---------------------- Risk Score Distribution ---------------------------- #

def create_risk_distribution(risk_scores: list | np.ndarray):
    colors = get_adaptive_colors()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=risk_scores, nbinsx=25,
        marker=dict(color=colors["primary"], line=dict(color="white", width=1)),
        hovertemplate="Risk Score: %{x}<br>Count: %{y}<extra></extra>",
        name="Risk Scores", opacity=0.95,
    ))

    fig.add_vline(x=_LOW_MAX, line_dash="dash", line_color=colors["low_risk"], line_width=2,
                  annotation_text=f"LOW→MED ({_LOW_MAX}%)", annotation_position="top right")
    fig.add_vline(x=_MED_MAX, line_dash="dash", line_color=colors["high_risk"], line_width=2,
                  annotation_text=f"MED→HIGH ({_MED_MAX}%)", annotation_position="top left")
    fig.add_vline(x=_OPT_THR_PCT, line_dash="dot", line_color="#000", line_width=2,
                  annotation_text=f"Model Thr ({_OPT_THR_PCT:.2f}%)", annotation_position="top")

    fig.add_vrect(x0=0, x1=_LOW_MAX, fillcolor=colors["low_risk"], opacity=0.10, line_width=0)
    fig.add_vrect(x0=_LOW_MAX, x1=_MED_MAX, fillcolor=colors["medium_risk"], opacity=0.10, line_width=0)
    fig.add_vrect(x0=_MED_MAX, x1=100, fillcolor=colors["high_risk"], opacity=0.10, line_width=0)

    fig.update_layout(
        title="📊 Risk Score Distribution",
        title_font_size=16,
        xaxis_title="Risk Score (0–100)",
        yaxis_title="Number of Orders",
        height=420,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False,
    )
    return configure_plotly_figure(fig)


# --------------------- Feature Importance Bars ----------------------------- #

def create_feature_impact_bars(feature_importance_df: pd.DataFrame, top_n: int = 10):
    colors = get_adaptive_colors()
    top = feature_importance_df.head(top_n)

    fig = go.Figure(go.Bar(
        x=top["Importance"], y=top["Feature"], orientation="h",
        marker=dict(color=top["Importance"], colorscale="Viridis", line=dict(color="white", width=1)),
        text=top["Importance"].round(4), textposition="auto",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        name="Importance",
    ))
    fig.update_layout(
        title=f"🎯 Top {top_n} Most Important Features",
        title_font_size=16,
        xaxis_title="Importance",
        yaxis_title="",
        height=520,
        margin=dict(l=250, r=50, t=80, b=50),
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    return configure_plotly_figure(fig)


# ----------------------- Batch Summary Dashboard --------------------------- #

def create_batch_summary_dashboard(predictions_df: pd.DataFrame):
    """
    Expects columns: Prediction, Late_Probability, Risk_Score, risk_level
    """
    from plotly.subplots import make_subplots
    colors = get_adaptive_colors()

    df = predictions_df.copy()
    if "risk_level" not in df.columns:
        def _lvl(s):
            return "LOW" if s < _LOW_MAX else ("MEDIUM" if s < _MED_MAX else "HIGH")
        df["risk_level"] = df["Risk_Score"].apply(_lvl)

    ordered_levels = ["LOW", "MEDIUM", "HIGH"]
    risk_counts = df["risk_level"].value_counts().reindex(ordered_levels).fillna(0)
    pred_counts = df["Prediction"].value_counts().reindex(["On-Time", "Late"]).fillna(0)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Prediction Distribution", "Risk Level Distribution",
                        "Risk Score Distribution", "Summary (avg prob by level)"),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "bar"}]],
        horizontal_spacing=0.12, vertical_spacing=0.15,
    )

    fig.add_trace(go.Pie(labels=pred_counts.index.tolist(),
                         values=pred_counts.values.tolist(),
                         marker=dict(colors=[colors["low_risk"], colors["high_risk"]]),
                         hole=0.35, sort=False, showlegend=False),
                  row=1, col=1)

    fig.add_trace(go.Bar(x=risk_counts.index.tolist(),
                         y=risk_counts.values.tolist(),
                         marker=dict(color=[colors["low_risk"], colors["medium_risk"], colors["high_risk"]]),
                         text=risk_counts.values.tolist(), textposition="auto", showlegend=False),
                  row=1, col=2)

    fig.add_trace(go.Histogram(x=df["Risk_Score"], nbinsx=25,
                               marker=dict(color=colors["primary"]), showlegend=False),
                  row=2, col=1)

    if "Late_Probability" in df.columns:
        avg = df.groupby("risk_level")["Late_Probability"].mean().reindex(ordered_levels).fillna(0.0)
        fig.add_trace(go.Bar(x=avg.index.tolist(),
                             y=(avg.values * 100).round(1),
                             marker=dict(color=[colors["low_risk"], colors["medium_risk"], colors["high_risk"]]),
                             text=[f"{v:.1f}%" for v in (avg.values * 100)],
                             textposition="auto", showlegend=False),
                      row=2, col=2)
    else:
        fig.add_trace(go.Bar(x=["LOW", "MEDIUM", "HIGH"], y=[0, 0, 0], showlegend=False),
                      row=2, col=2)

    fig.update_layout(
        height=760,
        title_text="📊 Batch Prediction Summary Dashboard",
        title_font_size=18,
        margin=dict(l=50, r=50, t=90, b=50),
    )
    fig.update_xaxes(title_text="Risk Level", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Risk Score (0–100)", row=2, col=1)
    fig.update_yaxes(title_text="Orders", row=2, col=1)
    fig.update_xaxes(title_text="Risk Level", row=2, col=2)
    fig.update_yaxes(title_text="Avg Late Probability (%)", row=2, col=2)

    return configure_plotly_figure(fig)
