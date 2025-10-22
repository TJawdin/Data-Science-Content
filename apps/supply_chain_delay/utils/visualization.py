"""
Reusable Visualization Components (theme-aware & metadata-driven)
All risk thresholds & bands come from utils.constants → final_metadata.json
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.theme_adaptive import get_adaptive_colors, configure_plotly_figure
from utils.constants import OPTIMAL_THRESHOLD, RISK_BANDS, FRIENDLY_FEATURE_NAMES

# ============================================================================ #
# Risk Gauge (Speedometer) - METADATA DRIVEN
# ============================================================================ #
def create_risk_gauge(risk_score: int, risk_level: str) -> go.Figure:
    """
    Gauge for risk score (0..100). Regions:
      LOW:    0 .. low_max
      MEDIUM: low_max+1 .. med_max
      HIGH:   >= med_max  (starts at operating threshold)
    """
    colors = get_adaptive_colors()
    low_max = int(RISK_BANDS["low_max"])
    med_max = int(RISK_BANDS["med_max"])
    thr_pct = int(round(OPTIMAL_THRESHOLD * 100))

    bar_color = {
        "LOW": colors["low_risk"],
        "MEDIUM": colors["medium_risk"],
        "HIGH": colors["high_risk"],
    }.get(risk_level, colors["primary"])

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={
                "text": (
                    f"<b>Late Delivery Risk</b>"
                    f"<br><span style='font-size:0.8em'>{risk_level} RISK</span>"
                )
            },
            number={"font": {"size": 44}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color, "thickness": 0.75},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 2,
                "steps": [
                    {"range": [0, low_max], "color": "rgba(46,204,113,0.20)"},    # LOW
                    {"range": [low_max, med_max], "color": "rgba(243,156,18,0.20)"},  # MED
                    {"range": [med_max, 100], "color": "rgba(231,76,60,0.20)"},  # HIGH
                ],
                "threshold": {
                    "value": risk_score,
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.8,
                },
            },
        )
    )
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=80, b=20))
    # Visual markers for cut points & operating threshold
    fig.add_vline(x=low_max, line_dash="dash", line_color="green", line_width=2)
    fig.add_vline(x=med_max, line_dash="dash", line_color="red", line_width=2)
    fig.add_annotation(
        x=int(thr_pct), y=0.02, yref="paper",
        text=f"Operating Threshold: {thr_pct}%", showarrow=False
    )
    return fig

# ============================================================================ #
# SHAP Waterfall Chart (Feature Contribution)
# ============================================================================ #
def create_shap_waterfall(feature_contributions: dict, base_value: float, prediction_value: float) -> go.Figure:
    """
    Waterfall for top-10 absolute SHAP contributions.
    feature_contributions: {technical_feature: shap_value}
    """
    # Map technical -> friendly names if available
    fc_items = []
    for k, v in feature_contributions.items():
        fc_items.append((FRIENDLY_FEATURE_NAMES.get(k, k), float(v)))

    # Top-10 by |value|
    sorted_features = sorted(fc_items, key=lambda x: abs(x[1]), reverse=True)[:10]
    feature_names = [n for n, _ in sorted_features]
    contributions = [c for _, c in sorted_features]

    fig = go.Figure(
        go.Waterfall(
            name="Feature Contributions",
            orientation="v",
            measure=["relative"] * len(feature_names) + ["total"],
            x=feature_names + ["Final<br>Prediction"],
            text=[f"{c:+.3f}" for c in contributions] + [f"{prediction_value:.3f}"],
            y=contributions + [prediction_value - base_value],
            increasing={"marker": {"color": "#E74C3C"}},
            decreasing={"marker": {"color": "#2ECC71"}},
            totals={"marker": {"color": "#3498DB"}},
            connector={"line": {"color": "rgb(63,63,63)"}},
        )
    )
    fig.update_layout(
        title="🔍 Feature Contribution to Prediction (SHAP Waterfall)",
        title_font_size=16,
        showlegend=False,
        height=520,
        xaxis_title="Features",
        yaxis_title="SHAP Value (Impact on Prediction)",
        margin=dict(l=60, r=40, t=90, b=100),
    )
    fig.update_xaxes(tickangle=-45)
    return fig

# ============================================================================ #
# Feature Correlation Heatmap
# ============================================================================ #
def create_correlation_heatmap(features_df: pd.DataFrame, friendly: bool = True) -> go.Figure:
    """
    Correlation heatmap for a feature matrix.
    If friendly=True, labels use business-friendly names where available.
    """
    corr = features_df.corr(numeric_only=True)
    if friendly:
        corr.index = [FRIENDLY_FEATURE_NAMES.get(c, c) for c in corr.index]
        corr.columns = [FRIENDLY_FEATURE_NAMES.get(c, c) for c in corr.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu_r",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(
        title="📊 Feature Correlation Heatmap",
        title_font_size=16,
        height=780,
        margin=dict(l=220, r=40, t=90, b=160),
    )
    fig.update_xaxes(tickangle=-45)
    return fig

# ============================================================================ #
# Risk Distribution Histogram - METADATA DRIVEN
# ============================================================================ #
def create_risk_distribution(risk_scores) -> go.Figure:
    """
    Histogram of risk scores (0..100) with shaded LOW/MED/HIGH regions
    based on RISK_BANDS + OPTIMAL_THRESHOLD.
    """
    low_max = int(RISK_BANDS["low_max"])
    med_max = int(RISK_BANDS["med_max"])

    fig = go.Figure(
        go.Histogram(
            x=risk_scores,
            nbinsx=20,
            marker=dict(color="#3498DB", line=dict(color="white", width=1)),
            hovertemplate="Risk Score: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="📊 Risk Score Distribution",
        title_font_size=16,
        xaxis_title="Risk Score (0–100)",
        yaxis_title="Number of Orders",
        height=420,
        margin=dict(l=60, r=40, t=80, b=60),
        showlegend=False,
    )
    # Shaded regions & cut lines
    fig.add_vrect(x0=0, x1=low_max, fillcolor="green", opacity=0.08, line_width=0)
    fig.add_vrect(x0=low_max, x1=med_max, fillcolor="orange", opacity=0.08, line_width=0)
    fig.add_vrect(x0=med_max, x1=100, fillcolor="red", opacity=0.08, line_width=0)
    fig.add_vline(x=low_max, line_dash="dash", line_color="green", line_width=2,
                  annotation_text=f"LOW/MED ({low_max}%)", annotation_position="top right")
    fig.add_vline(x=med_max, line_dash="dash", line_color="red", line_width=2,
                  annotation_text=f"MED/HIGH ({med_max}%)", annotation_position="top left")
    return fig

# ============================================================================ #
# Feature Impact Bar Chart
# ============================================================================ #
def create_feature_impact_bars(feature_importance_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Horizontal bar chart of feature importance.
    Expects columns: ['Feature','Importance'] with technical names in 'Feature'.
    """
    # Map to friendly labels for display
    df = feature_importance_df.copy()
    df["Feature"] = df["Feature"].map(lambda c: FRIENDLY_FEATURE_NAMES.get(c, c))
    top_df = df.head(top_n)

    fig = go.Figure(
        go.Bar(
            x=top_df["Importance"],
            y=top_df["Feature"],
            orientation="h",
            marker=dict(color=top_df["Importance"], colorscale="Viridis", line=dict(color="white", width=1)),
            text=top_df["Importance"].round(4),
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"🎯 Top {top_n} Most Important Features",
        title_font_size=16,
        xaxis_title="Feature Importance",
        yaxis_title="",
        height=520,
        margin=dict(l=280, r=40, t=90, b=60),
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    return fig

# ============================================================================ #
# Brazil State Heat “Bar” (Top-N) – simple, no geojson dependency
# ============================================================================ #
def create_brazil_state_heatmap(state_late_rates: dict, top_n: int = 15) -> go.Figure:
    """
    Bar ranking of late-rate by Brazilian state (top N).
    state_late_rates: {state_code: late_rate_percent}
    """
    state_mapping = {
        'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas',
        'BA': 'Bahia', 'CE': 'Ceará', 'DF': 'Distrito Federal', 'ES': 'Espírito Santo',
        'GO': 'Goiás', 'MA': 'Maranhão', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul',
        'MG': 'Minas Gerais', 'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná',
        'PE': 'Pernambuco', 'PI': 'Piauí', 'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte',
        'RS': 'Rio Grande do Sul', 'RO': 'Rondônia', 'RR': 'Roraima', 'SC': 'Santa Catarina',
        'SP': 'São Paulo', 'SE': 'Sergipe', 'TO': 'Tocantins'
    }
    rows = []
    for code, rate in state_late_rates.items():
        rows.append({"State Code": code, "State Name": state_mapping.get(code, code), "Late Rate": float(rate)})

    df = pd.DataFrame(rows).sort_values("Late Rate", ascending=False).head(top_n)

    fig = px.bar(
        df, x="State Code", y="Late Rate", color="Late Rate",
        hover_data=["State Name"], color_continuous_scale=["green", "yellow", "red"],
        title=f"🗺️ Late Delivery Rates by Brazilian State (Top {top_n})",
        labels={"Late Rate": "Late Delivery Rate (%)"}
    )
    fig.update_layout(height=520, margin=dict(l=50, r=40, t=100, b=60))
    return fig

# ============================================================================ #
# Batch Results Dashboard
# ============================================================================ #
def create_batch_summary_dashboard(predictions_df: pd.DataFrame) -> go.Figure:
    """
    Multi-panel dashboard for batch results.
    Expects columns: ['Prediction','Risk_Score','Risk_Level'] and optionally 'Late_Probability'.
    """
    from plotly.subplots import make_subplots

    # Safety: normalize expected columns
    df = predictions_df.copy()
    if "Risk_Level" not in df.columns and "risk_level" in df.columns:
        df["Risk_Level"] = df["risk_level"].str.upper()

    # Subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "bar"}]],
        subplot_titles=("Prediction Distribution", "Risk Level Distribution",
                        "Risk Score Distribution", "Summary Stats")
    )

    # 1) Prediction pie
    pred_counts = df["Prediction"].value_counts(dropna=False)
    fig.add_trace(go.Pie(labels=pred_counts.index, values=pred_counts.values,
                         marker=dict(colors=["#2ECC71", "#E74C3C"])),
                  row=1, col=1)

    # 2) Risk level bar
    risk_counts = df["Risk_Level"].value_counts(dropna=False)
    fig.add_trace(go.Bar(x=risk_counts.index, y=risk_counts.values,
                         marker=dict(color=["#2ECC71", "#F39C12", "#E74C3C"])),
                  row=1, col=2)

    # 3) Risk score histogram
    fig.add_trace(go.Histogram(x=df["Risk_Score"], nbinsx=20,
                               marker=dict(color="#3498DB")),
                  row=2, col=1)

    # 4) Summary stats (bars)
    stats_vals = {
        "Avg Risk": float(np.mean(df["Risk_Score"])) if len(df) else 0.0,
        "≥ Threshold": int((df["Risk_Score"] >= int(round(OPTIMAL_THRESHOLD * 100))).sum()),
        "Total Orders": int(len(df)),
    }
    fig.add_trace(go.Bar(x=list(stats_vals.keys()), y=list(stats_vals.values()),
                         marker=dict(color=["#3498DB", "#E74C3C", "#9B59B6"])),
                  row=2, col=2)

    fig.update_layout(
        height=760,
        title_text="📊 Batch Prediction Summary Dashboard",
        title_font_size=18,
        showlegend=False,
        margin=dict(l=60, r=40, t=100, b=60),
    )
    return fig
