# utils/visualization.py
# Purpose: Reusable, theme-aware visualizations that read thresholds & bands
#          from metadata (via utils.model_loader.load_metadata).
#
# References (latest docs):
# - Python: https://docs.python.org/3/
# - Pandas: https://pandas.pydata.org/docs/
# - Plotly: https://plotly.com/python/

from __future__ import annotations  # postpone annotations for clarity
from typing import Dict, Iterable, Optional  # type hints for better DX
import numpy as np                           # numeric utilities
import pandas as pd                          # tabular utilities
import plotly.graph_objects as go            # low-level plotly API
import plotly.express as px                  # convenience charts

# Theme helpers (already in your repo)
from utils.theme_adaptive import get_adaptive_colors, configure_plotly_figure  # adaptive palette + layout tweaks
# Metadata loader (replaces old utils.constants)
from utils.model_loader import load_metadata  # read final_metadata.json


# =============================================================================
# Internal helpers
# =============================================================================

def _bands_meta() -> Dict[str, int]:
    """Read risk bands (percent cutpoints) from metadata with safe defaults."""
    meta = load_metadata()                                 # load metadata dict
    rb = meta.get("risk_bands", {})                        # extract bands
    return {"low_max": int(rb.get("low_max", 30)),         # default 30
            "med_max": int(rb.get("med_max", 67))}         # default 67

def _threshold_meta() -> float:
    """Read operating threshold (0..1) from metadata with a safe default."""
    meta = load_metadata()                                 # load metadata dict
    return float(meta.get("optimal_threshold", 0.5))       # default 0.50

def _friendly_map_or_identity(mapping: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Return mapping or an identity map if None (handled lazily in callers)."""
    return mapping or {}                                   # empty dict = identity in .get(name, name)


# =============================================================================
# 1) Risk Gauge (Speedometer) — METADATA DRIVEN
# =============================================================================

def create_risk_gauge(prob_0_1: float, risk_level: str) -> go.Figure:
    """
    Speedometer gauge for a single predicted probability (0..1).

    Regions are shaded by current metadata bands (LOW/MED/HIGH).
    - prob_0_1: model probability (0..1)
    - risk_level: label ("Low", "Medium", "High") used for the bar color
    """
    colors = get_adaptive_colors()                         # theme palette
    bands = _bands_meta()                                  # {'low_max': int, 'med_max': int}
    thr_pct = int(round(_threshold_meta() * 100))          # threshold in percent
    low_max, med_max = int(bands["low_max"]), int(bands["med_max"])  # cut points

    score_100 = max(0, min(100, int(round(float(prob_0_1) * 100))))  # clamp to 0..100

    # Map risk level to bar color (theme-consistent)
    level = str(risk_level).strip().upper()                # normalize for mapping
    bar_color = {
        "LOW": colors["low_risk"],
        "MEDIUM": colors["medium_risk"],
        "HIGH": colors["high_risk"],
    }.get(level, colors["primary"])

    # Build the gauge
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",                           # gauge with numeric display
            value=score_100,                               # 0..100
            title={"text": "<b>Late Delivery Risk</b>"},   # title HTML allowed
            number={"font": {"size": 42}},                 # big number
            gauge={
                "axis": {"range": [0, 100]},               # axis range
                "bar": {"color": bar_color, "thickness": 0.75},  # colored bar
                "bgcolor": "rgba(0,0,0,0)",                # transparent background
                "borderwidth": 2,                          # subtle border
                "steps": [                                  # shaded regions for bands
                    {"range": [0, low_max], "color": "rgba(46,204,113,0.20)"},
                    {"range": [low_max, med_max], "color": "rgba(243,156,18,0.20)"},
                    {"range": [med_max, 100], "color": "rgba(231,76,60,0.20)"},
                ],
                "threshold": {                              # pointer line at current score
                    "value": score_100,
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.8,
                },
            },
        )
    )
    # Layout & annotations
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=60, b=20))  # compact layout
    fig.add_vline(x=low_max, line_dash="dash", line_color="green", line_width=2)  # LOW/MED cut
    fig.add_vline(x=med_max, line_dash="dash", line_color="red", line_width=2)    # MED/HIGH cut
    fig.add_annotation(
        x=thr_pct, y=0.02, yref="paper", text=f"Operating Threshold: {thr_pct}%", showarrow=False  # threshold note
    )
    return fig  # return figure object


# =============================================================================
# 2) SHAP Waterfall — Feature Contribution (optional friendly names)
# =============================================================================

def create_shap_waterfall(
    feature_contributions: Dict[str, float],
    base_value: float,
    prediction_value: float,
    friendly_map: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Waterfall for top-10 absolute SHAP contributions.
    - feature_contributions: {technical_feature: shap_value}
    - base_value: model base value (e.g., expected value)
    - prediction_value: final model output for the row
    - friendly_map: optional mapping tech→business labels
    """
    fmap = _friendly_map_or_identity(friendly_map)         # mapping or identity

    # Convert to (friendly_name, shap_val) and sort by |impact|
    pairs = [(fmap.get(k, k), float(v)) for k, v in feature_contributions.items()]  # apply mapping
    pairs = sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)[:10]             # top 10 by |SHAP|
    feature_names = [p[0] for p in pairs]                   # names for x-axis
    contributions = [p[1] for p in pairs]                   # SHAP values

    # Waterfall specification
    fig = go.Figure(
        go.Waterfall(
            name="Feature Contributions",                   # series name
            orientation="v",                                # vertical waterfall
            measure=["relative"] * len(feature_names) + ["total"],  # relative steps + final total
            x=feature_names + ["Final<br>Prediction"],      # categories + final label
            text=[f"{c:+.3f}" for c in contributions] + [f"{prediction_value:.3f}"],  # tooltips
            y=contributions + [prediction_value - base_value],  # step sizes and total delta
            increasing={"marker": {"color": "#E74C3C"}},    # positive impact color
            decreasing={"marker": {"color": "#2ECC71"}},    # negative impact color
            totals={"marker": {"color": "#3498DB"}},        # final bar color
            connector={"line": {"color": "rgb(63,63,63)"}}, # connectors
        )
    )
    fig.update_layout(
        title="🔍 Feature Contribution to Prediction (SHAP Waterfall)",  # chart title
        title_font_size=16,                        # title size
        showlegend=False,                          # hide legend
        height=520,                                # chart height
        xaxis_title="Features",                    # x-label
        yaxis_title="SHAP Value (Impact on Prediction)",  # y-label
        margin=dict(l=60, r=40, t=90, b=100),      # margins
    )
    fig.update_xaxes(tickangle=-45)                # rotate labels for readability
    return fig                                     # return figure


# =============================================================================
# 3) Feature Correlation Heatmap (friendly labels optional)
# =============================================================================

def create_correlation_heatmap(features_df: pd.DataFrame, friendly_map: Optional[Dict[str, str]] = None) -> go.Figure:
    """
    Correlation heatmap for a matrix of features.
    - features_df: numeric (and/or mixed) DataFrame
    - friendly_map: optional mapping tech→business labels
    """
    fmap = _friendly_map_or_identity(friendly_map)         # mapping or identity
    corr = features_df.corr(numeric_only=True)             # numeric-only correlation

    # Apply friendly labels if provided
    corr.index = [fmap.get(c, c) for c in corr.index]      # map row labels
    corr.columns = [fmap.get(c, c) for c in corr.columns]  # map col labels

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,                                 # correlation matrix values
            x=corr.columns,                                # x labels
            y=corr.index,                                  # y labels
            colorscale="RdBu_r",                           # diverging palette
            zmid=0,                                        # center at 0
            text=corr.round(2).values,                     # show numbers
            texttemplate="%{text}",                        # render numbers
            textfont={"size": 9},                          # font size
            colorbar=dict(title="Correlation"),            # colorbar label
        )
    )
    fig.update_layout(
        title="📊 Feature Correlation Heatmap",             # title
        title_font_size=16,                                 # font size
        height=780,                                         # height
        margin=dict(l=220, r=40, t=90, b=160),              # margins
    )
    fig.update_xaxes(tickangle=-45)                         # rotate x labels
    return fig                                              # return figure


# =============================================================================
# 4) Risk Distribution Histogram — METADATA DRIVEN
# =============================================================================

def create_risk_distribution(probabilities_0_1: Iterable[float]) -> go.Figure:
    """
    Histogram of predicted probabilities (0..1), displayed in 0..100 with
    shaded LOW/MED/HIGH regions from metadata bands.
    """
    bands = _bands_meta()                                   # get band cutpoints (%)
    low_max, med_max = int(bands["low_max"]), int(bands["med_max"])  # cut points

    # Convert to 0..100 integers for nicer bins
    probs = np.asarray(list(probabilities_0_1), dtype=float)         # ensure array
    scores_100 = np.clip((probs * 100.0).round().astype(int), 0, 100)  # clamp to [0,100]

    fig = go.Figure(
        go.Histogram(
            x=scores_100,                                  # histogram over 0..100
            nbinsx=20,                                     # bin count
            marker=dict(color="#3498DB", line=dict(color="white", width=1)),  # style
            hovertemplate="Risk Score: %{x}<br>Count: %{y}<extra></extra>",   # hover format
        )
    )
    fig.update_layout(
        title="📊 Risk Score Distribution",                 # chart title
        title_font_size=16,                                 # font size
        xaxis_title="Risk Score (0–100)",                   # x label
        yaxis_title="Number of Orders",                     # y label
        height=420,                                         # height
        margin=dict(l=60, r=40, t=80, b=60),                # margins
        showlegend=False,                                   # no legend
    )
    # Shade band regions
    fig.add_vrect(x0=0, x1=low_max, fillcolor="green", opacity=0.08, line_width=0)        # LOW region
    fig.add_vrect(x0=low_max, x1=med_max, fillcolor="orange", opacity=0.08, line_width=0) # MED region
    fig.add_vrect(x0=med_max, x1=100, fillcolor="red", opacity=0.08, line_width=0)        # HIGH region
    # Cut lines
    fig.add_vline(x=low_max, line_dash="dash", line_color="green", line_width=2,
                  annotation_text=f"LOW/MED ({low_max}%)", annotation_position="top right")
    fig.add_vline(x=med_max, line_dash="dash", line_color="red", line_width=2,
                  annotation_text=f"MED/HIGH ({med_max}%)", annotation_position="top left")
    return fig                                             # return figure


# =============================================================================
# 5) Feature Importance Bars (friendly labels optional)
# =============================================================================

def create_feature_impact_bars(
    feature_importance_df: pd.DataFrame,
    top_n: int = 10,
    friendly_map: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Horizontal bar chart for feature importance.
    Expects columns: ['Feature','Importance'] with technical names in 'Feature'.
    - friendly_map: optional mapping tech→business labels
    """
    fmap = _friendly_map_or_identity(friendly_map)          # mapping or identity
    df = feature_importance_df.copy()                       # copy to avoid mutation
    df["Feature"] = df["Feature"].map(lambda c: fmap.get(c, c))  # map labels
    top_df = df.head(int(top_n))                            # take top N rows

    fig = go.Figure(
        go.Bar(
            x=top_df["Importance"],                         # importance values
            y=top_df["Feature"],                            # feature labels
            orientation="h",                                # horizontal bars
            marker=dict(color=top_df["Importance"], colorscale="Viridis",
                        line=dict(color="white", width=1)), # styled markers
            text=top_df["Importance"].round(4),             # show values on bars
            textposition="auto",                            # auto placement
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",  # hover format
        )
    )
    fig.update_layout(
        title=f"🎯 Top {int(top_n)} Most Important Features",  # dynamic title
        title_font_size=16,                                  # font size
        xaxis_title="Feature Importance",                    # x label
        yaxis_title="",                                      # no y-axis title
        height=520,                                          # chart height
        margin=dict(l=280, r=40, t=90, b=60),                # margins
        showlegend=False,                                    # no legend
    )
    fig.update_yaxes(autorange="reversed")                   # highest at top
    return fig                                               # return figure


# =============================================================================
# 6) Brazil State Heat “Bar” (Top-N) — simple, no geojson dependency
# =============================================================================

def create_brazil_state_heatmap(
    state_late_rates: Dict[str, float],
    top_n: int = 15
) -> go.Figure:
    """
    Bar ranking of late-rate by Brazilian state (top N).
    state_late_rates: {state_code: late_rate_percent (0..100)}
    """
    # Mapping of state codes to names
    state_mapping = {
        'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas',
        'BA': 'Bahia', 'CE': 'Ceará', 'DF': 'Distrito Federal', 'ES': 'Espírito Santo',
        'GO': 'Goiás', 'MA': 'Maranhão', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul',
        'MG': 'Minas Gerais', 'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná',
        'PE': 'Pernambuco', 'PI': 'Piauí', 'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte',
        'RS': 'Rio Grande do Sul', 'RO': 'Rondônia', 'RR': 'Roraima', 'SC': 'Santa Catarina',
        'SP': 'São Paulo', 'SE': 'Sergipe', 'TO': 'Tocantins'
    }
    # Build rows DataFrame
    rows = [{"State Code": code,
             "State Name": state_mapping.get(code, code),
             "Late Rate": float(rate)} for code, rate in state_late_rates.items()]
    df = pd.DataFrame(rows).sort_values("Late Rate", ascending=False).head(int(top_n))  # sort & trim

    # Bar chart (no geojson dependency)
    fig = px.bar(
        df, x="State Code", y="Late Rate", color="Late Rate",
        hover_data=["State Name"], color_continuous_scale=["green", "yellow", "red"],
        title=f"🗺️ Late Delivery Rates by Brazilian State (Top {int(top_n)})",
        labels={"Late Rate": "Late Delivery Rate (%)"}
    )
    fig.update_layout(height=520, margin=dict(l=50, r=40, t=100, b=60))  # layout
    return fig  # return figure


# =============================================================================
# 7) Batch Results Dashboard (assumes modern columns: score, risk_band, meets_threshold)
# =============================================================================

def create_batch_summary_dashboard(predictions_df: pd.DataFrame) -> go.Figure:
    """
    Multi-panel dashboard for batch results.
    Expects modern columns: ['score','risk_band','meets_threshold'].
    We compute distributions and summary stats from these.
    """
    from plotly.subplots import make_subplots                      # import subplots

    thr = _threshold_meta()                                        # threshold 0..1
    df = predictions_df.copy()                                     # work on a copy

    # Normalize columns (defensive)
    if "risk_band" in df.columns:
        df["risk_band"] = df["risk_band"].astype(str).str.title()  # "Low/Medium/High"
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce").clip(0, 1)  # 0..1

    # Panels: (1) meets vs not, (2) band distribution, (3) score histogram, (4) summary bars
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "bar"}]],
        subplot_titles=("≥ Threshold vs < Threshold", "Risk Band Distribution",
                        "Score Distribution", "Summary Stats")
    )

    # (1) Threshold pie
    meets = int(df.get("meets_threshold", pd.Series([], dtype=bool)).sum()) if "meets_threshold" in df.columns else 0
    total = int(len(df))
    not_meets = max(0, total - meets)
    fig.add_trace(
        go.Pie(labels=["≥ Threshold", "< Threshold"], values=[meets, not_meets],
               marker=dict(colors=["#E74C3C", "#2ECC71"])),  # red for high-risk share
        row=1, col=1
    )

    # (2) Risk band bar
    if "risk_band" in df.columns:
        band_counts = df["risk_band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)
        fig.add_trace(
            go.Bar(x=band_counts.index.tolist(), y=band_counts.values.tolist(),
                   marker=dict(color=["#2ECC71", "#F39C12", "#E74C3C"])),
            row=1, col=2
        )

    # (3) Score histogram (0..1)
    fig.add_trace(
        go.Histogram(x=df["score"] if "score" in df.columns else [], nbinsx=20,
                     marker=dict(color="#3498DB")),
        row=2, col=1
    )

    # (4) Summary stats
    avg_score = float(df["score"].mean()) if "score" in df.columns and total > 0 else 0.0
    stats_vals = {
        "Avg Prob": avg_score,
        "≥ Threshold": meets,
        "Total Rows": total,
        "Threshold": thr,
    }
    fig.add_trace(
        go.Bar(x=list(stats_vals.keys()), y=list(stats_vals.values()),
               marker=dict(color=["#3498DB", "#E74C3C", "#9B59B6", "#95A5A6"])),
        row=2, col=2
    )

    # Layout polish
    fig.update_layout(
        height=760, title_text="📊 Batch Prediction Summary Dashboard",
        title_font_size=18, showlegend=False,
        margin=dict(l=60, r=40, t=100, b=60),
    )
    return fig
