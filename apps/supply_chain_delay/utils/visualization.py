"""
Reusable Visualization Components
Interactive charts and maps for the Streamlit app
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from utils.theme_adaptive import get_adaptive_colors, configure_plotly_figure

# ============================================================================
# Risk Gauge (Speedometer) - FIXED TEXT CUTOFF
# ============================================================================

def create_risk_gauge(risk_score, risk_level):
    """
    Create an interactive gauge chart showing risk score
    """
    
    colors = get_adaptive_colors()
    
    # Color based on risk level
    if risk_level == 'LOW':
        bar_color = colors['low_risk']
    elif risk_level == 'MEDIUM':
        bar_color = colors['medium_risk']
    else:
        bar_color = colors['high_risk']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<b>Late Delivery Risk</b><br><span style='font-size:0.7em;color:{bar_color}'>{risk_level} RISK</span>", 
            'font': {'size': 20}
        },
        delta={'reference': 50, 'increasing': {'color': colors['high_risk']}, 'decreasing': {'color': colors['low_risk']}},
        number={'font': {'size': 50}},
        gauge={
            'axis': {
                'range': [None, 100], 
                'tickwidth': 2, 
                'tickcolor': "gray",
                'tickmode': 'linear',
                'tick0': 0,
                'dtick': 20
            },
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': colors['bg_transparent'],
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(46, 204, 113, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(243, 156, 18, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(231, 76, 60, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    # Apply adaptive configuration
    fig.update_layout(
        height=350,
        margin=dict(l=30, r=30, t=100, b=30),
        paper_bgcolor=colors['bg_transparent'],
        plot_bgcolor=colors['bg_transparent']
    )
    
    return fig


# ============================================================================
# SHAP Waterfall Chart (Feature Contribution)
# ============================================================================

def create_shap_waterfall(feature_contributions, base_value, prediction_value):
    """
    Create waterfall chart showing feature contributions to prediction
    
    Parameters:
    -----------
    feature_contributions : dict of {feature_name: contribution_value}
    base_value : float (baseline prediction)
    prediction_value : float (final prediction)
    """
    
    # Sort by absolute contribution
    sorted_features = sorted(
        feature_contributions.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )[:10]  # Top 10 features
    
    feature_names = [f[0] for f in sorted_features]
    contributions = [f[1] for f in sorted_features]
    
    # Build waterfall
    fig = go.Figure(go.Waterfall(
        name="Feature Contributions",
        orientation="v",
        measure=["relative"] * len(feature_names) + ["total"],
        x=feature_names + ["Final<br>Prediction"],
        textposition="outside",
        text=[f"{c:+.3f}" for c in contributions] + [f"{prediction_value:.3f}"],
        y=contributions + [prediction_value - base_value],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#E74C3C"}},
        decreasing={"marker": {"color": "#2ECC71"}},
        totals={"marker": {"color": "#3498DB"}}
    ))
    
    fig.update_layout(
        title="🔍 Feature Contribution to Prediction (SHAP Waterfall)",
        title_font_size=16,
        showlegend=False,
        height=500,
        xaxis_title="Features",
        yaxis_title="SHAP Value (Impact on Prediction)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=100, b=100)
    )
    
    fig.update_xaxes(tickangle=-45)
    
    return fig


# ============================================================================
# Feature Correlation Heatmap - FIXED CUTOFF
# ============================================================================

def create_correlation_heatmap(features_df, feature_names_mapping=None):
    """
    Create interactive correlation heatmap
    
    Parameters:
    -----------
    features_df : pd.DataFrame with features
    feature_names_mapping : dict mapping technical to friendly names
    """
    
    # Calculate correlation
    corr_matrix = features_df.corr()
    
    # Rename if mapping provided
    if feature_names_mapping:
        corr_matrix.index = [feature_names_mapping.get(f, f) for f in corr_matrix.index]
        corr_matrix.columns = [feature_names_mapping.get(f, f) for f in corr_matrix.columns]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="📊 Feature Correlation Heatmap",
        title_font_size=16,
        height=700,
        xaxis_title="",
        yaxis_title="",
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=200, r=50, t=100, b=150)
    )
    
    fig.update_xaxes(tickangle=-45)
    
    return fig


# ============================================================================
# Risk Distribution Histogram
# ============================================================================

def create_risk_distribution(risk_scores):
    """
    Create histogram showing distribution of risk scores
    
    Parameters:
    -----------
    risk_scores : list or array of risk scores (0-100)
    """
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=risk_scores,
        nbinsx=20,
        marker=dict(
            color=risk_scores,
            colorscale=[
                [0, '#2ECC71'],
                [0.3, '#F39C12'],
                [0.7, '#E74C3C']
            ],
            line=dict(color='white', width=1)
        ),
        hovertemplate='Risk Score: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title="📊 Risk Score Distribution",
        title_font_size=16,
        xaxis_title="Risk Score (0-100)",
        yaxis_title="Number of Orders",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add vertical lines for risk thresholds
    fig.add_vline(x=30, line_dash="dash", line_color="green", 
                  annotation_text="Low/Medium", annotation_position="top")
    fig.add_vline(x=70, line_dash="dash", line_color="red", 
                  annotation_text="Medium/High", annotation_position="top")
    
    return fig


# ============================================================================
# Feature Impact Bar Chart - FIXED CUTOFF
# ============================================================================

def create_feature_impact_bars(feature_importance_df, top_n=10):
    """
    Create horizontal bar chart of feature importance
    
    Parameters:
    -----------
    feature_importance_df : pd.DataFrame with 'Feature' and 'Importance' columns
    top_n : int, number of top features to show
    """
    
    top_features = feature_importance_df.head(top_n)
    
    fig = go.Figure(go.Bar(
        x=top_features['Importance'],
        y=top_features['Feature'],
        orientation='h',
        marker=dict(
            color=top_features['Importance'],
            colorscale='Viridis',
            line=dict(color='white', width=1)
        ),
        text=top_features['Importance'].round(4),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"🎯 Top {top_n} Most Important Features",
        title_font_size=16,
        xaxis_title="Feature Importance",
        yaxis_title="",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=250, r=50, t=80, b=50)
    )
    
    fig.update_yaxes(autorange="reversed")
    
    return fig


# ============================================================================
# Brazil State Heatmap (Choropleth)
# ============================================================================

def create_brazil_state_heatmap(state_late_rates):
    """
    Create choropleth map of Brazil showing late delivery rates by state
    
    Parameters:
    -----------
    state_late_rates : dict of {state_code: late_rate_percentage}
    """
    
    # Brazil state codes and full names
    state_mapping = {
        'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas',
        'BA': 'Bahia', 'CE': 'Ceará', 'DF': 'Distrito Federal', 'ES': 'Espírito Santo',
        'GO': 'Goiás', 'MA': 'Maranhão', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul',
        'MG': 'Minas Gerais', 'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná',
        'PE': 'Pernambuco', 'PI': 'Piauí', 'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte',
        'RS': 'Rio Grande do Sul', 'RO': 'Rondônia', 'RR': 'Roraima', 'SC': 'Santa Catarina',
        'SP': 'São Paulo', 'SE': 'Sergipe', 'TO': 'Tocantins'
    }
    
    # Create dataframe
    df = pd.DataFrame([
        {'State Code': code, 'State Name': state_mapping[code], 'Late Rate': rate}
        for code, rate in state_late_rates.items()
    ])
    
    fig = px.bar(
        df.sort_values('Late Rate', ascending=False).head(15),
        x='State Code',
        y='Late Rate',
        color='Late Rate',
        color_continuous_scale=['green', 'yellow', 'red'],
        title='🗺️ Late Delivery Rates by Brazilian State (Top 15)',
        hover_data=['State Name'],
        labels={'Late Rate': 'Late Delivery Rate (%)'}
    )
    
    fig.update_layout(
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font_size=16,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    return fig


# ============================================================================
# Batch Results Dashboard
# ============================================================================

def create_batch_summary_dashboard(predictions_df):
    """
    Create multi-chart dashboard for batch prediction results
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame with prediction results
    """
    
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Prediction Distribution',
            'Risk Level Distribution',
            'Risk Score Distribution',
            'Summary Statistics'
        ),
        specs=[
            [{'type': 'pie'}, {'type': 'bar'}],
            [{'type': 'histogram'}, {'type': 'bar'}]
        ]
    )
    
    # 1. Prediction distribution (pie)
    pred_counts = predictions_df['Prediction'].value_counts()
    fig.add_trace(
        go.Pie(labels=pred_counts.index, values=pred_counts.values, 
               marker=dict(colors=['#2ECC71', '#E74C3C'])),
        row=1, col=1
    )
    
    # 2. Risk level distribution (bar)
    risk_counts = predictions_df['Risk_Level'].value_counts()
    fig.add_trace(
        go.Bar(x=risk_counts.index, y=risk_counts.values,
               marker=dict(color=['#2ECC71', '#F39C12', '#E74C3C'])),
        row=1, col=2
    )
    
    # 3. Risk score histogram
    fig.add_trace(
        go.Histogram(x=predictions_df['Risk_Score'], nbinsx=20,
                     marker=dict(color='#3498DB')),
        row=2, col=1
    )
    
    # 4. Placeholder
    fig.add_trace(
        go.Bar(x=['Stats'], y=[0]),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="📊 Batch Prediction Summary Dashboard",
        title_font_size=18,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    return fig
