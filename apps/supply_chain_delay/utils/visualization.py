"""
Visualization Module
Creates all charts and plots for the application
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from .theme_adaptive import get_risk_color, format_probability


def plot_risk_gauge(probability, threshold):
    """
    Create an animated risk gauge chart
    
    Args:
        probability: Delay probability (0-1)
        threshold: Classification threshold
    
    Returns:
        plotly figure
    """
    prob_pct = probability * 100
    
    # Determine risk level and color
    if prob_pct <= 30:
        risk_level = "Low"
        color = "#28a745"
    elif prob_pct <= 67:
        risk_level = "Medium"
        color = "#ffc107"
    else:
        risk_level = "High"
        color = "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Delay Risk: {risk_level}", 'font': {'size': 24}},
        delta={'reference': threshold * 100, 'suffix': '%'},
        number={'suffix': '%', 'font': {'size': 48}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 67], 'color': '#fff3cd'},
                {'range': [67, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        font={'size': 14}
    )
    
    return fig


def plot_feature_importance(feature_names, feature_values, top_n=10):
    """
    Create horizontal bar chart of feature importance
    
    Args:
        feature_names: List of feature names
        feature_values: List of importance values
        top_n: Number of top features to show
    
    Returns:
        plotly figure
    """
    # Create DataFrame and sort
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_values
    })
    df = df.nlargest(top_n, 'Importance')
    df = df.sort_values('Importance', ascending=True)
    
    # Create plot
    fig = go.Figure(go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h',
        marker=dict(
            color=df['Importance'],
            colorscale='Reds',
            showscale=False
        ),
        text=df['Importance'].round(3),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Most Important Features",
        xaxis_title="Importance Score",
        yaxis_title="",
        height=max(400, top_n * 40),
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


def plot_shap_waterfall(shap_values, feature_names, feature_values, base_value, max_display=10):
    """
    Create SHAP waterfall plot showing feature contributions
    
    Args:
        shap_values: SHAP values array
        feature_names: Feature names
        feature_values: Feature values
        base_value: Model's base prediction value
        max_display: Maximum features to display
    
    Returns:
        plotly figure
    """
    # Get top features by absolute SHAP value
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-max_display:][::-1]
    
    # Prepare data
    names = [feature_names[i] for i in top_indices]
    values = [shap_values[i] for i in top_indices]
    feat_vals = [feature_values[i] for i in top_indices]
    
    # Calculate cumulative values for waterfall
    cumulative = [base_value]
    for val in values:
        cumulative.append(cumulative[-1] + val)
    
    # Create colors
    colors = ['red' if v < 0 else 'green' for v in values]
    
    # Create plot
    fig = go.Figure()
    
    # Add base value
    fig.add_trace(go.Bar(
        x=[base_value],
        y=['Base Value'],
        orientation='h',
        marker=dict(color='lightgray'),
        text=[f'{base_value:.3f}'],
        textposition='outside',
        name='Base Value',
        hovertemplate='Base Value: %{x:.4f}<extra></extra>'
    ))
    
    # Add feature contributions
    for i, (name, val, feat_val, color) in enumerate(zip(names, values, feat_vals, colors)):
        fig.add_trace(go.Bar(
            x=[val],
            y=[f'{name}<br>= {feat_val:.2f}'],
            orientation='h',
            marker=dict(color=color),
            text=[f'{val:+.3f}'],
            textposition='outside',
            name=name,
            hovertemplate=f'<b>{name}</b><br>Value: {feat_val:.2f}<br>SHAP: {val:+.4f}<extra></extra>'
        ))
    
    # Add final prediction
    final_pred = cumulative[-1]
    fig.add_trace(go.Bar(
        x=[final_pred],
        y=['Final Prediction'],
        orientation='h',
        marker=dict(color='darkblue'),
        text=[f'{final_pred:.3f}'],
        textposition='outside',
        name='Final Prediction',
        hovertemplate='Final Prediction: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="SHAP Waterfall: Feature Contributions to Prediction",
        xaxis_title="SHAP Value",
        yaxis_title="",
        height=max(500, (max_display + 2) * 40),
        showlegend=False,
        barmode='overlay',
        margin=dict(l=20, r=100, t=60, b=40)
    )
    
    return fig


def plot_probability_distribution(probabilities, threshold, bins=50):
    """
    Plot distribution of delay probabilities
    
    Args:
        probabilities: Array of probabilities
        threshold: Classification threshold
        bins: Number of histogram bins
    
    Returns:
        plotly figure
    """
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=probabilities * 100,
        nbinsx=bins,
        name='Probability Distribution',
        marker=dict(
            color='lightblue',
            line=dict(color='darkblue', width=1)
        ),
        hovertemplate='Probability: %{x:.1f}%<br>Count: %{y}<extra></extra>'
    ))
    
    # Add threshold line
    fig.add_vline(
        x=threshold * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold*100:.1f}%",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Distribution of Delay Probabilities",
        xaxis_title="Delay Probability (%)",
        yaxis_title="Count",
        height=400,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def plot_risk_distribution(risk_levels):
    """
    Create pie chart of risk level distribution
    
    Args:
        risk_levels: List of risk levels ('Low', 'Medium', 'High')
    
    Returns:
        plotly figure
    """
    # Count risk levels
    risk_counts = pd.Series(risk_levels).value_counts()
    
    # Define colors
    colors_map = {
        'Low': '#28a745',
        'Medium': '#ffc107',
        'High': '#dc3545'
    }
    colors = [colors_map.get(level, 'gray') for level in risk_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        marker=dict(colors=colors),
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Risk Level Distribution",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def plot_geographic_map(df, probability_col='delay_probability', location_col='customer_state'):
    """
    Create choropleth map of Brazil showing delay probabilities by state
    
    Args:
        df: DataFrame with geographic data
        probability_col: Column with probabilities
        location_col: Column with state codes
    
    Returns:
        plotly figure
    """
    # Aggregate by state
    state_data = df.groupby(location_col)[probability_col].agg(['mean', 'count']).reset_index()
    state_data.columns = ['state', 'avg_delay_prob', 'count']
    state_data['avg_delay_prob_pct'] = state_data['avg_delay_prob'] * 100
    
    # Create map
    fig = px.choropleth(
        state_data,
        locations='state',
        locationmode='geojson-id',
        color='avg_delay_prob_pct',
        hover_data={'state': True, 'avg_delay_prob_pct': ':.1f', 'count': True},
        color_continuous_scale='Reds',
        range_color=[0, 100],
        labels={'avg_delay_prob_pct': 'Avg Delay %', 'count': 'Orders'}
    )
    
    fig.update_layout(
        title="Average Delay Probability by State",
        height=600,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    return fig


def plot_time_series(df, date_col, value_col, title="Time Series"):
    """
    Create time series line plot
    
    Args:
        df: DataFrame with time series data
        date_col: Date column name
        value_col: Value column name
        title: Plot title
    
    Returns:
        plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[value_col],
        mode='lines+markers',
        line=dict(color='#FF6B6B', width=2),
        marker=dict(size=6),
        hovertemplate='%{x}<br>Value: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=date_col,
        yaxis_title=value_col,
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='x unified'
    )
    
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=['On Time', 'Delayed']):
    """
    Create confusion matrix heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
    
    Returns:
        plotly figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Reds',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig
