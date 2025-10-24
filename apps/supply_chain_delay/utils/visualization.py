"""
Visualization utilities for supply chain delay predictions
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st


def plot_risk_gauge(probability_pct, risk_category):
    """
    Create enhanced gauge chart with arrow pointer, zone highlighting, and labels
    
    Combines:
    - Prominent zone highlighting
    - Clean arrow pointer
    - Zone labels for clarity
    - Professional styling
    
    Args:
        probability_pct: Probability as percentage (0-100)
        risk_category: Risk category ('Low', 'Medium', 'High')
    
    Returns:
        plotly figure
    """
    # Define risk bands (thresholds)
    low_max = 30
    med_max = 67
    
    # Determine which zone we're in
    if probability_pct <= low_max:
        current_zone = 'low'
        zone_color = '#00CC96'
        zone_name = 'Low Risk'
    elif probability_pct <= med_max:
        current_zone = 'medium'
        zone_color = '#FFA500'
        zone_name = 'Medium Risk'
    else:
        current_zone = 'high'
        zone_color = '#EF553B'
        zone_name = 'High Risk'
    
    fig = go.Figure()
    
    # Add the main gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=probability_pct,
        domain={'x': [0, 1], 'y': [0.12, 1]},
        number={
            'suffix': '%',
            'font': {'size': 52, 'color': zone_color, 'family': 'Arial Black'}
        },
        title={
            'text': f"<b>{zone_name}</b><br><span style='font-size:15px; color:#666'>Delay Probability</span>",
            'font': {'size': 26, 'color': zone_color}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2.5,
                'tickcolor': 'darkgray',
                'tickmode': 'array',
                'tickvals': [0, low_max, med_max, 100],
                'ticktext': ['0%', f'{low_max:.0f}%', f'{med_max:.0f}%', '100%'],
                'tickfont': {'size': 15, 'family': 'Arial', 'color': '#333'}
            },
            'bar': {'color': zone_color, 'thickness': 0.25},  # Visible progress bar
            'bgcolor': 'white',
            'borderwidth': 2.5,
            'bordercolor': '#AAAAAA',
            'steps': [
                # Low risk zone
                {
                    'range': [0, low_max],
                    'color': '#00CC96' if current_zone == 'low' else '#E8F7F2',
                    'thickness': 0.75 if current_zone == 'low' else 0.65,
                    'line': {'width': 3 if current_zone == 'low' else 1, 'color': '#00CC96'}
                },
                # Medium risk zone
                {
                    'range': [low_max, med_max],
                    'color': '#FFA500' if current_zone == 'medium' else '#FFF2E0',
                    'thickness': 0.75 if current_zone == 'medium' else 0.65,
                    'line': {'width': 3 if current_zone == 'medium' else 1, 'color': '#FFA500'}
                },
                # High risk zone
                {
                    'range': [med_max, 100],
                    'color': '#EF553B' if current_zone == 'high' else '#FFEBEB',
                    'thickness': 0.75 if current_zone == 'high' else 0.65,
                    'line': {'width': 3 if current_zone == 'high' else 1, 'color': '#EF553B'}
                }
            ],
            'threshold': {
                'line': {'color': zone_color, 'width': 7},
                'thickness': 0.95,
                'value': probability_pct
            }
        }
    ))
    
    # Add arrow pointer annotation
    angle = 180 - (probability_pct / 100) * 180  # Convert to degrees
    angle_rad = np.radians(angle)
    
    # Arrow pointing from center to value
    arrow_length = 0.38
    arrow_x = 0.5 + arrow_length * np.cos(angle_rad)
    arrow_y = 0.47 + arrow_length * np.sin(angle_rad)
    
    fig.add_annotation(
        x=arrow_x,
        y=arrow_y,
        ax=0.5,
        ay=0.47,
        xref='paper',
        yref='paper',
        axref='paper',
        ayref='paper',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.8,
        arrowwidth=5,
        arrowcolor=zone_color,
    )
    
    # Add center dot where arrow originates
    fig.add_shape(
        type='circle',
        xref='paper',
        yref='paper',
        x0=0.475, y0=0.445,
        x1=0.525, y1=0.495,
        fillcolor=zone_color,
        line_color='white',
        line_width=2
    )
    
    # Add zone labels at bottom
    fig.add_annotation(
        text="<b>Low</b>",
        x=0.15, y=0.04,
        xref='paper', yref='paper',
        showarrow=False,
        font={
            'size': 15 if current_zone == 'low' else 13,
            'color': '#00CC96' if current_zone == 'low' else '#999',
            'family': 'Arial Black' if current_zone == 'low' else 'Arial'
        }
    )
    
    fig.add_annotation(
        text="<b>Medium</b>",
        x=0.5, y=0.01,
        xref='paper', yref='paper',
        showarrow=False,
        font={
            'size': 15 if current_zone == 'medium' else 13,
            'color': '#FFA500' if current_zone == 'medium' else '#999',
            'family': 'Arial Black' if current_zone == 'medium' else 'Arial'
        }
    )
    
    fig.add_annotation(
        text="<b>High</b>",
        x=0.85, y=0.04,
        xref='paper', yref='paper',
        showarrow=False,
        font={
            'size': 15 if current_zone == 'high' else 13,
            'color': '#EF553B' if current_zone == 'high' else '#999',
            'family': 'Arial Black' if current_zone == 'high' else 'Arial'
        }
    )
    
    # Add subtle glow effect to active zone label
    if current_zone == 'low':
        label_x, label_y = 0.15, 0.04
    elif current_zone == 'medium':
        label_x, label_y = 0.5, 0.01
    else:
        label_x, label_y = 0.85, 0.04
    
    fig.add_shape(
        type='circle',
        xref='paper', yref='paper',
        x0=label_x - 0.03, y0=label_y - 0.015,
        x1=label_x + 0.03, y1=label_y + 0.015,
        fillcolor=zone_color,
        opacity=0.1,
        line_width=0
    )
    
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=90, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial, sans-serif'}
    )
    
    return fig


def plot_feature_importance(feature_names, importance_values, top_n=10):
    """
    Create horizontal bar chart of feature importance
    
    Args:
        feature_names: List of feature names
        importance_values: List of importance values
        top_n: Number of top features to display
    
    Returns:
        plotly figure
    """
    # Sort and get top N
    sorted_idx = np.argsort(importance_values)[-top_n:]
    top_features = [feature_names[i] for i in sorted_idx]
    top_importance = [importance_values[i] for i in sorted_idx]
    
    fig = go.Figure(go.Bar(
        x=top_importance,
        y=top_features,
        orientation='h',
        marker=dict(
            color=top_importance,
            colorscale='Blues',
            showscale=False
        ),
        text=[f'{v:.3f}' for v in top_importance],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Most Important Features',
        xaxis_title='Importance',
        yaxis_title='',
        height=400,
        margin=dict(l=150, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_probability_distribution(probabilities, threshold):
    """
    Create histogram of probability distribution with threshold line
    
    Args:
        probabilities: Array of probabilities
        threshold: Classification threshold
    
    Returns:
        plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=probabilities * 100,
        nbinsx=50,
        name='Distribution',
        marker_color='steelblue',
        opacity=0.7
    ))
    
    fig.add_vline(
        x=threshold * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold*100:.1f}%",
        annotation_position="top"
    )
    
    fig.update_layout(
        title='Delay Probability Distribution',
        xaxis_title='Delay Probability (%)',
        yaxis_title='Count',
        height=350,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_risk_breakdown(risk_counts):
    """
    Create pie chart of risk category breakdown
    
    Args:
        risk_counts: Dictionary of risk category counts
    
    Returns:
        plotly figure
    """
    colors = {
        'Low': '#00CC96',
        'Medium': '#FFA500',
        'High': '#EF553B'
    }
    
    labels = list(risk_counts.keys())
    values = list(risk_counts.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=[colors[label] for label in labels]),
        hole=0.4,
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig.update_layout(
        title='Risk Category Distribution',
        height=350,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_geographic_heatmap(df, location_col, metric_col, title):
    """
    Create geographic heatmap by state/city
    
    Args:
        df: Dataframe with location and metric data
        location_col: Column name for location
        metric_col: Column name for metric to plot
        title: Chart title
    
    Returns:
        plotly figure
    """
    # Aggregate by location
    agg_df = df.groupby(location_col)[metric_col].mean().reset_index()
    agg_df = agg_df.sort_values(metric_col, ascending=False).head(20)
    
    fig = go.Figure(data=[go.Bar(
        x=agg_df[location_col],
        y=agg_df[metric_col],
        marker=dict(
            color=agg_df[metric_col],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Risk %")
        ),
        text=[f'{v:.1f}%' for v in agg_df[metric_col]],
        textposition='outside'
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title=location_col.replace('_', ' ').title(),
        yaxis_title='Average Delay Risk (%)',
        height=400,
        xaxis_tickangle=-45,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_time_trends(df, date_col, metric_col, groupby='month'):
    """
    Create time series trend plot
    
    Args:
        df: Dataframe with date and metric columns
        date_col: Column name for date
        metric_col: Column name for metric
        groupby: Grouping period ('month', 'dayofweek', 'hour')
    
    Returns:
        plotly figure
    """
    # Aggregate by time period
    if groupby == 'month':
        df['period'] = pd.to_datetime(df[date_col]).dt.to_period('M').astype(str)
        xlabel = 'Month'
    elif groupby == 'dayofweek':
        df['period'] = pd.to_datetime(df[date_col]).dt.day_name()
        xlabel = 'Day of Week'
    else:  # hour
        df['period'] = pd.to_datetime(df[date_col]).dt.hour
        xlabel = 'Hour of Day'
    
    trend_df = df.groupby('period')[metric_col].mean().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trend_df['period'],
        y=trend_df[metric_col],
        mode='lines+markers',
        line=dict(color='steelblue', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(70, 130, 180, 0.2)'
    ))
    
    fig.update_layout(
        title=f'Delay Risk Trend by {xlabel}',
        xaxis_title=xlabel,
        yaxis_title='Average Delay Risk (%)',
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_metrics_cards(metrics_dict):
    """
    Display metrics in card format using Streamlit columns
    
    Args:
        metrics_dict: Dictionary of metric name to value
    """
    num_metrics = len(metrics_dict)
    cols = st.columns(min(num_metrics, 4))
    
    for idx, (label, value) in enumerate(metrics_dict.items()):
        col_idx = idx % 4
        with cols[col_idx]:
            st.metric(label=label, value=value)


def plot_correlation_heatmap(df, features):
    """
    Create correlation heatmap for selected features
    
    Args:
        df: Dataframe with feature data
        features: List of feature names
    
    Returns:
        plotly figure
    """
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        height=500,
        width=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_comparison_chart(scenarios_data):
    """
    Create grouped bar chart comparing multiple scenarios
    
    Args:
        scenarios_data: List of dicts with 'name' and 'probability' keys
    
    Returns:
        plotly figure
    """
    names = [s['name'] for s in scenarios_data]
    probs = [s['probability'] for s in scenarios_data]
    colors = [s['color'] for s in scenarios_data]
    
    fig = go.Figure(data=[go.Bar(
        x=names,
        y=probs,
        marker=dict(color=colors),
        text=[f'{p:.1f}%' for p in probs],
        textposition='outside'
    )])
    
    fig.update_layout(
        title='Scenario Risk Comparison',
        xaxis_title='Scenario',
        yaxis_title='Delay Risk (%)',
        height=400,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.add_hline(
        y=30, line_dash="dash", line_color="green",
        annotation_text="Low Risk Threshold"
    )
    fig.add_hline(
        y=67, line_dash="dash", line_color="red",
        annotation_text="High Risk Threshold"
    )
    
    return fig
