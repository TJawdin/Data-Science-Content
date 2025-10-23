import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_gauge_chart(value, threshold):
    """Create a gauge chart for risk visualization"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Delay Risk %"},
        delta = {'reference': threshold},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 67], 'color': "yellow"},
                {'range': [67, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

def create_risk_distribution(predictions_df):
    """Create distribution plot of risk levels"""
    
    fig = px.histogram(predictions_df, x='risk_probability', 
                       color='risk_level',
                       color_discrete_map={'Low': 'green', 'Medium': 'yellow', 'High': 'red'},
                       nbins=30,
                       title="Risk Distribution")
    
    fig.update_layout(
        xaxis_title="Delay Probability (%)",
        yaxis_title="Number of Orders",
        showlegend=True
    )
    
    return fig

def create_feature_impact_chart(feature_values, feature_importance):
    """Create a chart showing feature impact on prediction"""
    
    # Combine values and importance
    impact_df = pd.DataFrame({
        'Feature': feature_importance['feature'].head(10),
        'Importance': feature_importance['importance'].head(10),
        'Value': feature_values
    })
    
    fig = px.bar(impact_df, x='Feature', y='Importance',
                 hover_data=['Value'],
                 title="Top Features Impact",
                 color='Importance',
                 color_continuous_scale='Blues')
    
    fig.update_layout(height=400)
    return fig
