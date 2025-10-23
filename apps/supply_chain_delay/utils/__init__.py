"""
Utility functions for Supply Chain Delay Prediction App
"""
import streamlit as st
import pandas as pd
import numpy as np

# Import from utility modules
from .model_loader import (
    load_model_artifacts,
    load_metadata,
    predict_delay_risk,
    get_risk_category,
    get_risk_color,
    get_model_performance
)
from .feature_engineering import (
    prepare_single_prediction_input,
    prepare_batch_prediction_input,
    validate_input_data,
    get_feature_ranges,
    create_sample_scenarios
)
from .visualization import (
    plot_risk_gauge,
    plot_feature_importance,
    plot_probability_distribution,
    plot_risk_breakdown,
    plot_geographic_heatmap,
    plot_time_trends,
    create_metrics_cards
)
from .theme_adaptive import (
    apply_custom_css,
    get_risk_color_scheme,
    style_dataframe
)
from .pdf_generator import (
    generate_prediction_report,
    generate_batch_report
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def display_info_banner(title, message, icon="ℹ️"):
    """Display an info banner in the app"""
    st.info(f"{icon} **{title}**: {message}")


def show_page_header(title, description, icon="📊"):
    """
    Display a standardized page header
    
    Args:
        title: Page title
        description: Page description
        icon: Emoji icon for the page
    """
    st.title(f"{icon} {title}")
    st.markdown(f"### {description}")
    st.markdown("---")


def display_risk_badge(risk_level, probability):
    """
    Display a colored risk badge
    
    Args:
        risk_level: Risk level ('Low', 'Medium', 'High')
        probability: Probability value (0-1 or 0-100)
    """
    # Normalize probability to 0-100 range
    if probability <= 1:
        probability = probability * 100
    
    colors = {
        'Low': '#00CC96',
        'Medium': '#FFA500',
        'High': '#EF553B'
    }
    
    color = colors.get(risk_level, '#888888')
    
    st.markdown(f"""
    <div style="
        background-color: {color}20;
        border-left: 5px solid {color};
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    ">
        <h2 style="color: {color}; margin: 0;">
            {risk_level} Risk
        </h2>
        <p style="font-size: 1.5rem; margin: 0.5rem 0 0 0;">
            <strong>{probability:.1f}%</strong> Delay Probability
        </p>
    </div>
    """, unsafe_allow_html=True)


def calculate_temporal_features(datetime_obj):
    """
    Calculate temporal features from a datetime object
    
    Args:
        datetime_obj: datetime object
        
    Returns:
        dict: Dictionary with temporal features
    """
    hour = datetime_obj.hour
    
    return {
        'purch_year': datetime_obj.year,
        'purch_month': datetime_obj.month,
        'purch_dayofweek': datetime_obj.weekday(),
        'purch_hour': hour,
        'purch_is_weekend': 1 if datetime_obj.weekday() >= 5 else 0,
        'purch_hour_sin': np.sin(2 * np.pi * hour / 24),
        'purch_hour_cos': np.cos(2 * np.pi * hour / 24)
    }


def get_feature_descriptions():
    """
    Get human-readable descriptions for features
    
    Returns:
        dict: Dictionary mapping feature names to descriptions
    """
    return {
        'n_items': 'Number of items in the order',
        'n_sellers': 'Number of different sellers',
        'n_products': 'Number of unique products',
        'sum_price': 'Total price of all items',
        'sum_freight': 'Total shipping cost',
        'total_payment': 'Total payment amount',
        'n_payment_records': 'Number of payment transactions',
        'max_installments': 'Maximum payment installments',
        'avg_weight_g': 'Average product weight (grams)',
        'avg_length_cm': 'Average product length (cm)',
        'avg_height_cm': 'Average product height (cm)',
        'avg_width_cm': 'Average product width (cm)',
        'n_seller_states': 'Number of seller states',
        'purch_year': 'Purchase year',
        'purch_month': 'Purchase month',
        'purch_dayofweek': 'Day of week (0=Monday)',
        'purch_hour': 'Hour of purchase (0-23)',
        'purch_is_weekend': 'Weekend purchase indicator',
        'est_lead_days': 'Estimated delivery lead time',
        'n_categories': 'Number of product categories',
        'customer_state': 'Customer state code',
        'customer_city': 'Customer city',
        'seller_state_mode': 'Primary seller state'
    }


# ============================================================================
# ALIAS FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================================

def predict_delay(model, input_data, threshold):
    """
    Alias for predict_delay_risk that returns arrays for batch compatibility
    
    Args:
        model: Trained model
        input_data: DataFrame with features
        threshold: Classification threshold
        
    Returns:
        tuple: (predictions_array, probabilities_array, risk_levels_array)
    """
    # Handle both single and batch predictions
    if len(input_data) == 1:
        prob_pct, risk_category, is_high_risk = predict_delay_risk(model, input_data, threshold)
        return (
            np.array([1 if is_high_risk else 0]),
            np.array([prob_pct / 100]),
            np.array([risk_category])
        )
    else:
        # Batch prediction
        probs = model.predict_proba(input_data)[:, 1]
        predictions = (probs >= threshold).astype(int)
        risk_levels = np.array([get_risk_category(p * 100) for p in probs])
        return predictions, probs, risk_levels


def prepare_features(order_data, feature_names_or_metadata):
    """
    Alias for prepare_single_prediction_input
    Handles both just feature names (list) or full feature_metadata (dict)
    
    Args:
        order_data: Dictionary or DataFrame with order data
        feature_names_or_metadata: Either list of feature names OR full feature_metadata dict
        
    Returns:
        DataFrame: Prepared features
    """
    # Debug: Check what we received
    if order_data is None:
        raise ValueError("order_data is None. Check that create_example_order is returning data correctly.")
    
    # Handle both cases: just feature names list or full metadata dict
    if isinstance(feature_names_or_metadata, list):
        # We have just feature names - need to load full metadata
        _, feature_metadata = load_metadata()
    elif isinstance(feature_names_or_metadata, dict):
        # We have full metadata already
        feature_metadata = feature_names_or_metadata
    else:
        raise ValueError(f"feature_names_or_metadata must be a list or dict, got {type(feature_names_or_metadata)}")
    
    # Process the data
    if isinstance(order_data, dict):
        return prepare_single_prediction_input(order_data, feature_metadata)
    elif isinstance(order_data, pd.DataFrame):
        return prepare_batch_prediction_input(order_data, feature_metadata)
    else:
        raise ValueError(f"order_data must be a dict or DataFrame, got {type(order_data).__name__}")


def create_example_order(scenario_name=None):
    """
    Alias for create_sample_scenarios
    
    Args:
        scenario_name: Name of scenario ('low_risk', 'typical', 'high_risk')
        
    Returns:
        dict: Order data for the scenario
    """
    scenarios = create_sample_scenarios()
    
    # If scenarios is a dict of scenarios, extract the requested one
    if isinstance(scenarios, dict) and scenario_name:
        # Try exact match first
        if scenario_name in scenarios:
            return scenarios[scenario_name]
        # If not found, return typical or first available
        return scenarios.get('typical', list(scenarios.values())[0] if scenarios else {})
    
    # If no scenario name provided, return all scenarios
    if scenario_name is None:
        return scenarios
    
    # If scenarios itself is the data (not a dict of scenarios), return it
    return scenarios


def validate_input(df, required_features):
    """
    Validate that input DataFrame has all required features
    
    Args:
        df: Input DataFrame
        required_features: List of required feature names
        
    Returns:
        tuple: (is_valid, error_message)
    """
    return validate_input_data(df, {'feature_names': required_features})


def plot_risk_distribution(risk_levels):
    """
    Plot distribution of risk levels
    
    Args:
        risk_levels: Array of risk level strings
        
    Returns:
        plotly figure
    """
    import plotly.graph_objects as go
    
    # Count risk levels
    risk_counts = pd.Series(risk_levels).value_counts()
    
    colors = {
        'Low': '#00CC96',
        'Medium': '#FFA500',
        'High': '#EF553B'
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker=dict(
                color=[colors.get(level, '#888888') for level in risk_counts.index]
            ),
            text=risk_counts.values,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Risk Level Distribution",
        xaxis_title="Risk Level",
        yaxis_title="Number of Orders",
        height=400
    )
    
    return fig


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Model functions
    'load_model_artifacts',
    'load_metadata',
    'predict_delay_risk',
    'get_risk_category',
    'get_risk_color',
    'get_model_performance',
    
    # Feature engineering
    'prepare_single_prediction_input',
    'prepare_batch_prediction_input',
    'validate_input_data',
    'get_feature_ranges',
    'create_sample_scenarios',
    
    # Visualization
    'plot_risk_gauge',
    'plot_feature_importance',
    'plot_probability_distribution',
    'plot_risk_breakdown',
    'plot_geographic_heatmap',
    'plot_time_trends',
    'create_metrics_cards',
    'plot_risk_distribution',
    
    # Theme
    'apply_custom_css',
    'get_risk_color_scheme',
    'style_dataframe',
    
    # PDF generation
    'generate_prediction_report',
    'generate_batch_report',
    
    # Helper functions
    'display_info_banner',
    'show_page_header',
    'display_risk_badge',
    'calculate_temporal_features',
    'get_feature_descriptions',
    
    # Aliases for backward compatibility
    'predict_delay',
    'prepare_features',
    'create_example_order',
    'validate_input'
]
