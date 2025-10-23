"""
Utility functions for Supply Chain Delay Prediction App
"""
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

# Helper function for info banners
def display_info_banner(title, message, icon="ℹ️"):
    """Display an info banner in the app"""
    import streamlit as st
    st.info(f"{icon} **{title}**: {message}")

# Alias functions for backward compatibility
def predict_delay(model, input_data, threshold):
    """Alias for predict_delay_risk"""
    return predict_delay_risk(model, input_data, threshold)

def prepare_features(order_data, feature_metadata):
    """Alias for prepare_single_prediction_input"""
    return prepare_single_prediction_input(order_data, feature_metadata)

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
    
    # Theme
    'apply_custom_css',
    'get_risk_color_scheme',
    'style_dataframe',
    
    # PDF generation
    'generate_prediction_report',
    'generate_batch_report',
    
    # Helper functions
    'display_info_banner',
    
    # Aliases for backward compatibility
    'predict_delay',
    'prepare_features'
]
