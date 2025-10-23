"""
Supply Chain Delay Prediction System
Main Dashboard and Application Entry Point
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    load_model_artifacts,
    get_model_performance,
    apply_custom_css,
    show_page_header,
    create_three_column_layout,
    display_info_box,
    get_risk_color,
    get_risk_icon
)

# Page configuration
st.set_page_config(
    page_title="Supply Chain Delay Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_custom_css()

# Load model artifacts (cached)
model, final_metadata, feature_metadata, threshold = load_model_artifacts()

# Header
show_page_header(
    title="Supply Chain Delay Prediction System",
    description="AI-powered tool to predict and prevent delivery delays in e-commerce supply chains",
    icon="📦"
)

# Main content
st.markdown("""
### Welcome to the Supply Chain Delay Prediction System

This intelligent system uses machine learning to predict the likelihood of delivery delays 
in e-commerce orders. By analyzing 32 different features including order characteristics, 
geographic data, and temporal patterns, the system provides actionable insights to help 
optimize your supply chain operations.
""")

st.markdown("---")

# Key Metrics Overview
st.subheader("📊 System Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Model AUC-ROC",
        value=f"{final_metadata['best_model_auc']:.3f}",
        help="Area Under the ROC Curve - measures model discrimination ability"
    )

with col2:
    st.metric(
        label="Precision",
        value=f"{final_metadata['best_model_precision']:.3f}",
        help="Proportion of predicted delays that are actual delays"
    )

with col3:
    st.metric(
        label="Recall",
        value=f"{final_metadata['best_model_recall']:.3f}",
        help="Proportion of actual delays that are correctly predicted"
    )

with col4:
    st.metric(
        label="F1-Score",
        value=f"{final_metadata['best_model_f1']:.3f}",
        help="Harmonic mean of precision and recall"
    )

st.markdown("---")

# Feature sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Key Features")
    st.markdown("""
    - **Single Order Prediction**: Predict delay risk for individual orders with detailed explanations
    - **Batch Processing**: Upload CSV files to predict delays for multiple orders at once
    - **Interactive Scenarios**: Explore pre-built scenarios to understand risk factors
    - **Geographic Analysis**: Visualize delay patterns across Brazilian states
    - **Temporal Trends**: Analyze how delays vary over time
    - **SHAP Explanations**: Understand which features contribute to each prediction
    """)

with col2:
    st.subheader("📈 Model Information")
    st.markdown(f"""
    - **Algorithm**: {final_metadata['best_model']} (Gradient Boosting)
    - **Features**: {final_metadata['n_features']} predictive features
    - **Training Samples**: {final_metadata['n_samples_train']:,}
    - **Test Samples**: {final_metadata['n_samples_test']:,}
    - **Training Date**: {final_metadata['training_date']}
    - **Optimal Threshold**: {threshold*100:.1f}%
    """)

st.markdown("---")

# Risk Level Explanation
st.subheader("🚦 Risk Level Definitions")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style="background-color: {get_risk_color('Low')}20; padding: 20px; border-radius: 10px; border-left: 5px solid {get_risk_color('Low')}">
        <h3>{get_risk_icon('Low')} Low Risk (0-30%)</h3>
        <p>Orders with low probability of delay. Standard processing recommended.</p>
        <ul>
            <li>Proceed normally</li>
            <li>Standard monitoring</li>
            <li>Regular fulfillment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background-color: {get_risk_color('Medium')}20; padding: 20px; border-radius: 10px; border-left: 5px solid {get_risk_color('Medium')}">
        <h3>{get_risk_icon('Medium')} Medium Risk (30-67%)</h3>
        <p>Orders requiring enhanced attention and monitoring.</p>
        <ul>
            <li>Increased monitoring</li>
            <li>Verify inventory</li>
            <li>Check logistics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="background-color: {get_risk_color('High')}20; padding: 20px; border-radius: 10px; border-left: 5px solid {get_risk_color('High')}">
        <h3>{get_risk_icon('High')} High Risk (67-100%)</h3>
        <p>Orders at high risk of delay. Immediate action required.</p>
        <ul>
            <li>Immediate review</li>
            <li>Proactive communication</li>
            <li>Consider alternatives</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Getting Started
st.subheader("🚀 Getting Started")

display_info_box(
    title="Quick Start Guide",
    content="""
    1. **Single Prediction**: Go to '📊 Single Prediction' to predict delay risk for one order
    2. **Example Scenarios**: Visit '🎯 Example Scenarios' to see pre-built examples
    3. **Batch Processing**: Use '📦 Batch Predictions' to analyze multiple orders from CSV
    4. **Analyze Trends**: Explore '📈 Time Trends' and '🗺️ Geographic Map' for insights
    """,
    box_type="info"
)

st.markdown("---")

# Dataset Information
with st.expander("📋 Dataset & Feature Information"):
    st.markdown("""
    ### Feature Categories
    
    **Order Characteristics** (9 features)
    - Number of items, sellers, products, and categories
    - Product dimensions (weight, length, height, width)
    - Category information
    
    **Financial Features** (5 features)
    - Sum of prices and freight costs
    - Total payment amount
    - Number of payment records
    - Maximum installments
    
    **Geographic Features** (4 features)
    - Customer city and state
    - Seller state mode
    - Number of seller states
    
    **Temporal Features** (9 features)
    - Purchase year, month, day of week, hour
    - Weekend indicator
    - Cyclical time encodings (sine/cosine)
    - Estimated lead days
    
    **Payment Type Features** (5 features)
    - Boleto, Credit Card, Debit Card, Voucher, Not Defined
    """)
    
    st.markdown(f"""
    ### Target Distribution
    - **On-Time Deliveries**: {feature_metadata['target_distribution']['on_time']:,} ({feature_metadata['target_distribution']['on_time']/feature_metadata['n_samples']*100:.1f}%)
    - **Late Deliveries**: {feature_metadata['target_distribution']['late']:,} ({feature_metadata['target_distribution']['late']/feature_metadata['n_samples']*100:.1f}%)
    - **Imbalance Ratio**: {feature_metadata['target_distribution']['imbalance_ratio']:.1f}:1
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <p>📦 Supply Chain Delay Prediction System | Powered by LightGBM & SHAP</p>
    <p><small>Navigate using the sidebar to access different features →</small></p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/FF6B6B/FFFFFF?text=Supply+Chain+AI", use_container_width=True)
    st.markdown("---")
    
    st.markdown("### 📍 Navigation")
    st.markdown("""
    Use the pages above to:
    - 🎯 View example scenarios
    - 📊 Make single predictions
    - 📦 Process batch predictions
    - 📈 Analyze time trends
    - 🗺️ View geographic patterns
    """)
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.info("""
    This system predicts supply chain delays using machine learning 
    trained on Brazilian e-commerce data.
    
    **Model**: LightGBM  
    **Features**: 32  
    **AUC-ROC**: 0.789
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔗 Resources")
    st.markdown("""
    - [Documentation](#)
    - [API Guide](#)
    - [Contact Support](#)
    """)
