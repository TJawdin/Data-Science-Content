"""
Supply Chain Delay Prediction - Main Dashboard
A machine learning application for predicting delivery delays in e-commerce supply chains
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    load_model_artifacts,
    load_metadata,
    apply_custom_css,
    get_risk_color,
    create_metrics_cards,
    display_info_banner
)
from utils.model_loader import get_model_performance

# Page configuration
st.set_page_config(
    page_title="Supply Chain Delay Prediction",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to rename "app" to "Home" in sidebar  
st.markdown("""
/* HOME LABEL FIX - Add this section */
        /* Hide the default "app" label in sidebar */
        [data-testid="stSidebarNav"] ul li:first-child {
            display: none;
        }
        
        /* Add "Home" label at top of sidebar navigation */
        [data-testid="stSidebarNav"]::before {
            content: "🏠 Home";
            margin-left: 1rem;
            margin-top: 1.2rem;
            margin-bottom: 0.5rem;
            font-size: 1rem;
            position: relative;
            display: block;
            font-weight: 600;
        }
        /* END HOME LABEL FIX */
# Apply custom styling
apply_custom_css()

# Load metadata (lightweight for homepage)
final_metadata, feature_metadata = load_metadata()

# Header
st.title("📦 Supply Chain Delay Prediction System")
st.markdown("### Predict and prevent delivery delays with machine learning")

# Introduction section
with st.container():
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("""
        Welcome to the **Supply Chain Delay Prediction System**! This application uses 
        advanced machine learning to predict the likelihood of delivery delays in your 
        e-commerce orders.
        
        #### 🎯 Key Features:
        - **Real-time Predictions**: Get instant risk assessments for orders
        - **Batch Processing**: Analyze multiple orders simultaneously
        - **Visual Insights**: Interactive charts and geographic analysis
        - **Actionable Reports**: Download detailed prediction reports
        """)
    
    with col2:
        st.metric(
            label="Model Accuracy (AUC)",
            value=f"{final_metadata['best_model_auc']:.1%}"
        )
        st.metric(
            label="Features Used",
            value=final_metadata['n_features']
        )
    
    with col3:
        st.metric(
            label="Training Samples",
            value=f"{final_metadata['n_samples_train']:,}"
        )
        st.metric(
            label="Model Type",
            value=final_metadata['best_model']
        )

st.divider()

# Model Performance Section
st.subheader("📊 Model Performance Metrics")

performance = get_model_performance(final_metadata)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("AUC-ROC", performance['AUC-ROC'])
with col2:
    st.metric("Precision", performance['Precision'])
with col3:
    st.metric("Recall", performance['Recall'])
with col4:
    st.metric("F1-Score", performance['F1-Score'])

st.divider()

# Risk Categories Explanation
st.subheader("🎨 Risk Categories")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background-color: #E8F8F5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #00CC96;">
        <h4 style="color: #00CC96; margin-top: 0;">🟢 Low Risk (0-30%)</h4>
        <p>Orders with low probability of delay. Standard monitoring recommended.</p>
        <ul>
            <li>Typical delivery expected</li>
            <li>Minimal intervention needed</li>
            <li>Standard customer communication</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color: #FFF4E6; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #FFA500;">
        <h4 style="color: #FFA500; margin-top: 0;">🟡 Medium Risk (30-67%)</h4>
        <p>Orders requiring attention. Enhanced monitoring advised.</p>
        <ul>
            <li>Potential delivery issues</li>
            <li>Proactive customer updates</li>
            <li>Review logistics planning</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background-color: #FADBD8; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #EF553B;">
        <h4 style="color: #EF553B; margin-top: 0;">🔴 High Risk (67-100%)</h4>
        <p>Orders at high risk of delay. Immediate action required.</p>
        <ul>
            <li>Likely delivery delays</li>
            <li>Immediate intervention needed</li>
            <li>Alternative logistics options</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Feature Importance Overview
st.subheader("📈 Key Factors Affecting Delivery")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    #### Order Characteristics
    - **Number of items**: More items increase complexity
    - **Multiple sellers**: Coordination challenges
    - **Order value**: High-value orders may need special handling
    - **Product dimensions**: Large items face logistics constraints
    """)

with col2:
    st.markdown("""
    #### Geographic & Temporal Factors
    - **Customer location**: Remote areas have higher risk
    - **Seller location**: Distance affects delivery time
    - **Purchase timing**: Weekend/late-night orders
    - **Estimated lead time**: Longer estimates indicate complexity
    """)

st.divider()

# Navigation Guide
st.subheader("🗺️ Getting Started")

st.markdown("""
Choose from the following pages in the sidebar to get started:

1. **🎯 Example Scenarios** - Explore pre-configured scenarios to understand the model
2. **📊 Single Prediction** - Make predictions for individual orders with detailed insights
3. **📦 Batch Predictions** - Upload and analyze multiple orders at once
4. **📈 Time Trends** - Analyze how delay risk varies over time
5. **🗺️ Geographic Map** - Visualize risk patterns across different locations

---

*Select a page from the sidebar to begin!*
""")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📚 Documentation**
    - Model trained on Brazilian e-commerce data
    - Uses LightGBM gradient boosting
    - Optimized threshold for business needs
    """)

with col2:
    st.markdown(f"""
    **📅 Model Information**
    - Training Date: {final_metadata['training_date']}
    - Optimal Threshold: {final_metadata['optimal_threshold']:.4f}
    - Total Features: {final_metadata['n_features']}
    """)

with col3:
    st.markdown("""
    **💡 Tips**
    - Start with Example Scenarios to understand the model
    - Use Single Prediction for detailed analysis
    - Batch mode for operational efficiency
    """)

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    
    st.markdown("""
    This application predicts the likelihood of delivery delays 
    in e-commerce supply chains using machine learning.
    
    **Developed by:** Data Science Team  
    **Model Version:** 1.0  
    **Last Updated:** Oct 2025
    """)
    
    st.divider()
    
    st.subheader("🎯 Quick Stats")
    st.info(f"""
    - **Total Features:** {feature_metadata['n_features']}
    - **Training Samples:** {feature_metadata['n_samples']:,}
    - **Imbalance Ratio:** {feature_metadata['target_distribution']['imbalance_ratio']:.1f}:1
    """)
    
    st.divider()
    
    st.subheader("📊 Target Distribution")
    on_time = feature_metadata['target_distribution']['on_time']
    late = feature_metadata['target_distribution']['late']
    total = on_time + late
    
    st.markdown(f"""
    - **On-Time Deliveries:** {on_time:,} ({on_time/total*100:.1f}%)
    - **Late Deliveries:** {late:,} ({late/total*100:.1f}%)
    """)
    
    st.divider()
    
    # Runtime information
    import sys, importlib
    st.markdown("### 🔧 Runtime")
    st.write("Python", ".".join(map(str, sys.version_info[:3])))
    
    for lib in ("joblib", "lightgbm", "sklearn", "numpy", "streamlit"):
        try:
            m = importlib.import_module(lib)
            st.write(f"{lib}", getattr(m, "__version__", "unknown"))
        except Exception as e:
            st.write(f"{lib}", f"not importable ({e})")
