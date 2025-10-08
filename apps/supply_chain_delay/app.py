"""
Supply Chain Delay Risk Prediction App
Main landing page with model overview and navigation
"""

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_loader import load_model
from utils.theme_adaptive import apply_adaptive_theme

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Supply Chain Delay Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply adaptive theme
apply_adaptive_theme()

# ============================================================================
# Load Model Metadata
# ============================================================================

@st.cache_data
def load_metadata():
    """Load model metadata from artifacts"""
    try:
        metadata_path = Path(__file__).parent / "artifacts" / "final_metadata.json"
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"⚠️ Could not load model metadata: {str(e)}")
        return {
            'best_model': 'XGBoost',
            'best_model_auc': 0.8500,
            'n_features': 30,
            'n_samples_train': 80000,
            'training_date': 'October 2025'
        }

metadata = load_metadata()

# ============================================================================
# Main Landing Page
# ============================================================================

# Header
st.markdown('<div class="main-header">📦 Supply Chain Delay Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Late Delivery Risk Assessment</div>', unsafe_allow_html=True)

st.markdown("---")

# Introduction
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div class="info-box">
    <h3>🎯 What This App Does</h3>
    <p>This application uses machine learning to predict whether an e-commerce order will be delivered late, 
    enabling proactive intervention to improve customer satisfaction and reduce operational costs.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# How to Use
st.markdown("### 🚀 How to Use This App")

st.markdown("""
1. **📊 Single Prediction**: Use the sidebar to navigate to the Single Prediction page and enter order details
2. **📦 Batch Predictions**: Upload a CSV file with multiple orders for bulk processing
3. **🔍 Model Insights**: Explore what features drive late delivery predictions

**Navigation:** Use the sidebar on the left to switch between pages.
""")

col1, col2, col3 = st.columns(3)

# Features Section
st.markdown("### ✨ Key Features")

with col1:
    st.markdown("""
    <div class="metric-card">
    <h4>📊 Single Prediction</h4>
    <p>Enter order details to get an instant late delivery risk assessment with explanations.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
    <h4>📦 Batch Processing</h4>
    <p>Upload a CSV file to analyze multiple orders at once and download results.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
    <h4>🔍 Model Insights</h4>
    <p>Explore feature importance, SHAP analysis, and understand what drives predictions.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("---")
# Key Metrics
st.markdown("### 📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    auc_value = metadata.get('best_model_auc', 0)
    st.metric(
        label="AUC-ROC Score",
        value=f"{auc_value:.4f}",
        #delta="✅ Target: ≥0.85" if auc_value >= 0.85 else "⚠️ Target: ≥0.85"
    )

with col2:
    st.metric(
        label="Model Type",
        value=metadata.get('best_model', 'N/A')
    )

with col3:
    st.metric(
        label="Features Used",
        value=metadata.get('n_features', 0)
    )

with col4:
    st.metric(
        label="Training Samples",
        value=f"{metadata.get('n_samples_train', 0):,}"
    )

st.markdown("---")

st.markdown("### 🏆 Optimization Results")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Precision", "23.4%", "+33.6% vs baseline")
with col2:
    st.metric("Optimal Threshold", "18.44%")
with col3:
    st.metric("Optimization Method", "Bayesian (Optuna)")
    

# Business Impact
st.markdown("### 💼 Business Impact")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="success-box">
    <h4>📈 Expected Benefits</h4>
    <ul>
        <li><strong>20% reduction</strong> in late deliveries</li>
        <li><strong>$400K+ annual savings</strong> in refunds and lost customers</li>
        <li><strong>15% improvement</strong> in customer satisfaction</li>
        <li><strong>Proactive intervention</strong> on high-risk orders</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
    <h4>🎯 Use Cases</h4>
    <ul>
        <li><strong>Operations:</strong> Prioritize high-risk orders</li>
        <li><strong>Logistics:</strong> Optimize carrier selection</li>
        <li><strong>Customer Service:</strong> Proactive communication</li>
        <li><strong>Finance:</strong> Budget for expedited shipping</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Footer
st.markdown("### 📚 About")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Project:** Data Science Capstone - Supply Chain Delay Prediction  
    **Author:** Trecorrus Jordan   
    **Date:** November 2025  
    **Dataset:** Olist Brazilian E-Commerce (100k+ orders)
    """)

with col2:
    st.warning("""
    **Model Information:**  
    - Algorithm: XGBoost (Gradient Boosting)  
    - Features: 30 domain-engineered features  
    - Training: 5-fold cross-validation  
    - Metrics: AUC-ROC, Precision, Recall, F1-Score
    """)

# Sidebar
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    st.info("""
    Use the pages above to:
    - 📊 Make single predictions
    - 📦 Analyze order batches
    - 🔍 Explore model insights
    """)
    
    st.markdown("---")
    
    st.markdown("## ℹ️ Quick Start")
    st.markdown("""
    **First Time User?**
    1. Click "📊 Single Prediction"
    2. Fill in order details
    3. Get instant risk score!
    """)
    
    st.markdown("---")
    
    st.markdown("## 📊 Model Info")
    st.markdown(f"""
    **Trained:** {metadata.get('training_date', 'October 2025')}  
    **Model:** {metadata.get('best_model', 'XGBoost')}  
    **AUC-ROC:** {metadata.get('best_model_auc', 0.85):.4f}  
    **Features:** {metadata.get('n_features', 30)}
    """)
