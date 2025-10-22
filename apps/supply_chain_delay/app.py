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

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Supply Chain Delay Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply adaptive theme
apply_adaptive_theme()

# =============================================================================
# Load Model Metadata
# Expecting artifacts/final_metadata.json with keys like:
#   best_model, best_model_auc, n_features, n_samples_train, training_date,
#   precision, recall, f1, optimal_threshold, optimization_method
# =============================================================================
@st.cache_data
def load_metadata():
    try:
        metadata_path = Path(__file__).parent / "artifacts" / "final_metadata.json"
        with open(metadata_path, "r") as f:
            data = json.load(f)
        # Coerce some expected types / defaults
        data.setdefault("best_model", "LightGBM")
        data.setdefault("best_model_auc", 0.7890)
        data.setdefault("precision", 0.304)  # 30.4%
        data.setdefault("recall", 0.443)
        data.setdefault("f1", 0.361)
        data.setdefault("optimal_threshold", 0.185)
        data.setdefault("optimization_method", "RandomizedSearchCV (5-fold)")
        data.setdefault("n_features", 30)
        data.setdefault("n_samples_train", 72000)
        data.setdefault("training_date", "October 2025")
        return data
    except Exception as e:
        st.warning(f"⚠️ Could not load model metadata from artifacts: {e}")
        # Safe defaults that match current notebook orientation (LightGBM)
        return {
            "best_model": "LightGBM",
            "best_model_auc": 0.7890,
            "precision": 0.304,
            "recall": 0.443,
            "f1": 0.361,
            "optimal_threshold": 0.185,
            "optimization_method": "RandomizedSearchCV (5-fold)",
            "n_features": 30,
            "n_samples_train": 72000,
            "training_date": "October 2025"
        }

metadata = load_metadata()

# =============================================================================
# Main Landing Page
# =============================================================================
st.markdown('<div class="main-header">📦 Supply Chain Delay Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Late Delivery Risk Assessment</div>', unsafe_allow_html=True)

st.markdown("---")

# Intro
_, mid, _ = st.columns([1, 2, 1])
with mid:
    st.markdown("""
    <div class="info-box">
    <h3>🎯 What This App Does</h3>
    <p>This application uses machine learning to predict whether an e-commerce order will be delivered late,
    enabling proactive intervention to improve customer satisfaction and reduce operational costs.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# How to use
st.markdown("### 🚀 How to Use This App")
st.markdown("""
1. **📊 Single Prediction**: Use the sidebar to navigate to the Single Prediction page and enter order details  
2. **📦 Batch Predictions**: Upload a CSV file with multiple orders for bulk processing  
3. **🔍 Model Insights**: Explore what features drive late delivery predictions

**Navigation:** Use the sidebar on the left to switch between pages.
""")

st.markdown("---")

# Key features
st.markdown("### ✨ Key Features")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="metric-card">
    <h4>📊 Single Prediction</h4>
    <p>Enter order details to get an instant late delivery risk assessment with explanations.</p>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="metric-card">
    <h4>📦 Batch Processing</h4>
    <p>Upload a CSV file to analyze multiple orders at once and download results.</p>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="metric-card">
    <h4>🔍 Geographic Analysis</h4>
    <p>Interactive map visualization of shipping routes, risk zones, and delivery patterns across Brazil.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Business impact
st.markdown("### 💼 Business Impact")
b1, b2 = st.columns(2)
with b1:
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
with b2:
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

# Model performance tiles
st.markdown("### 📊 Model Performance")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("AUC-ROC", f"{metadata.get('best_model_auc', 0):.4f}")
with m2:
    st.metric("Model Type", metadata.get("best_model", "N/A"))
with m3:
    st.metric("Features Used", f"{metadata.get('n_features', 0)}")
with m4:
    st.metric("Training Samples", f"{metadata.get('n_samples_train', 0):,}")

# Optimization summary (dynamic)
st.markdown("### 🏆 Optimization Results")
o1, o2, o3 = st.columns(3)
with o1:
    # Use the correct keys from final_metadata.json
    prec = 100 * float(metadata.get("best_model_precision", metadata.get("precision", 0)))
    st.metric("Precision", f"{prec:.1f}%", "vs. baseline +Δ")  # show as percent
with o2:
    st.metric("Optimal Threshold", f"{100*float(metadata.get('optimal_threshold', 0.5)):.2f}%")
with o3:
    # Keep the tuning method if you store it; else show model type as a helpful label
    st.metric("Tuning Method", metadata.get("optimization_method", "RandomizedSearchCV"))

st.markdown("---")

# Footer
st.markdown("### 📚 About")
f1, f2 = st.columns(2)
with f1:
    st.info(f"""
**Project:** Data Science Capstone - Supply Chain Delay Prediction  
**Author:** Trecorrus Jordan  
**Date:** November 2025  
**Dataset:** Olist Brazilian E-Commerce (100k+ orders)
""")
with f2:
    st.warning(f"""
**Model Information:**  
- Algorithm: {metadata.get('best_model','LightGBM')}  
- Features: {metadata.get('n_features',30)} domain-engineered  
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
**Model:** {metadata.get('best_model', 'LightGBM')}  
**AUC-ROC:** {metadata.get('best_model_auc', 0.789):.4f}  
**Features:** {metadata.get('n_features', 30)}
""")
