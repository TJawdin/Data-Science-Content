# streamlit_app.py
# ============================================================
# Supply-Chain Delay Predictor — Streamlit App
# Author: TJawdin
# Last Updated: 2024-12-19
# 
# Features:
#   - Batch predictions from CSV upload
#   - Single record prediction via form
#   - Model explainability (SHAP, feature importance)
#   - Adjustable decision threshold
#   - Downloadable results
#   - Future: Interactive maps, heatmaps, time series analysis
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

# ----------------------- Page Config -------------------------
st.set_page_config(
    page_title="Supply-Chain Delay Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- Constants ---------------------------
ARTIFACTS = Path("artifacts")
FEATURES_JSON = ARTIFACTS / "feature_names.json"
GLOBAL_PI_PNG = ARTIFACTS / "global_importance_permutation.png"
SHAP_SUMMARY_BEE = ARTIFACTS / "shap_summary_beeswarm.png"
SHAP_SUMMARY_BAR = ARTIFACTS / "shap_summary_bar.png"

# ----------------------- Helper Functions --------------------
def unwrap_model(m):
    """Extract the actual estimator from a Pipeline if wrapped."""
    if hasattr(m, "named_steps") and "clf" in m.named_steps:
        return m.named_steps["clf"]
    return m

# ----------------------- Cached Loaders ----------------------
@st.cache_resource(show_spinner=False)
def load_model_and_features():
    """Load the trained model and feature schema from artifacts."""
    assert ARTIFACTS.exists(), "artifacts/ directory not found."
    
    # Find model file
    candidates = sorted(ARTIFACTS.glob("model_*.pkl"))
    assert candidates, "No model_*.pkl found in artifacts/."
    model_path = candidates[0]
    
    # Load model
    model = joblib.load(model_path)
    
    # Load features
    assert FEATURES_JSON.exists(), "feature_names.json not found in artifacts/."
    with open(FEATURES_JSON, "r") as f:
        features = json.load(f)
    
    # Get base model for introspection
    base_model = unwrap_model(model)
    
    return model, base_model, model_path.name, features

@st.cache_data(show_spinner=False)
def align_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Align input DataFrame to training feature schema."""
    df = df.copy()
    
    # Add missing expected columns
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
    
    # Keep only expected features, in correct order
    df = df[features]
    
    # Sanitize: handle inf, nan
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Ensure numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    df = df.astype("float32")
    return df

def predict_batch(model, df_input: pd.DataFrame, features: list, threshold: float = 0.5):
    """Make predictions on a batch of records."""
    X = align_features(df_input, features)
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        # Fallback: decision_function or predict
        if hasattr(model, "decision_function"):
            raw = model.decision_function(X)
        else:
            raw = model.predict(X)
        raw = np.asarray(raw, dtype="float32")
        # Normalize to [0,1]
        proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    
    pred = (proba >= threshold).astype(int)
    return proba, pred

# ----------------------- Custom CSS --------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------- Sidebar -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/package.png", width=80)
    st.title("⚙️ Settings")
    
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help="Orders with predicted probability ≥ threshold are flagged as 'Late'"
    )
    
    st.markdown("---")
    
    show_importance = st.checkbox("Show Feature Importance", value=True)
    show_advanced = st.checkbox("Show Advanced Options", value=False)
    
    if show_advanced:
        try_shap = st.checkbox("Enable Ad-hoc SHAP Analysis", value=False)
        download_format = st.selectbox("Download Format", ["CSV", "Excel", "JSON"])
    else:
        try_shap = False
        download_format = "CSV"
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Info")
    try:
        model, base_model, model_name, TRAIN_FEATURES = load_model_and_features()
        st.success(f"**Model:** {model_name.replace('model_', '').replace('.pkl', '').title()}")
        st.info(f"**Features:** {len(TRAIN_FEATURES)}")
        st.info(f"**Type:** {type(base_model).__name__}")
    except Exception as e:
        st.error("⚠️ Model not loaded")
    
    st.markdown("---")
    
    st.caption("💡 **Tip:** Adjust the threshold to balance precision vs recall")
    st.caption("📚 [Documentation](#) | [GitHub](https://github.com/TJawdin)")

# ----------------------- Header ------------------------------
st.markdown('<div class="main-header">📦 Supply-Chain Delay Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered delivery delay prediction at order time</div>', unsafe_allow_html=True)

# ----------------------- Load Artifacts ----------------------
try:
    model, base_model, model_name, TRAIN_FEATURES = load_model_and_features()
    st.markdown(f"""
    <div class="success-box">
        ✅ <strong>Model Ready:</strong> {model_name} | {len(TRAIN_FEATURES)} features loaded
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"❌ **Failed to load artifacts:** {e}")
    st.stop()

# ----------------------- Tabs Layout -------------------------
tab_pred, tab_single, tab_explain, tab_analytics = st.tabs([
    "📊 Batch Predictions", 
    "📝 Single Prediction", 
    "🔍 Model Explainability",
    "📈 Analytics Dashboard"  # New tab for future features
])

# ===================== TAB 1: Batch Predict ==================
with tab_pred:
    st.header("📊 Batch Predictions from CSV")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Upload a CSV file with order-level data. The app will automatically:
        - Align columns to the training schema
        - Fill missing features with zeros
        - Generate predictions for all rows
        """)
    
    with col2:
        sample_available = Path("data/sample_input.csv").exists()
        if sample_available:
            st.info("📥 **Demo Available**")
            with open("data/sample_input.csv", "rb") as f:
                st.download_button(
                    "⬇️ Download Sample CSV",
                    data=f,
                    file_name="sample_input.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    uploaded = st.file_uploader("Upload CSV File", type=["csv"], key="batch_upload")
    
    with st.expander("ℹ️ What columns do I need?"):
        st.write(f"**Training features ({len(TRAIN_FEATURES)}):**")
        
        # Show features in a nice formatted way
        feature_categories = {}
        for feat in TRAIN_FEATURES:
            category = feat.split('.')[0] if '.' in feat else "Other"
            if category not in feature_categories:
                feature_categories[category] = []
            feature_categories[category].append(feat)
        
        for category, feats in sorted(feature_categories.items()):
            with st.expander(f"📁 {category} ({len(feats)} features)"):
                st.write(", ".join(feats[:20]))
                if len(feats) > 20:
                    st.caption(f"... and {len(feats) - 20} more")

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            
            # Validation
            if len(df_in) == 0:
                st.warning("⚠️ Uploaded CSV is empty!")
                st.stop()
            
            if len(df_in) > 50000:
                st.warning(f"⚠️ Large file detected ({len(df_in):,} rows). Processing may take a moment...")
            
            st.markdown("### 📋 Input Data Preview")
            st.dataframe(df_in.head(10), use_container_width=True)
            
            # Predict
            with st.spinner("🔮 Making predictions..."):
                proba, pred = predict_batch(model, df_in, TRAIN_FEATURES, threshold)
            
            # Add results to output
            out = df_in.copy()
            out["late_probability"] = proba.round(4)
            out["late_pred"] = pred
            out["late_pred_label"] = out["late_pred"].map({0: "On-time", 1: "Late"})
            out["risk_category"] = pd.cut(
                proba, 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=["Low", "Medium", "High"]
            )
            
            # Summary metrics
            st.markdown("### 📊 Prediction Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Orders", f"{len(out):,}")
            with col2:
                late_count = (pred == 1).sum()
                late_pct = late_count / len(out) * 100
                st.metric("Predicted Late", f"{late_count:,}", f"{late_pct:.1f}%")
            with col3:
                ontime_count = (pred == 0).sum()
                ontime_pct = ontime_count / len(out) * 100
                st.metric("Predicted On-time", f"{ontime_count:,}", f"{ontime_pct:.1f}%")
            with col4:
                avg_prob = proba.mean()
                st.metric("Avg Risk Score", f"{avg_prob:.3f}")
            
            # Risk distribution
            st.markdown("### 📈 Risk Distribution")
            risk_counts = out["risk_category"].value_counts()
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.bar_chart(risk_counts)
            with col2:
                st.dataframe(
                    pd.DataFrame({
                        "Risk Level": risk_counts.index,
                        "Count": risk_counts.values,
                        "Percentage": (risk_counts.values / len(out) * 100).round(1)
                    }),
                    hide_index=True
                )
            
            # Results table
            st.markdown("### 🎯 Detailed Results")
            
            # Filter options
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                risk_filter = st.multiselect(
                    "Filter by Risk Category",
                    options=["Low", "Medium", "High"],
                    default=["Low", "Medium", "High"]
                )
            with filter_col2:
                pred_filter = st.multiselect(
                    "Filter by Prediction",
                    options=["On-time", "Late"],
                    default=["On-time", "Late"]
                )
            
            # Apply filters
            filtered_out = out[
                (out["risk_category"].isin(risk_filter)) &
                (out["late_pred_label"].isin(pred_filter))
            ]
            
            st.dataframe(
                filtered_out.head(100), 
                use_container_width=True,
                height=400
            )
            
            if len(filtered_out) > 100:
                st.info(f"ℹ️ Showing first 100 of {len(filtered_out):,} filtered results. Download full results below.")
            
            # Download section
            st.markdown("### 📥 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Full Results (CSV)",
                    data=csv,
                    file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # High risk only
                high_risk = out[out["risk_category"] == "High"]
                if len(high_risk) > 0:
                    csv_high = high_risk.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⚠️ Download High Risk Only",
                        data=csv_high,
                        file_name=f"high_risk_orders_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            import traceback
            with st.expander("🐛 Show error details"):
                st.code(traceback.format_exc())

# ===================== TAB 2: Single Predict =================
with tab_single:
    st.header("📝 Single Order Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Enter feature values manually or paste a JSON object. 
        Any omitted features will default to 0.0.
        """)
    
    with col2:
        st.info("💡 **Quick Tip**\n\nUse the JSON input for faster entry of multiple features")
    
    # Show top N features in form
    N_TOP = min(12, len(TRAIN_FEATURES))
    
    with st.form(key="single_form"):
        st.markdown(f"**Enter values for top {N_TOP} features:**")
        cols = st.columns(2)
        inputs = {}
        
        for i, feat in enumerate(TRAIN_FEATURES[:N_TOP]):
            col = cols[i % 2]
            inputs[feat] = col.number_input(
                feat, 
                value=0.0, 
                step=0.1, 
                format="%.4f",
                help=f"Feature {i+1} of {len(TRAIN_FEATURES)}"
            )
        
        st.markdown("---")
        st.markdown("**🔧 Advanced: JSON Override**")
        
        example_json = {TRAIN_FEATURES[0]: 1.2, TRAIN_FEATURES[1]: 0.3}
        raw_json = st.text_area(
            "JSON input (optional)",
            value="",
            height=120,
            placeholder=f"Example: {json.dumps(example_json)}",
            help="Any values here will override the form inputs above"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
        with col2:
            reset = st.form_submit_button("🔄 Reset", use_container_width=True)

    if submitted:
        try:
            # Build base feature dict
            base = {c: 0.0 for c in TRAIN_FEATURES}
            
            # Add form inputs
            for k, v in inputs.items():
                base[k] = float(v)
            
            # Override with JSON if provided
            if raw_json.strip():
                try:
                    j = json.loads(raw_json)
                    assert isinstance(j, dict), "JSON must be a single object (key/value pairs)"
                    for k, v in j.items():
                        base[k] = float(v)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
                    st.stop()
            
            # Make prediction
            df_one = pd.DataFrame([base])
            proba, pred = predict_batch(model, df_one, TRAIN_FEATURES, threshold)
            
            # Display results with visual appeal
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Late Probability", 
                    f"{proba[0]:.3f}",
                    delta=f"{(proba[0] - 0.5):.3f} from threshold" if proba[0] != 0.5 else None
                )
            
            with col2:
                pred_label = "🔴 Late" if pred[0] == 1 else "🟢 On-time"
                st.metric("Prediction", pred_label)
            
            with col3:
                if proba[0] >= 0.7:
                    risk = "🔴 High Risk"
                elif proba[0] >= 0.3:
                    risk = "🟡 Medium Risk"
                else:
                    risk = "🟢 Low Risk"
                st.metric("Risk Level", risk)
            
            # Progress bar for visual representation
            st.markdown("**Risk Score Visualization:**")
            st.progress(float(proba[0]))
            
            # Show feature values used
            with st.expander("📋 Feature values used in prediction"):
                # Show non-zero features first
                non_zero = {k: v for k, v in base.items() if v != 0.0}
                if non_zero:
                    st.markdown("**Non-zero features:**")
                    st.json(non_zero)
                
                with st.expander("Show all features"):
                    st.json(base)
                    
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            import traceback
            with st.expander("🐛 Show error details"):
                st.code(traceback.format_exc())

# ===================== TAB 3: Explainability ================
with tab_explain:
    st.header("🔍 Model Explainability")
    st.caption("Understand which features drive predictions and model behavior")

    # Global importance section
    st.markdown("### 🌍 Global Feature Importance")
    
    cols = st.columns(2)
    
    # Permutation importance from artifacts
    with cols[0]:
        st.markdown("#### Permutation Importance")
        if GLOBAL_PI_PNG.exists() and show_importance:
            st.image(str(GLOBAL_PI_PNG), caption="Model-agnostic importance (from Step 8)")
        else:
            st.info("📊 Permutation importance plot not found. Run Step 8 to generate.")
    
    # Native model importance
    with cols[1]:
        st.markdown("#### Native Model Importance")
        if show_importance:
            if hasattr(base_model, "feature_importances_"):
                imp = pd.Series(base_model.feature_importances_, index=TRAIN_FEATURES)
                imp = imp.sort_values(ascending=False)
                
                # Interactive top-k selector
                top_k = st.slider("Show top N features", 5, 50, 20, key="importance_topk")
                st.bar_chart(imp.head(top_k))
                
                with st.expander("📊 View importance values"):
                    st.dataframe(
                        imp.head(top_k).reset_index().rename(columns={"index": "Feature", 0: "Importance"}),
                        hide_index=True
                    )
                
            elif hasattr(base_model, "coef_"):
                coefs = np.ravel(base_model.coef_)
                imp = pd.Series(np.abs(coefs), index=TRAIN_FEATURES)
                imp = imp.sort_values(ascending=False)
                
                top_k = st.slider("Show top N features", 5, 50, 20, key="coef_topk")
                st.bar_chart(imp.head(top_k))
                
                with st.expander("📊 View coefficient values"):
                    st.dataframe(
                        imp.head(top_k).reset_index().rename(columns={"index": "Feature", 0: "|Coefficient|"}),
                        hide_index=True
                    )
            else:
                st.info("ℹ️ Native feature importances not available for this model type.")
        else:
            st.info("Enable 'Show Feature Importance' in sidebar.")

    # SHAP section
    st.markdown("---")
    st.markdown("### 🎯 SHAP Analysis")
    st.caption("SHapley Additive exPlanations - Shows how each feature contributes to predictions")
    
    shap_cols = st.columns(2)
    
    with shap_cols[0]:
        if SHAP_SUMMARY_BEE.exists():
            st.image(str(SHAP_SUMMARY_BEE), caption="SHAP Beeswarm - Feature impact distribution")
        else:
            st.info("📊 SHAP beeswarm plot not found. Run Step 8 with SHAP to generate.")
    
    with shap_cols[1]:
        if SHAP_SUMMARY_BAR.exists():
            st.image(str(SHAP_SUMMARY_BAR), caption="SHAP Bar - Average feature importance")
        else:
            st.info("📊 SHAP bar plot not found. Run Step 8 with SHAP to generate.")

    # Optional: Ad-hoc SHAP
    if try_shap:
        st.markdown("---")
        st.markdown("### 🔬 Ad-hoc SHAP Analysis")
        st.warning("⚠️ This generates SHAP values on-the-fly and may be slow. For production-quality SHAP, use artifacts from Step 8.")
        
        if st.button("🚀 Generate SHAP Analysis", use_container_width=True):
            try:
                import shap
                import matplotlib.pyplot as plt
                
                with st.spinner("Computing SHAP values... This may take a minute..."):
                    # Create synthetic sample
                    rng = np.random.RandomState(42)
                    n_samples = 200
                    synth = pd.DataFrame(0.0, index=range(n_samples), columns=TRAIN_FEATURES)
                    
                    # Add variance
                    n_vary = min(5, len(TRAIN_FEATURES))
                    synth.iloc[:, :n_vary] = rng.randn(n_samples, n_vary)
                    
                    Xs = align_features(synth, TRAIN_FEATURES)
                    
                    # Choose explainer
                    model_type_lower = type(base_model).__name__.lower()
                    
                    if "tree" in model_type_lower or "forest" in model_type_lower or \
                       "xgb" in model_type_lower or "catboost" in model_type_lower:
                        explainer = shap.TreeExplainer(base_model)
                        shap_values = explainer.shap_values(Xs)
                        
                        if isinstance(shap_values, list) and len(shap_values) == 2:
                            shap_values = shap_values[1]
                            
                    elif "logistic" in model_type_lower or "linear" in model_type_lower:
                        explainer = shap.LinearExplainer(base_model, Xs)
                        shap_values = explainer.shap_values(Xs)
                        
                    else:
                        st.warning("Using KernelExplainer (very slow)...")
                        background = shap.sample(Xs, 50)
                        explainer = shap.KernelExplainer(model.predict_proba, background)
                        shap_values = explainer.shap_values(Xs.sample(100, random_state=42))
                        
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]
                    
                    # Display
                    st.success("✅ SHAP values computed!")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, Xs, show=False, max_display=20)
                    st.pyplot(fig)
                    plt.close()
                    
            except ImportError:
                st.error("❌ SHAP package not installed. Install with: `pip install shap`")
            except Exception as e:
                st.error(f"❌ SHAP analysis failed: {e}")
                with st.expander("🐛 Show error details"):
                    import traceback
                    st.code(traceback.format_exc())

# ===================== TAB 4: Analytics Dashboard ===========
with tab_analytics:
    st.header("📈 Analytics Dashboard")
    st.info("🚧 **Coming Soon!** This section will include:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🗺️ Geographic Analysis
        - **Interactive Maps**: Visualize delivery delays by region
        - **Heatmaps**: Identify high-risk geographic clusters
        - **Route Optimization**: Analyze shipping routes and delays
        
        ### 📊 Time Series Analysis
        - **Trend Analysis**: Delay patterns over time
        - **Seasonality**: Identify peak delay periods
        - **Forecasting**: Predict future delay rates
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Performance Metrics
        - **Model Monitoring**: Track prediction accuracy over time
        - **Feature Drift**: Detect changes in feature distributions
        - **A/B Testing**: Compare model versions
        
        ### 💼 Business Intelligence
        - **Cost Analysis**: Financial impact of delays
        - **Customer Segments**: Delay patterns by customer type
        - **Vendor Performance**: Analyze seller/carrier reliability
        """)
    
    st.markdown("---")
    
    # Placeholder for future visualizations
    st.markdown("### 🎨 Preview: Sample Visualizations")
    
    tab_map, tab_time, tab_corr = st.tabs(["🗺️ Map View", "📈 Time Series", "🔥 Correlation"])
    
    with tab_map:
        st.info("Geographic heatmap will be displayed here using Plotly or Folium")
        st.code("""
        # Future implementation:
        import plotly.express as px
        fig = px.density_mapbox(
            data, lat='latitude', lon='longitude', 
            z='late_probability', radius=10,
            mapbox_style="carto-positron", zoom=3
        )
        st.plotly_chart(fig)
        """, language="python")
    
    with tab_time:
        st.info("Time series analysis will be displayed here")
        st.code("""
        # Future implementation:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=delay_rate, mode='lines+markers'))
        fig.update_layout(title='Delay Rate Over Time')
        st.plotly_chart(fig)
        """, language="python")
    
    with tab_corr:
        st.info("Feature correlation heatmap will be displayed here")
        st.code("""
        # Future implementation:
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        """, language="python")

# ----------------------- Footer ------------------------------
st.markdown("---")
st.markdown("### 💡 Tips & Best Practices")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🎯 Threshold Tuning**
    - **Lower threshold** (e.g., 0.3): Catch more potential delays (high recall)
    - **Higher threshold** (e.g., 0.7): Fewer false alarms (high precision)
    - **Default (0.5)**: Balanced approach
    """)

with col2:
    st.markdown("""
    **📊 Batch Processing**
    - Files < 10K rows: Process instantly
    - Files 10K-50K rows: May take 10-30 seconds
    - Files > 50K rows: Consider splitting into batches
    - Maximum file size: 200MB
    """)

with col3:
    st.markdown("""
    **🔍 Interpreting Results**
    - **Probability**: Confidence in the prediction (0-1)
    - **Risk Category**: Low (<0.3), Medium (0.3-0.7), High (>0.7)
    - **SHAP Values**: Feature contribution to prediction
    """)

st.markdown("---")

# Footer info
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("📦 **Supply-Chain Delay Predictor**")
    st.caption("Version 1.0.0 | Built with Streamlit")

with footer_col2:
    st.caption("👤 **Author:** TJawdin")
    st.caption("[GitHub](https://github.com/TJawdin) | [LinkedIn](#)")

with footer_col3:
    st.caption("📅 **Last Updated:** 2024-12-19")
    st.caption(f"🎯 **Model:** {model_name.replace('model_', '').replace('.pkl', '').upper()}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Trained on Olist Brazilian E-Commerce data")


