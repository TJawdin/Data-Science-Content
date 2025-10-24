"""
Diagnostic Page - Model Testing & Debugging
Shows what the model actually predicts for different scenarios
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    load_model_artifacts,
    prepare_features,
    predict_delay,
    create_example_order,
    apply_custom_css
)

# Page config
st.set_page_config(page_title="Model Diagnostics", page_icon="🔬", layout="wide")
apply_custom_css()

# Header
st.title("🔬 Model Diagnostics & Testing")
st.markdown("### Debug and understand what the model actually predicts")
st.markdown("---")

# Load model
model, final_metadata, feature_metadata, threshold = load_model_artifacts()

# Display model info
st.subheader("📊 Model Information")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Type", final_metadata['best_model'])
with col2:
    st.metric("AUC", f"{final_metadata['best_model_auc']:.1%}")
with col3:
    st.metric("Optimal Threshold", f"{threshold:.4f}")
with col4:
    st.metric("Training Date", final_metadata['training_date'])

st.markdown("---")

# Test scenarios
st.subheader("🧪 Scenario Testing")
st.markdown("Testing each pre-built scenario to see what the model predicts:")

results_data = []

for scenario_name in ['low_risk', 'typical', 'high_risk']:
    st.markdown(f"### {scenario_name.replace('_', ' ').title()}")
    
    # Get scenario
    scenario_data = create_example_order(scenario_name)
    
    # Display key features
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Key Features:**")
        key_features = {
            'purch_month': scenario_data['purch_month'],
            'est_lead_days': scenario_data['est_lead_days'],
            'customer_state': scenario_data['customer_state'],
            'sum_freight': scenario_data['sum_freight'],
            'n_items': scenario_data['n_items'],
            'n_sellers': scenario_data['n_sellers'],
            'avg_weight_g': scenario_data['avg_weight_g']
        }
        
        for feat, val in key_features.items():
            st.text(f"  {feat}: {val}")
    
    # Make prediction
    features_df = prepare_features(scenario_data, feature_metadata['feature_names'])
    predictions, probabilities, risk_levels = predict_delay(model, features_df, threshold)
    
    prob_pct = probabilities[0] * 100
    
    with col2:
        st.markdown("**Prediction Results:**")
        st.metric("Delay Probability", f"{prob_pct:.2f}%")
        st.metric("Risk Level", risk_levels[0])
        st.metric("Binary Prediction", "Delayed" if predictions[0] == 1 else "On-Time")
        
        # Color code based on risk
        if risk_levels[0] == 'Low':
            st.success(f"✅ Below threshold ({threshold:.4f})")
        elif risk_levels[0] == 'Medium':
            st.warning(f"⚠️ Moderate risk")
        else:
            st.error(f"🚨 Above threshold ({threshold:.4f})")
    
    # Store for comparison
    results_data.append({
        'Scenario': scenario_name.replace('_', ' ').title(),
        'Month': scenario_data['purch_month'],
        'Lead Days': scenario_data['est_lead_days'],
        'State': scenario_data['customer_state'],
        'Freight': scenario_data['sum_freight'],
        'Probability': f"{prob_pct:.2f}%",
        'Risk Level': risk_levels[0],
        'Above Threshold': probabilities[0] >= threshold
    })
    
    st.markdown("---")

# Comparison table
st.subheader("📋 Scenario Comparison")
comparison_df = pd.DataFrame(results_data)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Feature importance from model
st.subheader("📈 Model Feature Importance")

if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_metadata['feature_names'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.markdown("**Top 20 Most Important Features:**")
    
    # Display as table
    top_features = feature_importance.head(20).copy()
    top_features['Importance'] = top_features['Importance'].apply(lambda x: f"{x:.6f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(
            top_features.head(10).reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.dataframe(
            top_features.tail(10).reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("Feature importance not available for this model type")

st.markdown("---")

# Individual feature testing
st.subheader("🔧 Interactive Feature Testing")
st.markdown("Test how changing individual features affects predictions")

# Start with low_risk baseline
baseline = create_example_order('low_risk')

col1, col2, col3 = st.columns(3)

with col1:
    test_month = st.selectbox(
        "Purchase Month",
        options=list(range(1, 13)),
        index=6  # Default to July (month 7, index 6)
    )

with col2:
    test_lead_days = st.slider(
        "Est. Lead Days",
        min_value=1.0,
        max_value=30.0,
        value=5.0,
        step=1.0
    )

with col3:
    test_state = st.selectbox(
        "Customer State",
        options=['SP', 'RJ', 'MG', 'RS', 'PR', 'BA', 'AM'],
        index=0  # Default to SP
    )

col4, col5 = st.columns(2)

with col4:
    test_freight = st.slider(
        "Sum Freight",
        min_value=5.0,
        max_value=100.0,
        value=25.0,
        step=5.0
    )

with col5:
    test_n_sellers = st.slider(
        "Number of Sellers",
        min_value=1,
        max_value=5,
        value=1
    )

if st.button("🔮 Run Test Prediction", type="primary"):
    # Create test scenario
    test_scenario = baseline.copy()
    test_scenario['purch_month'] = test_month
    test_scenario['est_lead_days'] = test_lead_days
    test_scenario['customer_state'] = test_state
    test_scenario['sum_freight'] = test_freight
    test_scenario['n_sellers'] = test_n_sellers
    
    # Update sin/cos for consistency (though likely not important)
    hour = test_scenario['purch_hour']
    test_scenario['purch_hour_sin'] = np.sin(2 * np.pi * hour / 24)
    test_scenario['purch_hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Make prediction
    features_df = prepare_features(test_scenario, feature_metadata['feature_names'])
    predictions, probabilities, risk_levels = predict_delay(model, features_df, threshold)
    
    prob_pct = probabilities[0] * 100
    
    st.markdown("### Test Results:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Delay Probability", f"{prob_pct:.2f}%")
    with col2:
        st.metric("Risk Level", risk_levels[0])
    with col3:
        st.metric("Prediction", "Delayed" if predictions[0] == 1 else "On-Time")
    
    # Visual indicator
    if risk_levels[0] == 'High':
        st.error(f"🚨 High Risk: {prob_pct:.2f}% probability of delay")
    elif risk_levels[0] == 'Medium':
        st.warning(f"⚠️ Medium Risk: {prob_pct:.2f}% probability of delay")
    else:
        st.success(f"✅ Low Risk: {prob_pct:.2f}% probability of delay")

st.markdown("---")

# Expected behavior notes
st.subheader("📝 Expected Behavior (Based on SHAP Analysis)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Features that INCREASE delay risk:**
    - Months: 2, 3, 4, 11, 12 (early year & holidays)
    - Higher est_lead_days
    - Customer states: MG, PR, BA (not SP!)
    - LOWER sum_freight (counterintuitive!)
    - More sellers
    """)

with col2:
    st.markdown("""
    **Features that DECREASE delay risk:**
    - Months: 5-10 (mid-year)
    - Lower est_lead_days
    - Customer state: SP (São Paulo)
    - HIGHER sum_freight (premium shipping)
    - Single seller
    """)

# Footer
st.markdown("---")
st.info("💡 Use this page to verify the model is working correctly and scenarios are properly configured")
