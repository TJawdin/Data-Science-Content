"""
Example Scenarios Page
Pre-built scenarios demonstrating different risk levels
"""

import streamlit as st
import pandas as pd
from utils import (
    load_model_artifacts,
    predict_delay,
    prepare_features,
    create_example_order,
    apply_custom_css,
    show_page_header,
    display_risk_badge,
    plot_risk_gauge,
    plot_feature_importance,
    get_feature_descriptions
)

# Page config
st.set_page_config(page_title="Example Scenarios", page_icon="🎯", layout="wide")
apply_custom_css()

# Load model
model, final_metadata, feature_metadata, threshold = load_model_artifacts()

# Header
show_page_header(
    title="Example Scenarios",
    description="Explore pre-built order scenarios to understand how different factors affect delay risk",
    icon="🎯"
)

# Scenario selection
st.markdown("### 📋 Select a Scenario")

col1, col2, col3 = st.columns(3)

with col1:
    low_risk_btn = st.button("✅ Low Risk Order", use_container_width=True, type="secondary")

with col2:
    typical_btn = st.button("⚡ Typical Order", use_container_width=True, type="secondary")

with col3:
    high_risk_btn = st.button("🚨 High Risk Order", use_container_width=True, type="secondary")

# Initialize session state
if 'selected_scenario' not in st.session_state:
    st.session_state.selected_scenario = 'typical'

# Update scenario based on button clicks
if low_risk_btn:
    st.session_state.selected_scenario = 'low_risk'
if typical_btn:
    st.session_state.selected_scenario = 'typical'
if high_risk_btn:
    st.session_state.selected_scenario = 'high_risk'

# Get scenario data
scenario = st.session_state.selected_scenario
scenario_data = create_example_order(scenario)

st.markdown("---")

# Scenario description
scenario_descriptions = {
    'low_risk': {
        'title': '✅ Low Risk Order',
        'description': """
        This scenario represents an ideal order with minimal delay risk:
        - Single seller, single item
        - Small, lightweight product
        - Same-state delivery (São Paulo)
        - Credit card payment
        - Short estimated lead time
        - Weekday purchase during business hours
        """
    },
    'typical': {
        'title': '⚡ Typical Order',
        'description': """
        This scenario represents a standard e-commerce order:
        - Moderate number of items from one seller
        - Medium-sized products
        - Standard payment installments
        - Within-state or nearby delivery
        - Normal business day purchase
        - Average lead time
        """
    },
    'high_risk': {
        'title': '🚨 High Risk Order',
        'description': """
        This scenario represents a complex order with elevated delay risk:
        - Multiple items from multiple sellers
        - Large, heavy products
        - Cross-country delivery (São Paulo to Manaus)
        - Multiple product categories
        - Weekend/late night purchase
        - Long estimated lead time
        - Boleto payment (slower processing)
        """
    }
}

st.markdown(f"## {scenario_descriptions[scenario]['title']}")
st.markdown(scenario_descriptions[scenario]['description'])

st.markdown("---")

# Make prediction
features_df = prepare_features(scenario_data, feature_metadata['feature_names'])
predictions, probabilities, risk_levels = predict_delay(model, features_df, threshold)

prediction = predictions[0]
probability = probabilities[0]
risk_level = risk_levels[0]

# Display prediction results
st.markdown("### 🎯 Prediction Results")

col1, col2 = st.columns([1, 2])

with col1:
    display_risk_badge(risk_level, probability)
    
    st.markdown("#### Key Metrics")
    st.metric("Delay Probability", f"{probability*100:.1f}%")
    st.metric("Classification", "Delayed" if prediction == 1 else "On Time")
    st.metric("Risk Level", risk_level)

with col2:
    # Risk gauge
    fig_gauge = plot_risk_gauge(probability, threshold)
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# Order details
st.markdown("### 📦 Order Details")

# Organize features into categories
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📊 Order Characteristics")
    st.write(f"**Items**: {scenario_data['n_items']}")
    st.write(f"**Sellers**: {scenario_data['n_sellers']}")
    st.write(f"**Products**: {scenario_data['n_products']}")
    st.write(f"**Categories**: {scenario_data['n_categories']}")
    
    st.markdown("#### 💰 Financial")
    st.write(f"**Price**: R$ {scenario_data['sum_price']:.2f}")
    st.write(f"**Freight**: R$ {scenario_data['sum_freight']:.2f}")
    st.write(f"**Total**: R$ {scenario_data['total_payment']:.2f}")
    st.write(f"**Installments**: {scenario_data['max_installments']}")

with col2:
    st.markdown("#### 📐 Product Dimensions")
    st.write(f"**Weight**: {scenario_data['avg_weight_g']:.0f}g")
    st.write(f"**Length**: {scenario_data['avg_length_cm']:.0f}cm")
    st.write(f"**Height**: {scenario_data['avg_height_cm']:.0f}cm")
    st.write(f"**Width**: {scenario_data['avg_width_cm']:.0f}cm")
    
    st.markdown("#### 🗺️ Geographic")
    st.write(f"**Customer**: {scenario_data['customer_city'].title()}, {scenario_data['customer_state']}")
    st.write(f"**Seller State**: {scenario_data['seller_state_mode']}")
    st.write(f"**Seller States**: {scenario_data['n_seller_states']}")

with col3:
    st.markdown("#### ⏰ Temporal")
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    st.write(f"**Purchase Hour**: {scenario_data['purch_hour']}:00")
    st.write(f"**Day of Week**: {days_of_week[scenario_data['purch_dayofweek']]}")
    st.write(f"**Weekend**: {'Yes' if scenario_data['purch_is_weekend'] else 'No'}")
    st.write(f"**Est. Lead Days**: {scenario_data['est_lead_days']:.0f}")
    
    st.markdown("#### 💳 Payment")
    payment_types = []
    if scenario_data['paytype_credit_card']: payment_types.append("Credit Card")
    if scenario_data['paytype_debit_card']: payment_types.append("Debit Card")
    if scenario_data['paytype_boleto']: payment_types.append("Boleto")
    if scenario_data['paytype_voucher']: payment_types.append("Voucher")
    if scenario_data['paytype_not_defined']: payment_types.append("Not Defined")
    st.write(f"**Type**: {', '.join(payment_types)}")

st.markdown("---")

# Feature importance for this prediction
st.markdown("### 📈 Feature Importance")
st.markdown("*Top features contributing to the model's predictions*")

# Get feature importance from model
if hasattr(model, 'feature_importances_'):
    feature_importance = model.feature_importances_
    fig_importance = plot_feature_importance(
        feature_metadata['feature_names'],
        feature_importance,
        top_n=15
    )
    st.plotly_chart(fig_importance, use_container_width=True)
else:
    st.info("Feature importance visualization not available for this model type.")

st.markdown("---")

# Recommendations
st.markdown("### 💡 Recommendations")

if risk_level == 'High':
    st.error("⚠️ **High Risk - Immediate Action Required**")
    st.markdown("""
    **Recommended Actions:**
    1. 🚀 Consider expedited shipping options
    2. 📞 Proactively communicate with customer about delivery timeline
    3. 🔍 Review and verify seller inventory and shipping capacity
    4. 📊 Monitor this order closely throughout the fulfillment process
    5. 💼 Evaluate splitting multi-seller orders if feasible
    6. 🎯 Prioritize this order in warehouse operations
    
    **Key Risk Factors:**
    - Multiple sellers/states increase coordination complexity
    - Long-distance or cross-country delivery
    - Large/heavy items require special handling
    - Weekend/late-night orders may face processing delays
    """)

elif risk_level == 'Medium':
    st.warning("⚠️ **Medium Risk - Enhanced Monitoring Recommended**")
    st.markdown("""
    **Recommended Actions:**
    1. 👀 Monitor order progress more frequently than usual
    2. ✅ Verify adequate inventory at fulfillment centers
    3. 🚛 Confirm seller shipping capabilities and timelines
    4. 📱 Set realistic customer expectations for delivery
    5. 📋 Review logistics partner performance for this route
    
    **Key Risk Factors:**
    - Some complexity in order composition or logistics
    - Moderate distance or multiple items
    - Potential for minor delays
    """)

else:
    st.success("✅ **Low Risk - Standard Processing**")
    st.markdown("""
    **Recommended Actions:**
    1. ✨ Proceed with standard fulfillment process
    2. 📊 Continue routine monitoring for any unexpected issues
    3. 🎯 Maintain current shipping and handling practices
    4. 💚 This order is well-positioned for on-time delivery
    
    **Positive Factors:**
    - Simple order composition
    - Short delivery distance
    - Optimal timing and payment method
    - Favorable logistics characteristics
    """)

st.markdown("---")

# Compare scenarios
with st.expander("🔄 Compare All Scenarios"):
    st.markdown("### Scenario Comparison")
    
    # Get predictions for all scenarios
    scenarios_comparison = []
    for scn in ['low_risk', 'typical', 'high_risk']:
        scn_data = create_example_order(scn)
        scn_features = prepare_features(scn_data, feature_metadata['feature_names'])
        scn_pred, scn_prob, scn_risk = predict_delay(model, scn_features, threshold)
        
        scenarios_comparison.append({
            'Scenario': scn.replace('_', ' ').title(),
            'Delay Probability': f"{scn_prob[0]*100:.1f}%",
            'Risk Level': scn_risk[0],
            'Prediction': 'Delayed' if scn_pred[0] == 1 else 'On Time',
            'Items': scn_data['n_items'],
            'Sellers': scn_data['n_sellers'],
            'Total Payment': f"R$ {scn_data['total_payment']:.2f}",
            'Est. Lead Days': scn_data['est_lead_days']
        })
    
    comparison_df = pd.DataFrame(scenarios_comparison)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>💡 Try different scenarios to understand how various factors influence delay risk</p>
</div>
""", unsafe_allow_html=True)
