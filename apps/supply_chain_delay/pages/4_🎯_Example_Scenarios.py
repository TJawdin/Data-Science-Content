"""
Example Scenarios Page
Pre-loaded scenarios for quick testing and demos
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_engineering import calculate_features, get_feature_descriptions
from utils.model_loader import load_model, predict_single
from utils.visualization import create_risk_gauge

# Page config
st.set_page_config(
    page_title="Example Scenarios",
    page_icon="🎯",
    layout="wide"
)
from utils.theme_adaptive import apply_adaptive_theme

# Apply theme right after page config
apply_adaptive_theme()
# ============================================================================
# Header
# ============================================================================

st.title("🎯 Example Scenarios")
st.markdown("""
Quick-test the model with pre-configured realistic scenarios.
Perfect for demos, training, and understanding model behavior!
""")

st.markdown("---")

# ============================================================================
# Load Model
# ============================================================================

model = load_model()

if model is None:
    st.error("⚠️ Model not found. Please copy your trained model to the artifacts folder.")
    st.stop()

# ============================================================================
# Define Example Scenarios
# ============================================================================

scenarios = {
    "🟢 LOW RISK: Standard Local Order": {
        'description': "Single item, local delivery, generous timeframe, low shipping cost",
        'data': {
            'num_items': 1,
            'num_sellers': 1,
            'num_products': 1,
            'total_order_value': 120.0,
            'avg_item_price': 120.0,
            'max_item_price': 120.0,
            'total_shipping_cost': 8.0,
            'avg_shipping_cost': 8.0,
            'total_weight_g': 800,
            'avg_weight_g': 800,
            'max_weight_g': 800,
            'avg_length_cm': 25.0,
            'avg_height_cm': 18.0,
            'avg_width_cm': 12.0,
            'avg_shipping_distance_km': 80,
            'max_shipping_distance_km': 80,
            'is_cross_state': 0,
            'order_weekday': 2,
            'order_month': 5,
            'order_hour': 10,
            'is_weekend_order': 0,
            'is_holiday_season': 0,
            'estimated_days': 12
        },
        'color': 'green'
    },
    
    "🟡 MEDIUM RISK: Multi-Item Cross-State": {
        'description': "Multiple items, different state, moderate distance",
        'data': {
            'num_items': 3,
            'num_sellers': 2,
            'num_products': 3,
            'total_order_value': 200.0,
            'avg_item_price': 66.67,
            'max_item_price': 100.0,
            'total_shipping_cost': 35.0,
            'avg_shipping_cost': 11.67,
            'total_weight_g': 3000,
            'avg_weight_g': 1000,
            'max_weight_g': 1500,
            'avg_length_cm': 35.0,
            'avg_height_cm': 25.0,
            'avg_width_cm': 20.0,
            'avg_shipping_distance_km': 600,
            'max_shipping_distance_km': 800,
            'is_cross_state': 1,
            'order_weekday': 5,
            'order_month': 6,
            'order_hour': 18,
            'is_weekend_order': 1,
            'estimated_days': 10
        },
        'color': 'orange'
    },
    
    "🔴 HIGH RISK: Rush Long-Distance Multi-Seller": {
        'description': "Rush order, very long distance, multiple sellers, heavy items",
        'data': {
            'num_items': 5,
            'num_sellers': 3,
            'num_products': 5,
            'total_order_value': 500.0,
            'avg_item_price': 100.0,
            'max_item_price': 200.0,
            'total_shipping_cost': 80.0,
            'avg_shipping_cost': 16.0,
            'total_weight_g': 8000,
            'avg_weight_g': 1600,
            'max_weight_g': 3000,
            'avg_length_cm': 50.0,
            'avg_height_cm': 40.0,
            'avg_width_cm': 30.0,
            'avg_shipping_distance_km': 1500,
            'max_shipping_distance_km': 1800,
            'is_cross_state': 1,
            'order_weekday': 6,
            'order_month': 12,
            'order_hour': 20,
            'is_weekend_order': 1,
            'is_holiday_season': 1,
            'estimated_days': 5
        },
        'color': 'red'
    },
    
    "🟢 LOW RISK: Same-State Small Package": {
        'description': "Small, light, same state, plenty of time",
        'data': {
            'num_items': 1,
            'num_sellers': 1,
            'num_products': 1,
            'total_order_value': 85.0,
            'avg_item_price': 85.0,
            'max_item_price': 85.0,
            'total_shipping_cost': 6.0,
            'avg_shipping_cost': 6.0,
            'total_weight_g': 300,
            'avg_weight_g': 300,
            'max_weight_g': 300,
            'avg_length_cm': 18.0,
            'avg_height_cm': 12.0,
            'avg_width_cm': 8.0,
            'avg_shipping_distance_km': 65,
            'max_shipping_distance_km': 65,
            'is_cross_state': 0,
            'order_weekday': 1,
            'order_month': 4,
            'order_hour': 11,
            'is_weekend_order': 0,
            'is_holiday_season': 0,
            'estimated_days': 18
        },
        'color': 'green'
    },
    
    "🟡 MEDIUM RISK: Holiday Season Order": {
        'description': "Holiday season, moderate complexity",
        'data': {
            'num_items': 4,
            'num_sellers': 2,
            'num_products': 4,
            'total_order_value': 300.0,
            'avg_item_price': 75.0,
            'max_item_price': 120.0,
            'total_shipping_cost': 40.0,
            'avg_shipping_cost': 10.0,
            'total_weight_g': 4000,
            'avg_weight_g': 1000,
            'max_weight_g': 1800,
            'avg_length_cm': 40.0,
            'avg_height_cm': 30.0,
            'avg_width_cm': 25.0,
            'avg_shipping_distance_km': 500,
            'max_shipping_distance_km': 600,
            'is_cross_state': 1,
            'order_weekday': 4,
            'order_month': 11,
            'order_hour': 16,
            'is_weekend_order': 0,
            'is_holiday_season': 1,
            'estimated_days': 8
        },
        'color': 'orange'
    }
}
# ============================================================================
# Scenario Selection
# ============================================================================

st.markdown("## 📋 Select a Scenario to Test")

# Create buttons for each scenario
selected_scenario = None

cols = st.columns(len(scenarios))

for idx, (scenario_name, scenario_data) in enumerate(scenarios.items()):
    with cols[idx]:
        if st.button(
            scenario_name,
            use_container_width=True,
            type="secondary"
        ):
            selected_scenario = scenario_name

# If no scenario selected yet, show first one by default
if selected_scenario is None:
    selected_scenario = list(scenarios.keys())[0]

st.markdown("---")

# ============================================================================
# Display Selected Scenario
# ============================================================================

st.markdown(f"## {selected_scenario}")

scenario = scenarios[selected_scenario]

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Scenario Description")
    st.info(scenario['description'])
    
    st.markdown("### 📊 Order Details")
    
    # Show key details in a nice format
    key_details = {
        'Number of Items': scenario['data']['num_items'],
        'Number of Sellers': scenario['data']['num_sellers'],
        'Total Order Value': f"${scenario['data']['total_order_value']:.2f}",
        'Total Shipping Cost': f"${scenario['data']['total_shipping_cost']:.2f}",
        'Total Weight': f"{scenario['data']['total_weight_g']}g",
        'Shipping Distance': f"{scenario['data']['avg_shipping_distance_km']}km",
        'Cross-State': 'Yes' if scenario['data'].get('is_cross_state', 0) == 1 else 'No',
        'Weekend Order': 'Yes' if scenario['data'].get('is_weekend_order', 0) == 1 else 'No',
        'Holiday Season': 'Yes' if scenario['data'].get('is_holiday_season', 0) == 1 else 'No',
        'Estimated Delivery': f"{scenario['data']['estimated_days']} days"
    }
    
    # Create table
    details_df = pd.DataFrame({
        'Attribute': list(key_details.keys()),
        'Value': list(key_details.values())
    })
    
    st.table(details_df)

with col2:
    # Make prediction
    with st.spinner("Calculating risk..."):
        try:
            features_df = calculate_features(scenario['data'])
            result = predict_single(model, features_df)
            
            if result:
                # Display risk gauge
                fig = create_risk_gauge(result['risk_score'], result['risk_level'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Prediction", result['prediction_label'])
                
                with col_b:
                    st.metric("Risk Score", f"{result['risk_score']}/100")
                
                with col_c:
                    st.metric("Risk Level", result['risk_level'])
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            result = None

st.markdown("---")

# ============================================================================
# Recommendations
# ============================================================================

if result:
    st.markdown("### 💡 Recommended Actions")
    
    if result['risk_level'] == 'HIGH':
        st.error("""
        **🚨 HIGH RISK - Immediate Action Required:**
        - ⚡ Upgrade to expedited shipping immediately
        - 📞 Proactively contact customer with realistic timeline
        - 🏷️ Flag order for priority processing in warehouse
        - 📦 Consider splitting order across warehouses if possible
        - 💰 Budget for potential refund/compensation
        - 📊 Daily monitoring until delivery confirmed
        """)
    
    elif result['risk_level'] == 'MEDIUM':
        st.warning("""
        **⚠️ MEDIUM RISK - Monitor Closely:**
        - 👀 Add to daily monitoring watchlist
        - 📧 Send automated tracking updates to customer
        - 🚚 Ensure optimal carrier selection for route
        - 📊 Review shipping route for potential bottlenecks
        - 💬 Prepare customer service team for potential inquiries
        """)
    
    else:
        st.success("""
        **✅ LOW RISK - Standard Processing:**
        - ✓ Proceed with normal shipping workflow
        - ✓ Standard customer communication
        - ✓ No special intervention required
        - ✓ Monitor as part of regular batch processing
        """)

st.markdown("---")

# ============================================================================
# Feature Breakdown
# ============================================================================

with st.expander("🔍 View Detailed Feature Values"):
    try:
        feature_descriptions = get_feature_descriptions()
        
        if 'features_df' in locals():
            feature_display = []
            for col in features_df.columns:
                feature_display.append({
                    'Feature': feature_descriptions.get(col, col),
                    'Technical Name': col,
                    'Value': features_df[col].values[0]
                })
            
            st.dataframe(
                pd.DataFrame(feature_display),
                use_container_width=True,
                height=400
            )
    except Exception as e:
        st.error(f"Could not display feature breakdown: {str(e)}")

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## 🎯 Example Scenarios")
    st.info("""
    **Purpose:**
    - Quick model testing
    - Demo for stakeholders
    - Training new users
    - Understanding risk factors
    """)
    
    st.markdown("---")
    
    st.markdown("## 📊 Scenario Summary")
    st.markdown(f"""
    **Total Scenarios:** {len(scenarios)}
    - 🟢 Low Risk: 2
    - 🟡 Medium Risk: 2
    - 🔴 High Risk: 1
    """)
    
    st.markdown("---")
    
    st.markdown("## 💡 Tips")
    st.success("""
    **Compare scenarios** to understand:
    - How features impact risk
    - Which factors matter most
    - When to intervene
    - Expected model behavior
    """)
