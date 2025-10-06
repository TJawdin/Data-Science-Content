"""
Single Order Prediction Page
User enters order details and gets instant prediction
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_engineering import calculate_features, get_feature_descriptions
from utils.model_loader import load_model, predict_single

# Page config
st.set_page_config(
    page_title="Single Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .risk-box-low {
        background-color: #D5F4E6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 10px solid #27AE60;
        margin: 1rem 0;
    }
    .risk-box-medium {
        background-color: #FEF5E7;
        padding: 2rem;
        border-radius: 10px;
        border-left: 10px solid #F39C12;
        margin: 1rem 0;
    }
    .risk-box-high {
        background-color: #FADBD8;
        padding: 2rem;
        border-radius: 10px;
        border-left: 10px solid #E74C3C;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Header
# ============================================================================

st.title("📊 Single Order Prediction")
st.markdown("Enter order details below to get an instant late delivery risk assessment.")

st.markdown("---")

# ============================================================================
# Load Model
# ============================================================================

model = load_model()

if model is None:
    st.error("""
    ⚠️ **Model Not Found**
    
    Please copy your trained model file from the notebook to:
    `apps/supply_chain_delay/artifacts/best_model_*.pkl`
    
    Steps:
    1. Find the model in your notebook's artifacts folder
    2. Copy to this app's artifacts folder
    3. Refresh this page
    """)
    st.stop()

# ============================================================================
# Input Form
# ============================================================================

st.markdown("### 📝 Order Details")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📦 Order Information")
    
    num_items = st.number_input(
        "Number of Items",
        min_value=1,
        max_value=20,
        value=2,
        help="How many items are in this order?"
    )
    
    num_sellers = st.number_input(
        "Number of Sellers",
        min_value=1,
        max_value=10,
        value=1,
        help="How many different sellers are involved?"
    )
    
    total_order_value = st.number_input(
        "Total Order Value ($)",
        min_value=0.0,
        max_value=10000.0,
        value=150.0,
        step=10.0,
        help="Total price of all items"
    )
    
    total_shipping_cost = st.number_input(
        "Total Shipping Cost ($)",
        min_value=0.0,
        max_value=500.0,
        value=20.0,
        step=5.0,
        help="Total shipping/freight cost"
    )
    
    st.markdown("#### 📏 Product Details")
    
    total_weight_g = st.number_input(
        "Total Weight (grams)",
        min_value=0,
        max_value=50000,
        value=2000,
        step=100,
        help="Total weight of all items"
    )
    
    avg_length_cm = st.number_input(
        "Average Length (cm)",
        min_value=0.0,
        max_value=200.0,
        value=30.0,
        step=5.0
    )
    
    avg_height_cm = st.number_input(
        "Average Height (cm)",
        min_value=0.0,
        max_value=200.0,
        value=20.0,
        step=5.0
    )
    
    avg_width_cm = st.number_input(
        "Average Width (cm)",
        min_value=0.0,
        max_value=200.0,
        value=15.0,
        step=5.0
    )

with col2:
    st.markdown("#### 🗺️ Geographic Information")
    
    avg_shipping_distance_km = st.number_input(
        "Shipping Distance (km)",
        min_value=0,
        max_value=5000,
        value=500,
        step=50,
        help="Distance between seller and customer"
    )
    
    is_cross_state = st.selectbox(
        "Cross-State Shipping?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Is customer in different state than seller?"
    )
    
    st.markdown("#### 📅 Timing Information")
    
    order_weekday = st.selectbox(
        "Order Day of Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                               'Friday', 'Saturday', 'Sunday'][x],
        index=2,
        help="Day of the week order was placed"
    )
    
    order_month = st.selectbox(
        "Order Month",
        options=list(range(1, 13)),
        format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1],
        index=5,
        help="Month order was placed"
    )
    
    order_hour = st.slider(
        "Order Hour (0-23)",
        min_value=0,
        max_value=23,
        value=14,
        help="Hour of day order was placed (0=midnight, 12=noon, 23=11pm)"
    )
    
    estimated_days = st.number_input(
        "Estimated Delivery Days",
        min_value=1,
        max_value=60,
        value=10,
        help="Promised delivery timeframe shown to customer"
    )

st.markdown("---")

# ============================================================================
# Predict Button
# ============================================================================

if st.button("🔮 Predict Late Delivery Risk", type="primary", use_container_width=True):
    
    with st.spinner("Calculating risk..."):
        
        # Prepare input data
        order_data = {
            'num_items': num_items,
            'num_sellers': num_sellers,
            'num_products': num_items,  # Simplified: assume 1 product per item
            'total_order_value': total_order_value,
            'avg_item_price': total_order_value / num_items,
            'max_item_price': total_order_value / num_items,
            'total_shipping_cost': total_shipping_cost,
            'avg_shipping_cost': total_shipping_cost / num_items,
            'total_weight_g': total_weight_g,
            'avg_weight_g': total_weight_g / num_items,
            'max_weight_g': total_weight_g / num_items,
            'avg_length_cm': avg_length_cm,
            'avg_height_cm': avg_height_cm,
            'avg_width_cm': avg_width_cm,
            'avg_shipping_distance_km': avg_shipping_distance_km,
            'max_shipping_distance_km': avg_shipping_distance_km,
            'is_cross_state': is_cross_state,
            'order_weekday': order_weekday,
            'order_month': order_month,
            'order_hour': order_hour,
            'is_weekend_order': 1 if order_weekday >= 5 else 0,
            'is_holiday_season': 1 if order_month in [11, 12] else 0,
            'estimated_days': estimated_days
        }
        
        # Calculate features
        features_df = calculate_features(order_data)
        
        # Make prediction
        result = predict_single(model, features_df)
        
        if result:
            # Display result
            st.markdown("---")
            st.markdown("## 🎯 Prediction Result")
            
            # Risk box based on level
            risk_box_class = f"risk-box-{result['risk_level'].lower()}"
            
            st.markdown(f"""
            <div class="{risk_box_class}">
                <div class="prediction-label" style="color: {result['risk_color']};">
                    {result['prediction_label'].upper()}
                </div>
                <h2 style="text-align: center;">Risk Score: {result['risk_score']}/100</h2>
                <h3 style="text-align: center;">Risk Level: {result['risk_level']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", result['prediction_label'])
            
            with col2:
                st.metric("Late Probability", f"{result['probability']:.1%}")
            
            with col3:
                st.metric("Risk Level", result['risk_level'])
            
            st.markdown("---")
            
            # Recommendations
            st.markdown("### 💡 Recommended Actions")
            
            if result['risk_level'] == 'HIGH':
                st.error("""
                **🚨 HIGH RISK - Immediate Action Required:**
                - ⚡ Upgrade to expedited shipping
                - 📞 Proactively contact customer to manage expectations
                - 🏷️ Flag order for priority processing
                - 📦 Consider splitting order across multiple warehouses
                - 💰 Budget for potential refund/compensation
                """)
            
            elif result['risk_level'] == 'MEDIUM':
                st.warning("""
                **⚠️ MEDIUM RISK - Monitor Closely:**
                - 👀 Add to watchlist for daily monitoring
                - 📧 Send automated tracking updates
                - 🚚 Ensure optimal carrier selection
                - 📊 Review shipping route for potential delays
                """)
            
            else:
                st.success("""
                **✅ LOW RISK - Standard Processing:**
                - ✓ Proceed with normal shipping workflow
                - ✓ Standard customer communication
                - ✓ No special intervention required
                """)
            
            st.markdown("---")
            
            # Feature values used
            with st.expander("🔍 View Feature Values Used"):
                st.dataframe(
                    features_df.T.rename(columns={0: 'Value'}),
                    use_container_width=True
                )

# ============================================================================
# Sidebar Info
# ============================================================================

with st.sidebar:
    st.markdown("## ℹ️ How It Works")
    st.info("""
    This tool uses machine learning to predict late delivery risk based on:
    
    - **Order complexity** (items, sellers)
    - **Financial** (order value, shipping cost)
    - **Physical** (weight, dimensions)
    - **Geographic** (distance, cross-state)
    - **Temporal** (day, time, season)
    """)
    
    st.markdown("---")
    
    st.markdown("## 📊 Risk Levels")
    st.markdown("""
    - **LOW** (0-30): Proceed normally
    - **MEDIUM** (30-70): Monitor closely
    - **HIGH** (70-100): Take immediate action
    """)
    
    st.markdown("---")
    
    st.markdown("## 🎯 Tips")
    st.markdown("""
    **High-risk factors:**
    - Long shipping distances
    - Multi-seller orders
    - Heavy/bulky items
    - Cross-state shipping
    - Weekend/holiday orders
    - Rush deliveries (<7 days)
    """)
