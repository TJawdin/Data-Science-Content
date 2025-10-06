"""
Single Prediction Page
Interactive form for predicting late delivery risk for a single order
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_engineering import calculate_features, get_feature_descriptions
from utils.model_loader import load_model, predict_single
from utils.visualization import create_risk_gauge

# Page config
st.set_page_config(
    page_title="Single Prediction",
    page_icon="📊",
    layout="wide"
)
from utils.theme_adaptive import apply_adaptive_theme

# Apply theme right after page config
apply_adaptive_theme()
# ============================================================================
# Header
# ============================================================================

st.title("📊 Single Order Prediction")
st.markdown("""
Enter order details below to predict late delivery risk in real-time.
Get instant risk assessment, probability score, and actionable recommendations!
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
# Input Form
# ============================================================================

st.markdown("## 📝 Enter Order Details")

with st.form("order_input_form"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 Order Information")
        
        num_items = st.number_input(
            "Number of Items",
            min_value=1,
            max_value=20,
            value=1,
            help="Total number of items in the order"
        )
        
        num_sellers = st.number_input(
            "Number of Sellers",
            min_value=1,
            max_value=10,
            value=1,
            help="Number of different sellers fulfilling this order"
        )
        
        total_order_value = st.number_input(
            "Total Order Value ($)",
            min_value=0.0,
            value=120.0,
            step=10.0,
            help="Total monetary value of all items"
        )
        
        total_shipping_cost = st.number_input(
            "Total Shipping Cost ($)",
            min_value=0.0,
            value=8.0,
            step=5.0,
            help="Total cost of shipping"
        )
        
        st.markdown("### 📏 Physical Characteristics")
        
        total_weight_g = st.number_input(
            "Total Weight (grams)",
            min_value=0,
            value=800,
            step=100,
            help="Combined weight of all items"
        )
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            avg_length_cm = st.number_input(
                "Avg Length (cm)",
                min_value=0.0,
                value=25.0,
                step=1.0
            )
        
        with col_b:
            avg_height_cm = st.number_input(
                "Avg Height (cm)",
                min_value=0.0,
                value=18.0,
                step=1.0
            )
        
        with col_c:
            avg_width_cm = st.number_input(
                "Avg Width (cm)",
                min_value=0.0,
                value=12.0,
                step=1.0
            )
    
    with col2:
        st.markdown("### 🗺️ Geographic Information")
        
        avg_shipping_distance_km = st.number_input(
            "Shipping Distance (km)",
            min_value=0,
            value=80,
            step=50,
            help="Distance from warehouse to delivery address"
        )
        
        is_cross_state = st.selectbox(
            "Cross-State Shipping?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Does this order cross state boundaries?"
        )
        
        st.markdown("### 📅 Timing Information")
        
        estimated_days = st.number_input(
            "Estimated Delivery Days",
            min_value=1,
            max_value=60,
            value=12,
            help="Promised delivery timeframe"
        )
        
        order_weekday = st.selectbox(
            "Order Day of Week",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                   'Friday', 'Saturday', 'Sunday'][x],
            index=2,
            help="Day when order was placed"
        )
        
        order_month = st.selectbox(
            "Order Month",
            options=list(range(1, 13)),
            format_func=lambda x: ['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November', 'December'][x-1],
            index=4,
            help="Month when order was placed"
        )
        
        order_hour = st.slider(
            "Order Hour (24-hour format)",
            min_value=0,
            max_value=23,
            value=14,
            help="Time of day when order was placed"
        )
    
    # Submit button
    submitted = st.form_submit_button(
        "🎯 Predict Late Delivery Risk",
        use_container_width=True,
        type="primary"
    )

# ============================================================================
# Process Prediction
# ============================================================================

if submitted:
    
    with st.spinner("Calculating risk..."):
        
        # Prepare order data
        order_data = {
            'num_items': num_items,
            'num_sellers': num_sellers,
            'num_products': num_items,
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
            
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            # Display risk gauge and metrics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Risk gauge
                fig = create_risk_gauge(result['risk_score'], result['risk_level'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Key Metrics")
                
                st.metric(
                    label="Prediction",
                    value=result['prediction_label'],
                    help="Binary prediction: Late or On-Time"
                )
                
                st.metric(
                    label="Risk Score",
                    value=f"{result['risk_score']}/100",
                    help="Probability-based risk score (0-100)"
                )
                
                st.metric(
                    label="Risk Level",
                    value=result['risk_level'],
                    help="LOW (<30), MEDIUM (30-70), HIGH (>70)"
                )
                
                st.metric(
                    label="Late Probability",
                    value=f"{result['probability']:.1%}",
                    help="Model confidence in late delivery"
                )
            
            st.markdown("---")
            
            # Recommendations based on risk level
            st.markdown("## 💡 Recommended Actions")
            
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
            
            # ================================================================
            # PDF Report Export
            # ================================================================
            
            st.markdown("### 📥 Download Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    from utils.pdf_generator import generate_risk_report
                    
                    pdf_bytes = generate_risk_report(
                        order_data=order_data,
                        prediction_result=result,
                        features_df=features_df
                    )
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary"
                    )
                except Exception as e:
                    st.error(f"⚠️ PDF generation error: {str(e)}")
                    st.info("Make sure fpdf2 is installed: pip install fpdf2")
            
            with col2:
                import json
                
                json_data = {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'order_data': order_data,
                    'prediction': result,
                    'risk_score': result['risk_score'],
                    'risk_level': result['risk_level']
                }
                
                                # Add CSS for button visibility
                st.markdown("""
                <style>
                    div[data-testid="stDownloadButton"] button {
                        background-color: #0068C9 !important;
                        color: white !important;
                        border: 2px solid #0068C9 !important;
                        font-weight: 600 !important;
                    }
                    div[data-testid="stDownloadButton"] button:hover {
                        background-color: #0056a3 !important;
                        border-color: #0056a3 !important;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                st.download_button(
                    label="📊 Download Data (JSON)",
                    data=json.dumps(json_data, indent=2, default=str),
                    file_name=f"prediction_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # ================================================================
            # Feature Breakdown
            # ================================================================
            
            with st.expander("🔍 View Detailed Feature Breakdown"):
                st.markdown("#### Calculated Features Used in Prediction")
                
                feature_descriptions = get_feature_descriptions()
                
                feature_display = []
                for col in features_df.columns:
                    feature_display.append({
                        'Feature': feature_descriptions.get(col, col),
                        'Technical Name': col,
                        'Value': f"{features_df[col].values[0]:.2f}"
                    })
                
                st.dataframe(
                    pd.DataFrame(feature_display),
                    use_container_width=True,
                    height=400
                )
                
                st.info("""
                **Understanding Features:**
                - These 29 features were engineered from your input
                - They capture complexity, financial, physical, geographic, and temporal aspects
                - The model uses all features to calculate risk probability
                """)

# ============================================================================
# Sidebar
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
    
    st.markdown("## 💡 Tips")
    st.success("""
    **For accurate predictions:**
    - Enter realistic values
    - Consider all fields
    - Review recommendations
    - Download reports for records
    """)
    
    st.markdown("---")
    
    st.markdown("## 📊 Risk Levels")
    st.markdown("""
    - 🟢 **LOW** (0-30): Standard processing
    - 🟡 **MEDIUM** (30-70): Monitor closely
    - 🔴 **HIGH** (70-100): Immediate action
    """)
