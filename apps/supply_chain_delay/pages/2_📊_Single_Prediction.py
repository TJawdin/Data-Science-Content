"""
Single Prediction Page
Interactive form for predicting delay risk for individual orders
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils import (
    load_model_artifacts,
    predict_delay,
    prepare_features,
    calculate_temporal_features,
    apply_custom_css,
    show_page_header,
    display_risk_badge,
    plot_risk_gauge,
    generate_prediction_report
)

# Page config
st.set_page_config(page_title="Single Prediction", page_icon="📊", layout="wide")
apply_custom_css()

# Load model
model, final_metadata, feature_metadata, threshold = load_model_artifacts()

# Header
show_page_header(
    title="Single Order Prediction",
    description="Enter order details to predict delivery delay risk with AI-powered insights",
    icon="📊"
)

# Instructions
st.info("📝 Fill in the order details below. All fields are required for accurate predictions.")

# Create tabs for input organization
tab1, tab2, tab3, tab4 = st.tabs(["📦 Order Details", "💰 Financial", "🗺️ Geographic", "⏰ Temporal"])

# Initialize session state for form data
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

# Tab 1: Order Details
with tab1:
    st.markdown("### Order Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_items = st.number_input(
            "Number of Items",
            min_value=1,
            max_value=50,
            value=2,
            help="Total number of items in the order"
        )
        
        n_sellers = st.number_input(
            "Number of Sellers",
            min_value=1,
            max_value=20,
            value=1,
            help="Number of different sellers in this order"
        )
        
        n_products = st.number_input(
            "Number of Unique Products",
            min_value=1,
            max_value=50,
            value=2,
            help="Number of distinct products"
        )
        
        n_categories = st.number_input(
            "Number of Categories",
            min_value=1,
            max_value=20,
            value=1,
            help="Number of different product categories"
        )
        
        mode_category_count = st.number_input(
            "Most Common Category Count",
            min_value=1,
            max_value=50,
            value=2,
            help="Count of items in the most frequent category"
        )
    
    with col2:
        mode_category = st.selectbox(
            "Primary Product Category",
            options=[
                'electronics', 'furniture_decor', 'health_beauty',
                'sports_leisure', 'computers_accessories', 'housewares',
                'watches_gifts', 'bed_bath_table', 'toys', 'auto',
                'telephony', 'books_general_interest', 'cool_stuff',
                'home_appliances', 'garden_tools', 'baby', 'fashion_shoes',
                'perfumery', 'stationery', 'fashion_bags_accessories'
            ],
            help="Primary category of products in the order"
        )
        
        st.markdown("### Product Dimensions (Averages)")
        
        avg_weight_g = st.number_input(
            "Average Weight (grams)",
            min_value=1.0,
            max_value=50000.0,
            value=2000.0,
            step=100.0
        )
        
        avg_length_cm = st.number_input(
            "Average Length (cm)",
            min_value=1.0,
            max_value=200.0,
            value=30.0,
            step=1.0
        )
        
        avg_height_cm = st.number_input(
            "Average Height (cm)",
            min_value=1.0,
            max_value=200.0,
            value=15.0,
            step=1.0
        )
        
        avg_width_cm = st.number_input(
            "Average Width (cm)",
            min_value=1.0,
            max_value=200.0,
            value=20.0,
            step=1.0
        )

# Tab 2: Financial
with tab2:
    st.markdown("### Financial Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sum_price = st.number_input(
            "Total Price (R$)",
            min_value=0.01,
            max_value=100000.0,
            value=150.0,
            step=10.0,
            help="Sum of all item prices"
        )
        
        sum_freight = st.number_input(
            "Total Freight Cost (R$)",
            min_value=0.0,
            max_value=5000.0,
            value=25.0,
            step=5.0,
            help="Total shipping/freight cost"
        )
        
        total_payment = st.number_input(
            "Total Payment (R$)",
            min_value=0.01,
            max_value=100000.0,
            value=sum_price + sum_freight,
            step=10.0,
            help="Total amount paid (usually price + freight)"
        )
    
    with col2:
        n_payment_records = st.number_input(
            "Number of Payment Transactions",
            min_value=1,
            max_value=10,
            value=1,
            help="Number of separate payment records"
        )
        
        max_installments = st.number_input(
            "Maximum Installments",
            min_value=1,
            max_value=24,
            value=3,
            help="Maximum number of payment installments"
        )
        
        st.markdown("### Payment Type")
        payment_type = st.selectbox(
            "Select Payment Method",
            options=['credit_card', 'debit_card', 'boleto', 'voucher', 'not_defined'],
            format_func=lambda x: x.replace('_', ' ').title()
        )

# Tab 3: Geographic
with tab3:
    st.markdown("### Geographic Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Customer Location")
        
        customer_state = st.selectbox(
            "Customer State",
            options=[
                'SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'ES', 'GO',
                'PE', 'CE', 'PA', 'MT', 'MA', 'MS', 'PB', 'RN', 'AL', 'PI',
                'SE', 'RO', 'TO', 'AC', 'AM', 'AP', 'RR'
            ],
            help="Brazilian state code (e.g., SP for São Paulo)"
        )
        
        # Common cities by state
        city_options = {
            'SP': ['sao paulo', 'campinas', 'santos', 'sorocaba', 'ribeirao preto'],
            'RJ': ['rio de janeiro', 'niteroi', 'duque de caxias', 'nova iguacu'],
            'MG': ['belo horizonte', 'uberlandia', 'contagem', 'juiz de fora'],
            'default': ['capital city', 'major city', 'other']
        }
        
        customer_city = st.selectbox(
            "Customer City",
            options=city_options.get(customer_state, city_options['default'])
        )
    
    with col2:
        st.markdown("#### Seller Information")
        
        seller_state_mode = st.selectbox(
            "Primary Seller State",
            options=[
                'SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'ES', 'GO',
                'PE', 'CE', 'PA', 'MT', 'MA', 'MS', 'PB', 'RN', 'AL', 'PI',
                'SE', 'RO', 'TO', 'AC', 'AM', 'AP', 'RR'
            ],
            help="State where most sellers are located"
        )
        
        n_seller_states = st.number_input(
            "Number of Seller States",
            min_value=1,
            max_value=10,
            value=1,
            help="Number of different states sellers are from"
        )

# Tab 4: Temporal
with tab4:
    st.markdown("### Temporal Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        purchase_date = st.date_input(
            "Purchase Date",
            value=datetime.now(),
            help="Date when the order was placed"
        )
        
        purchase_time = st.time_input(
            "Purchase Time",
            value=datetime.now().time(),
            help="Time when the order was placed"
        )
        
        # Combine date and time
        purchase_datetime = datetime.combine(purchase_date, purchase_time)
        
        # Calculate temporal features
        temporal_features = calculate_temporal_features(purchase_datetime)
    
    with col2:
        est_lead_days = st.slider(
            "Estimated Lead Time (days)",
            min_value=1.0,
            max_value=30.0,
            value=7.0,
            step=0.5,
            help="Expected delivery lead time in days"
        )
        
        st.markdown("### Calculated Temporal Features")
        st.write(f"**Year**: {temporal_features['purch_year']}")
        st.write(f"**Month**: {temporal_features['purch_month']}")
        st.write(f"**Day of Week**: {temporal_features['purch_dayofweek']} (0=Mon)")
        st.write(f"**Hour**: {temporal_features['purch_hour']}")
        st.write(f"**Is Weekend**: {'Yes' if temporal_features['purch_is_weekend'] else 'No'}")

st.markdown("---")

# Prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Delay Risk", use_container_width=True, type="primary")

# Make prediction when button is clicked
if predict_button:
    # Create payment type one-hot encoding
    payment_features = {
        'paytype_boleto': 1 if payment_type == 'boleto' else 0,
        'paytype_credit_card': 1 if payment_type == 'credit_card' else 0,
        'paytype_debit_card': 1 if payment_type == 'debit_card' else 0,
        'paytype_not_defined': 1 if payment_type == 'not_defined' else 0,
        'paytype_voucher': 1 if payment_type == 'voucher' else 0
    }
    
    # Compile all features
    order_data = {
        'n_items': n_items,
        'n_sellers': n_sellers,
        'n_products': n_products,
        'sum_price': sum_price,
        'sum_freight': sum_freight,
        'total_payment': total_payment,
        'n_payment_records': n_payment_records,
        'max_installments': max_installments,
        'avg_weight_g': avg_weight_g,
        'avg_length_cm': avg_length_cm,
        'avg_height_cm': avg_height_cm,
        'avg_width_cm': avg_width_cm,
        'n_seller_states': n_seller_states,
        'est_lead_days': est_lead_days,
        'n_categories': n_categories,
        'mode_category_count': mode_category_count,
        'mode_category': mode_category,
        'seller_state_mode': seller_state_mode,
        'customer_city': customer_city,
        'customer_state': customer_state,
        **temporal_features,
        **payment_features
    }
    
    # Prepare features and make prediction
    try:
        features_df = prepare_features(order_data, feature_metadata['feature_names'])
        predictions, probabilities, risk_levels = predict_delay(model, features_df, threshold)
        
        prediction = predictions[0]
        probability = probabilities[0]
        risk_level = risk_levels[0]
        
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        # Display results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            
            
            st.markdown("### Key Metrics")
            st.metric("Delay Probability", f"{probability*100:.1f}%")
            st.metric("Classification", "Delayed" if prediction == 1 else "On Time")
            st.metric("Risk Level", risk_level)
            st.metric("Threshold", f"{threshold*100:.1f}%")
        
        with col2:
            fig_gauge = plot_risk_gauge(probability, threshold)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if risk_level == 'High':
            st.error("🚨 **High Risk - Immediate Action Required**")
            st.markdown("""
            **Priority Actions:**
            1. 🚀 Consider expedited shipping
            2. 📞 Proactive customer communication
            3. 🔍 Verify seller capacity and inventory
            4. 📊 Enhanced monitoring throughout fulfillment
            5. 💼 Evaluate order optimization opportunities
            """)
        elif risk_level == 'Medium':
            st.warning("⚠️ **Medium Risk - Enhanced Monitoring**")
            st.markdown("""
            **Recommended Actions:**
            1. 👀 Increase monitoring frequency
            2. ✅ Verify inventory and logistics
            3. 📱 Set realistic customer expectations
            4. 🚛 Confirm shipping capabilities
            """)
        else:
            st.success("✅ **Low Risk - Standard Process**")
            st.markdown("""
            **Standard Actions:**
            1. ✨ Proceed with normal fulfillment
            2. 📊 Routine monitoring
            3. 💚 Order well-positioned for on-time delivery
            """)
        
        st.markdown("---")
        
        # Generate PDF report
        st.markdown("### 📄 Generate Report")
        
        report_data = {
            'risk_level': risk_level,
            'probability': probability,
            'prediction': prediction,
            'order_details': order_data
        }
        
        if st.button("📥 Download PDF Report"):
            try:
                pdf_buffer = generate_prediction_report(report_data)
                st.download_button(
                    label="💾 Download Report",
                    data=pdf_buffer,
                    file_name=f"delay_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
                st.success("✅ Report generated successfully!")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.info("Please check that all fields are filled correctly and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>💡 Tip: All fields must be filled for accurate predictions. Use realistic values based on your actual orders.</p>
</div>
""", unsafe_allow_html=True)
