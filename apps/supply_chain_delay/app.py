import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from utils.model_loader import ModelLoader
from utils.feature_engineering import FeatureEngineer
from utils.visualization import create_gauge_chart, create_risk_distribution

# Page configuration
st.set_page_config(
    page_title="Supply Chain Delay Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }
    .on-time {
        background: linear-gradient(135deg, #D4EDDA, #C3E6CB);
        border: 2px solid #28A745;
    }
    .delayed {
        background: linear-gradient(135deg, #F8D7DA, #F5C6CB);
        border: 2px solid #DC3545;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize model loader
@st.cache_resource
def init_model_loader():
    return ModelLoader(artifacts_path="./artifacts")

# Header
st.markdown('<h1 class="main-header">📦 Supply Chain Delay Prediction System</h1>', unsafe_allow_html=True)
st.markdown("### Brazilian E-Commerce Delivery Performance Analyzer")

# Load model and metadata
model_loader = init_model_loader()
model = model_loader.load_model()
metadata, feature_metadata = model_loader.load_metadata()

# Display model performance metrics
with st.expander("📊 Model Performance Metrics", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AUC Score", f"{metadata['best_model_auc']:.3f}")
    with col2:
        st.metric("Precision", f"{metadata['best_model_precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metadata['best_model_recall']:.3f}")
    with col4:
        st.metric("F1 Score", f"{metadata['best_model_f1']:.3f}")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🎯 Quick Prediction", "📊 Detailed Analysis", "ℹ️ About"])

with tab1:
    st.header("Quick Delivery Delay Prediction")
    
    # Create three columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📦 Order Details")
        n_items = st.number_input("Number of Items", min_value=1, max_value=50, value=2)
        n_products = st.number_input("Number of Different Products", min_value=1, max_value=20, value=1)
        n_sellers = st.number_input("Number of Sellers", min_value=1, max_value=10, value=1)
        n_categories = st.number_input("Number of Categories", min_value=1, max_value=10, value=1)
        
        st.subheader("💰 Payment")
        sum_price = st.number_input("Total Price (R$)", min_value=0.0, max_value=10000.0, value=100.0, step=10.0)
        sum_freight = st.number_input("Freight Cost (R$)", min_value=0.0, max_value=500.0, value=20.0, step=5.0)
        payment_type = st.selectbox("Payment Type", 
                                   ["credit_card", "boleto", "voucher", "debit_card", "not_defined"])
        max_installments = st.slider("Max Installments", 0, 24, 1)
    
    with col2:
        st.subheader("📏 Product Dimensions (Average)")
        avg_weight_g = st.number_input("Weight (g)", min_value=0.0, max_value=50000.0, value=500.0, step=100.0)
        avg_length_cm = st.number_input("Length (cm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        avg_height_cm = st.number_input("Height (cm)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
        avg_width_cm = st.number_input("Width (cm)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
        
        st.subheader("📅 Order Timing")
        order_date = st.date_input("Order Date", value=datetime.now())
        order_hour = st.slider("Order Hour", 0, 23, 12)
        est_lead_days = st.number_input("Estimated Lead Time (days)", min_value=1, max_value=60, value=10)
    
    with col3:
        st.subheader("📍 Location")
        customer_state = st.selectbox("Customer State", 
                                     ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "ES", "GO", 
                                      "PE", "CE", "PA", "MT", "MS", "MA", "RN", "PB", "AL", "PI", 
                                      "SE", "RO", "TO", "AC", "AM", "RR", "AP"])
        
        customer_city = st.text_input("Customer City", value="sao paulo")
        seller_state_mode = st.selectbox("Main Seller State", 
                                        ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "ES", "GO", 
                                         "PE", "CE", "PA", "MT", "MS", "MA", "RN", "PB", "AL", "PI"])
        n_seller_states = st.number_input("Number of Seller States", min_value=1, max_value=10, value=1)
        
        st.subheader("🏷️ Product Category")
        mode_category = st.selectbox("Main Category", 
                                    ["cama_mesa_banho", "beleza_saude", "esporte_lazer", 
                                     "informatica_acessorios", "moveis_decoracao", "utilidades_domesticas",
                                     "relogios_presentes", "telefonia", "ferramentas_jardim", "outros"])
        mode_category_count = st.number_input("Items in Main Category", min_value=1, max_value=20, value=1)
    
    # Predict button
    if st.button("🔮 Predict Delay Risk", type="primary", use_container_width=True):
        
        # Prepare features
        features = prepare_features(
            n_items, n_sellers, n_products, sum_price, sum_freight,
            payment_type, max_installments, avg_weight_g, avg_length_cm,
            avg_height_cm, avg_width_cm, n_seller_states, order_date,
            order_hour, est_lead_days, n_categories, mode_category_count,
            mode_category, seller_state_mode, customer_city, customer_state
        )
        
        # Make prediction
        predictions, probabilities, risk_levels = model_loader.predict_with_probability(features)
        
        if predictions is not None:
            prediction = predictions[0]
            probability = probabilities[0]
            risk_level = risk_levels[0]
            
            st.markdown("---")
            
            # Display results in columns
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                if prediction == 1:
                    st.markdown(
                        f'<div class="prediction-box delayed">'
                        f'<h2 style="color: #DC3545;">⚠️ HIGH DELAY RISK</h2>'
                        f'<p style="font-size: 20px;">This shipment has a {probability*100:.1f}% chance of being delayed</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-box on-time">'
                        f'<h2 style="color: #28A745;">✅ LOW DELAY RISK</h2>'
                        f'<p style="font-size: 20px;">This shipment has only a {probability*100:.1f}% chance of delay</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            with result_col2:
                # Risk gauge
                fig = create_gauge_chart(probability * 100, metadata['optimal_threshold'] * 100)
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk level indicator
            risk_class = f"risk-{risk_level.lower()}"
            st.markdown(f'<div class="{risk_class}"><b>Risk Level: {risk_level}</b></div>', 
                       unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("📋 Recommendations")
            if prediction == 1:
                st.warning("""
                **High Risk Actions:**
                - 🚚 Consider expedited shipping options
                - 📞 Proactively contact customer about potential delay
                - 📦 Prioritize order processing
                - 🔍 Monitor shipment closely throughout delivery
                - 💡 Review if items can be sourced from closer warehouses
                """)
            else:
                st.success("""
                **Low Risk - Standard Processing:**
                - ✅ Proceed with standard shipping
                - 📊 Order within normal delivery parameters
                - 📧 Send standard shipping confirmation
                """)

with tab2:
    st.header("📊 Detailed Analysis")
    
    # Feature importance
    importance_df = model_loader.get_feature_importance()
    if importance_df is not None:
        st.subheader("Feature Importance")
        
        top_features = importance_df.head(15)
        fig = px.bar(top_features, x='importance', y='feature', orientation='h',
                    title="Top 15 Most Important Features",
                    labels={'importance': 'Importance Score', 'feature': 'Feature'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model details
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Configuration")
        st.json({
            "Model Type": metadata['best_model'],
            "Optimal Threshold": f"{metadata['optimal_threshold']:.3f}",
            "Training Date": metadata['training_date'],
            "Training Samples": metadata['n_samples_train'],
            "Test Samples": metadata['n_samples_test'],
            "Number of Features": metadata['n_features']
        })
    
    with col2:
        st.subheader("Risk Bands")
        risk_data = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High'],
            'Probability Range': [
                f"0% - {metadata['risk_bands']['low_max']}%",
                f"{metadata['risk_bands']['low_max']}% - {metadata['risk_bands']['med_max']}%",
                f"{metadata['risk_bands']['med_max']}% - 100%"
            ],
            'Action': [
                'Standard Processing',
                'Monitor Closely',
                'Expedite & Alert'
            ]
        })
        st.dataframe(risk_data, use_container_width=True)

with tab3:
    st.header("ℹ️ About This System")
    st.markdown("""
    ### 🎯 Purpose
    This system predicts the likelihood of delivery delays in Brazilian e-commerce orders using advanced machine learning.
    
    ### 🤖 Model Details
    - **Algorithm**: LightGBM (Gradient Boosting)
    - **Training Data**: 77,129 historical orders
    - **Features**: 32 engineered features covering order details, payment, geography, and timing
    - **Performance**: {:.1f}% AUC Score
    
    ### 📊 Key Factors for Delays
    1. **Geographic Distance**: Orders crossing multiple states
    2. **Product Complexity**: Multiple sellers, categories, or heavy items
    3. **Payment Method**: Boleto payments show different patterns
    4. **Timing**: Weekend and holiday orders
    5. **Freight Cost**: Higher freight often indicates complex delivery
    
    ### 🔄 Updates
    - Model trained on: {}
    - Last updated: {}
    - Version: 1.0
    """.format(
        metadata['best_model_auc'] * 100,
        metadata['training_date'],
        datetime.now().strftime("%Y-%m-%d")
    ))

# Helper function to prepare features
def prepare_features(n_items, n_sellers, n_products, sum_price, sum_freight,
                    payment_type, max_installments, avg_weight_g, avg_length_cm,
                    avg_height_cm, avg_width_cm, n_seller_states, order_date,
                    order_hour, est_lead_days, n_categories, mode_category_count,
                    mode_category, seller_state_mode, customer_city, customer_state):
    """Prepare features in the exact format expected by the model"""
    
    # Calculate derived features
    total_payment = sum_price + sum_freight
    n_payment_records = 1 if max_installments == 0 else max_installments
    
    # Extract date features
    purch_year = order_date.year
    purch_month = order_date.month
    purch_dayofweek = order_date.weekday()
    purch_is_weekend = 1 if purch_dayofweek >= 5 else 0
    
    # Cyclic encoding for hour
    purch_hour_sin = np.sin(2 * np.pi * order_hour / 24)
    purch_hour_cos = np.cos(2 * np.pi * order_hour / 24)
    
    # Create feature dictionary in exact order
    features = {
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
        'purch_year': purch_year,
        'purch_month': purch_month,
        'purch_dayofweek': purch_dayofweek,
        'purch_hour': order_hour,
        'purch_is_weekend': purch_is_weekend,
        'purch_hour_sin': purch_hour_sin,
        'purch_hour_cos': purch_hour_cos,
        'est_lead_days': est_lead_days,
        'n_categories': n_categories,
        'mode_category_count': mode_category_count,
        'paytype_boleto': 1 if payment_type == 'boleto' else 0,
        'paytype_credit_card': 1 if payment_type == 'credit_card' else 0,
        'paytype_debit_card': 1 if payment_type == 'debit_card' else 0,
        'paytype_not_defined': 1 if payment_type == 'not_defined' else 0,
        'paytype_voucher': 1 if payment_type == 'voucher' else 0,
        'mode_category': mode_category,
        'seller_state_mode': seller_state_mode,
        'customer_city': customer_city.lower(),
        'customer_state': customer_state
    }
    
    # Convert to DataFrame with single row
    df = pd.DataFrame([features])
    
    # Ensure column order matches model training
    expected_columns = feature_metadata['feature_names']
    df = df[expected_columns]
    
    return df
