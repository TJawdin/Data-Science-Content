import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from utils.model_loader import ModelLoader
from utils.visualization import create_gauge_chart, create_feature_impact_chart

st.set_page_config(page_title="Single Prediction", page_icon="📊", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .prediction-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .input-section {
        background-color: #f7f7f7;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Single Order Delay Prediction")
st.markdown("Enter order details to predict delivery delay risk")

# Initialize model
@st.cache_resource
def init_model_loader():
    return ModelLoader(artifacts_path="./artifacts")

model_loader = init_model_loader()
model = model_loader.load_model()
metadata, feature_metadata = model_loader.load_metadata()

# Create input form with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📦 Order Info", "💳 Payment", "📏 Product Details", "🚚 Logistics"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        n_items = st.number_input("Number of Items", min_value=1, max_value=50, value=2, 
                                 help="Total quantity of items in the order")
        n_products = st.number_input("Number of Different Products", min_value=1, max_value=20, value=1,
                                    help="How many unique products")
        n_sellers = st.number_input("Number of Sellers", min_value=1, max_value=10, value=1,
                                   help="Number of different sellers involved")
    with col2:
        n_categories = st.number_input("Number of Categories", min_value=1, max_value=10, value=1,
                                      help="Product category diversity")
        mode_category = st.selectbox("Main Product Category", 
                                    ["beleza_saude", "informatica_acessorios", "esporte_lazer",
                                     "cama_mesa_banho", "moveis_decoracao", "utilidades_domesticas",
                                     "relogios_presentes", "telefonia", "ferramentas_jardim",
                                     "automotivo", "brinquedos", "cool_stuff", "perfumaria",
                                     "bebes", "eletronicos", "papelaria", "fashion_bolsas_e_acessorios",
                                     "pet_shop", "outros"],
                                    help="Primary category of products")
        mode_category_count = st.number_input("Items in Main Category", min_value=1, max_value=20, 
                                             value=min(n_items, 1))

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        sum_price = st.number_input("Product Total (R$)", min_value=0.0, max_value=50000.0, 
                                   value=150.0, step=10.0,
                                   help="Total price of products (excluding freight)")
        sum_freight = st.number_input("Freight Cost (R$)", min_value=0.0, max_value=1000.0, 
                                     value=20.0, step=5.0,
                                     help="Shipping cost")
        total_payment = sum_price + sum_freight
        st.metric("Total Order Value", f"R$ {total_payment:.2f}")
    
    with col2:
        payment_type = st.selectbox("Payment Method", 
                                   ["credit_card", "boleto", "debit_card", "voucher", "not_defined"],
                                   help="Payment method used")
        max_installments = st.slider("Installments", min_value=0, max_value=24, value=1,
                                    help="Number of payment installments (0 for single payment)")
        n_payment_records = 1 if max_installments == 0 else max_installments

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Average Product Dimensions")
        avg_weight_g = st.number_input("Weight (grams)", min_value=0.0, max_value=100000.0, 
                                      value=500.0, step=100.0)
        avg_length_cm = st.number_input("Length (cm)", min_value=0.0, max_value=200.0, 
                                       value=20.0, step=1.0)
    with col2:
        st.markdown("#### ")  # Empty header for alignment
        avg_height_cm = st.number_input("Height (cm)", min_value=0.0, max_value=200.0, 
                                       value=10.0, step=1.0)
        avg_width_cm = st.number_input("Width (cm)", min_value=0.0, max_value=200.0, 
                                      value=15.0, step=1.0)
    
    # Volume calculation
    volume = (avg_length_cm * avg_height_cm * avg_width_cm) / 1000  # in liters
    st.info(f"📦 Package Volume: {volume:.1f} liters | Weight: {avg_weight_g/1000:.2f} kg")

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Delivery Location")
        customer_state = st.selectbox("Customer State", 
                                     ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "ES", "GO", 
                                      "PE", "CE", "PA", "MT", "MS", "MA", "RN", "PB", "AL", "PI", 
                                      "SE", "RO", "TO", "AC", "AM", "RR", "AP"])
        customer_city = st.text_input("Customer City", value="sao paulo",
                                     help="City name (lowercase)")
        
        st.markdown("#### Seller Location")
        seller_state_mode = st.selectbox("Primary Seller State", 
                                        ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "ES", "GO"])
        n_seller_states = st.number_input("Number of Seller States", min_value=1, max_value=10, 
                                         value=1, help="Geographic distribution of sellers")
    
    with col2:
        st.markdown("#### Order Timing")
        order_datetime = st.datetime_input("Order Date & Time", value=datetime.now())
        purch_year = order_datetime.year
        purch_month = order_datetime.month
        purch_dayofweek = order_datetime.weekday()
        purch_hour = order_datetime.hour
        purch_is_weekend = 1 if purch_dayofweek >= 5 else 0
        
        # Display timing info
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        st.info(f"📅 {day_names[purch_dayofweek]} | {'Weekend' if purch_is_weekend else 'Weekday'} | {purch_hour:02d}:00")
        
        est_lead_days = st.number_input("Estimated Lead Time (days)", min_value=1, max_value=60, 
                                       value=10, help="Expected delivery time in days")
        
        # Calculate expected delivery date
        expected_delivery = order_datetime + timedelta(days=est_lead_days)
        st.info(f"📅 Expected Delivery: {expected_delivery.strftime('%Y-%m-%d')}")

# Predict button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🔮 Predict Delivery Risk", type="primary", use_container_width=True)

if predict_button:
    # Prepare features
    purch_hour_sin = np.sin(2 * np.pi * purch_hour / 24)
    purch_hour_cos = np.cos(2 * np.pi * purch_hour / 24)
    
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
        'purch_hour': purch_hour,
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
    
    # Create DataFrame and ensure correct column order
    features_df = pd.DataFrame([features])
    features_df = features_df[feature_metadata['feature_names']]
    
    # Get predictions
    with st.spinner("Analyzing order risk..."):
        predictions, probabilities, risk_levels = model_loader.predict_with_probability(features_df)
    
    prediction = predictions[0]
    probability = probabilities[0]
    risk_level = risk_levels[0]
    
    # Results section
    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")
    
    # Main result display
    result_col1, result_col2, result_col3 = st.columns([2, 1, 1])
    
    with result_col1:
        if prediction == 1:
            st.error("### ⚠️ HIGH RISK OF DELAY")
            st.markdown(f"**Delay Probability:** {probability*100:.1f}%")
            st.markdown(f"**Risk Level:** {risk_level}")
            
            # Calculate days at risk
            risk_days = int(est_lead_days * 0.3)  # Assume 30% delay
            new_delivery = expected_delivery + timedelta(days=risk_days)
            st.warning(f"⏰ Potential delivery by: {new_delivery.strftime('%Y-%m-%d')} (+{risk_days} days)")
        else:
            st.success("### ✅ LOW RISK OF DELAY")
            st.markdown(f"**Delay Probability:** {probability*100:.1f}%")
            st.markdown(f"**Risk Level:** {risk_level}")
            st.info(f"📅 Expected on-time delivery: {expected_delivery.strftime('%Y-%m-%d')}")
    
    with result_col2:
        fig = create_gauge_chart(probability * 100, metadata['optimal_threshold'] * 100)
        st.plotly_chart(fig, use_container_width=True)
    
    with result_col3:
        # Confidence meter
        confidence = abs(probability - 0.5) * 200  # Convert to confidence percentage
        st.metric("Model Confidence", f"{confidence:.1f}%")
        
        # Risk factors count
        risk_factors = 0
        if n_seller_states > 2: risk_factors += 1
        if avg_weight_g > 5000: risk_factors += 1
        if est_lead_days > 15: risk_factors += 1
        if purch_is_weekend: risk_factors += 1
        if payment_type == 'boleto': risk_factors += 1
        
        st.metric("Risk Factors", f"{risk_factors}/5")
    
    # Detailed Analysis
    with st.expander("📊 Detailed Risk Analysis", expanded=True):
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown("#### 🚨 Risk Factors")
            
            risk_analysis = []
            
            # Geographic risk
            if customer_state in ['AC', 'RR', 'AP', 'AM', 'RO']:
                risk_analysis.append(("🌍 Remote delivery location", "High"))
            elif n_seller_states > 2:
                risk_analysis.append(("📍 Multiple seller locations", "Medium"))
            
            # Product complexity
            if avg_weight_g > 10000:
                risk_analysis.append(("⚖️ Very heavy package", "High"))
            elif avg_weight_g > 5000:
                risk_analysis.append(("📦 Heavy package", "Medium"))
            
            # Timing risk
            if purch_is_weekend:
                risk_analysis.append(("📅 Weekend order", "Medium"))
            if purch_month in [11, 12]:
                risk_analysis.append(("🎄 Holiday season", "High"))
            
            # Payment risk
            if payment_type == 'boleto':
                risk_analysis.append(("💳 Boleto payment processing", "Medium"))
            
            # Lead time risk
            if est_lead_days > 20:
                risk_analysis.append(("⏰ Long estimated lead time", "High"))
            
            if risk_analysis:
                for factor, severity in risk_analysis:
                    if severity == "High":
                        st.markdown(f"🔴 {factor}")
                    else:
                        st.markdown(f"🟡 {factor}")
            else:
                st.success("✅ No significant risk factors identified")
        
        with analysis_col2:
            st.markdown("#### 💡 Recommendations")
            
            if prediction == 1:
                st.info("""
                **Mitigation Strategies:**
                - 🚀 Consider express shipping upgrade
                - 📞 Set up proactive customer communication
                - 📦 Prioritize order processing
                - 🔍 Enable real-time tracking
                - 📧 Send delay risk notification
                """)
            else:
                st.success("""
                **Standard Processing:**
                - ✅ Process with normal priority
                - 📧 Send standard confirmation
                - 🚚 Regular shipping method suitable
                """)
    
    # Feature importance for this prediction
    with st.expander("🔍 Feature Impact Analysis", expanded=False):
        importance_df = model_loader.get_feature_importance()
        
        if importance_df is not None:
            # Get top 10 features
            top_features = importance_df.head(10)
            
            # Create feature values for comparison
            feature_values = []
            for feat in top_features['feature']:
                if feat in features:
                    feature_values.append(features[feat])
                else:
                    feature_values.append(0)
            
            # Bar chart of importance
            fig = px.bar(top_features, x='importance', y='feature', 
                        orientation='h',
                        title="Top 10 Most Important Features",
                        labels={'importance': 'Feature Importance', 'feature': 'Feature'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Save prediction option
    with st.expander("💾 Save Prediction", expanded=False):
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'prediction': 'Delayed' if prediction == 1 else 'On Time',
            'probability': probability,
            'risk_level': risk_level,
            **features
        }
        
        st.json(prediction_data)
        
        # Download button
        df_download = pd.DataFrame([prediction_data])
        csv = df_download.to_csv(index=False)
        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
