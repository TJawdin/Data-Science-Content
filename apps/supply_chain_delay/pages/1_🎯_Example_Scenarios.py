import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.model_loader import ModelLoader
from utils.visualization import create_gauge_chart

st.set_page_config(page_title="Example Scenarios", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .scenario-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #3B82F6;
    }
    .risk-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        margin: 5px;
    }
    .risk-high { background-color: #f8d7da; color: #721c24; }
    .risk-medium { background-color: #fff3cd; color: #856404; }
    .risk-low { background-color: #d4edda; color: #155724; }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Example Delivery Scenarios")
st.markdown("Explore how different order characteristics affect delivery delay risk")

# Initialize model loader
@st.cache_resource
def init_model_loader():
    return ModelLoader(artifacts_path="./artifacts")

model_loader = init_model_loader()
model = model_loader.load_model()
metadata, feature_metadata = model_loader.load_metadata()

# Define example scenarios
scenarios = {
    "🏃 Express Urban Delivery": {
        "description": "Small, lightweight item delivered within the same state with express shipping",
        "features": {
            "n_items": 1, "n_sellers": 1, "n_products": 1, "sum_price": 150.0,
            "sum_freight": 15.0, "total_payment": 165.0, "n_payment_records": 1,
            "max_installments": 0, "avg_weight_g": 200, "avg_length_cm": 15,
            "avg_height_cm": 10, "avg_width_cm": 10, "n_seller_states": 1,
            "purch_year": 2024, "purch_month": 3, "purch_dayofweek": 2,
            "purch_hour": 14, "purch_is_weekend": 0, 
            "purch_hour_sin": np.sin(2 * np.pi * 14 / 24),
            "purch_hour_cos": np.cos(2 * np.pi * 14 / 24),
            "est_lead_days": 3, "n_categories": 1, "mode_category_count": 1,
            "paytype_boleto": 0, "paytype_credit_card": 1, "paytype_debit_card": 0,
            "paytype_not_defined": 0, "paytype_voucher": 0,
            "mode_category": "informatica_acessorios", "seller_state_mode": "SP",
            "customer_city": "sao paulo", "customer_state": "SP"
        }
    },
    "🏔️ Remote Rural Delivery": {
        "description": "Heavy furniture delivered to remote location across multiple states",
        "features": {
            "n_items": 3, "n_sellers": 2, "n_products": 3, "sum_price": 2500.0,
            "sum_freight": 350.0, "total_payment": 2850.0, "n_payment_records": 10,
            "max_installments": 10, "avg_weight_g": 25000, "avg_length_cm": 150,
            "avg_height_cm": 80, "avg_width_cm": 60, "n_seller_states": 2,
            "purch_year": 2024, "purch_month": 3, "purch_dayofweek": 5,
            "purch_hour": 20, "purch_is_weekend": 0,
            "purch_hour_sin": np.sin(2 * np.pi * 20 / 24),
            "purch_hour_cos": np.cos(2 * np.pi * 20 / 24),
            "est_lead_days": 25, "n_categories": 2, "mode_category_count": 2,
            "paytype_boleto": 1, "paytype_credit_card": 0, "paytype_debit_card": 0,
            "paytype_not_defined": 0, "paytype_voucher": 0,
            "mode_category": "moveis_decoracao", "seller_state_mode": "SP",
            "customer_city": "rio branco", "customer_state": "AC"
        }
    },
    "🎄 Holiday Season Order": {
        "description": "Multiple items ordered during peak holiday season",
        "features": {
            "n_items": 5, "n_sellers": 3, "n_products": 5, "sum_price": 500.0,
            "sum_freight": 45.0, "total_payment": 545.0, "n_payment_records": 3,
            "max_installments": 3, "avg_weight_g": 800, "avg_length_cm": 30,
            "avg_height_cm": 20, "avg_width_cm": 25, "n_seller_states": 2,
            "purch_year": 2024, "purch_month": 12, "purch_dayofweek": 6,
            "purch_hour": 22, "purch_is_weekend": 1,
            "purch_hour_sin": np.sin(2 * np.pi * 22 / 24),
            "purch_hour_cos": np.cos(2 * np.pi * 22 / 24),
            "est_lead_days": 15, "n_categories": 4, "mode_category_count": 2,
            "paytype_boleto": 0, "paytype_credit_card": 1, "paytype_debit_card": 0,
            "paytype_not_defined": 0, "paytype_voucher": 0,
            "mode_category": "relogios_presentes", "seller_state_mode": "SP",
            "customer_city": "belo horizonte", "customer_state": "MG"
        }
    },
    "💊 Healthcare Essentials": {
        "description": "Urgent healthcare and beauty products with expedited shipping",
        "features": {
            "n_items": 2, "n_sellers": 1, "n_products": 2, "sum_price": 180.0,
            "sum_freight": 25.0, "total_payment": 205.0, "n_payment_records": 1,
            "max_installments": 0, "avg_weight_g": 300, "avg_length_cm": 20,
            "avg_height_cm": 15, "avg_width_cm": 15, "n_seller_states": 1,
            "purch_year": 2024, "purch_month": 3, "purch_dayofweek": 1,
            "purch_hour": 10, "purch_is_weekend": 0,
            "purch_hour_sin": np.sin(2 * np.pi * 10 / 24),
            "purch_hour_cos": np.cos(2 * np.pi * 10 / 24),
            "est_lead_days": 5, "n_categories": 1, "mode_category_count": 2,
            "paytype_boleto": 0, "paytype_credit_card": 0, "paytype_debit_card": 1,
            "paytype_not_defined": 0, "paytype_voucher": 0,
            "mode_category": "beleza_saude", "seller_state_mode": "SP",
            "customer_city": "rio de janeiro", "customer_state": "RJ"
        }
    },
    "🛏️ Bulk Home Goods": {
        "description": "Large order of home goods and bedding from multiple sellers",
        "features": {
            "n_items": 8, "n_sellers": 4, "n_products": 6, "sum_price": 1200.0,
            "sum_freight": 120.0, "total_payment": 1320.0, "n_payment_records": 6,
            "max_installments": 6, "avg_weight_g": 1500, "avg_length_cm": 50,
            "avg_height_cm": 30, "avg_width_cm": 40, "n_seller_states": 3,
            "purch_year": 2024, "purch_month": 3, "purch_dayofweek": 3,
            "purch_hour": 16, "purch_is_weekend": 0,
            "purch_hour_sin": np.sin(2 * np.pi * 16 / 24),
            "purch_hour_cos": np.cos(2 * np.pi * 16 / 24),
            "est_lead_days": 12, "n_categories": 2, "mode_category_count": 5,
            "paytype_boleto": 1, "paytype_credit_card": 0, "paytype_debit_card": 0,
            "paytype_not_defined": 0, "paytype_voucher": 0,
            "mode_category": "cama_mesa_banho", "seller_state_mode": "SC",
            "customer_city": "porto alegre", "customer_state": "RS"
        }
    }
}

# Scenario selection
selected_scenario = st.selectbox(
    "Select a scenario to analyze:",
    list(scenarios.keys()),
    format_func=lambda x: x
)

# Display scenario details
scenario = scenarios[selected_scenario]
st.markdown(f'<div class="scenario-card">', unsafe_allow_html=True)
st.markdown(f"### {selected_scenario}")
st.markdown(f"**Description:** {scenario['description']}")
st.markdown('</div>', unsafe_allow_html=True)

# Make prediction for selected scenario
if st.button("🔮 Analyze Scenario", type="primary"):
    
    # Prepare features
    features_df = pd.DataFrame([scenario['features']])
    features_df = features_df[feature_metadata['feature_names']]
    
    # Get predictions
    predictions, probabilities, risk_levels = model_loader.predict_with_probability(features_df)
    
    prediction = predictions[0]
    probability = probabilities[0]
    risk_level = risk_levels[0]
    
    # Display results
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("📊 Prediction Results")
        
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK OF DELAY**")
        else:
            st.success(f"✅ **LOW RISK OF DELAY**")
        
        st.metric("Delay Probability", f"{probability*100:.1f}%")
        
        # Risk badge
        risk_class = f"risk-{risk_level.lower()}"
        st.markdown(f'<span class="risk-badge {risk_class}">Risk Level: {risk_level}</span>', 
                   unsafe_allow_html=True)
    
    with col2:
        # Gauge chart
        fig = create_gauge_chart(probability * 100, metadata['optimal_threshold'] * 100)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("📋 Key Factors")
        factors = []
        
        if scenario['features']['n_seller_states'] > 2:
            factors.append("❌ Multiple seller states")
        if scenario['features']['avg_weight_g'] > 5000:
            factors.append("❌ Heavy items")
        if scenario['features']['est_lead_days'] > 15:
            factors.append("❌ Long lead time")
        if scenario['features']['purch_is_weekend'] == 1:
            factors.append("⚠️ Weekend order")
        if scenario['features']['paytype_boleto'] == 1:
            factors.append("⚠️ Boleto payment")
        if scenario['features']['customer_state'] in ['AC', 'RR', 'AP', 'AM']:
            factors.append("❌ Remote location")
            
        if factors:
            for factor in factors:
                st.write(factor)
        else:
            st.write("✅ No major risk factors")
    
    # Detailed breakdown
    with st.expander("📋 Scenario Details"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📦 Order Details**")
            st.write(f"- Items: {scenario['features']['n_items']}")
            st.write(f"- Products: {scenario['features']['n_products']}")
            st.write(f"- Sellers: {scenario['features']['n_sellers']}")
            st.write(f"- Categories: {scenario['features']['n_categories']}")
            st.write(f"- Main Category: {scenario['features']['mode_category']}")
        
        with col2:
            st.markdown("**💰 Payment**")
            st.write(f"- Total: R$ {scenario['features']['total_payment']:.2f}")
            st.write(f"- Freight: R$ {scenario['features']['sum_freight']:.2f}")
            st.write(f"- Installments: {scenario['features']['max_installments']}")
            payment_type = [k.replace('paytype_', '') for k, v in scenario['features'].items() 
                          if k.startswith('paytype_') and v == 1][0]
            st.write(f"- Type: {payment_type}")
        
        with col3:
            st.markdown("**📍 Logistics**")
            st.write(f"- Weight: {scenario['features']['avg_weight_g']/1000:.1f} kg")
            st.write(f"- Lead Time: {scenario['features']['est_lead_days']} days")
            st.write(f"- From: {scenario['features']['seller_state_mode']}")
            st.write(f"- To: {scenario['features']['customer_city']}, {scenario['features']['customer_state']}")

# Comparison section
st.markdown("---")
st.subheader("🔄 Compare All Scenarios")

if st.button("Compare All Scenarios"):
    comparison_data = []
    
    for scenario_name, scenario_data in scenarios.items():
        features_df = pd.DataFrame([scenario_data['features']])
        features_df = features_df[feature_metadata['feature_names']]
        
        predictions, probabilities, risk_levels = model_loader.predict_with_probability(features_df)
        
        comparison_data.append({
            'Scenario': scenario_name,
            'Delay Risk (%)': probabilities[0] * 100,
            'Risk Level': risk_levels[0],
            'Prediction': 'Delayed' if predictions[0] == 1 else 'On Time',
            'Lead Time': scenario_data['features']['est_lead_days'],
            'Total Cost': scenario_data['features']['total_payment']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Delay Risk (%)', ascending=False)
    
    # Display comparison table
    st.dataframe(
        comparison_df.style.background_gradient(subset=['Delay Risk (%)'], cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    # Bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=comparison_df['Scenario'],
            y=comparison_df['Delay Risk (%)'],
            marker_color=['red' if x > metadata['optimal_threshold']*100 else 'green' 
                         for x in comparison_df['Delay Risk (%)']],
            text=comparison_df['Delay Risk (%)'].round(1),
            textposition='auto',
        )
    ])
    
    fig.add_hline(y=metadata['optimal_threshold']*100, line_dash="dash", 
                  line_color="red", annotation_text="Risk Threshold")
    
    fig.update_layout(
        title="Delay Risk Comparison Across Scenarios",
        xaxis_title="Scenario",
        yaxis_title="Delay Risk (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
