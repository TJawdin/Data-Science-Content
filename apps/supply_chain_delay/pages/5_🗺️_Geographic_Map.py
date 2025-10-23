import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.model_loader import ModelLoader
import json

st.set_page_config(page_title="Geographic Analysis", page_icon="🗺️", layout="wide")

st.title("🗺️ Geographic Delivery Risk Analysis")
st.markdown("Analyze delivery delay risks across Brazilian states and regions")

# Initialize model
@st.cache_resource
def init_model_loader():
    return ModelLoader(artifacts_path="./artifacts")

model_loader = init_model_loader()
model = model_loader.load_model()
metadata, feature_metadata = model_loader.load_metadata()

# Brazilian states with coordinates
brazil_states = {
    'AC': {'name': 'Acre', 'lat': -9.0238, 'lon': -70.8120, 'region': 'North'},
    'AL': {'name': 'Alagoas', 'lat': -9.5713, 'lon': -36.7820, 'region': 'Northeast'},
    'AP': {'name': 'Amapá', 'lat': 0.9020, 'lon': -52.0030, 'region': 'North'},
    'AM': {'name': 'Amazonas', 'lat': -3.4168, 'lon': -65.8561, 'region': 'North'},
    'BA': {'name': 'Bahia', 'lat': -12.5797, 'lon': -41.7007, 'region': 'Northeast'},
    'CE': {'name': 'Ceará', 'lat': -5.4984, 'lon': -39.3206, 'region': 'Northeast'},
    'DF': {'name': 'Distrito Federal', 'lat': -15.7998, 'lon': -47.8645, 'region': 'Central-West'},
    'ES': {'name': 'Espírito Santo', 'lat': -19.1834, 'lon': -40.3089, 'region': 'Southeast'},
    'GO': {'name': 'Goiás', 'lat': -15.8270, 'lon': -49.8362, 'region': 'Central-West'},
    'MA': {'name': 'Maranhão', 'lat': -4.9609, 'lon': -45.2744, 'region': 'Northeast'},
    'MT': {'name': 'Mato Grosso', 'lat': -12.6819, 'lon': -56.9211, 'region': 'Central-West'},
    'MS': {'name': 'Mato Grosso do Sul', 'lat': -20.7722, 'lon': -54.7852, 'region': 'Central-West'},
    'MG': {'name': 'Minas Gerais', 'lat': -18.5122, 'lon': -44.5550, 'region': 'Southeast'},
    'PA': {'name': 'Pará', 'lat': -1.9981, 'lon': -54.9306, 'region': 'North'},
    'PB': {'name': 'Paraíba', 'lat': -7.2400, 'lon': -36.7820, 'region': 'Northeast'},
    'PR': {'name': 'Paraná', 'lat': -25.2521, 'lon': -52.0215, 'region': 'South'},
    'PE': {'name': 'Pernambuco', 'lat': -8.8137, 'lon': -36.9541, 'region': 'Northeast'},
    'PI': {'name': 'Piauí', 'lat': -7.7183, 'lon': -42.7289, 'region': 'Northeast'},
    'RJ': {'name': 'Rio de Janeiro', 'lat': -22.9068, 'lon': -43.1729, 'region': 'Southeast'},
    'RN': {'name': 'Rio Grande do Norte', 'lat': -5.4026, 'lon': -36.9541, 'region': 'Northeast'},
    'RS': {'name': 'Rio Grande do Sul', 'lat': -30.0346, 'lon': -51.2177, 'region': 'South'},
    'RO': {'name': 'Rondônia', 'lat': -11.5057, 'lon': -63.5806, 'region': 'North'},
    'RR': {'name': 'Roraima', 'lat': 1.9957, 'lon': -61.3333, 'region': 'North'},
    'SC': {'name': 'Santa Catarina', 'lat': -27.2423, 'lon': -50.2189, 'region': 'South'},
    'SP': {'name': 'São Paulo', 'lat': -23.5505, 'lon': -46.6333, 'region': 'Southeast'},
    'SE': {'name': 'Sergipe', 'lat': -10.5741, 'lon': -37.3857, 'region': 'Northeast'},
    'TO': {'name': 'Tocantins', 'lat': -10.1753, 'lon': -48.2982, 'region': 'North'}
}

# Calculate risk for each state
@st.cache_data
def calculate_state_risks(origin_state='SP'):
    """Calculate delivery risks for each destination state from a given origin"""
    
    state_risks = []
    
    for state_code, state_info in brazil_states.items():
        # Base features
        features = {
            'n_items': 2, 'n_sellers': 1, 'n_products': 2,
            'sum_price': 200.0, 'sum_freight': 25.0, 'total_payment': 225.0,
            'n_payment_records': 1, 'max_installments': 2,
            'avg_weight_g': 800, 'avg_length_cm': 25, 
            'avg_height_cm': 15, 'avg_width_cm': 20,
            'n_seller_states': 1,
            'purch_year': 2024, 'purch_month': 3, 'purch_dayofweek': 2,
            'purch_hour': 14, 'purch_is_weekend': 0,
            'purch_hour_sin': np.sin(2 * np.pi * 14 / 24),
            'purch_hour_cos': np.cos(2 * np.pi * 14 / 24),
            'n_categories': 1, 'mode_category_count': 2,
            'paytype_boleto': 0, 'paytype_credit_card': 1, 
            'paytype_debit_card': 0, 'paytype_not_defined': 0, 'paytype_voucher': 0,
            'mode_category': 'informatica_acessorios',
            'seller_state_mode': origin_state,
            'customer_city': 'capital',  # Generic city
            'customer_state': state_code
        }
        
        # Adjust lead time based on distance
        if state_code == origin_state:
            features['est_lead_days'] = 3
        elif state_info['region'] == brazil_states[origin_state]['region']:
            features['est_lead_days'] = 7
        else:
            features['est_lead_days'] = 14
            
        # Remote states get longer lead times
        if state_code in ['AC', 'RR', 'AP', 'AM']:
            features['est_lead_days'] += 5
            features['sum_freight'] = 80.0
            features['total_payment'] = 280.0
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])[feature_metadata['feature_names']]
        
        # Get prediction
        _, prob, risk = model_loader.predict_with_probability(features_df)
        
        state_risks.append({
            'state': state_code,
            'name': state_info['name'],
            'lat': state_info['lat'],
            'lon': state_info['lon'],
            'region': state_info['region'],
            'risk_probability': prob[0] * 100,
            'risk_level': risk[0],
            'lead_days': features['est_lead_days']
        })
    
    return pd.DataFrame(state_risks)

# UI Controls
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("📍 Configuration")
    origin_state = st.selectbox(
        "Origin State (Seller)",
        options=list(brazil_states.keys()),
        index=list(brazil_states.keys()).index('SP'),
        format_func=lambda x: f"{x} - {brazil_states[x]['name']}"
    )
    
    view_type = st.radio(
        "View Type",
        ["Risk Heatmap", "Regional Analysis", "Route Analysis"]
    )
    
    show_labels = st.checkbox("Show State Labels", value=True)

# Calculate risks
df_risks = calculate_state_risks(origin_state)

with col2:
    if view_type == "Risk Heatmap":
        st.subheader(f"📊 Delivery Risk from {brazil_states[origin_state]['name']}")
        
        # Create map
        fig = go.Figure()
        
        # Add markers for each state
        fig.add_trace(go.Scattergeo(
            lon=df_risks['lon'],
            lat=df_risks['lat'],
            text=df_risks['name'],
            customdata=df_risks[['risk_probability', 'risk_level', 'lead_days']],
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=df_risks['risk_probability']/2,
                color=df_risks['risk_probability'],
                colorscale='RdYlGn_r',
                cmin=0,
                cmax=100,
                colorbar=dict(
                    title="Risk %",
                    thickness=20,
                    len=0.7
                ),
                line=dict(width=1, color='white')
            ),
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate='<b>%{text}</b><br>' +
                         'Risk: %{customdata[0]:.1f}%<br>' +
                         'Level: %{customdata[1]}<br>' +
                         'Lead Time: %{customdata[2]} days<br>' +
                         '<extra></extra>'
        ))
        
        # Add origin marker
        origin_info = brazil_states[origin_state]
        fig.add_trace(go.Scattergeo(
            lon=[origin_info['lon']],
            lat=[origin_info['lat']],
            text=[f"ORIGIN: {origin_info['name']}"],
            mode='markers+text',
            marker=dict(
                size=15,
                color='blue',
                symbol='star',
                line=dict(width=2, color='white')
            ),
            textposition="top center",
            textfont=dict(size=12, color='blue'),
            showlegend=False
        ))
        
        fig.update_layout(
            geo=dict(
                scope='south america',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
                coastlinecolor='rgb(204, 
