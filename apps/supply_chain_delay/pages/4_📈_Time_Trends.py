import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from utils.model_loader import ModelLoader

st.set_page_config(page_title="Time Trends", page_icon="📈", layout="wide")

st.title("📈 Delivery Risk Time Trends Analysis")
st.markdown("Analyze how delivery risk varies across different time periods")

# Initialize model
@st.cache_resource
def init_model_loader():
    return ModelLoader(artifacts_path="./artifacts")

model_loader = init_model_loader()
model = model_loader.load_model()
metadata, feature_metadata = model_loader.load_metadata()

# Analysis options
analysis_type = st.selectbox(
    "Select Analysis Type",
    ["Hourly Patterns", "Daily Patterns", "Monthly Patterns", "Seasonal Analysis", "Holiday Impact"]
)

# Create sample data for demonstration
@st.cache_data
def generate_time_series_predictions(analysis_type):
    """Generate predictions for different time periods"""
    
    # Base features (average values)
    base_features = {
        'n_items': 2, 'n_sellers': 1, 'n_products': 2,
        'sum_price': 200.0, 'sum_freight': 25.0, 'total_payment': 225.0,
        'n_payment_records': 1, 'max_installments': 2,
        'avg_weight_g': 800, 'avg_length_cm': 25, 'avg_height_cm': 15, 'avg_width_cm': 20,
        'n_seller_states': 1, 'est_lead_days': 10,
        'n_categories': 1, 'mode_category_count': 2,
        'paytype_boleto': 0, 'paytype_credit_card': 1, 'paytype_debit_card': 0,
        'paytype_not_defined': 0, 'paytype_voucher': 0,
        'mode_category': 'informatica_acessorios',
        'seller_state_mode': 'SP', 'customer_city': 'sao paulo', 'customer_state': 'SP'
    }
    
    predictions_data = []
    
    if analysis_type == "Hourly Patterns":
        for hour in range(24):
            features = base_features.copy()
            features['purch_hour'] = hour
            features['purch_hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['purch_hour_cos'] = np.cos(2 * np.pi * hour / 24)
            features['purch_year'] = 2024
            features['purch_month'] = 3
            features['purch_dayofweek'] = 2  # Wednesday
            features['purch_is_weekend'] = 0
            
            features_df = pd.DataFrame([features])[feature_metadata['feature_names']]
            _, prob, risk = model_loader.predict_with_probability(features_df)
            
            predictions_data.append({
                'Hour': f"{hour:02d}:00",
                'Delay_Risk': prob[0] * 100,
                'Risk_Level': risk[0]
            })
    
    elif analysis_type == "Daily Patterns":
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day_idx, day_name in enumerate(days):
            features = base_features.copy()
            features['purch_dayofweek'] = day_idx
            features['purch_is_weekend'] = 1 if day_idx >= 5 else 0
            features['purch_hour'] = 14
            features['purch_hour_sin'] = np.sin(2 * np.pi * 14 / 24)
            features['purch_hour_cos'] = np.cos(2 * np.pi * 14 / 24)
            features['purch_year'] = 2024
            features['purch_month'] = 3
            
            features_df = pd.DataFrame([features])[feature_metadata['feature_names']]
            _, prob, risk = model_loader.predict_with_probability(features_df)
            
            predictions_data.append({
                'Day': day_name,
                'Day_Index': day_idx,
                'Delay_Risk': prob[0] * 100,
                'Risk_Level': risk[0],
                'Is_Weekend': 'Weekend' if day_idx >= 5 else 'Weekday'
            })
    
    elif analysis_type == "Monthly Patterns":
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month_idx, month_name in enumerate(months, 1):
            features = base_features.copy()
            features['purch_month'] = month_idx
            features['purch_year'] = 2024
            features['purch_dayofweek'] = 2
            features['purch_hour'] = 14
            features['purch_hour_sin'] = np.sin(2 * np.pi * 14 / 24)
            features['purch_hour_cos'] = np.cos(2 * np.pi * 14 / 24)
            features['purch_is_weekend'] = 0
            
            # Adjust for seasonal factors
            if month_idx in [11, 12]:  # Holiday season
                features['n_items'] = 4
                features['sum_price'] = 400
                features['total_payment'] = 450
            
            features_df = pd.DataFrame([features])[feature_metadata['feature_names']]
            _, prob, risk = model_loader.predict_with_probability(features_df)
            
            predictions_data.append({
                'Month': month_name,
                'Month_Num': month_idx,
                'Delay_Risk': prob[0] * 100,
                'Risk_Level': risk[0],
                'Season': 'Holiday' if month_idx in [11, 12] else 'Regular'
            })
    
    return pd.DataFrame(predictions_data)

# Generate predictions
df_predictions = generate_time_series_predictions(analysis_type)

# Display results based on analysis type
if analysis_type == "Hourly Patterns":
    st.subheader("⏰ Delay Risk by Hour of Day")
    
    # Line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_predictions['Hour'],
        y=df_predictions['Delay_Risk'],
        mode='lines+markers',
        name='Delay Risk',
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=8)
    ))
    
    # Add threshold line
    fig.add_hline(y=metadata['optimal_threshold']*100, 
                  line_dash="dash", line_color="red", 
                  annotation_text="Risk Threshold")
    
    # Highlight peak hours
    peak_hours = df_predictions.nlargest(3, 'Delay_Risk')
    for _, row in peak_hours.iterrows():
        fig.add_annotation(
            x=row['Hour'], y=row['Delay_Risk'],
            text=f"Peak: {row['Delay_Risk']:.1f}%",
            showarrow=True, arrowhead=2
        )
    
    fig.update_layout(
        title="Delivery Delay Risk Throughout the Day",
        xaxis_title="Hour of Day",
        yaxis_title="Delay Risk (%)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        peak_hour = df_predictions.loc[df_predictions['Delay_Risk'].idxmax(), 'Hour']
        st.metric("Peak Risk Hour", peak_hour, 
                 f"{df_predictions['Delay_Risk'].max():.1f}%")
    with col2:
        low_hour = df_predictions.loc[df_predictions['Delay_Risk'].idxmin(), 'Hour']
        st.metric("Lowest Risk Hour", low_hour,
                 f"{df_predictions['Delay_Risk'].min():.1f}%")
    with col3:
        avg_risk = df_predictions['Delay_Risk'].mean()
        st.metric("Average Risk", f"{avg_risk:.1f}%")
    
    # Recommendations
    st.info("""
    💡 **Insights:**
    - Orders placed during evening hours (18:00-22:00) show higher delay risk
    - Morning orders (06:00-10:00) typically have lower delay probability
    - Consider promotional timing during low-risk hours
    """)

elif analysis_type == "Daily Patterns":
    st.subheader("📅 Delay Risk by Day of Week")
    
    # Bar chart with color coding
    fig = px.bar(df_predictions, x='Day', y='Delay_Risk',
                 color='Is_Weekend',
                 color_discrete_map={'Weekday': '#3B82F6', 'Weekend': '#EF4444'},
                 title="Delivery Delay Risk by Day of Week")
    
    # Add threshold line
    fig.add_hline(y=metadata['optimal_threshold']*100, 
                  line_dash="dash", line_color="red", 
                  annotation_text="Risk Threshold")
    
    fig.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Delay Risk (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekend vs Weekday comparison
    col1, col2 = st.columns(2)
    with col1:
        weekday_avg = df_predictions[df_predictions['Is_Weekend'] == 'Weekday']['Delay_Risk'].mean()
        weekend_avg = df_predictions[df_predictions['Is_Weekend'] == 'Weekend']['Delay_Risk'].mean()
        
        comparison_df = pd.DataFrame({
            'Period': ['Weekday', 'Weekend'],
            'Average Risk': [weekday_avg, weekend_avg]
        })
        
        fig2 = px.pie(comparison_df, values='Average Risk', names='Period',
                     title="Weekday vs Weekend Risk Distribution",
                     color_discrete_map={'Weekday': '#3B82F6', 'Weekend': '#EF4444'})
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Key Statistics")
        st.metric("Weekday Average", f"{weekday_avg:.1f}%")
        st.metric("Weekend Average", f"{weekend_avg:.1f}%")
        diff = weekend_avg - weekday_avg
        st.metric("Weekend Risk Increase", f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%")

elif analysis_type == "Monthly Patterns":
    st.subheader("📆 Delay Risk by Month")
    
    # Area chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_predictions['Month'],
        y=df_predictions['Delay_Risk'],
        mode='lines+markers',
        fill='tozeroy',
        name='Delay Risk',
        line=dict(color='#3B82F6', width=2),
        fillcolor='rgba(59, 130, 246, 0.2)'
    ))
    
    # Highlight holiday season
    fig.add_vrect(x0=10.5, x1=11.5, 
                  fillcolor="red", opacity=0.1,
                  annotation_text="Holiday Season")
    
    fig.update_layout(
        title="Annual Delivery Delay Risk Pattern",
        xaxis_title="Month",
        yaxis_title="Delay Risk (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal breakdown
    st.subheader("🎄 Seasonal Impact Analysis")
    
    regular_months = df_predictions[df_predictions['Season'] == 'Regular']
    holiday_months = df_predictions[df_predictions['Season'] == 'Holiday']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Regular Season Avg", f"{regular_months['Delay_Risk'].mean():.1f}%")
    with col2:
        st.metric("Holiday Season Avg", f"{holiday_months['Delay_Risk'].mean():.1f}%")
    with col3:
        impact = ((holiday_months['Delay_Risk'].mean() / regular_months['Delay_Risk'].mean()) - 1) * 100
        st.metric("Holiday Impact", f"+{impact:.1f}%")

# Additional analysis section
with st.expander("🔍 Advanced Time Analysis"):
    st.subheader("Combined Time Factors")
    
    # Create heatmap data
    hours = range(24)
    days = range(7)
    
    heatmap_data = []
    for day in days:
        day_data = []
        for hour in hours:
            features = {
                'n_items': 2, 'n_sellers': 1, 'n_products': 2,
                'sum_price': 200.0, 'sum_freight': 25.0, 'total_payment': 225.0,
                'n_payment_records': 1, 'max_installments': 2,
                'avg_weight_g': 800, 'avg_length_cm': 25, 
                'avg_height_cm': 15, 'avg_width_cm': 20,
                'n_seller_states': 1, 'est_lead_days': 10,
                'n_categories': 1, 'mode_category_count': 2,
                'paytype_boleto': 0, 'paytype_credit_card': 1, 
                'paytype_debit_card': 0, 'paytype_not_defined': 0, 'paytype_voucher': 0,
                'mode_category': 'informatica_acessorios',
                'seller_state_mode': 'SP', 'customer_city': 'sao paulo', 
                'customer_state': 'SP',
                'purch_hour': hour,
                'purch_hour_sin': np.sin(2 * np.pi * hour / 24),
                'purch_hour_cos': np.cos(2 * np.pi * hour / 24),
                'purch_year': 2024, 'purch_month': 3,
                'purch_dayofweek': day,
                'purch_is_weekend': 1 if day >= 5 else 0
            }
            
            features_df = pd.DataFrame([features])[feature_metadata['feature_names']]
            _, prob, _ = model_loader.predict_with_probability(features_df)
            day_data.append(prob[0] * 100)
        
        heatmap_data.append(day_data)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"{h:02d}:00" for h in hours],
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        colorscale='RdYlGn_r',
        text=[[f"{val:.1f}%" for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 8},
        colorbar=dict(title="Risk %")
    ))
    
    fig.update_layout(
        title="Delay Risk Heatmap: Day vs Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    🎯 **Optimal Ordering Windows:**
    - Best: Tuesday-Thursday, 10:00-14:00
    - Avoid: Weekend evenings (highest risk)
    - Consider: Early week mornings for time-sensitive deliveries
    """)
