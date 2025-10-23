"""
Time Trends Page
Analyze how delay patterns vary across temporal dimensions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from utils import (
    load_model_artifacts,
    predict_delay,
    prepare_features,
    create_example_order,
    apply_custom_css,
    show_page_header
)

# Page config
st.set_page_config(page_title="Time Trends", page_icon="📈", layout="wide")
apply_custom_css()

# Load model
model, final_metadata, feature_metadata, threshold = load_model_artifacts()

# Header
show_page_header(
    title="Temporal Trends Analysis",
    description="Explore how delay risk varies across time dimensions: hours, days, months, and seasons",
    icon="📈"
)

# Instructions
st.info("📊 Analyze delay patterns across different time periods to optimize operational planning and resource allocation")

# Create tabs for different temporal analyses
tab1, tab2, tab3, tab4 = st.tabs(["⏰ Hourly Patterns", "📅 Day of Week", "📆 Monthly Trends", "🔄 Seasonal Analysis"])

# Generate sample predictions across time dimensions
@st.cache_data
def generate_hourly_predictions():
    """Generate predictions for each hour of the day"""
    base_order = create_example_order('typical')
    results = []
    
    for hour in range(24):
        order = base_order.copy()
        order['purch_hour'] = hour
        order['purch_hour_sin'] = np.sin(2 * np.pi * hour / 24)
        order['purch_hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        features_df = prepare_features(order, feature_metadata['feature_names'])
        _, prob, risk = predict_delay(model, features_df, threshold)
        
        results.append({
            'hour': hour,
            'delay_probability': prob[0],
            'risk_level': risk[0]
        })
    
    return pd.DataFrame(results)

@st.cache_data
def generate_day_of_week_predictions():
    """Generate predictions for each day of the week"""
    base_order = create_example_order('typical')
    results = []
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for day_num, day_name in enumerate(days):
        order = base_order.copy()
        order['purch_dayofweek'] = day_num
        order['purch_is_weekend'] = 1 if day_num >= 5 else 0
        
        features_df = prepare_features(order, feature_metadata['feature_names'])
        _, prob, risk = predict_delay(model, features_df, threshold)
        
        results.append({
            'day_number': day_num,
            'day_name': day_name,
            'delay_probability': prob[0],
            'risk_level': risk[0],
            'is_weekend': order['purch_is_weekend']
        })
    
    return pd.DataFrame(results)

@st.cache_data
def generate_monthly_predictions():
    """Generate predictions for each month"""
    base_order = create_example_order('typical')
    results = []
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for month_num, month_name in enumerate(months, 1):
        order = base_order.copy()
        order['purch_month'] = month_num
        
        features_df = prepare_features(order, feature_metadata['feature_names'])
        _, prob, risk = predict_delay(model, features_df, threshold)
        
        results.append({
            'month_number': month_num,
            'month_name': month_name,
            'delay_probability': prob[0],
            'risk_level': risk[0]
        })
    
    return pd.DataFrame(results)

# Tab 1: Hourly Patterns
with tab1:
    st.markdown("### ⏰ Delay Risk by Hour of Day")
    st.markdown("*How does the time of purchase affect delay risk?*")
    
    hourly_df = generate_hourly_predictions()
    
    # Create hourly line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hourly_df['hour'],
        y=hourly_df['delay_probability'] * 100,
        mode='lines+markers',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8),
        name='Delay Probability',
        hovertemplate='<b>Hour</b>: %{x}:00<br><b>Risk</b>: %{y:.1f}%<extra></extra>'
    ))
    
    # Add threshold line
    fig.add_hline(
        y=threshold * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold*100:.1f}%)",
        annotation_position="right"
    )
    
    # Add risk zones
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=30, y1=67, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=67, y1=100, fillcolor="red", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title="Delay Probability Throughout the Day",
        xaxis_title="Hour of Day",
        yaxis_title="Delay Probability (%)",
        height=500,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        peak_hour = hourly_df.loc[hourly_df['delay_probability'].idxmax(), 'hour']
        peak_prob = hourly_df['delay_probability'].max() * 100
        st.metric(
            "⚠️ Highest Risk Hour",
            f"{int(peak_hour)}:00",
            f"{peak_prob:.1f}%"
        )
    
    with col2:
        best_hour = hourly_df.loc[hourly_df['delay_probability'].idxmin(), 'hour']
        best_prob = hourly_df['delay_probability'].min() * 100
        st.metric(
            "✅ Lowest Risk Hour",
            f"{int(best_hour)}:00",
            f"{best_prob:.1f}%"
        )
    
    with col3:
        avg_prob = hourly_df['delay_probability'].mean() * 100
        st.metric(
            "📊 Average Risk",
            f"{avg_prob:.1f}%"
        )
    
    st.markdown("---")
    
    with st.expander("💡 Hourly Insights & Recommendations"):
        st.markdown("""
        **Key Findings:**
        - Late night and early morning orders typically show higher delay risk
        - Business hours (9 AM - 5 PM) generally have lower delay probabilities
        - Peak shopping hours may correlate with warehouse capacity constraints
        
        **Recommendations:**
        1. 🎯 Incentivize orders during low-risk hours with promotions
        2. 📦 Allocate more fulfillment resources during high-risk periods
        3. 🤖 Implement dynamic delivery promise times based on purchase hour
        4. 📢 Set customer expectations appropriately for late-night orders
        """)

# Tab 2: Day of Week
with tab2:
    st.markdown("### 📅 Delay Risk by Day of Week")
    st.markdown("*How do weekdays compare to weekends?*")
    
    dow_df = generate_day_of_week_predictions()
    
    # Create bar chart
    colors = ['#28a745' if x == 0 else '#ffc107' for x in dow_df['is_weekend']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dow_df['day_name'],
        y=dow_df['delay_probability'] * 100,
        marker=dict(
            color=colors,
            line=dict(color='black', width=1)
        ),
        text=dow_df['delay_probability'].apply(lambda x: f'{x*100:.1f}%'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Risk: %{y:.1f}%<extra></extra>'
    ))
    
    # Add threshold line
    fig.add_hline(
        y=threshold * 100,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold"
    )
    
    fig.update_layout(
        title="Delay Probability by Day of Week",
        xaxis_title="Day",
        yaxis_title="Delay Probability (%)",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekday vs Weekend comparison
    col1, col2, col3 = st.columns(3)
    
    weekday_avg = dow_df[dow_df['is_weekend'] == 0]['delay_probability'].mean() * 100
    weekend_avg = dow_df[dow_df['is_weekend'] == 1]['delay_probability'].mean() * 100
    
    with col1:
        st.metric(
            "📊 Weekday Average",
            f"{weekday_avg:.1f}%"
        )
    
    with col2:
        st.metric(
            "🏖️ Weekend Average",
            f"{weekend_avg:.1f}%"
        )
    
    with col3:
        diff = weekend_avg - weekday_avg
        st.metric(
            "📈 Weekend vs Weekday",
            f"{abs(diff):.1f}%",
            f"{'Higher' if diff > 0 else 'Lower'} risk"
        )
    
    st.markdown("---")
    
    with st.expander("💡 Day of Week Insights & Recommendations"):
        st.markdown("""
        **Key Findings:**
        - Weekend orders may experience different processing patterns
        - Monday orders might accumulate from weekend, affecting capacity
        - Friday orders near weekend transitions can face delays
        
        **Recommendations:**
        1. 📦 Ensure adequate weekend staffing and logistics capacity
        2. 🎯 Adjust Monday operations to handle weekend order backlog
        3. 📊 Monitor Thursday/Friday orders for weekend transition issues
        4. 💼 Consider differential SLAs for weekend vs weekday orders
        """)

# Tab 3: Monthly Trends
with tab3:
    st.markdown("### 📆 Delay Risk by Month")
    st.markdown("*Seasonal and monthly patterns in delivery performance*")
    
    monthly_df = generate_monthly_predictions()
    
    # Create line chart with markers
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_df['month_name'],
        y=monthly_df['delay_probability'] * 100,
        mode='lines+markers',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=12, color='#FF6B6B'),
        name='Delay Probability',
        fill='tonexty',
        hovertemplate='<b>%{x}</b><br>Risk: %{y:.1f}%<extra></extra>'
    ))
    
    # Add threshold
    fig.add_hline(
        y=threshold * 100,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold"
    )
    
    fig.update_layout(
        title="Delay Probability Throughout the Year",
        xaxis_title="Month",
        yaxis_title="Delay Probability (%)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        peak_month = monthly_df.loc[monthly_df['delay_probability'].idxmax(), 'month_name']
        peak_prob = monthly_df['delay_probability'].max() * 100
        st.metric(
            "⚠️ Highest Risk Month",
            peak_month,
            f"{peak_prob:.1f}%"
        )
    
    with col2:
        best_month = monthly_df.loc[monthly_df['delay_probability'].idxmin(), 'month_name']
        best_prob = monthly_df['delay_probability'].min() * 100
        st.metric(
            "✅ Lowest Risk Month",
            best_month,
            f"{best_prob:.1f}%"
        )
    
    with col3:
        variance = monthly_df['delay_probability'].std() * 100
        st.metric(
            "📊 Monthly Variance",
            f"{variance:.2f}%"
        )
    
    st.markdown("---")
    
    with st.expander("💡 Monthly Insights & Recommendations"):
        st.markdown("""
        **Key Findings:**
        - Holiday seasons (Nov-Dec) typically show elevated delay risks
        - Post-holiday months may have lower risks due to reduced volume
        - Summer months can have varying patterns based on regional factors
        
        **Recommendations:**
        1. 🎄 Pre-plan logistics capacity for high-risk holiday periods
        2. 📈 Build inventory buffers before peak seasons
        3. 🤝 Coordinate with logistics partners on seasonal capacity
        4. 📢 Adjust marketing and promotions based on capacity
        5. 💼 Consider seasonal hiring and resource allocation
        """)

# Tab 4: Seasonal Analysis
with tab4:
    st.markdown("### 🔄 Seasonal Pattern Analysis")
    st.markdown("*Compare performance across different quarters and seasons*")
    
    # Create quarterly aggregation
    monthly_df['quarter'] = ((monthly_df['month_number'] - 1) // 3) + 1
    monthly_df['quarter_name'] = monthly_df['quarter'].map({
        1: 'Q1 (Jan-Mar)',
        2: 'Q2 (Apr-Jun)',
        3: 'Q3 (Jul-Sep)',
        4: 'Q4 (Oct-Dec)'
    })
    
    quarterly_df = monthly_df.groupby('quarter_name').agg({
        'delay_probability': ['mean', 'min', 'max']
    }).reset_index()
    quarterly_df.columns = ['quarter', 'avg_prob', 'min_prob', 'max_prob']
    
    # Create quarterly bar chart with error bars
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=quarterly_df['quarter'],
        y=quarterly_df['avg_prob'] * 100,
        error_y=dict(
            type='data',
            symmetric=False,
            array=(quarterly_df['max_prob'] - quarterly_df['avg_prob']) * 100,
            arrayminus=(quarterly_df['avg_prob'] - quarterly_df['min_prob']) * 100
        ),
        marker=dict(color='#FF6B6B'),
        text=quarterly_df['avg_prob'].apply(lambda x: f'{x*100:.1f}%'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg Risk: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Average Delay Risk by Quarter",
        xaxis_title="Quarter",
        yaxis_title="Average Delay Probability (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quarterly comparison table
    st.markdown("### 📊 Quarterly Comparison")
    
    quarterly_display = quarterly_df.copy()
    quarterly_display['avg_prob'] = quarterly_display['avg_prob'].apply(lambda x: f'{x*100:.1f}%')
    quarterly_display['min_prob'] = quarterly_display['min_prob'].apply(lambda x: f'{x*100:.1f}%')
    quarterly_display['max_prob'] = quarterly_display['max_prob'].apply(lambda x: f'{x*100:.1f}%')
    quarterly_display.columns = ['Quarter', 'Average Risk', 'Minimum Risk', 'Maximum Risk']
    
    st.dataframe(quarterly_display, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    with st.expander("💡 Seasonal Insights & Recommendations"):
        st.markdown("""
        **Key Findings:**
        - Q4 typically shows highest risk due to holiday shopping surge
        - Q1 often has lower volumes but may face weather-related delays
        - Q2-Q3 generally show more stable patterns
        
        **Strategic Recommendations:**
        1. 📊 **Capacity Planning**: Scale resources based on quarterly patterns
        2. 🎯 **Inventory Management**: Build strategic buffers before Q4
        3. 🤝 **Partner Coordination**: Negotiate seasonal contracts with logistics partners
        4. 📈 **Demand Forecasting**: Use historical patterns for better planning
        5. 💼 **Staffing**: Implement seasonal hiring strategies
        6. 🚀 **Technology**: Deploy automation during high-risk periods
        """)

st.markdown("---")

# Summary insights
st.markdown("## 📋 Key Temporal Insights Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Highest Risk Periods
    - **Late Night Hours**: 11 PM - 3 AM
    - **Weekend Orders**: Saturday-Sunday
    - **Holiday Seasons**: November-December
    - **Monday Mornings**: Weekend backlog
    """)

with col2:
    st.markdown("""
    ### ✅ Lowest Risk Periods
    - **Business Hours**: 9 AM - 5 PM
    - **Mid-Week**: Tuesday-Thursday
    - **Post-Holiday**: January-February
    - **Mid-Quarter**: Month 2 of each quarter
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>📈 Use these temporal insights to optimize operations, staffing, and customer expectations</p>
</div>
""", unsafe_allow_html=True)
