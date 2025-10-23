"""
Geographic Map Page
Visualize delay risk across different geographic regions
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent))

from config import *
from utils.model_loader import load_model, load_metadata, batch_predict
from utils.feature_engineering import create_sample_data

st.set_page_config(page_title="Geographic Map", page_icon="🗺️", layout="wide")

# Load resources
@st.cache_resource
def load_resources():
    model = load_model(str(MODEL_PATH))
    final_metadata = load_metadata(str(FINAL_METADATA_PATH))
    feature_metadata = load_metadata(str(FEATURE_METADATA_PATH))
    return model, final_metadata, feature_metadata

model, final_metadata, feature_metadata = load_resources()

# Title
st.title("🗺️ Geographic Risk Distribution")
st.markdown("""
Visualize how delivery delay risk varies across different geographic regions.
Identify high-risk areas and optimize your logistics strategy accordingly.
""")

# Data source selection
st.markdown("### 📊 Data Source")

data_source = st.radio(
    "Select data source:",
    options=["Upload CSV", "Use Sample Data", "Use Batch Results"],
    horizontal=True
)

df_to_analyze = None

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file with geographic data", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} orders")
            
            if 'probability' not in df.columns:
                with st.spinner("Generating predictions..."):
                    df_to_analyze = batch_predict(
                        model, df, feature_metadata['feature_names'],
                        OPTIMAL_THRESHOLD, RISK_BANDS
                    )
            else:
                df_to_analyze = df
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

elif data_source == "Use Sample Data":
    if st.button("Generate Sample Data", type="primary"):
        with st.spinner("Generating sample data..."):
            sample_df = create_sample_data(n_samples=100)
            df_to_analyze = batch_predict(
                model, sample_df, feature_metadata['feature_names'],
                OPTIMAL_THRESHOLD, RISK_BANDS
            )
            st.success(f"✅ Generated {len(df_to_analyze)} sample orders")

elif data_source == "Use Batch Results":
    if 'batch_results' in st.session_state:
        df_to_analyze = st.session_state['batch_results']
        st.success(f"✅ Using {len(df_to_analyze)} orders from batch predictions")
    else:
        st.warning("⚠️ No batch results found. Please run batch predictions first.")

# Analysis section
if df_to_analyze is not None and len(df_to_analyze) > 0:
    st.markdown("---")
    st.markdown("## 🗺️ Geographic Analysis")
    
    # Geography selection
    geo_level = st.radio(
        "Geographic Level:",
        options=["customer_state", "customer_city", "seller_state_mode"],
        format_func=lambda x: {
            'customer_state': 'Customer State',
            'customer_city': 'Customer City',
            'seller_state_mode': 'Seller State'
        }[x],
        horizontal=True
    )
    
    # Aggregate by geography
    geo_stats = df_to_analyze.groupby(geo_level).agg({
        'probability': ['mean', 'median', 'std', 'count'],
        'risk_category': lambda x: (x == 'high').sum()
    }).round(4)
    
    geo_stats.columns = ['Avg_Probability', 'Median_Probability', 'Std_Dev', 'Order_Count', 'High_Risk_Count']
    geo_stats['Avg_Probability_Pct'] = (geo_stats['Avg_Probability'] * 100).round(1)
    geo_stats['High_Risk_Pct'] = (geo_stats['High_Risk_Count'] / geo_stats['Order_Count'] * 100).round(1)
    geo_stats = geo_stats.reset_index()
    
    # Create choropleth/bar chart
    st.markdown("### 📊 Risk Distribution Map")
    
    # Sort by average probability
    geo_stats_sorted = geo_stats.sort_values('Avg_Probability', ascending=False)
    
    # Create bar chart (since we don't have actual map coordinates)
    fig = go.Figure()
    
    # Determine colors based on risk
    colors = []
    for prob in geo_stats_sorted['Avg_Probability']:
        if prob * 100 <= RISK_BANDS['low']['max']:
            colors.append(RISK_BANDS['low']['color'])
        elif prob * 100 <= RISK_BANDS['medium']['max']:
            colors.append(RISK_BANDS['medium']['color'])
        else:
            colors.append(RISK_BANDS['high']['color'])
    
    fig.add_trace(go.Bar(
        x=geo_stats_sorted[geo_level],
        y=geo_stats_sorted['Avg_Probability_Pct'],
        marker=dict(color=colors),
        text=geo_stats_sorted['Avg_Probability_Pct'],
        texttemplate='%{text:.1f}%',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg Delay Probability: %{y:.1f}%<br>Orders: ' + 
                     geo_stats_sorted['Order_Count'].astype(str) + '<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Average Delay Probability by {geo_level.replace('_', ' ').title()}",
        xaxis_title=geo_level.replace('_', ' ').title(),
        yaxis_title="Average Delay Probability (%)",
        height=500,
        showlegend=False,
        hovermode='x',
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top/Bottom performers
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Highest Risk Regions")
        top_risk = geo_stats_sorted.head(5)
        
        for idx, row in top_risk.iterrows():
            with st.container():
                st.markdown(f"""
                <div style='background-color: {RISK_BANDS['high']['color']}22; 
                            padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;
                            border-left: 4px solid {RISK_BANDS['high']['color']};'>
                    <strong>{row[geo_level]}</strong><br>
                    Avg Risk: {row['Avg_Probability_Pct']:.1f}% | 
                    Orders: {int(row['Order_Count'])} | 
                    High Risk: {int(row['High_Risk_Count'])}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🟢 Lowest Risk Regions")
        bottom_risk = geo_stats_sorted.tail(5)
        
        for idx, row in bottom_risk.iterrows():
            with st.container():
                st.markdown(f"""
                <div style='background-color: {RISK_BANDS['low']['color']}22; 
                            padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;
                            border-left: 4px solid {RISK_BANDS['low']['color']};'>
                    <strong>{row[geo_level]}</strong><br>
                    Avg Risk: {row['Avg_Probability_Pct']:.1f}% | 
                    Orders: {int(row['Order_Count'])} | 
                    High Risk: {int(row['High_Risk_Count'])}
                </div>
                """, unsafe_allow_html=True)
    
    # Detailed statistics
    st.markdown("---")
    st.markdown("### 📊 Detailed Geographic Statistics")
    
    # Format for display
    display_df = geo_stats_sorted[[
        geo_level, 'Order_Count', 'Avg_Probability_Pct', 
        'Median_Probability', 'Std_Dev', 'High_Risk_Count', 'High_Risk_Pct'
    ]].copy()
    
    display_df.columns = [
        'Location', 'Orders', 'Avg Risk (%)', 
        'Median Prob', 'Std Dev', 'High Risk Orders', 'High Risk %'
    ]
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Key insights
    st.markdown("---")
    st.markdown("### 💡 Geographic Insights")
    
    highest_risk_location = geo_stats_sorted.iloc[0]
    lowest_risk_location = geo_stats_sorted.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Highest Risk Region",
            highest_risk_location[geo_level],
            f"{highest_risk_location['Avg_Probability_Pct']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Lowest Risk Region",
            lowest_risk_location[geo_level],
            f"{lowest_risk_location['Avg_Probability_Pct']:.1f}%"
        )
    
    with col3:
        risk_range = highest_risk_location['Avg_Probability_Pct'] - lowest_risk_location['Avg_Probability_Pct']
        st.metric(
            "Risk Variance",
            f"{risk_range:.1f}%",
            "Difference between regions"
        )
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 🎯 Strategic Recommendations")
    
    # Identify patterns
    high_risk_regions = geo_stats_sorted[geo_stats_sorted['Avg_Probability'] > 0.4]
    low_volume_high_risk = geo_stats_sorted[
        (geo_stats_sorted['Avg_Probability'] > 0.4) & 
        (geo_stats_sorted['Order_Count'] < geo_stats_sorted['Order_Count'].median())
    ]
    
    if len(high_risk_regions) > 0:
        st.warning(f"""
        **High-Risk Regions Identified ({len(high_risk_regions)} locations)**
        
        Consider these actions for high-risk regions:
        - Establish local distribution centers or partnerships
        - Offer expedited shipping options
        - Set realistic delivery time expectations
        - Proactive customer communication about potential delays
        """)
    
    if len(low_volume_high_risk) > 0:
        st.info(f"""
        **Low Volume, High Risk ({len(low_volume_high_risk)} locations)**
        
        These regions have both low order volume and high delay risk:
        - Consider batching orders for efficiency
        - Partner with specialized logistics providers
        - Evaluate market potential vs operational costs
        """)
    
    # Distribution insights
    total_orders = geo_stats['Order_Count'].sum()
    top_3_pct = geo_stats_sorted.head(3)['Order_Count'].sum() / total_orders * 100
    
    st.success(f"""
    **Volume Distribution:**
    - Top 3 regions account for {top_3_pct:.1f}% of total orders
    - Focus optimization efforts on high-volume regions first
    - Consider region-specific logistics strategies
    """)
    
    # Download data
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Geographic Analysis",
            data=csv_data,
            file_name=f"geographic_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("👆 Select a data source and load data to begin geographic analysis")
    
    st.markdown("### 🗺️ What You Can Discover")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Regional Patterns
        - Which states have highest delay risk?
        - Urban vs rural delivery performance
        - Regional logistics challenges
        
        #### Volume Distribution
        - Where are most orders coming from?
        - High-volume vs high-risk regions
        - Market penetration opportunities
        """)
    
    with col2:
        st.markdown("""
        #### Optimization Opportunities
        - Where to establish warehouses?
        - Which regions need better logistics?
        - Partnership opportunities
        
        #### Strategic Planning
        - Market expansion priorities
        - Resource allocation decisions
        - Risk mitigation strategies
        """)
