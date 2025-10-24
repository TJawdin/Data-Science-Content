"""
Geographic Map Page
Visualize delay risk across different geographic regions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_model_artifacts,
    load_metadata,
    apply_custom_css,
    show_page_header,
    predict_delay,
    prepare_features,
    create_example_order,
    format_state_name,
    format_city_name
)

# Page config
st.set_page_config(page_title="Geographic Map", page_icon="🗺️", layout="wide")
apply_custom_css()

# Load model
model, final_metadata, feature_metadata, threshold = load_model_artifacts()

# Header
show_page_header(
    title="Geographic Risk Distribution",
    description="Visualize how delivery delay risk varies across different geographic regions",
    icon="🗺️"
)

# Get risk bands from metadata
risk_bands = {
    'low': {'max': final_metadata['risk_bands']['low_max'], 'color': '#00CC96'},
    'medium': {'max': final_metadata['risk_bands']['med_max'], 'color': '#FFA500'},
    'high': {'max': 100, 'color': '#EF553B'}
}

# Data source selection
st.markdown("### 📊 Data Source")

data_source = st.radio(
    "Select data source:",
    options=["Generate Sample Data", "Use Batch Results"],
    horizontal=True
)

df_to_analyze = None

if data_source == "Generate Sample Data":
    st.info("Generate sample data to explore geographic patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("Number of samples", 100, 500, 1000)
    
    with col2:
        if st.button("🔄 Generate Sample Data", type="primary", use_container_width=True):
            with st.spinner("Generating sample data..."):
                # Generate sample orders across different states
                brazilian_states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'ES', 'GO', 
                                   'PE', 'CE', 'PA', 'AM', 'MA', 'RN', 'PB', 'AL', 'PI', 'SE']
                
                cities_by_state = {
                    'SP': ['sao paulo', 'campinas', 'santos', 'sorocaba'],
                    'RJ': ['rio de janeiro', 'niteroi', 'duque de caxias'],
                    'MG': ['belo horizonte', 'uberlandia', 'contagem'],
                    'AM': ['manaus', 'itacoatiara'],
                    'BA': ['salvador', 'feira de santana'],
                }
                
                sample_data = []
                
                for _ in range(n_samples):
                    # Random state
                    state = np.random.choice(brazilian_states)
                    city = np.random.choice(cities_by_state.get(state, ['capital']))
                    
                    # Base scenario with variations
                    base = create_example_order('typical')
                    
                    # Modify based on state (remote states have higher risk)
                    if state in ['AM', 'RR', 'AP', 'AC', 'RO']:  # Amazon region
                        base['est_lead_days'] = np.random.uniform(10, 20)
                        base['sum_freight'] = np.random.uniform(50, 100)
                    else:
                        base['est_lead_days'] = np.random.uniform(3, 10)
                        base['sum_freight'] = np.random.uniform(10, 40)
                    
                    base['customer_state'] = state
                    base['customer_city'] = city
                    base['seller_state_mode'] = np.random.choice(['SP', 'RJ', 'MG'])
                    
                    sample_data.append(base)
                
                # Create dataframe
                df = pd.DataFrame(sample_data)
                
                # Make predictions
                predictions_list = []
                probabilities_list = []
                risk_levels_list = []
                
                for idx, row in df.iterrows():
                    features = prepare_features(row.to_dict(), feature_metadata['feature_names'])
                    preds, probs, risks = predict_delay(model, features, threshold)
                    predictions_list.append(preds[0])
                    probabilities_list.append(probs[0])
                    risk_levels_list.append(risks[0])
                
                df['prediction'] = predictions_list
                df['probability'] = probabilities_list
                df['risk_category'] = risk_levels_list
                
                # Store in session state to prevent loss on radio button click
                st.session_state['geo_analysis_df'] = df
                df_to_analyze = df
                st.success(f"✅ Generated {len(df_to_analyze)} sample orders")

elif data_source == "Use Batch Results":
    if 'prediction_results' in st.session_state:
        df_to_analyze = st.session_state['prediction_results']
        st.success(f"✅ Using {len(df_to_analyze)} orders from batch predictions")
    else:
        st.warning("⚠️ No batch results found. Please run batch predictions first on the Batch Predictions page.")

# Check if we have data in session state from previous interaction
if df_to_analyze is None and 'geo_analysis_df' in st.session_state:
    df_to_analyze = st.session_state['geo_analysis_df']

# Analysis section
if df_to_analyze is not None and len(df_to_analyze) > 0:
    st.markdown("---")
    st.markdown("## 🗺️ Geographic Analysis")
    
    # Initialize session state for geographic level if not exists
    if 'geo_level' not in st.session_state:
        st.session_state.geo_level = 'customer_state'
    
    # Geography selection - WITH SESSION STATE KEY TO PREVENT RESET
    geo_level = st.radio(
        "Geographic Level:",
        options=["customer_state", "customer_city"],
        format_func=lambda x: {
            'customer_state': 'Customer State',
            'customer_city': 'Customer City'
        }[x],
        key='geo_level',  # THIS IS THE CRITICAL FIX - prevents page reset!
        horizontal=True
    )
    
    # IMPORTANT: Aggregate FIRST with original names, then format for display
    # Aggregating with formatted names causes grouping issues
    
    # Aggregate by geography using ORIGINAL unformatted names
    geo_stats = df_to_analyze.groupby(geo_level).agg({
        'probability': ['mean', 'median', 'std', 'count'],
        'risk_category': lambda x: (x == 'High').sum()
    }).round(4)
    
    geo_stats.columns = ['Avg_Probability', 'Median_Probability', 'Std_Dev', 'Order_Count', 'High_Risk_Count']
    geo_stats['Avg_Probability_Pct'] = (geo_stats['Avg_Probability'] * 100).round(1)
    geo_stats['High_Risk_Pct'] = (geo_stats['High_Risk_Count'] / geo_stats['Order_Count'] * 100).round(1)
    geo_stats = geo_stats.reset_index()
    
    # NOW format location names AFTER aggregation for display purposes only
    if geo_level == 'customer_state':
        # Format state names: "SP" -> "SP - São Paulo"
        geo_stats['location_display'] = geo_stats[geo_level].apply(format_state_name)
    elif geo_level == 'customer_city':
        # Format city names: "sao paulo" -> "São Paulo"
        geo_stats['location_display'] = geo_stats[geo_level].apply(format_city_name)
    
    display_col = 'location_display'
    
    # Create choropleth/bar chart
    st.markdown("### 📊 Risk Distribution Map")
    
    # Sort by average probability
    geo_stats_sorted = geo_stats.sort_values('Avg_Probability', ascending=False)
    
    # Create bar chart
    fig = go.Figure()
    
    # Determine colors based on risk
    colors = []
    for prob in geo_stats_sorted['Avg_Probability']:
        if prob * 100 <= risk_bands['low']['max']:
            colors.append(risk_bands['low']['color'])
        elif prob * 100 <= risk_bands['medium']['max']:
            colors.append(risk_bands['medium']['color'])
        else:
            colors.append(risk_bands['high']['color'])
    
    fig.add_trace(go.Bar(
        x=geo_stats_sorted[display_col],
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
        xaxis_tickangle=-45  # Angle labels for better readability with long names
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top/Bottom performers
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Highest Risk Regions")
        top_risk = geo_stats_sorted.head(5)
        
        for idx, row in top_risk.iterrows():
            location_display = row[display_col] if display_col in row else row[geo_level]
            st.markdown(f"""
            <div style='background-color: {risk_bands['high']['color']}22; 
                        padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;
                        border-left: 4px solid {risk_bands['high']['color']};'>
                <strong>{location_display}</strong><br>
                Avg Risk: {row['Avg_Probability_Pct']:.1f}% | 
                Orders: {int(row['Order_Count'])} | 
                High Risk: {int(row['High_Risk_Count'])}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🟢 Lowest Risk Regions")
        bottom_risk = geo_stats_sorted.tail(5)
        
        for idx, row in bottom_risk.iterrows():
            location_display = row[display_col] if display_col in row else row[geo_level]
            st.markdown(f"""
            <div style='background-color: {risk_bands['low']['color']}22; 
                        padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;
                        border-left: 4px solid {risk_bands['low']['color']};'>
                <strong>{location_display}</strong><br>
                Avg Risk: {row['Avg_Probability_Pct']:.1f}% | 
                Orders: {int(row['Order_Count'])} | 
                High Risk: {int(row['High_Risk_Count'])}
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed statistics
    st.markdown("---")
    st.markdown("### 📊 Detailed Geographic Statistics")
    
    # Format for display
    display_stats_df = geo_stats_sorted[[
        display_col, 'Order_Count', 'Avg_Probability_Pct', 
        'Median_Probability', 'Std_Dev', 'High_Risk_Count', 'High_Risk_Pct'
    ]].copy()
    
    display_stats_df.columns = [
        'Location', 'Orders', 'Avg Risk (%)', 
        'Median Prob', 'Std Dev', 'High Risk Orders', 'High Risk %'
    ]
    
    st.dataframe(
        display_stats_df.style.format({
            'Avg Risk (%)': '{:.1f}',
            'Median Prob': '{:.4f}',
            'Std Dev': '{:.4f}',
            'High Risk %': '{:.1f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Key insights
    st.markdown("---")
    st.markdown("### 💡 Geographic Insights")
    
    highest_risk_location = geo_stats_sorted.iloc[0]
    lowest_risk_location = geo_stats_sorted.iloc[-1]
    
    highest_display = highest_risk_location[display_col] if display_col in highest_risk_location else highest_risk_location[geo_level]
    lowest_display = lowest_risk_location[display_col] if display_col in lowest_risk_location else lowest_risk_location[geo_level]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Highest Risk Region",
            highest_display,
            f"{highest_risk_location['Avg_Probability_Pct']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Lowest Risk Region",
            lowest_display,
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
        csv_data = display_stats_df.to_csv(index=False)
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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>🗺️ Use geographic insights to optimize your logistics network and reduce regional delays</p>
</div>
""", unsafe_allow_html=True)
