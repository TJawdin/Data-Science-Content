"""
Batch Predictions Page
Upload CSV file and get predictions for multiple orders
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_engineering import calculate_features
from utils.model_loader import load_model, predict_batch
from utils.theme_adaptive import apply_adaptive_theme, get_adaptive_colors

# Page config
st.set_page_config(
    page_title="Batch Predictions",
    page_icon="📦",
    layout="wide"
)

# Apply adaptive theme
apply_adaptive_theme()

# Get adaptive colors
colors = get_adaptive_colors()

# ============================================================================
# Header
# ============================================================================

st.title("📦 Batch Predictions")
st.markdown("""
Upload a CSV file containing multiple orders to get risk predictions for all of them at once.
Perfect for daily order processing and risk monitoring!
""")

st.markdown("---")

# ============================================================================
# Synthetic Data Generator Function
# ============================================================================

def generate_synthetic_orders(n_orders=100):
    """
    Generate synthetic order data with realistic distributions
    Includes mix of low, medium, and high risk orders
    
    Parameters:
    -----------
    n_orders : int
        Number of orders to generate
    
    Returns:
    --------
    pd.DataFrame : Synthetic order data
    """
    
    np.random.seed(None)  # Random seed for variety
    
    orders = []
    
    # Target distribution: 40% low, 40% medium, 20% high risk
    risk_categories = np.random.choice(
        ['low', 'medium', 'high'],
        size=n_orders,
        p=[0.4, 0.4, 0.2]
    )
    
    for i, risk_cat in enumerate(risk_categories):
        
        if risk_cat == 'low':
            # LOW RISK: Simple, local, standard timeline
            num_items = np.random.randint(1, 3)
            num_sellers = 1
            total_order_value = np.random.uniform(50, 200)
            total_shipping_cost = np.random.uniform(5, 15)
            total_weight_g = np.random.randint(200, 1500)
            avg_length_cm = np.random.uniform(15, 30)
            avg_height_cm = np.random.uniform(10, 20)
            avg_width_cm = np.random.uniform(8, 15)
            avg_shipping_distance_km = np.random.randint(20, 150)
            is_cross_state = 0
            estimated_days = np.random.randint(10, 20)
            order_weekday = np.random.choice([0, 1, 2, 3, 4])  # Weekdays
            order_month = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Non-holiday
            order_hour = np.random.randint(8, 18)  # Business hours
            
        elif risk_cat == 'medium':
            # MEDIUM RISK: Moderate complexity/distance
            num_items = np.random.randint(2, 5)
            num_sellers = np.random.choice([1, 2], p=[0.6, 0.4])
            total_order_value = np.random.uniform(150, 400)
            total_shipping_cost = np.random.uniform(15, 40)
            total_weight_g = np.random.randint(1000, 4000)
            avg_length_cm = np.random.uniform(25, 45)
            avg_height_cm = np.random.uniform(18, 30)
            avg_width_cm = np.random.uniform(12, 25)
            avg_shipping_distance_km = np.random.randint(150, 800)
            is_cross_state = np.random.choice([0, 1], p=[0.3, 0.7])
            estimated_days = np.random.randint(7, 15)
            order_weekday = np.random.randint(0, 7)  # Any day
            order_month = np.random.randint(1, 13)  # Any month
            order_hour = np.random.randint(6, 22)
            
        else:  # high risk
            # HIGH RISK: Complex, remote, rushed
            num_items = np.random.randint(4, 10)
            num_sellers = np.random.randint(2, 5)
            total_order_value = np.random.uniform(300, 800)
            total_shipping_cost = np.random.uniform(40, 120)
            total_weight_g = np.random.randint(3000, 10000)
            avg_length_cm = np.random.uniform(40, 70)
            avg_height_cm = np.random.uniform(30, 50)
            avg_width_cm = np.random.uniform(20, 40)
            avg_shipping_distance_km = np.random.randint(800, 2500)
            is_cross_state = 1
            estimated_days = np.random.randint(3, 8)  # Rush order
            order_weekday = np.random.choice([5, 6])  # Weekend
            order_month = np.random.choice([11, 12])  # Holiday season
            order_hour = np.random.choice(list(range(0, 6)) + list(range(20, 24)))  # Off hours
        
        # Calculate derived fields
        is_weekend_order = 1 if order_weekday >= 5 else 0
        is_holiday_season = 1 if order_month in [11, 12] else 0
        
        order = {
            'order_id': f'ORDER_{i+1:05d}',
            'num_items': num_items,
            'num_sellers': num_sellers,
            'num_products': num_items,
            'total_order_value': round(total_order_value, 2),
            'avg_item_price': round(total_order_value / num_items, 2),
            'max_item_price': round(total_order_value / num_items * 1.2, 2),
            'total_shipping_cost': round(total_shipping_cost, 2),
            'avg_shipping_cost': round(total_shipping_cost / num_items, 2),
            'total_weight_g': int(total_weight_g),
            'avg_weight_g': int(total_weight_g / num_items),
            'max_weight_g': int(total_weight_g / num_items * 1.3),
            'avg_length_cm': round(avg_length_cm, 1),
            'avg_height_cm': round(avg_height_cm, 1),
            'avg_width_cm': round(avg_width_cm, 1),
            'avg_shipping_distance_km': int(avg_shipping_distance_km),
            'max_shipping_distance_km': int(avg_shipping_distance_km * 1.1),
            'is_cross_state': is_cross_state,
            'order_weekday': order_weekday,
            'order_month': order_month,
            'order_hour': order_hour,
            'is_weekend_order': is_weekend_order,
            'is_holiday_season': is_holiday_season,
            'estimated_days': estimated_days
        }
        
        orders.append(order)
    
    df = pd.DataFrame(orders)
    return df

def create_risk_distribution(predictions_df):
    """
    Create a pie chart showing risk level distribution
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with prediction results including 'risk_level' column
    
    Returns:
    --------
    plotly.graph_objects.Figure : Pie chart figure
    """
    
    colors_dict = get_adaptive_colors()
    
    # Count risk levels
    risk_counts = predictions_df['risk_level'].value_counts()
    
    # Define colors for each risk level
    color_map = {
        'LOW': colors_dict['low_risk'],
        'MEDIUM': colors_dict['medium_risk'],
        'HIGH': colors_dict['high_risk']
    }
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        marker=dict(colors=[color_map.get(level, '#999999') for level in risk_counts.index]),
        textinfo='label+percent+value',
        textfont=dict(size=14),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': 'Risk Level Distribution',
            'font': {'size': 16}
        },
        height=400,
        plot_bgcolor=colors_dict['bg_transparent'],
        paper_bgcolor=colors_dict['bg_transparent'],
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# ============================================================================
# Load Model
# ============================================================================

model = load_model()

# ============================================================================
# Load Model
# ============================================================================

model = load_model()

if model is None:
    st.error("⚠️ Model not found. Please copy your trained model to the artifacts folder.")
    st.stop()

# ============================================================================
# Synthetic Data Generator
# ============================================================================

st.markdown("## 🎲 Generate Test Data")

col1, col2 = st.columns([3, 1])

with col1:
    st.info("""
    **Don't have a CSV file?** Generate synthetic test data with realistic order characteristics!
    
    The generated data will include a mix of:
    - 🟢 **Low Risk Orders** (~40%) - Simple, local, standard timeline
    - 🟡 **Medium Risk Orders** (~40%) - Moderate complexity/distance
    - 🔴 **High Risk Orders** (~20%) - Complex, remote, rushed
    """)

with col2:
    num_orders = st.number_input(
        "Number of Orders",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="How many test orders to generate"
    )
    
    if st.button("🎲 Generate Test Data", type="primary", use_container_width=True):
        
        with st.spinner(f"Generating {num_orders} synthetic orders..."):
            
            # Generate synthetic data
            synthetic_data = generate_synthetic_orders(num_orders)
            
            # Convert to CSV
            csv = synthetic_data.to_csv(index=False)
            
            # Success message
            st.success(f"✅ Generated {num_orders} orders with mixed risk levels!")
            
            # Show preview
            with st.expander("📊 Preview Generated Data (first 10 rows)"):
                st.dataframe(synthetic_data.head(10), use_container_width=True)
            
            # Download button
            st.download_button(
                label="📥 Download Test Data (CSV)",
                data=csv,
                file_name=f"synthetic_orders_{num_orders}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
            
            st.info("""
            **Next Step:** Use the file uploader below to upload this CSV and test batch predictions!
            """)

st.markdown("---")

# ============================================================================
# CSV Template Download
# ============================================================================

st.markdown("### 📄 Step 1: Prepare Your CSV File")

# Use HTML/CSS for guaranteed visibility
st.markdown("""
<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;">
    <p style="color: #262730; margin: 0; font-size: 14px;">
        <strong style="color: #262730;">Required Columns</strong> (minimum):<br>
        • <code>num_items</code>, <code>total_order_value</code>, <code>total_shipping_cost</code>, <code>total_weight_g</code><br>
        • <code>avg_length_cm</code>, <code>avg_height_cm</code>, <code>avg_width_cm</code><br>
        • <code>avg_shipping_distance_km</code>, <code>estimated_days</code>
    </p>
    <p style="color: #262730; margin-top: 10px; font-size: 14px;">
        <strong style="color: #262730;">Optional Columns</strong> (will use defaults if missing):<br>
        • <code>num_sellers</code>, <code>is_cross_state</code>, <code>order_weekday</code>, <code>order_month</code>, <code>order_hour</code>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Create sample CSV template
sample_data = {
    'order_id': ['ORD001', 'ORD002', 'ORD003'],
    'num_items': [2, 1, 5],
    'num_sellers': [1, 1, 2],
    'total_order_value': [150.0, 80.0, 300.0],
    'total_shipping_cost': [20.0, 15.0, 35.0],
    'total_weight_g': [2000, 500, 5000],
    'avg_length_cm': [30.0, 20.0, 40.0],
    'avg_height_cm': [20.0, 15.0, 30.0],
    'avg_width_cm': [15.0, 10.0, 25.0],
    'avg_shipping_distance_km': [500, 200, 800],
    'is_cross_state': [1, 0, 1],
    'order_weekday': [2, 4, 5],
    'order_month': [6, 6, 11],
    'order_hour': [14, 10, 18],
    'estimated_days': [10, 7, 15]
}

sample_df = pd.DataFrame(sample_data)

# Convert to CSV
csv_buffer = BytesIO()
sample_df.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

# Center the download button with custom styling
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    st.markdown("""
    <style>
        div[data-testid="stDownloadButton"] button {
            background-color: #0068C9 !important;
            color: white !important;
            border: none !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
        }
        div[data-testid="stDownloadButton"] button:hover {
            background-color: #0056a3 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=csv_buffer,
        file_name="sample_orders_template.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("""
<p style="text-align: center; color: #666666; font-size: 12px; margin-top: 10px;">
    💡 The sample template contains 3 example orders for reference
</p>
""", unsafe_allow_html=True)

st.markdown("---")# ============================================================================
# File Upload
# ============================================================================

st.markdown("### 📤 Step 2: Upload Your CSV File")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file with order data"
)

if uploaded_file is not None:
    
    try:
        # Read CSV
        input_df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! Found {len(input_df)} orders.")
        
        # Show preview
        with st.expander("👀 Preview Uploaded Data (First 10 rows)"):
            st.dataframe(input_df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # ========================================================================
        # Process Orders
        # ========================================================================
        
        st.markdown("### 🔄 Step 3: Process Predictions")
        
        if st.button("🚀 Run Batch Predictions", type="primary", use_container_width=True):
            
            with st.spinner(f"Processing {len(input_df)} orders..."):
                
                # Prepare features for each order
                all_features = []
                order_ids = []
                
                for idx, row in input_df.iterrows():
                    try:
                        # Extract order ID if present
                        order_id = row.get('order_id', f'Order_{idx+1}')
                        order_ids.append(order_id)
                        
                        # Prepare order data
                        order_data = {
                            'num_items': row.get('num_items', 1),
                            'num_sellers': row.get('num_sellers', 1),
                            'num_products': row.get('num_items', 1),
                            'total_order_value': row.get('total_order_value', 0),
                            'avg_item_price': row.get('total_order_value', 0) / max(row.get('num_items', 1), 1),
                            'max_item_price': row.get('total_order_value', 0) / max(row.get('num_items', 1), 1),
                            'total_shipping_cost': row.get('total_shipping_cost', 0),
                            'avg_shipping_cost': row.get('total_shipping_cost', 0) / max(row.get('num_items', 1), 1),
                            'total_weight_g': row.get('total_weight_g', 0),
                            'avg_weight_g': row.get('total_weight_g', 0) / max(row.get('num_items', 1), 1),
                            'max_weight_g': row.get('total_weight_g', 0) / max(row.get('num_items', 1), 1),
                            'avg_length_cm': row.get('avg_length_cm', 30),
                            'avg_height_cm': row.get('avg_height_cm', 20),
                            'avg_width_cm': row.get('avg_width_cm', 15),
                            'avg_shipping_distance_km': row.get('avg_shipping_distance_km', 500),
                            'max_shipping_distance_km': row.get('avg_shipping_distance_km', 500),
                            'is_cross_state': row.get('is_cross_state', 0),
                            'order_weekday': row.get('order_weekday', 2),
                            'order_month': row.get('order_month', 6),
                            'order_hour': row.get('order_hour', 14),
                            'is_weekend_order': 1 if row.get('order_weekday', 2) >= 5 else 0,
                            'is_holiday_season': 1 if row.get('order_month', 6) in [11, 12] else 0,
                            'estimated_days': row.get('estimated_days', 10)
                        }
                        
                        # Calculate features
                        features = calculate_features(order_data)
                        all_features.append(features)
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error processing row {idx+1}: {str(e)}")
                        continue
                
                # Combine all features
                if all_features:
                    features_df = pd.concat(all_features, ignore_index=True)
                    
                    # Make predictions
                    predictions = predict_batch(model, features_df)
                    
                    if predictions is not None:
                        
                        # Add order IDs
                        predictions.insert(0, 'Order_ID', order_ids[:len(predictions)])
                        
                        # Success message
                        st.success(f"✅ Successfully processed {len(predictions)} orders!")
                        
                        st.markdown("---")
                        
                        # ============================================================
                        # Interactive Dashboard
                        # ============================================================
                        
                        st.markdown("## 📊 Results Dashboard")
                        
                        # Key Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_orders = len(predictions)
                            st.metric("Total Orders", f"{total_orders:,}")
                        
                        with col2:
                            late_count = (predictions['Prediction'] == 'Late').sum()
                            late_pct = (late_count / total_orders * 100)
                            st.metric("Predicted Late", f"{late_count}", f"{late_pct:.1f}%")
                        
                        with col3:
                            high_risk = (predictions['risk_level'] == 'HIGH').sum()
                            st.metric("High Risk Orders", f"{high_risk}", 
                                     delta="Needs Attention" if high_risk > 0 else "All Clear",
                                     delta_color="inverse" if high_risk > 0 else "normal")
                        
                        with col4:
                            avg_risk = predictions['Risk_Score'].mean()
                            st.metric("Average Risk Score", f"{avg_risk:.1f}/100")
                        
                        st.markdown("---")
                        
                        # Visualizations
                        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Risk Analysis", "📋 Detailed Results"])
                        
                        with tab1:
                            st.markdown("### 📊 Prediction Overview")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Prediction pie chart
                                pred_counts = predictions['Prediction'].value_counts()
                                import plotly.graph_objects as go
                                
                                fig_pie = go.Figure(data=[go.Pie(
                                    labels=pred_counts.index,
                                    values=pred_counts.values,
                                    marker=dict(colors=['#2ECC71', '#E74C3C']),
                                    hole=0.4
                                )])
                                fig_pie.update_layout(
                                    title="Prediction Distribution",
                                    height=400
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            with col2:
                                # Risk level bar chart
                                risk_counts = predictions['risk_level'].value_counts()
                                risk_order = ['LOW', 'MEDIUM', 'HIGH']
                                risk_counts = risk_counts.reindex(risk_order, fill_value=0)
                                
                                fig_bar = go.Figure(data=[go.Bar(
                                    x=risk_counts.index,
                                    y=risk_counts.values,
                                    marker=dict(color=['#2ECC71', '#F39C12', '#E74C3C']),
                                    text=risk_counts.values,
                                    textposition='auto'
                                )])
                                fig_bar.update_layout(
                                    title="Risk Level Distribution",
                                    xaxis_title="Risk Level",
                                    yaxis_title="Number of Orders",
                                    height=400
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)
                        
                        with tab2:
                            st.markdown("### 📈 Risk Score Analysis")
                            
                           # Risk distribution histogram
                            fig_hist = create_risk_distribution(predictions)  # Pass full DataFrame, not just Risk_Score column
                            st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Risk statistics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Min Risk Score", f"{predictions['Risk_Score'].min()}")
                            with col2:
                                st.metric("Median Risk Score", f"{predictions['Risk_Score'].median():.0f}")
                            with col3:
                                st.metric("Max Risk Score", f"{predictions['Risk_Score'].max()}")
                        
                        with tab3:
                            st.markdown("### 📋 Detailed Prediction Results")
                            
                            # Add filters
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                filter_prediction = st.multiselect(
                                    "Filter by Prediction",
                                    options=['On-Time', 'Late'],
                                    default=['On-Time', 'Late']
                                )
                            
                            with col2:
                                filter_risk = st.multiselect(
                                    "Filter by Risk Level",
                                    options=['LOW', 'MEDIUM', 'HIGH'],
                                    default=['LOW', 'MEDIUM', 'HIGH']
                                )
                            
                            with col3:
                                sort_by = st.selectbox(
                                    "Sort by",
                                    options=['Risk_Score', 'Order_ID', 'Prediction'],
                                    index=0
                                )
                            
                            # Apply filters
                            filtered_df = predictions[
                                (predictions['Prediction'].isin(filter_prediction)) &
                                (predictions['risk_level'].isin(filter_risk))
                            ].sort_values(sort_by, ascending=False)
                            
                            # Display table
                            st.dataframe(
                                filtered_df.style.applymap(
                                    lambda x: 'background-color: #FADBD8' if x == 'HIGH' else 
                                             ('background-color: #FEF5E7' if x == 'MEDIUM' else 
                                              ('background-color: #D5F4E6' if x == 'LOW' else '')),
                                    subset=['risk_level']
                                ),
                                use_container_width=True,
                                height=400
                            )
                            
                            st.caption(f"Showing {len(filtered_df)} of {len(predictions)} orders")
                        
                        st.markdown("---")
                        
                        # ============================================================
                        # Download Results
                        # ============================================================
                        
                        st.markdown("### 📥 Step 4: Download Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Prepare enhanced CSV with recommendations
                            output_df = input_df.copy()
                            
                            # Add predictions
                            for col in predictions.columns:
                                if col != 'Order_ID':
                                    output_df[col] = predictions[col].values
                            
                            # Add recommendations
                            def get_recommendation(risk_level):
                                if risk_level == 'HIGH':
                                    return "URGENT: Expedite shipping, contact customer"
                                elif risk_level == 'MEDIUM':
                                    return "Monitor closely, ensure optimal routing"
                                else:
                                    return "Standard processing"
                            
                            output_df['Recommendation'] = predictions['risk_level'].apply(get_recommendation)
                            
                            # Convert to CSV
                            output_csv = output_df.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Download Complete Results (CSV)",
                                data=output_csv,
                                file_name=f"predictions_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Download high-risk orders only
                            high_risk_df = output_df[predictions['risk_level'] == 'HIGH']
                            
                            if len(high_risk_df) > 0:
                                high_risk_csv = high_risk_df.to_csv(index=False)
                                
                                st.download_button(
                                    label=f"⚠️ Download High-Risk Orders Only ({len(high_risk_df)})",
                                    data=high_risk_csv,
                                    file_name=f"high_risk_orders_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            else:
                                st.success("✅ No high-risk orders found!")
                
                else:
                    st.error("❌ No valid orders to process. Please check your CSV file.")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please make sure your CSV file matches the expected format.")

else:
    # Show instructions when no file uploaded
    st.info("👆 Upload a CSV file to get started!")
    
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - **Include all required columns** for accurate predictions
    - **Use consistent units**: Weights in grams, distances in km, prices in dollars
    - **Provide order IDs** for easy tracking (optional but recommended)
    - **Test with sample file** first to verify format
    """)

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## 📦 Batch Processing")
    st.info("""
    **Benefits:**
    - Process hundreds of orders in seconds
    - Identify high-risk orders quickly
    - Download actionable reports
    - Optimize resource allocation
    """)
    
    st.markdown("---")
    
    st.markdown("## 📊 Output Includes")
    st.markdown("""
    - Risk score (0-100)
    - Risk level (LOW/MEDIUM/HIGH)
    - Prediction (On-Time/Late)
    - Actionable recommendations
    - Interactive dashboard
    """)
    
    st.markdown("---")
    
    st.markdown("## 🎯 Use Cases")
    st.markdown("""
    - **Daily operations**: Morning batch processing
    - **Planning**: Identify resource needs
    - **Alerts**: Flag orders needing attention
    - **Reporting**: Share with stakeholders
    """)
