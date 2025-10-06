"""
Order Comparison Tool Page
Compare 2-3 orders side-by-side to understand risk differences
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_engineering import calculate_features, get_feature_descriptions
from utils.model_loader import load_model, predict_single
from utils.visualization import create_risk_gauge

# Page config
st.set_page_config(
    page_title="Compare Orders",
    page_icon="🔄",
    layout="wide"
)

# ============================================================================
# Header
# ============================================================================

st.title("🔄 Order Comparison Tool")
st.markdown("""
Compare up to 3 orders side-by-side to understand what drives different risk levels.
Perfect for training, analysis, and explaining model decisions!
""")

st.markdown("---")

# ============================================================================
# Load Model
# ============================================================================

model = load_model()

if model is None:
    st.error("⚠️ Model not found. Please copy your trained model to the artifacts folder.")
    st.stop()

# ============================================================================
# Number of Orders to Compare
# ============================================================================

st.markdown("## 🎛️ Comparison Settings")

num_orders = st.radio(
    "How many orders do you want to compare?",
    options=[2, 3],
    horizontal=True,
    help="Select 2 or 3 orders for side-by-side comparison"
)

st.markdown("---")

# ============================================================================
# Initialize Session State for Orders
# ============================================================================

if 'orders' not in st.session_state:
    st.session_state.orders = {}

# ============================================================================
# Create Input Forms for Each Order
# ============================================================================

st.markdown("## 📝 Enter Order Details")

# Create columns for each order
if num_orders == 2:
    cols = st.columns(2)
else:
    cols = st.columns(3)

orders_data = []

for idx, col in enumerate(cols):
    order_num = idx + 1
    
    with col:
        st.markdown(f"### 🛒 Order {order_num}")
        
        with st.form(f"order_{order_num}_form"):
            st.markdown("#### 📦 Order Info")
            
            num_items = st.number_input(
                "Number of Items",
                min_value=1,
                max_value=20,
                value=1 if order_num == 1 else (3 if order_num == 2 else 5),
                key=f"items_{order_num}"
            )
            
            num_sellers = st.number_input(
                "Number of Sellers",
                min_value=1,
                max_value=10,
                value=1 if order_num == 1 else (2 if order_num == 2 else 3),
                key=f"sellers_{order_num}"
            )
            
            total_order_value = st.number_input(
                "Total Order Value ($)",
                min_value=0.0,
                value=120.0 if order_num == 1 else (200.0 if order_num == 2 else 500.0),
                step=10.0,
                key=f"value_{order_num}"
            )
            
            total_shipping_cost = st.number_input(
                "Total Shipping Cost ($)",
                min_value=0.0,
                value=8.0 if order_num == 1 else (35.0 if order_num == 2 else 80.0),
                step=5.0,
                key=f"shipping_{order_num}"
            )
            
            st.markdown("#### 📏 Physical")
            
            total_weight_g = st.number_input(
                "Total Weight (g)",
                min_value=0,
                value=800 if order_num == 1 else (3000 if order_num == 2 else 8000),
                step=100,
                key=f"weight_{order_num}"
            )
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                avg_length_cm = st.number_input(
                    "Length (cm)",
                    min_value=0.0,
                    value=25.0 if order_num == 1 else (35.0 if order_num == 2 else 50.0),
                    key=f"length_{order_num}"
                )
            with col_b:
                avg_height_cm = st.number_input(
                    "Height (cm)",
                    min_value=0.0,
                    value=18.0 if order_num == 1 else (25.0 if order_num == 2 else 40.0),
                    key=f"height_{order_num}"
                )
            with col_c:
                avg_width_cm = st.number_input(
                    "Width (cm)",
                    min_value=0.0,
                    value=12.0 if order_num == 1 else (20.0 if order_num == 2 else 30.0),
                    key=f"width_{order_num}"
                )
            
            st.markdown("#### 🗺️ Geographic")
            
            avg_shipping_distance_km = st.number_input(
                "Shipping Distance (km)",
                min_value=0,
                value=80 if order_num == 1 else (600 if order_num == 2 else 1500),
                step=50,
                key=f"distance_{order_num}"
            )
            
            is_cross_state = st.selectbox(
                "Cross-State?",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=0 if order_num == 1 else 1,
                key=f"cross_state_{order_num}"
            )
            
            st.markdown("#### 📅 Timing")
            
            estimated_days = st.number_input(
                "Estimated Delivery Days",
                min_value=1,
                max_value=60,
                value=12 if order_num == 1 else (10 if order_num == 2 else 5),
                key=f"est_days_{order_num}"
            )
            
            order_weekday = st.selectbox(
                "Order Day",
                options=[0, 1, 2, 3, 4, 5, 6],
                format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x],
                index=2 if order_num == 1 else (5 if order_num == 2 else 6),
                key=f"weekday_{order_num}"
            )
            
            order_month = st.selectbox(
                "Order Month",
                options=list(range(1, 13)),
                format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1],
                index=4 if order_num == 1 else (5 if order_num == 2 else 11),
                key=f"month_{order_num}"
            )
            
            submitted = st.form_submit_button(
                f"✅ Set Order {order_num}",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                # Store order data
                order_data = {
                    'num_items': num_items,
                    'num_sellers': num_sellers,
                    'num_products': num_items,
                    'total_order_value': total_order_value,
                    'avg_item_price': total_order_value / num_items,
                    'max_item_price': total_order_value / num_items,
                    'total_shipping_cost': total_shipping_cost,
                    'avg_shipping_cost': total_shipping_cost / num_items,
                    'total_weight_g': total_weight_g,
                    'avg_weight_g': total_weight_g / num_items,
                    'max_weight_g': total_weight_g / num_items,
                    'avg_length_cm': avg_length_cm,
                    'avg_height_cm': avg_height_cm,
                    'avg_width_cm': avg_width_cm,
                    'avg_shipping_distance_km': avg_shipping_distance_km,
                    'max_shipping_distance_km': avg_shipping_distance_km,
                    'is_cross_state': is_cross_state,
                    'order_weekday': order_weekday,
                    'order_month': order_month,
                    'order_hour': 14,
                    'is_weekend_order': 1 if order_weekday >= 5 else 0,
                    'is_holiday_season': 1 if order_month in [11, 12] else 0,
                    'estimated_days': estimated_days
                }
                
                st.session_state.orders[order_num] = order_data
                st.success(f"✅ Order {order_num} saved!")

# ============================================================================
# Comparison Button
# ============================================================================

st.markdown("---")

if st.button("🔄 Compare Orders", type="primary", use_container_width=True):
    
    # Check if all orders are set
    if len(st.session_state.orders) < num_orders:
        st.error(f"⚠️ Please set all {num_orders} orders before comparing!")
    else:
        with st.spinner("Comparing orders..."):
            
            # Calculate features and predictions for each order
            results = {}
            
            for order_num in range(1, num_orders + 1):
                order_data = st.session_state.orders[order_num]
                features_df = calculate_features(order_data)
                prediction = predict_single(model, features_df)
                
                results[order_num] = {
                    'data': order_data,
                    'features': features_df,
                    'prediction': prediction
                }
            
            # ================================================================
            # Display Comparison Results
            # ================================================================
            
            st.markdown("---")
            st.markdown("## 📊 Comparison Results")
            
            # Risk Gauges Side-by-Side
            gauge_cols = st.columns(num_orders)
            
            for idx, (order_num, result) in enumerate(results.items()):
                with gauge_cols[idx]:
                    st.markdown(f"### 🛒 Order {order_num}")
                    if result['prediction']:
                        fig = create_risk_gauge(
                            result['prediction']['risk_score'],
                            result['prediction']['risk_level']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Metrics
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Prediction", result['prediction']['prediction_label'])
                        with col_b:
                            st.metric("Risk Level", result['prediction']['risk_level'])
            
            st.markdown("---")
            
            # ================================================================
            # Key Metrics Comparison Table
            # ================================================================
            
            st.markdown("## 📋 Key Metrics Comparison")
            
            comparison_data = {
                'Metric': [
                    'Risk Score',
                    'Prediction',
                    'Risk Level',
                    'Number of Items',
                    'Number of Sellers',
                    'Order Value ($)',
                    'Shipping Cost ($)',
                    'Weight (g)',
                    'Distance (km)',
                    'Cross-State',
                    'Estimated Days',
                    'Weekend Order'
                ]
            }
            
            for order_num, result in results.items():
                data = result['data']
                pred = result['prediction']
                
                comparison_data[f'Order {order_num}'] = [
                    f"{pred['risk_score']}/100",
                    pred['prediction_label'],
                    pred['risk_level'],
                    data['num_items'],
                    data['num_sellers'],
                    f"${data['total_order_value']:.2f}",
                    f"${data['total_shipping_cost']:.2f}",
                    f"{data['total_weight_g']}g",
                    f"{data['avg_shipping_distance_km']}km",
                    'Yes' if data['is_cross_state'] == 1 else 'No',
                    f"{data['estimated_days']} days",
                    'Yes' if data['is_weekend_order'] == 1 else 'No'
                ]
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, height=500)
            
            st.markdown("---")
            
            # ================================================================
            # Feature Comparison Chart
            # ================================================================
            
            st.markdown("## 📊 Feature Value Comparison")
            
            # Select top features to compare
            feature_names = list(results[1]['features'].columns)[:10]  # Top 10
            
            fig = go.Figure()
            
            colors = ['#3498DB', '#E74C3C', '#2ECC71']
            
            for idx, (order_num, result) in enumerate(results.items()):
                feature_values = [result['features'][f].values[0] for f in feature_names]
                
                fig.add_trace(go.Bar(
                    name=f'Order {order_num}',
                    x=feature_names,
                    y=feature_values,
                    marker_color=colors[idx],
                    text=[f"{v:.1f}" for v in feature_values],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Feature Values Comparison (Top 10 Features)",
                xaxis_title="Features",
                yaxis_title="Value",
                barmode='group',
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig.update_xaxes(tickangle=-45)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ================================================================
            # Risk Comparison Radar Chart
            # ================================================================
            
            st.markdown("## 🎯 Risk Factor Comparison (Radar Chart)")
            
            # Select key risk dimensions
            risk_dimensions = {
                'Complexity': 'num_items',
                'Value': 'total_order_value',
                'Weight': 'total_weight_g',
                'Distance': 'avg_shipping_distance_km',
                'Timeline': 'estimated_days'
            }
            
            fig = go.Figure()
            
            for idx, (order_num, result) in enumerate(results.items()):
                features = result['features']
                
                # Normalize values to 0-100 scale for comparison
                values = []
                for dim, feature in risk_dimensions.items():
                    raw_value = features[feature].values[0]
                    # Simple normalization (could be improved with actual min/max from training data)
                    if feature == 'num_items':
                        normalized = min(raw_value * 10, 100)
                    elif feature == 'total_order_value':
                        normalized = min(raw_value / 5, 100)
                    elif feature == 'total_weight_g':
                        normalized = min(raw_value / 100, 100)
                    elif feature == 'avg_shipping_distance_km':
                        normalized = min(raw_value / 20, 100)
                    elif feature == 'estimated_days':
                        normalized = 100 - min(raw_value * 5, 100)  # Inverse (less time = more risk)
                    else:
                        normalized = raw_value
                    
                    values.append(normalized)
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=list(risk_dimensions.keys()),
                    fill='toself',
                    name=f'Order {order_num}',
                    line=dict(color=colors[idx])
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title="Risk Dimensions Comparison (Normalized 0-100)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ================================================================
            # Key Differences Analysis
            # ================================================================
            
            # ================================================================
# Key Differences Analysis
# ================================================================

st.markdown("## 🔍 Key Differences Analysis")

if num_orders == 2:
    order1 = results[1]
    order2 = results[2]
    
    risk_diff = order2['prediction']['risk_score'] - order1['prediction']['risk_score']
    
    # Identify key differences
    differences = []
    
    # Check complexity
    if order2['data']['num_items'] > order1['data']['num_items']:
        diff = order2['data']['num_items'] - order1['data']['num_items']
        differences.append(f"✓ Order 2 has {diff} more items (increased complexity)")
    elif order1['data']['num_items'] > order2['data']['num_items']:
        diff = order1['data']['num_items'] - order2['data']['num_items']
        differences.append(f"✓ Order 1 has {diff} more items (increased complexity)")
    
    # Check distance
    if order2['data']['avg_shipping_distance_km'] > order1['data']['avg_shipping_distance_km']:
        diff = order2['data']['avg_shipping_distance_km'] - order1['data']['avg_shipping_distance_km']
        differences.append(f"✓ Order 2 has {diff:.0f}km longer shipping distance")
    elif order1['data']['avg_shipping_distance_km'] > order2['data']['avg_shipping_distance_km']:
        diff = order1['data']['avg_shipping_distance_km'] - order2['data']['avg_shipping_distance_km']
        differences.append(f"✓ Order 1 has {diff:.0f}km longer shipping distance")
    
    # Check timeline
    if order2['data']['estimated_days'] < order1['data']['estimated_days']:
        diff = order1['data']['estimated_days'] - order2['data']['estimated_days']
        differences.append(f"✓ Order 2 has {diff} fewer days (rush order)")
    elif order1['data']['estimated_days'] < order2['data']['estimated_days']:
        diff = order2['data']['estimated_days'] - order1['data']['estimated_days']
        differences.append(f"✓ Order 1 has {diff} fewer days (rush order)")
    
    # Check weekend
    if order2['data']['is_weekend_order'] > order1['data']['is_weekend_order']:
        differences.append("✓ Order 2 is placed on weekend (higher risk)")
    elif order1['data']['is_weekend_order'] > order2['data']['is_weekend_order']:
        differences.append("✓ Order 1 is placed on weekend (higher risk)")
    
    # Check holiday season
    if order2['data']['is_holiday_season'] > order1['data']['is_holiday_season']:
        differences.append("✓ Order 2 is during holiday season (higher risk)")
    elif order1['data']['is_holiday_season'] > order2['data']['is_holiday_season']:
        differences.append("✓ Order 1 is during holiday season (higher risk)")
    
    # Check cross-state
    if order2['data']['is_cross_state'] > order1['data']['is_cross_state']:
        differences.append("✓ Order 2 requires cross-state shipping")
    elif order1['data']['is_cross_state'] > order2['data']['is_cross_state']:
        differences.append("✓ Order 1 requires cross-state shipping")
    
    # Check sellers
    if order2['data']['num_sellers'] > order1['data']['num_sellers']:
        diff = order2['data']['num_sellers'] - order1['data']['num_sellers']
        differences.append(f"✓ Order 2 has {diff} more sellers (coordination complexity)")
    elif order1['data']['num_sellers'] > order2['data']['num_sellers']:
        diff = order1['data']['num_sellers'] - order2['data']['num_sellers']
        differences.append(f"✓ Order 1 has {diff} more sellers (coordination complexity)")
    
    # Check weight
    if order2['data']['total_weight_g'] > order1['data']['total_weight_g'] * 1.5:
        differences.append(f"✓ Order 2 is significantly heavier ({order2['data']['total_weight_g']}g vs {order1['data']['total_weight_g']}g)")
    elif order1['data']['total_weight_g'] > order2['data']['total_weight_g'] * 1.5:
        differences.append(f"✓ Order 1 is significantly heavier ({order1['data']['total_weight_g']}g vs {order2['data']['total_weight_g']}g)")
    
    # Display analysis
    if abs(risk_diff) < 10:
        st.info(f"""
        **Similar Risk Levels** (Difference: {abs(risk_diff):.0f} points)
        
        Both orders have comparable late delivery risk despite some different characteristics.
        
        **Notable Differences:**
        {chr(10).join(['- ' + d for d in differences]) if differences else '- Orders are very similar across all key dimensions'}
        """)
    
    elif risk_diff > 0:
        st.error(f"""
        **⚠️ Order 2 is {abs(risk_diff):.0f} points HIGHER RISK than Order 1**
        
        **Risk Scores:**
        - Order 1: {order1['prediction']['risk_score']}/100 ({order1['prediction']['risk_level']})
        - Order 2: {order2['prediction']['risk_score']}/100 ({order2['prediction']['risk_level']})
        
        **Main drivers of increased risk in Order 2:**
        {chr(10).join(['- ' + d for d in differences]) if differences else '- Multiple small factors combine to increase risk'}
        
        **Recommendation:** Order 2 requires closer monitoring and potential intervention.
        """)
    
    else:
        st.error(f"""
        **⚠️ Order 1 is {abs(risk_diff):.0f} points HIGHER RISK than Order 2**
        
        **Risk Scores:**
        - Order 1: {order1['prediction']['risk_score']}/100 ({order1['prediction']['risk_level']})
        - Order 2: {order2['prediction']['risk_score']}/100 ({order2['prediction']['risk_level']})
        
        **Main drivers of increased risk in Order 1:**
        {chr(10).join(['- ' + d for d in differences]) if differences else '- Multiple small factors combine to increase risk'}
        
        **Recommendation:** Order 1 requires closer monitoring and potential intervention.
        """)
    
    # Side-by-side metrics for easy comparison
    st.markdown("---")
    st.markdown("### 📊 Quick Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🛒 Order 1")
        st.metric("Items", order1['data']['num_items'])
        st.metric("Sellers", order1['data']['num_sellers'])
        st.metric("Value", f"${order1['data']['total_order_value']:.2f}")
        st.metric("Distance", f"{order1['data']['avg_shipping_distance_km']}km")
        st.metric("Timeline", f"{order1['data']['estimated_days']} days")
        st.metric("**RISK SCORE**", f"{order1['prediction']['risk_score']}/100", 
                 delta=f"{order1['prediction']['risk_level']}")
    
    with col2:
        st.markdown("#### 🛒 Order 2")
        st.metric("Items", order2['data']['num_items'])
        st.metric("Sellers", order2['data']['num_sellers'])
        st.metric("Value", f"${order2['data']['total_order_value']:.2f}")
        st.metric("Distance", f"{order2['data']['avg_shipping_distance_km']}km")
        st.metric("Timeline", f"{order2['data']['estimated_days']} days")
        st.metric("**RISK SCORE**", f"{order2['prediction']['risk_score']}/100",
                 delta=f"{risk_diff:+.0f} vs Order 1",
                 delta_color="inverse")

elif num_orders == 3:
    # For 3-way comparison
    st.info("""
    **3-Way Comparison Analysis**
    
    Review the charts above to understand differences across all three orders.
    Key metrics table and radar chart provide comprehensive comparison.
    """)
    
    # Show risk ranking
    risk_ranking = sorted(
        [(num, res['prediction']['risk_score']) for num, res in results.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    st.success(f"""
    **Risk Ranking (Highest to Lowest):**
    
    1. 🥇 Order {risk_ranking[0][0]}: {risk_ranking[0][1]}/100
    2. 🥈 Order {risk_ranking[1][0]}: {risk_ranking[1][1]}/100
    3. 🥉 Order {risk_ranking[2][0]}: {risk_ranking[2][1]}/100
    """)
# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## 🔄 Compare Orders")
    st.info("""
    **Purpose:**
    - Understand risk drivers
    - Compare "what-if" scenarios
    - Training & education
    - Explain model decisions
    """)
    
    st.markdown("---")
    
    st.markdown("## 💡 Tips")
    st.success("""
    **Best Practices:**
    - Start with defaults
    - Change one variable at a time
    - Compare similar orders
    - Analyze differences
    """)
    
    st.markdown("---")
    
    st.markdown("## 🎯 Example Uses")
    st.markdown("""
    1. **Same order, different times**
       - Weekday vs Weekend
    2. **Distance impact**
       - Local vs Cross-country
    3. **Rush vs Standard**
       - 5 days vs 15 days
    """)
