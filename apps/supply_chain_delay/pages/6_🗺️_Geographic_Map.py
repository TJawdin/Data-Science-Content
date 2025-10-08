"""
Geographic Map Visualization Page
Interactive map showing shipping routes, warehouse locations, and risk zones
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="Geographic Map",
    page_icon="🗺️",
    layout="wide"
)

# ============================================================================
# Header
# ============================================================================

st.title("🗺️ Geographic Shipping Analysis")
st.markdown("""
Interactive map visualization of shipping routes, risk zones, and delivery patterns across Brazil.
Explore geographic factors affecting late deliveries!
""")

st.markdown("---")

# ============================================================================
# Brazilian Cities & Coordinates (Major hubs)
# ============================================================================

@st.cache_data
def get_brazil_locations():
    """Get major Brazilian cities with coordinates and risk data"""
    return pd.DataFrame([
        # Major distribution hubs
        {'city': 'São Paulo', 'state': 'SP', 'lat': -23.5505, 'lng': -46.6333, 
         'type': 'warehouse', 'late_rate': 4.9, 'orders': 45000},
        {'city': 'Rio de Janeiro', 'state': 'RJ', 'lat': -22.9068, 'lng': -43.1729, 
         'type': 'warehouse', 'late_rate': 6.1, 'orders': 28000},
        {'city': 'Belo Horizonte', 'state': 'MG', 'lat': -19.9167, 'lng': -43.9345, 
         'type': 'warehouse', 'late_rate': 6.5, 'orders': 15000},
        {'city': 'Curitiba', 'state': 'PR', 'lat': -25.4284, 'lng': -49.2733, 
         'type': 'warehouse', 'late_rate': 5.8, 'orders': 12000},
        {'city': 'Porto Alegre', 'state': 'RS', 'lat': -30.0346, 'lng': -51.2177, 
         'type': 'warehouse', 'late_rate': 5.5, 'orders': 10000},
        
        # High-risk remote areas
        {'city': 'Manaus', 'state': 'AM', 'lat': -3.1190, 'lng': -60.0217, 
         'type': 'customer', 'late_rate': 11.3, 'orders': 5000},
        {'city': 'Belém', 'state': 'PA', 'lat': -1.4558, 'lng': -48.5039, 
         'type': 'customer', 'late_rate': 10.8, 'orders': 4500},
        {'city': 'Recife', 'state': 'PE', 'lat': -8.0476, 'lng': -34.8770, 
         'type': 'customer', 'late_rate': 8.3, 'orders': 8000},
        {'city': 'Fortaleza', 'state': 'CE', 'lat': -3.7319, 'lng': -38.5267, 
         'type': 'customer', 'late_rate': 8.1, 'orders': 7500},
        {'city': 'Salvador', 'state': 'BA', 'lat': -12.9714, 'lng': -38.5014, 
         'type': 'customer', 'late_rate': 7.8, 'orders': 9000},
        {'city': 'Brasília', 'state': 'DF', 'lat': -15.8267, 'lng': -47.9218, 
         'type': 'customer', 'late_rate': 5.2, 'orders': 11000},
        {'city': 'Goiânia', 'state': 'GO', 'lat': -16.6869, 'lng': -49.2648, 
         'type': 'customer', 'late_rate': 6.8, 'orders': 6500},
        
        # Additional customer locations
        {'city': 'Florianópolis', 'state': 'SC', 'lat': -27.5954, 'lng': -48.5480, 
         'type': 'customer', 'late_rate': 5.3, 'orders': 5500},
        {'city': 'Natal', 'state': 'RN', 'lat': -5.7945, 'lng': -35.2110, 
         'type': 'customer', 'late_rate': 8.9, 'orders': 4000},
        {'city': 'Campo Grande', 'state': 'MS', 'lat': -20.4697, 'lng': -54.6201, 
         'type': 'customer', 'late_rate': 7.2, 'orders': 4500},
    ])

locations = get_brazil_locations()

# ============================================================================
# Map Visualization Options
# ============================================================================

st.markdown("## 🎛️ Map Controls")

col1, col2, col3 = st.columns(3)

with col1:
    map_view = st.selectbox(
        "Map View",
        ["Risk Heatmap", "Shipping Routes", "Warehouse Coverage"],
        help="Choose what to visualize on the map"
    )

with col2:
    show_markers = st.checkbox("Show City Markers", value=True)

with col3:
    show_labels = st.checkbox("Show Risk Labels", value=True)

st.markdown("---")

# ============================================================================
# Create Interactive Folium Map
# ============================================================================

def create_brazil_map(view_type, show_markers, show_labels):
    """Create Folium map with different visualizations"""
    
    # Center on Brazil
    m = folium.Map(
        location=[-15.7801, -47.9292],  # Center of Brazil
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # ========== RISK HEATMAP VIEW ==========
    if view_type == "Risk Heatmap":
        
        # Add circle markers for each city
        for _, row in locations.iterrows():
            # Color based on risk level
            if row['late_rate'] < 6:
                color = 'green'
                icon_color = 'white'
                risk_level = 'LOW'
            elif row['late_rate'] < 9:
                color = 'orange'
                icon_color = 'white'
                risk_level = 'MEDIUM'
            else:
                color = 'red'
                icon_color = 'white'
                risk_level = 'HIGH'
            
            # Circle size based on order volume
            radius = (row['orders'] / 1000) * 2
            
            if show_markers:
                # Add circle marker
                folium.CircleMarker(
                    location=[row['lat'], row['lng']],
                    radius=radius,
                    popup=f"""
                    <b>{row['city']}, {row['state']}</b><br>
                    Late Rate: {row['late_rate']:.1f}%<br>
                    Risk Level: {risk_level}<br>
                    Orders: {row['orders']:,}<br>
                    Type: {row['type'].title()}
                    """,
                    tooltip=f"{row['city']}: {row['late_rate']:.1f}% late rate",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6,
                    weight=2
                ).add_to(m)
                
                # Add icon marker
                icon = 'warehouse' if row['type'] == 'warehouse' else 'home'
                folium.Marker(
                    location=[row['lat'], row['lng']],
                    icon=folium.Icon(color=color, icon=icon, prefix='fa'),
                    popup=f"<b>{row['city']}</b><br>{row['late_rate']:.1f}% late rate"
                ).add_to(m)
            
            # Add risk labels
            if show_labels:
                folium.Marker(
                    location=[row['lat'], row['lng']],
                    icon=folium.DivIcon(html=f"""
                        <div style="font-size: 10px; color: {color}; font-weight: bold; 
                                    text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, 
                                    -1px 1px 0 #fff, 1px 1px 0 #fff;">
                            {row['city']}<br>{row['late_rate']:.1f}%
                        </div>
                    """)
                ).add_to(m)
    
    # ========== SHIPPING ROUTES VIEW ==========
    elif view_type == "Shipping Routes":
        
        # Get warehouses
        warehouses = locations[locations['type'] == 'warehouse']
        customers = locations[locations['type'] == 'customer']
        
        # Add warehouse markers (green)
        for _, wh in warehouses.iterrows():
            folium.Marker(
                location=[wh['lat'], wh['lng']],
                icon=folium.Icon(color='green', icon='warehouse', prefix='fa'),
                popup=f"<b>Warehouse: {wh['city']}</b><br>Late Rate: {wh['late_rate']:.1f}%",
                tooltip=f"Warehouse: {wh['city']}"
            ).add_to(m)
        
        # Add customer markers (blue)
        for _, cust in customers.iterrows():
            folium.Marker(
                location=[cust['lat'], cust['lng']],
                icon=folium.Icon(color='blue', icon='home', prefix='fa'),
                popup=f"<b>Customer: {cust['city']}</b><br>Late Rate: {cust['late_rate']:.1f}%",
                tooltip=f"Customer: {cust['city']}"
            ).add_to(m)
        
        # Draw shipping routes (from each warehouse to customers)
        for _, wh in warehouses.iterrows():
            for _, cust in customers.iterrows():
                # Calculate distance-based risk
                distance = np.sqrt((wh['lat'] - cust['lat'])**2 + (wh['lng'] - cust['lng'])**2)
                
                # Color based on combined risk
                combined_risk = (wh['late_rate'] + cust['late_rate']) / 2
                if combined_risk < 6:
                    color = 'green'
                    weight = 1
                elif combined_risk < 9:
                    color = 'orange'
                    weight = 2
                else:
                    color = 'red'
                    weight = 3
                
                # Draw route
                folium.PolyLine(
                    locations=[[wh['lat'], wh['lng']], [cust['lat'], cust['lng']]],
                    color=color,
                    weight=weight,
                    opacity=0.4,
                    popup=f"{wh['city']} → {cust['city']}<br>Combined Risk: {combined_risk:.1f}%"
                ).add_to(m)
    
    # ========== WAREHOUSE COVERAGE VIEW ==========
    elif view_type == "Warehouse Coverage":
        
        warehouses = locations[locations['type'] == 'warehouse']
        
        for _, wh in warehouses.iterrows():
            # Add warehouse marker
            folium.Marker(
                location=[wh['lat'], wh['lng']],
                icon=folium.Icon(color='green', icon='warehouse', prefix='fa'),
                popup=f"<b>{wh['city']} Distribution Center</b><br>Late Rate: {wh['late_rate']:.1f}%<br>Orders: {wh['orders']:,}",
                tooltip=f"DC: {wh['city']}"
            ).add_to(m)
            
            # Add coverage circles (500km, 1000km, 1500km)
            for radius, color, opacity in [(500, 'green', 0.1), (1000, 'orange', 0.05), (1500, 'red', 0.02)]:
                folium.Circle(
                    location=[wh['lat'], wh['lng']],
                    radius=radius * 1000,  # Convert km to meters
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=opacity,
                    weight=1,
                    popup=f"{wh['city']} - {radius}km radius"
                ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="margin: 0; font-weight: bold;">Risk Levels</p>
    <p style="margin: 5px 0;"><span style="color: green;">●</span> Low Risk</p>
    <p style="margin: 5px 0;"><span style="color: orange;">●</span> Medium Risk</p>
    <p style="margin: 5px 0;"><span style="color: red;">●</span> High Risk</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# Create and display map
st.markdown(f"## 🗺️ {map_view}")

with st.spinner("Loading interactive map..."):
    folium_map = create_brazil_map(map_view, show_markers, show_labels)
    st_folium(folium_map, width=1400, height=600)

st.markdown("---")

# ============================================================================
# Statistics & Analysis
# ============================================================================

st.markdown("## 📊 Geographic Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_risk = locations['late_rate'].mean()
    st.metric("Average Late Rate", f"{avg_risk:.1f}%")

with col2:
    high_risk_cities = len(locations[locations['late_rate'] > 9])
    st.metric("High-Risk Cities", high_risk_cities)

with col3:
    total_orders = locations['orders'].sum()
    st.metric("Total Orders Mapped", f"{total_orders:,}")

with col4:
    warehouses_count = len(locations[locations['type'] == 'warehouse'])
    st.metric("Distribution Centers", warehouses_count)

st.markdown("---")

# ============================================================================
# Distance vs Risk Analysis
# ============================================================================

st.markdown("## 📏 Distance vs Late Delivery Risk")

# Calculate distances from São Paulo (main hub)
sp_lat, sp_lng = -23.5505, -46.6333

locations['distance_from_sp_km'] = locations.apply(
    lambda row: np.sqrt((row['lat'] - sp_lat)**2 + (row['lng'] - sp_lng)**2) * 111,  # Approx km
    axis=1
)

# Scatter plot
fig = px.scatter(
    locations,
    x='distance_from_sp_km',
    y='late_rate',
    size='orders',
    color='type',
    hover_name='city',
    hover_data={'state': True, 'orders': ':,', 'distance_from_sp_km': ':.0f', 'late_rate': ':.1f'},
    labels={
        'distance_from_sp_km': 'Distance from São Paulo (km)',
        'late_rate': 'Late Delivery Rate (%)',
        'orders': 'Order Volume',
        'type': 'Location Type'
    },
    title='Distance vs Late Delivery Risk (from São Paulo hub)',
    color_discrete_map={'warehouse': '#2ECC71', 'customer': '#3498DB'}
)

fig.add_hline(y=locations['late_rate'].mean(), line_dash="dash", 
              annotation_text=f"Average: {locations['late_rate'].mean():.1f}%",
              annotation_position="right")

fig.update_layout(
    height=500,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

st.plotly_chart(fig, use_container_width=True)

# Correlation analysis
correlation = locations['distance_from_sp_km'].corr(locations['late_rate'])
st.info(f"""
**Correlation Analysis:**
- **Distance vs Late Rate Correlation**: {correlation:.3f}
- **Interpretation**: {'Strong positive correlation - longer distances significantly increase late risk' if correlation > 0.7 else 'Moderate correlation - distance is a factor but not the only driver' if correlation > 0.4 else 'Weak correlation - other factors may be more important'}
""")

# ============================================================================
# Top Risk Cities Table
# ============================================================================

st.markdown("---")
st.markdown("## 🎯 Top Risk Locations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔴 Highest Risk Cities")
    high_risk = locations.nlargest(5, 'late_rate')[['city', 'state', 'late_rate', 'orders', 'distance_from_sp_km']]
    high_risk.columns = ['City', 'State', 'Late Rate (%)', 'Orders', 'Distance (km)']
    st.dataframe(
        high_risk.style.background_gradient(cmap='Reds', subset=['Late Rate (%)']),
        use_container_width=True
    )

with col2:
    st.markdown("### 🟢 Best Performing Cities")
    low_risk = locations.nsmallest(5, 'late_rate')[['city', 'state', 'late_rate', 'orders', 'distance_from_sp_km']]
    low_risk.columns = ['City', 'State', 'Late Rate (%)', 'Orders', 'Distance (km)']
    st.dataframe(
        low_risk.style.background_gradient(cmap='Greens_r', subset=['Late Rate (%)']),
        use_container_width=True
    )

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## 🗺️ Map Features")
    st.info("""
    **Interactive Elements:**
    - Click markers for details
    - Hover for quick info
    - Zoom/pan to explore
    - Switch views for insights
    """)
    
    st.markdown("---")
    
    st.markdown("## 💡 Key Insights")
    st.success(f"""
    **Geographic Patterns:**
    - {len(locations[locations['type'] == 'warehouse'])} major warehouses
    - {len(locations[locations['late_rate'] > 9])} high-risk zones
    - {correlation:.0%} distance correlation
    """)
    
    st.markdown("---")
    
    st.markdown("## 📦 Legend")
    st.markdown("""
    **Markers:**
    - 🟢 Green: Low Risk
    - 🟠 Orange: Medium
    - 🔴 Red: High Risk
    
    **Icons:**
    - 🏭 Warehouse/DC
    - 🏠 Customer Location
    """)
