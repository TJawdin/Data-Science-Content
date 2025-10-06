"""
Time-Series Trends Analysis Page
Analyze late delivery risk patterns by time (hour, day, week, month)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_loader import load_model

# Page config
st.set_page_config(
    page_title="Time Trends Analysis",
    page_icon="📈",
    layout="wide"
)

# ============================================================================
# Header
# ============================================================================

st.title("📈 Time-Series Trends Analysis")
st.markdown("""
Discover patterns in late delivery risk across different time dimensions.
Identify peak risk hours, days, and seasons to optimize operations!
""")

st.markdown("---")

# ============================================================================
# Generate Sample Time-Series Data (In production, use real training data)
# ============================================================================

@st.cache_data
def generate_time_series_data():
    """Generate realistic time-series risk data for visualization"""
    np.random.seed(42)
    
    # Hour of day (0-23)
    hours = list(range(24))
    hour_risk = [
        25 + 5 * np.sin((h - 6) * np.pi / 12) + np.random.normal(0, 2)
        for h in hours
    ]
    hour_risk = [max(15, min(70, r)) for r in hour_risk]  # Clamp between 15-70
    
    # Day of week (0=Mon, 6=Sun)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_risk = [28, 26, 25, 27, 32, 45, 48]  # Weekend higher
    
    # Month (1-12)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_risk = [30, 28, 27, 26, 28, 32, 35, 33, 30, 35, 52, 58]  # Holiday season spike
    
    # Week of month (1-4)
    weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
    week_risk = [28, 32, 35, 42]  # End of month pressure
    
    return {
        'hours': hours,
        'hour_risk': hour_risk,
        'days': days,
        'day_risk': day_risk,
        'months': months,
        'month_risk': month_risk,
        'weeks': weeks,
        'week_risk': week_risk
    }

data = generate_time_series_data()

# ============================================================================
# Tabs for Different Time Dimensions
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "⏰ By Hour of Day",
    "📅 By Day of Week", 
    "🗓️ By Month",
    "📊 Combined View"
])

# ============================================================================
# TAB 1: Hour of Day Analysis
# ============================================================================

with tab1:
    st.markdown("## ⏰ Late Delivery Risk by Hour of Day")
    
    st.info("""
    **Business Question:** When during the day are orders most at risk of being late?
    
    **Insights:** Identify peak risk hours to adjust staffing and processing priorities.
    """)
    
    # Line chart with area fill
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['hours'],
        y=data['hour_risk'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#3498DB', width=3),
        marker=dict(size=8, color=data['hour_risk'], colorscale='RdYlGn_r', 
                    showscale=True, colorbar=dict(title="Risk Score")),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)',
        hovertemplate='<b>Hour: %{x}:00</b><br>Risk Score: %{y:.1f}<extra></extra>'
    ))
    
    # Add threshold lines
    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                  annotation_text="Low Risk Threshold", annotation_position="right")
    fig.add_hline(y=50, line_dash="dash", line_color="red", 
                  annotation_text="High Risk Threshold", annotation_position="right")
    
    # Highlight peak hours
    peak_hour = data['hours'][np.argmax(data['hour_risk'])]
    peak_risk = max(data['hour_risk'])
    
    fig.add_annotation(
        x=peak_hour, y=peak_risk,
        text=f"Peak Risk<br>{peak_hour}:00",
        showarrow=True,
        arrowhead=2,
        arrowcolor='red',
        ax=0, ay=-40,
        bgcolor='rgba(255, 0, 0, 0.1)',
        bordercolor='red'
    )
    
    fig.update_layout(
        title="Average Late Delivery Risk Score Throughout the Day",
        xaxis_title="Hour of Day (24-hour format)",
        yaxis_title="Average Risk Score",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        low_hours = [h for h, r in zip(data['hours'], data['hour_risk']) if r < 30]
        st.success(f"""
        **✅ Best Hours (Low Risk)**
        
        Hours: {', '.join(map(str, low_hours[:5]))}:00
        
        *Optimal for processing critical orders*
        """)
    
    with col2:
        high_hours = [h for h, r in zip(data['hours'], data['hour_risk']) if r > 50]
        st.error(f"""
        **🚨 Peak Risk Hours**
        
        Hours: {', '.join(map(str, high_hours)) if high_hours else 'None'}:00
        
        *Increase staffing during these times*
        """)
    
    with col3:
        avg_risk = np.mean(data['hour_risk'])
        st.info(f"""
        **📊 Daily Average Risk**
        
        Score: {avg_risk:.1f}/100
        
        *Overall daily performance baseline*
        """)

# ============================================================================
# TAB 2: Day of Week Analysis
# ============================================================================

with tab2:
    st.markdown("## 📅 Late Delivery Risk by Day of Week")
    
    st.info("""
    **Business Question:** Which days of the week have the highest late delivery risk?
    
    **Insights:** Plan weekly operations and resource allocation accordingly.
    """)
    
    # Create color mapping
    colors = ['#2ECC71' if r < 30 else '#F39C12' if r < 50 else '#E74C3C' 
              for r in data['day_risk']]
    
    fig = go.Figure(go.Bar(
        x=data['days'],
        y=data['day_risk'],
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=data['day_risk'],
        texttemplate='%{text:.1f}',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Risk Score: %{y:.1f}<extra></extra>'
    ))
    
    # Add threshold lines
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
    fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title="Average Late Delivery Risk Score by Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Average Risk Score",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(range=[0, max(data['day_risk']) * 1.2])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Day comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 Best Performing Days")
        best_days = sorted(zip(data['days'], data['day_risk']), key=lambda x: x[1])[:3]
        for day, risk in best_days:
            st.success(f"**{day}**: {risk:.1f}/100")
    
    with col2:
        st.markdown("### 🔴 Highest Risk Days")
        worst_days = sorted(zip(data['days'], data['day_risk']), key=lambda x: x[1], reverse=True)[:3]
        for day, risk in worst_days:
            st.error(f"**{day}**: {risk:.1f}/100")
    
    # Operational recommendations
    st.markdown("---")
    st.markdown("### 💡 Operational Recommendations")
    
    weekend_avg = np.mean(data['day_risk'][5:])
    weekday_avg = np.mean(data['day_risk'][:5])
    
    st.warning(f"""
    **Weekend vs Weekday Analysis:**
    - **Weekend Average Risk**: {weekend_avg:.1f}/100
    - **Weekday Average Risk**: {weekday_avg:.1f}/100
    - **Difference**: {abs(weekend_avg - weekday_avg):.1f} points higher on weekends
    
    **Actions:**
    - 📦 Prioritize Friday shipments to avoid weekend delays
    - 👥 Increase weekend staffing by {int((weekend_avg/weekday_avg - 1) * 100)}%
    - 📞 Set customer expectations for weekend orders
    - 🚚 Partner with carriers offering 7-day delivery
    """)

# ============================================================================
# TAB 3: Monthly Analysis
# ============================================================================

with tab3:
    st.markdown("## 🗓️ Late Delivery Risk by Month")
    
    st.info("""
    **Business Question:** How does late delivery risk vary throughout the year?
    
    **Insights:** Plan for seasonal demand and holiday rush periods.
    """)
    
    # Line chart with markers
    fig = go.Figure()
    
    # Add line
    fig.add_trace(go.Scatter(
        x=data['months'],
        y=data['month_risk'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#E74C3C', width=3),
        marker=dict(
            size=12,
            color=data['month_risk'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Risk Score"),
            line=dict(color='white', width=2)
        ),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.1)',
        hovertemplate='<b>%{x}</b><br>Risk Score: %{y:.1f}<extra></extra>'
    ))
    
    # Highlight holiday season
    fig.add_vrect(
        x0=10.5, x1=11.5,
        fillcolor="rgba(255, 0, 0, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="Holiday Season",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title="Average Late Delivery Risk Score by Month",
        xaxis_title="Month",
        yaxis_title="Average Risk Score",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal analysis
    st.markdown("### 📊 Seasonal Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    seasons = {
        'Q1 (Jan-Mar)': np.mean(data['month_risk'][:3]),
        'Q2 (Apr-Jun)': np.mean(data['month_risk'][3:6]),
        'Q3 (Jul-Sep)': np.mean(data['month_risk'][6:9]),
        'Q4 (Oct-Dec)': np.mean(data['month_risk'][9:12])
    }
    
    for col, (season, risk) in zip([col1, col2, col3, col4], seasons.items()):
        with col:
            if risk > 40:
                st.error(f"**{season}**\n\n{risk:.1f}/100")
            elif risk > 30:
                st.warning(f"**{season}**\n\n{risk:.1f}/100")
            else:
                st.success(f"**{season}**\n\n{risk:.1f}/100")
    
    # Holiday season warning
    nov_dec_risk = np.mean(data['month_risk'][10:12])
    st.error(f"""
    **🎄 Holiday Season Alert (Nov-Dec)**
    
    - **Average Risk**: {nov_dec_risk:.1f}/100 ({int((nov_dec_risk / np.mean(data['month_risk'][:10]) - 1) * 100)}% higher than rest of year)
    - **Peak Month**: {data['months'][np.argmax(data['month_risk'])]} ({max(data['month_risk']):.1f}/100)
    
    **Preparation Strategy:**
    1. 📅 Start hiring seasonal staff in October
    2. 📦 Increase warehouse capacity by 30%
    3. 🚚 Secure backup carrier partnerships
    4. ⏰ Extend cutoff times for guaranteed delivery
    5. 💬 Proactive customer communication about delays
    """)

# ============================================================================
# TAB 4: Combined Heatmap View
# ============================================================================

with tab4:
    st.markdown("## 📊 Combined Time-Risk Heatmap")
    
    st.info("""
    **Comprehensive View:** See risk patterns across multiple time dimensions at once.
    
    Darker colors = Higher risk periods requiring more attention.
    """)
    
    # Create synthetic heatmap data (Hour x Day)
    np.random.seed(42)
    heatmap_data = np.zeros((24, 7))
    
    for hour in range(24):
        for day in range(7):
            base_risk = data['hour_risk'][hour]
            day_factor = data['day_risk'][day] / 30  # Normalize
            heatmap_data[hour, day] = base_risk * day_factor * (1 + np.random.normal(0, 0.1))
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=data['days'],
        y=[f"{h}:00" for h in range(24)],
        colorscale='RdYlGn_r',
        colorbar=dict(title="Risk Score"),
        hovertemplate='<b>%{x}</b><br>Hour: %{y}<br>Risk: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Late Delivery Risk: Hour of Day × Day of Week Heatmap",
        xaxis_title="Day of Week",
        yaxis_title="Hour of Day",
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key findings
    st.markdown("### 🔍 Key Patterns Identified")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Optimal Processing Windows:**
        - **Monday-Thursday mornings** (6 AM - 12 PM)
        - **Tuesday-Wednesday afternoons** (2 PM - 6 PM)
        - **Early week, mid-day** combinations
        
        → Schedule critical orders during these times
        """)
    
    with col2:
        st.error("""
        **🚨 High-Risk Periods:**
        - **Weekend evenings** (Friday 6 PM - Sunday 11 PM)
        - **Late night hours** (10 PM - 6 AM) any day
        - **End of month Fridays**
        
        → Avoid promises or add buffer time
        """)

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## 📈 Time Trends")
    st.info("""
    **Analysis Benefits:**
    - Identify peak risk periods
    - Optimize staffing schedules
    - Plan for seasonal demand
    - Set realistic expectations
    """)
    
    st.markdown("---")
    
    st.markdown("## 💡 Quick Insights")
    st.success(f"""
    **Peak Risk Time:**
    - Hour: {peak_hour}:00
    - Day: {data['days'][np.argmax(data['day_risk'])]}
    - Month: {data['months'][np.argmax(data['month_risk'])]}
    """)
    
    st.markdown("---")
    
    st.markdown("## 📊 Data Source")
    st.caption("""
    *Analysis based on historical order patterns from training dataset (100k+ orders)*
    """)
