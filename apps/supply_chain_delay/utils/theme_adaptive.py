"""
Theme and Formatting Utilities
Consistent styling across the application
"""

import streamlit as st


def get_risk_color(risk_level):
    """
    Get color code for risk level
    
    Args:
        risk_level: 'Low', 'Medium', or 'High'
    
    Returns:
        str: Hex color code
    """
    colors = {
        "Low": "#28a745",      # Green
        "Medium": "#ffc107",   # Yellow/Orange
        "High": "#dc3545"      # Red
    }
    return colors.get(risk_level, "#6c757d")


def get_risk_icon(risk_level):
    """
    Get emoji icon for risk level
    
    Args:
        risk_level: 'Low', 'Medium', or 'High'
    
    Returns:
        str: Emoji icon
    """
    icons = {
        "Low": "✅",
        "Medium": "⚠️",
        "High": "🚨"
    }
    return icons.get(risk_level, "❓")


def format_probability(probability):
    """
    Format probability as percentage string
    
    Args:
        probability: Float probability (0-1)
    
    Returns:
        str: Formatted percentage
    """
    return f"{probability * 100:.1f}%"


def format_currency(amount):
    """
    Format amount as Brazilian Real currency
    
    Args:
        amount: Numeric amount
    
    Returns:
        str: Formatted currency
    """
    return f"R$ {amount:,.2f}"


def create_metric_card(label, value, delta=None, help_text=None):
    """
    Create a styled metric card
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
        help_text: Optional help text
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(label=label, value=value, delta=delta, help=help_text)


def display_risk_badge(risk_level, probability):
    """
    Display a styled risk badge
    
    Args:
        risk_level: 'Low', 'Medium', or 'High'
        probability: Delay probability (0-1)
    """
    color = get_risk_color(risk_level)
    icon = get_risk_icon(risk_level)
    prob_pct = format_probability(probability)
    
    st.markdown(
        f"""
        <div style="
            background-color: {color};
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        ">
            {icon} {risk_level} Risk - {prob_pct}
        </div>
        """,
        unsafe_allow_html=True
    )


def display_info_box(title, content, box_type="info"):
    """
    Display a styled information box
    
    Args:
        title: Box title
        content: Box content
        box_type: 'info', 'success', 'warning', or 'error'
    """
    colors = {
        "info": "#0dcaf0",
        "success": "#28a745",
        "warning": "#ffc107",
        "error": "#dc3545"
    }
    
    bg_color = colors.get(box_type, colors["info"])
    
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color}20;
            border-left: 5px solid {bg_color};
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        ">
            <h4 style="margin: 0 0 10px 0; color: {bg_color};">{title}</h4>
            <p style="margin: 0;">{content}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def apply_custom_css():
    """
    Apply custom CSS styling to the app
    """
    st.markdown(
        """
        <style>
        /* Main container */
        .main {
            padding: 2rem;
        }
        
        /* Headers */
        h1 {
            color: #1f1f1f;
            font-weight: 700;
        }
        
        h2, h3 {
            color: #2c3e50;
            font-weight: 600;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #FF6B6B;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #ff5252;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
        }
        
        /* Cards */
        .element-container {
            border-radius: 5px;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        
        /* Success/Error messages */
        .stSuccess, .stError, .stWarning, .stInfo {
            border-radius: 5px;
            padding: 1rem;
        }
        
        /* DataFrame */
        .dataframe {
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def show_page_header(title, description, icon="📊"):
    """
    Display a consistent page header
    
    Args:
        title: Page title
        description: Page description
        icon: Page icon emoji
    """
    st.markdown(f"# {icon} {title}")
    st.markdown(f"*{description}*")
    st.markdown("---")


def create_two_column_layout():
    """
    Create a standard two-column layout
    
    Returns:
        tuple: (left_column, right_column)
    """
    return st.columns(2)


def create_three_column_layout():
    """
    Create a standard three-column layout
    
    Returns:
        tuple: (col1, col2, col3)
    """
    return st.columns(3)
