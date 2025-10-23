"""
Theme and styling utilities for the Streamlit app
"""

import streamlit as st


def apply_custom_css():
    """
    Apply custom CSS styling to the app
    """
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1f77b4;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 500;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #34495e;
        font-weight: 500;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #555;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Risk badges */
    .risk-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .risk-low {
        background-color: #d4edda;
        color: #155724;
    }
    
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Cards */
    .prediction-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #1f77b4;
        border-radius: 0.5rem;
        padding: 2rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 500;
        font-size: 1rem;
    }
    
    /* Success/Warning/Error messages */
    .element-container .stSuccess {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    .element-container .stWarning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    
    .element-container .stError {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main {
            padding: 0rem 0.5rem;
        }
        
        h1 {
            font-size: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def get_risk_color_scheme():
    """
    Get color scheme for risk categories
    
    Returns:
        dict: Risk category to color mapping
    """
    return {
        'Low': {
            'primary': '#00CC96',
            'light': '#d4edda',
            'dark': '#155724',
            'bg': '#E8F8F5'
        },
        'Medium': {
            'primary': '#FFA500',
            'light': '#fff3cd',
            'dark': '#856404',
            'bg': '#FFF4E6'
        },
        'High': {
            'primary': '#EF553B',
            'light': '#f8d7da',
            'dark': '#721c24',
            'bg': '#FADBD8'
        }
    }


def style_dataframe(df, risk_column='risk_category'):
    """
    Apply styling to dataframe based on risk categories
    
    Args:
        df: DataFrame to style
        risk_column: Column name containing risk categories
    
    Returns:
        Styled dataframe
    """
    def highlight_risk(row):
        if risk_column in row:
            risk = row[risk_column]
            colors = get_risk_color_scheme()
            
            if risk == 'Low':
                return ['background-color: ' + colors['Low']['light']] * len(row)
            elif risk == 'Medium':
                return ['background-color: ' + colors['Medium']['light']] * len(row)
            elif risk == 'High':
                return ['background-color: ' + colors['High']['light']] * len(row)
        
        return [''] * len(row)
    
    if risk_column in df.columns:
        return df.style.apply(highlight_risk, axis=1)
    else:
        return df


def create_risk_badge_html(risk_category, probability):
    """
    Create HTML for risk badge
    
    Args:
        risk_category: Risk level ('Low', 'Medium', 'High')
        probability: Probability percentage
    
    Returns:
        str: HTML string
    """
    colors = get_risk_color_scheme()
    color_scheme = colors.get(risk_category, colors['Medium'])
    
    html = f"""
    <div style="
        background-color: {color_scheme['light']};
        color: {color_scheme['dark']};
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {color_scheme['primary']};
        font-weight: 600;
        text-align: center;
        margin: 0.5rem 0;
    ">
        <div style="font-size: 0.9rem; opacity: 0.8;">Risk Level</div>
        <div style="font-size: 1.5rem; margin: 0.25rem 0;">{risk_category}</div>
        <div style="font-size: 1.2rem; color: {color_scheme['primary']};">{probability:.1f}%</div>
    </div>
    """
    return html


def create_feature_card_html(title, value, icon="📊"):
    """
    Create HTML for feature card
    
    Args:
        title: Card title
        value: Card value
        icon: Emoji icon
    
    Returns:
        str: HTML string
    """
    html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 0.9rem; opacity: 0.9;">{title}</div>
        <div style="font-size: 1.5rem; font-weight: 600; margin-top: 0.25rem;">{value}</div>
    </div>
    """
    return html


def display_info_banner(message, banner_type='info'):
    """
    Display styled information banner
    
    Args:
        message: Message to display
        banner_type: Type of banner ('info', 'success', 'warning', 'error')
    """
    colors = {
        'info': {'bg': '#e3f2fd', 'border': '#1976d2', 'icon': 'ℹ️'},
        'success': {'bg': '#e8f5e9', 'border': '#4caf50', 'icon': '✅'},
        'warning': {'bg': '#fff3e0', 'border': '#ff9800', 'icon': '⚠️'},
        'error': {'bg': '#ffebee', 'border': '#f44336', 'icon': '❌'}
    }
    
    style = colors.get(banner_type, colors['info'])
    
    st.markdown(f"""
    <div style="
        background-color: {style['bg']};
        border-left: 4px solid {style['border']};
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    ">
        <strong>{style['icon']} {message}</strong>
    </div>
    """, unsafe_allow_html=True)
