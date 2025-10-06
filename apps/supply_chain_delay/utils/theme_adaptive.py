"""
Theme Adaptive Utility
Automatically adjusts colors and styling based on user's theme preference
"""

import streamlit as st

def apply_adaptive_theme():
    """
    Apply CSS that adapts to both light and dark modes
    Uses CSS variables and media queries for automatic adaptation
    """
    
    st.markdown("""
    <style>
        /* Adaptive color variables */
        :root {
            --text-primary: #262730;
            --text-secondary: #555555;
            --bg-primary: #FFFFFF;
            --bg-secondary: #F0F2F6;
            --border-color: #E0E0E0;
        }
        
        /* Dark mode adjustments */
        @media (prefers-color-scheme: dark) {
            :root {
                --text-primary: #FAFAFA;
                --text-secondary: #B0B0B0;
                --bg-primary: #0E1117;
                --bg-secondary: #262730;
                --border-color: #3D3D3D;
            }
        }
        
        /* Ensure Streamlit respects theme */
        [data-testid="stAppViewContainer"] {
            background-color: var(--bg-primary);
        }
        
        [data-testid="stSidebar"] {
            background-color: var(--bg-secondary);
        }
        
        /* Text colors that adapt */
        .stMarkdown p, .stMarkdown li, .stMarkdown span {
            color: var(--text-primary) !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
        }
        
        /* Metric labels - adaptive */
        [data-testid="stMetricLabel"] {
            color: var(--text-secondary) !important;
        }
        
        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
        }
        
        /* Info/Warning/Error boxes - adaptive backgrounds */
        .stAlert {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        /* Form inputs - adaptive */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {
            color: var(--text-primary) !important;
            background-color: var(--bg-secondary) !important;
            border-color: var(--border-color) !important;
        }
        
        /* Tables - adaptive */
        [data-testid="stDataFrame"] {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Code blocks - adaptive */
        .stCodeBlock {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        /* Expander - adaptive */
        .streamlit-expanderHeader {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Tabs - adaptive */
        .stTabs [data-baseweb="tab-list"] {
            background-color: var(--bg-secondary) !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary) !important;
        }
        
        .stTabs [aria-selected="true"] {
            color: var(--text-primary) !important;
        }
        
        /* Buttons - keep original colors but ensure text is visible */
        .stButton > button {
            color: #FFFFFF !important;
        }
        
                /* Download buttons - always visible with blue background */
        div[data-testid="stDownloadButton"] button {
            background-color: #0068C9 !important;
            color: white !important;
            border: 2px solid #0068C9 !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
            margin: 0 !important;
            vertical-align: middle !important;
        }
        
        div[data-testid="stDownloadButton"] button:hover {
            background-color: #0056a3 !important;
            border-color: #0056a3 !important;
        }
        
        div[data-testid="stDownloadButton"] button * {
            color: white !important;
        }
        
        /* Align download button containers */
        div[data-testid="stDownloadButton"] {
            display: flex !important;
            align-items: center !important;
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)


def get_plotly_template():
    """
    Returns appropriate Plotly template based on Streamlit theme
    
    Returns:
    --------
    str : 'plotly_white' or 'plotly_dark'
    """
    
    # Try to detect theme (Streamlit doesn't expose this directly yet)
    # Default to a neutral template that works for both
    return 'plotly'  # Neutral template


def get_adaptive_colors():
    """
    Returns color scheme that works in both light and dark modes
    
    Returns:
    --------
    dict : Color mappings for various elements
    """
    
    return {
        # Risk level colors (vibrant, work in both modes)
        'low_risk': '#2ECC71',      # Green
        'medium_risk': '#F39C12',   # Orange  
        'high_risk': '#E74C3C',     # Red
        
        # Chart colors (high contrast)
        'primary': '#3498DB',       # Blue
        'secondary': '#9B59B6',     # Purple
        'accent': '#1ABC9C',        # Teal
        
        # Neutral colors
        'text_dark': '#2C3E50',     # Dark blue-gray
        'text_light': '#ECF0F1',    # Light gray
        
        # Background colors (transparent for adaptation)
        'bg_transparent': 'rgba(0,0,0,0)',
        'bg_light': 'rgba(255,255,255,0.05)',
        'bg_dark': 'rgba(0,0,0,0.05)'
    }


def configure_plotly_figure(fig, title=None):
    """
    Configure Plotly figure to be theme-adaptive
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The figure to configure
    title : str, optional
        Figure title
    
    Returns:
    --------
    fig : Modified figure
    """
    
    colors = get_adaptive_colors()
    
    fig.update_layout(
        # Transparent backgrounds (inherits from page)
        plot_bgcolor=colors['bg_transparent'],
        paper_bgcolor=colors['bg_transparent'],
        
        # Template for adaptive styling
        template='plotly',
        
        # Title styling
        title=dict(
            text=title,
            font=dict(size=16)
        ) if title else None,
        
        # Legend styling
        legend=dict(
            bgcolor=colors['bg_transparent'],
            bordercolor=colors['bg_transparent']
        ),
        
        # Hover styling
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.9)',
            font_color='black'
        )
    )
    
    return fig
