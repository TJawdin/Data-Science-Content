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
        /* ================================================================ */
        /* ADAPTIVE COLOR VARIABLES */
        /* ================================================================ */
        
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
        
        /* ================================================================ */
        /* LAYOUT - ADAPTIVE BACKGROUNDS */
        /* ================================================================ */
        
        [data-testid="stAppViewContainer"] {
            background-color: var(--bg-primary);
        }
        
        [data-testid="stSidebar"] {
            background-color: var(--bg-secondary);
        }
        
        [data-testid="stHeader"] {
            background-color: var(--bg-primary);
        }
        
        /* ================================================================ */
        /* TEXT - ADAPTIVE COLORS */
        /* ================================================================ */
        
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
        
        [data-testid="stMetricDelta"] {
            color: var(--text-secondary) !important;
        }
        
        /* Caption text */
        .stCaption {
            color: var(--text-secondary) !important;
        }
        
        /* ================================================================ */
        /* ALERT BOXES - ADAPTIVE */
        /* ================================================================ */
        
        .stAlert {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        /* Info boxes */
        div[data-baseweb="notification"] {
            background-color: var(--bg-secondary) !important;
            border-color: var(--border-color) !important;
        }
        
        /* ================================================================ */
        /* FORM INPUTS - ADAPTIVE */
        /* ================================================================ */
        
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select,
        .stTextArea textarea {
            color: var(--text-primary) !important;
            background-color: var(--bg-secondary) !important;
            border-color: var(--border-color) !important;
        }
        
        /* Input labels */
        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stTextArea label {
            color: var(--text-primary) !important;
        }
        
        /* ================================================================ */
        /* TABLES - ADAPTIVE */
        /* ================================================================ */
        
        [data-testid="stDataFrame"] {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }
        
        [data-testid="stTable"] {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }
        
        /* ================================================================ */
        /* CODE BLOCKS - ADAPTIVE */
        /* ================================================================ */
        
        .stCodeBlock {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        code {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        /* ================================================================ */
        /* EXPANDERS - ADAPTIVE */
        /* ================================================================ */
        
        .streamlit-expanderHeader {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        details[open] > summary {
            border-bottom-color: var(--border-color) !important;
        }
        
        /* ================================================================ */
        /* TABS - ADAPTIVE */
        /* ================================================================ */
        
        .stTabs [data-baseweb="tab-list"] {
            background-color: var(--bg-secondary) !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary) !important;
        }
        
        .stTabs [aria-selected="true"] {
            color: var(--text-primary) !important;
        }
        
        /* ================================================================ */
        /* BUTTONS - ALWAYS VISIBLE (HIGHEST PRIORITY) */
        /* ================================================================ */
        
        /* Primary buttons (type="primary") - Blue with white text */
        button[kind="primary"],
        button[data-testid="baseButton-primary"],
        .stButton > button[kind="primary"] {
            background-color: #0068C9 !important;
            color: white !important;
            border: 2px solid #0068C9 !important;
            font-weight: 600 !important;
        }
        
        button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover,
        .stButton > button[kind="primary"]:hover {
            background-color: #0056a3 !important;
            border-color: #0056a3 !important;
            color: white !important;
        }
        
        button[kind="primary"] p,
        button[data-testid="baseButton-primary"] p,
        .stButton > button[kind="primary"] p {
            color: white !important;
        }
        
        /* Secondary buttons (type="secondary") - Gray with dark text */
        button[kind="secondary"],
        button[data-testid="baseButton-secondary"],
        .stButton > button[kind="secondary"] {
            background-color: #F0F2F6 !important;
            color: #262730 !important;
            border: 2px solid #E0E0E0 !important;
            font-weight: 600 !important;
        }
        
        button[kind="secondary"]:hover,
        button[data-testid="baseButton-secondary"]:hover,
        .stButton > button[kind="secondary"]:hover {
            background-color: #E0E0E0 !important;
            border-color: #C0C0C0 !important;
            color: #262730 !important;
        }
        
        button[kind="secondary"] p,
        button[data-testid="baseButton-secondary"] p,
        .stButton > button[kind="secondary"] p {
            color: #262730 !important;
        }
        
        /* Regular buttons (no type specified) - Gray with dark text */
        .stButton > button {
            background-color: #F0F2F6 !important;
            color: #262730 !important;
            border: 2px solid #E0E0E0 !important;
            font-weight: 600 !important;
        }
        
        .stButton > button:hover {
            background-color: #E0E0E0 !important;
            border-color: #C0C0C0 !important;
            color: #262730 !important;
        }
        
        .stButton > button p,
        .stButton > button span,
        .stButton > button div {
            color: #262730 !important;
        }
        
        /* Download buttons - Always blue with white text */
        div[data-testid="stDownloadButton"] {
            display: flex !important;
            align-items: center !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        div[data-testid="stDownloadButton"] button {
            background-color: #0068C9 !important;
            color: white !important;
            border: 2px solid #0068C9 !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
            margin: 0 !important;
            min-height: 48px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        div[data-testid="stDownloadButton"] button:hover {
            background-color: #0056a3 !important;
            border-color: #0056a3 !important;
            color: white !important;
        }
        
        div[data-testid="stDownloadButton"] button *,
        div[data-testid="stDownloadButton"] button p,
        div[data-testid="stDownloadButton"] button span,
        div[data-testid="stDownloadButton"] button div {
            color: white !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Form submit buttons */
        .stForm button[type="submit"] {
            background-color: #0068C9 !important;
            color: white !important;
            border: 2px solid #0068C9 !important;
            font-weight: 600 !important;
        }
        
        .stForm button[type="submit"]:hover {
            background-color: #0056a3 !important;
            border-color: #0056a3 !important;
            color: white !important;
        }
        
        .stForm button[type="submit"] p {
            color: white !important;
        }
        
        /* Ensure all button text is visible */
        button * {
            font-weight: inherit !important;
        }
        
        /* ================================================================ */
        /* FILE UPLOADER - ADAPTIVE */
        /* ================================================================ */
        
        [data-testid="stFileUploader"] {
            background-color: var(--bg-secondary) !important;
        }
        
        [data-testid="stFileUploader"] label {
            color: var(--text-primary) !important;
        }
        
        /* ================================================================ */
        /* SIDEBAR SPECIFIC */
        /* ================================================================ */
        
        [data-testid="stSidebar"] .stMarkdown {
            color: var(--text-primary) !important;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--text-primary) !important;
        }
        
        /* ================================================================ */
        /* PROGRESS BAR - ADAPTIVE */
        /* ================================================================ */
        
        .stProgress > div > div > div {
            background-color: #0068C9 !important;
        }
        
        /* ================================================================ */
        /* SPINNER - ADAPTIVE */
        /* ================================================================ */
        
        .stSpinner > div {
            border-top-color: #0068C9 !important;
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
