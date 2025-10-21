"""
Theme Adaptive Utility
Automatically adjusts colors and styling based on user's theme preference
"""

import streamlit as st


def apply_adaptive_theme():
    """
    Apply CSS that adapts to both light and dark modes.
    Adds utility classes used across the app: main-header, sub-header,
    metric-card, info-box, success-box, badge styles, etc.
    """
    st.markdown(
        """
<style>
/* ================================================================ */
/* ADAPTIVE COLOR VARIABLES                                         */
/* ================================================================ */

:root {
  --text-primary: #262730;
  --text-secondary: #555555;
  --bg-primary: #FFFFFF;
  --bg-secondary: #F0F2F6;
  --border-color: #E0E0E0;

  --blue: #0068C9;
  --blue-dark: #0056a3;

  --green: #2ECC71;
  --orange: #F39C12;
  --red: #E74C3C;
  --purple: #9B59B6;
  --teal: #1ABC9C;
  --indigo: #3498DB;

  --card-shadow: 0 2px 10px rgba(0,0,0,0.06);
}

@media (prefers-color-scheme: dark) {
  :root {
    --text-primary: #FAFAFA;
    --text-secondary: #B0B0B0;
    --bg-primary: #0E1117;
    --bg-secondary: #262730;
    --border-color: #3D3D3D;

    --card-shadow: 0 2px 10px rgba(0,0,0,0.35);
  }
}

/* ================================================================ */
/* LAYOUT - ADAPTIVE BACKGROUNDS                                    */
/* ================================================================ */
[data-testid="stAppViewContainer"] { background-color: var(--bg-primary); }
[data-testid="stSidebar"] { background-color: var(--bg-secondary); }
[data-testid="stHeader"] { background-color: var(--bg-primary); }

/* ================================================================ */
/* TEXT - ADAPTIVE COLORS                                           */
/* ================================================================ */
.stMarkdown p, .stMarkdown li, .stMarkdown span { color: var(--text-primary) !important; }
h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; }
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; }
[data-testid="stMetricDelta"] { color: var(--text-secondary) !important; }
.stCaption { color: var(--text-secondary) !important; }

/* ================================================================ */
/* HEADERS USED IN MAIN PAGE                                        */
/* ================================================================ */
.main-header {
  font-size: 2rem;
  font-weight: 800;
  color: var(--text-primary);
  margin: 0.25rem 0 0.25rem 0;
}
.sub-header {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 0.75rem;
}

/* ================================================================ */
/* CARDS / BOXES                                                     */
/* ================================================================ */
.metric-card,
.info-box,
.success-box {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 16px 18px;
  box-shadow: var(--card-shadow);
}

.metric-card h4,
.info-box h3, .info-box h4,
.success-box h3, .success-box h4 {
  margin: 0 0 8px 0;
  color: var(--text-primary);
  font-weight: 700;
}

.metric-card p,
.info-box p, .success-box p,
.info-box li, .success-box li {
  color: var(--text-primary);
}

/* Subtle list spacing inside cards */
.info-box ul, .success-box ul { margin: 0.25rem 0 0 1.15rem; }

/* ================================================================ */
/* BADGES / CHIPS (e.g., risk levels)                               */
/* ================================================================ */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.85rem;
  border: 1px solid var(--border-color);
}
.badge-low   { background: rgba(46, 204, 113, 0.15); color: var(--green); }
.badge-mid   { background: rgba(243, 156, 18, 0.18); color: var(--orange); }
.badge-high  { background: rgba(231, 76, 60, 0.18); color: var(--red); }

/* ================================================================ */
/* ALERT BOXES - ADAPTIVE                                           */
/* ================================================================ */
.stAlert { background-color: var(--bg-secondary) !important; border: 1px solid var(--border-color) !important; }
div[data-baseweb="notification"] { background-color: var(--bg-secondary) !important; border-color: var(--border-color) !important; }

/* ================================================================ */
/* FORM INPUTS - ADAPTIVE                                           */
/* ================================================================ */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > select,
.stTextArea textarea {
  color: var(--text-primary) !important;
  background-color: var(--bg-secondary) !important;
  border-color: var(--border-color) !important;
}
.stTextInput label, .stNumberInput label, .stSelectbox label, .stTextArea label {
  color: var(--text-primary) !important;
}

/* ================================================================ */
/* TABLES - ADAPTIVE                                                */
/* ================================================================ */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  background-color: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--border-color);
}
[data-testid="stDataFrame"] table, [data-testid="stTable"] table {
  border-collapse: collapse;
}
[data-testid="stDataFrame"] th, [data-testid="stTable"] th {
  background: var(--bg-secondary) !important;
  color: var(--text-primary) !important;
}

/* ================================================================ */
/* CODE BLOCKS - ADAPTIVE                                           */
/* ================================================================ */
.stCodeBlock { background-color: var(--bg-secondary) !important; border: 1px solid var(--border-color) !important; }
code { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }

/* ================================================================ */
/* EXPANDERS - ADAPTIVE                                             */
/* ================================================================ */
.streamlit-expanderHeader { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; }
details[open] > summary { border-bottom-color: var(--border-color) !important; }

/* ================================================================ */
/* TABS - ADAPTIVE                                                  */
/* ================================================================ */
.stTabs [data-baseweb="tab-list"] { background-color: var(--bg-secondary) !important; }
.stTabs [data-baseweb="tab"] { color: var(--text-secondary) !important; }
.stTabs [aria-selected="true"] { color: var(--text-primary) !important; }

/* ================================================================ */
/* BUTTONS - ALWAYS VISIBLE                                         */
/* ================================================================ */
button[kind="primary"],
button[data-testid="baseButton-primary"],
.stButton > button[kind="primary"] {
  background-color: var(--blue) !important;
  color: white !important;
  border: 2px solid var(--blue) !important;
  font-weight: 600 !important;
}
button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {
  background-color: var(--blue-dark) !important;
  border-color: var(--blue-dark) !important;
  color: white !important;
}
button[kind="primary"] p,
button[data-testid="baseButton-primary"] p,
.stButton > button[kind="primary"] p { color: white !important; }

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
button * { font-weight: inherit !important; }

/* Download button */
div[data-testid="stDownloadButton"] { display: flex !important; align-items: center !important; }
div[data-testid="stDownloadButton"] button {
  background-color: var(--blue) !important;
  color: white !important;
  border: 2px solid var(--blue) !important;
  font-weight: 600 !important;
  padding: 0.5rem 1rem !important;
  min-height: 48px !important;
}
div[data-testid="stDownloadButton"] button:hover {
  background-color: var(--blue-dark) !important;
  border-color: var(--blue-dark) !important;
}

/* ================================================================ */
/* FILE UPLOADER - ADAPTIVE                                         */
/* ================================================================ */
[data-testid="stFileUploader"] { background-color: var(--bg-secondary) !important; }
[data-testid="stFileUploader"] label { color: var(--text-primary) !important; }

/* ================================================================ */
/* SIDEBAR                                                          */
/* ================================================================ */
[data-testid="stSidebar"] .stMarkdown { color: var(--text-primary) !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--text-primary) !important; }

/* ================================================================ */
/* PROGRESS / SPINNER                                               */
/* ================================================================ */
.stProgress > div > div > div { background-color: var(--blue) !important; }
.stSpinner > div { border-top-color: var(--blue) !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


def get_plotly_template():
    """
    Returns a neutral Plotly template that behaves nicely in both themes.
    """
    return "plotly"  # neutral; works with our transparent backgrounds


def get_adaptive_colors():
    """
    Color scheme used across charts and UI elements (theme-safe).
    """
    return {
        # Risk level colors
        "low_risk": "#2ECC71",
        "medium_risk": "#F39C12",
        "high_risk": "#E74C3C",

        # Chart palette
        "primary": "#3498DB",
        "secondary": "#9B59B6",
        "accent": "#1ABC9C",

        # Neutral text
        "text_dark": "#2C3E50",
        "text_light": "#ECF0F1",

        # Transparent BGs
        "bg_transparent": "rgba(0,0,0,0)",
        "bg_light": "rgba(255,255,255,0.05)",
        "bg_dark": "rgba(0,0,0,0.05)",
    }


def configure_plotly_figure(fig, title=None):
    """
    Configure Plotly figure to be theme-adaptive (transparent backgrounds,
    neutral template, tidy legend/hover).
    """
    colors = get_adaptive_colors()
    fig.update_layout(
        plot_bgcolor=colors["bg_transparent"],
        paper_bgcolor=colors["bg_transparent"],
        template="plotly",
        title=dict(text=title, font=dict(size=16)) if title else None,
        legend=dict(bgcolor=colors["bg_transparent"], bordercolor=colors["bg_transparent"]),
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.9)", font_color="black"),
    )
    return fig
