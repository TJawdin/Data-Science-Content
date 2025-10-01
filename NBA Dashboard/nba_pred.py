
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from predict import predict_pts

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="NBA Player Dashboard",
    page_icon="🏀",
    layout="wide"
)

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess NBA data"""
    # Determine file path relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'cleaned_nba_data.csv')
    
    # Fallback paths for deployment
    if not os.path.exists(data_path):
        data_path = 'cleaned_nba_data.csv'
    if not os.path.exists(data_path):
        data_path = 'cleaned_nba_data.csv'
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"❌ Data file not found. Please ensure 'cleaned_nba_data.csv' is in the correct location.")
        st.stop()
    
    # Clean column names
    df = df.rename(columns={
        'FG%': 'FG_pct', 
        '3P%': '3P_pct', 
        'FT%': 'FT_pct', 
        'Data': 'Date'
    })
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['Date'])
    
    return df

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Check if dataframe is empty
if df.empty:
    st.error("❌ No data available to display.")
    st.stop()

# Sidebar for user input
st.sidebar.title("🏀 Player Selection")
player = st.sidebar.selectbox(
    'Select a Player:', 
    sorted(df['Player'].unique()),
    help="Choose a player to view their stats and make predictions"
)

# Filter data for the selected player
player_df = df[df['Player'] == player].copy()

# Check if player has data
if player_df.empty:
    st.warning(f"No data available for {player}")
    st.stop()

# Main title
st.title(f'🏀 NBA Player Performance Dashboard')
st.header(f'{player}')

# Add some metrics at the top
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Games Played", len(player_df))
with col2:
    st.metric("Avg Points", f"{player_df['PTS'].mean():.1f}")
with col3:
    st.metric("Avg Minutes", f"{player_df['MP'].mean():.1f}")
with col4:
    st.metric("Avg Assists", f"{player_df['AST'].mean():.1f}")

st.divider()

# --- Visualizations ---
st.subheader("📊 Game-by-Game Performance")

# Sort by date for proper line chart
player_df_sorted = player_df.sort_values('Date')

fig = px.line(
    player_df_sorted, 
    x='Date', 
    y='PTS', 
    title='Points Over Time',
    markers=True
)
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Points",
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Game Stats Table ---
st.subheader("📋 Game Stats Table")

# Select columns that exist in the dataframe
display_cols = ['Date', 'PTS', 'MP', 'FGA', '3PA', 'FTA', 'TRB', 'AST', 'TOV']
existing_cols = [col for col in display_cols if col in player_df.columns]

if existing_cols:
    st.dataframe(
        player_df[existing_cols].sort_values(by='Date', ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("Unable to display stats table - columns missing from dataset")

st.divider()

# --- Prediction Section ---
st.subheader("🔮 Predict Points for a Game")

st.markdown("Enter game statistics to predict how many points the player would score:")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        mp = st.number_input("Minutes Played", min_value=0.0, max_value=48.0, value=30.0, step=1.0)
        fga = st.number_input("Field Goal Attempts", min_value=0, value=15, step=1)
        fg_pct = st.slider("Field Goal %", min_value=0.0, max_value=1.0, value=0.45, step=0.01)
        three_pa = st.number_input("3-Point Attempts", min_value=0, value=5, step=1)
        three_p_pct = st.slider("3-Point %", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
    
    with col2:
        fta = st.number_input("Free Throw Attempts", min_value=0, value=6, step=1)
        ft_pct = st.slider("Free Throw %", min_value=0.0, max_value=1.0, value=0.80, step=0.01)
        trb = st.number_input("Total Rebounds", min_value=0, value=5, step=1)
        ast = st.number_input("Assists", min_value=0, value=5, step=1)
        tov = st.number_input("Turnovers", min_value=0, value=2, step=1)

    submit = st.form_submit_button("🎯 Predict Points", use_container_width=True)

    if submit:
        try:
            input_data = pd.DataFrame([{
                'MP': mp,
                'FGA': fga,
                'FG_pct': fg_pct,
                '3PA': three_pa,
                '3P_pct': three_p_pct,
                'FTA': fta,
                'FT_pct': ft_pct,
                'TRB': trb,
                'AST': ast,
                'TOV': tov
            }])
            
            prediction = predict_pts(input_data)
            st.success(f"🎯 Predicted Points: **{prediction:.2f}**")
            
            # Add comparison to player's average
            avg_pts = player_df['PTS'].mean()
            diff = prediction - avg_pts
            if diff > 0:
                st.info(f"📈 That's **{diff:.2f}** points above {player}'s average of {avg_pts:.2f}")
            else:
                st.info(f"📉 That's **{abs(diff):.2f}** points below {player}'s average of {avg_pts:.2f}")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please check that the predict.py file and model are properly configured.")

# Footer
st.divider()
st.caption("📊 NBA Player Performance Dashboard | Data updates may vary")