# pages/5_🗺️_Geographic_Map.py
# Purpose: Show a simple geographic view of risk by city/state using folium.
# Approach:
#   - Upload a SCORED CSV (or generate demo).
#   - Aggregate by (customer_city, customer_state) and plot markers sized by count and colored by risk band share.
#   - We use a small built-in geocoding dictionary for common Brazilian cities used in demos.
#
# Dependencies: folium, streamlit-folium

from __future__ import annotations
import io
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from utils.model_loader import predict_batch

st.set_page_config(page_title="Geographic Map", page_icon="🗺️", layout="wide")
st.title("🗺️ Geographic Map")
st.caption("Plot risk patterns by city/state. Upload a scored CSV or generate a demo dataset.")

st.markdown("---")

# Minimal city lat/lon dictionary (extend as needed)
CITY_COORDS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("sao paulo", "SP"): (-23.5505, -46.6333),
    ("rio de janeiro", "RJ"): (-22.9068, -43.1729),
    ("belo horizonte", "MG"): (-19.9167, -43.9345),
    ("curitiba", "PR"): (-25.4284, -49.2733),
    ("campinas", "SP"): (-22.9099, -47.0626),
    ("porto alegre", "RS"): (-30.0346, -51.2177),
}

def _color_for_band(band: str) -> str:
    b = str(band).lower()
    if b == "low":
        return "#2ECC71"
    if b == "medium":
        return "#F39C12"
    return "#E74C3C"

def _demo(n=300, seed=123) -> pd.DataFrame:
    np.random.seed(int(seed))
    from datetime import timedelta
    base = pd.Timestamp("2017-06-01")
    # RAW demo
    raw = pd.DataFrame({
        "order_purchase_timestamp": [(base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%dT%H:%M:%S") for d in np.random.randint(0, 365, size=n)],
        "estimated_delivery_date": [(base + pd.Timedelta(days=int(d)+np.random.randint(3, 20))).strftime("%Y-%m-%dT%H:%M:%S") for d in np.random.randint(0, 365, size=n)],
        "sum_price": np.round(np.random.gamma(2.0, 60.0, size=n), 2),
        "sum_freight": np.round(np.random.gamma(1.5, 12.0, size=n), 2),
        "n_items": np.clip(np.random.poisson(2, size=n) + 1, 1, 6),
        "n_sellers": np.clip(np.random.poisson(1, size=n) + 1, 1, 3),
        "payment_type": np.random.choice(["credit_card","boleto","debit_card","voucher","not_defined"], p=[0.65,0.2,0.08,0.05,0.02], size=n),
        "max_installments": np.clip((np.random.exponential(1.2, size=n)).astype(int) + 1, 1, 12),
        "mode_category": np.random.choice(["bed_bath_table","health_beauty","sports_leisure","computers_accessories","furniture_decor","watches_gifts","housewares","auto","toys","stationery"], size=n),
        "customer_city": np.random.choice([k[0] for k in CITY_COORDS.keys()], size=n),
        "customer_state": np.array([CITY_COORDS.keys()])[:,0] if False else np.random.choice([k[1] for k in CITY_COORDS.keys()], size=n),
    })
    # The line above for states is intentionally weird-safe; we just sample states.
    # Score it
    return predict_batch(raw)

# --- Input: upload scored or generate demo ---
left, right = st.columns([2, 1])
with left:
    uploaded = st.file_uploader("Upload a SCORED CSV (from Batch page)", type=["csv"])
with right:
    gen = st.button("Generate & Score Demo")

df = None
if uploaded:
    try:
        df = pd.read_csv(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
elif gen:
    with st.spinner("Generating & scoring demo…"):
        df = _demo(n=300, seed=123)

if df is None:
    st.info("Upload a scored CSV or click the demo button.")
    st.stop()

# Basic cleaning
for c in ["customer_city", "customer_state", "risk_band"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()
if "customer_city" not in df.columns or "customer_state" not in df.columns:
    st.error("Missing required columns 'customer_city' and/or 'customer_state'.")
    st.stop()

# Aggregate stats by (city, state)
grp = df.groupby(["customer_city", "customer_state"], dropna=False).agg(
    n=("score", "size"),
    avg_prob=("score", "mean"),
    share_high=("risk_band", lambda s: (s.astype(str).str.lower() == "high").mean() if "risk_band" in df.columns else np.nan)
).reset_index()

st.success(f"Aggregated {len(grp)} city/state locations.")

# Build a Folium map
# Center on Brazil roughly
m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4, tiles="cartodbpositron")

for _, row in grp.iterrows():
    city = str(row["customer_city"]).lower()
    state = str(row["customer_state"]).upper()
    key = (city, state)
    latlon = CITY_COORDS.get(key)
    if not latlon:
        # skip unknown city/state pairs to avoid geocoding
        continue

    lat, lon = latlon
    size = max(6, min(30, int(row["n"] ** 0.5 * 6)))  # sqrt scaling
    band_color = _color_for_band("High" if row["share_high"] >= 0.5 else ("Medium" if row["share_high"] >= 0.2 else "Low"))

    popup = folium.Popup(
        html=f"<b>{city.title()}, {state}</b><br/>Rows: {int(row['n']):,}<br/>Avg prob: {row['avg_prob']:.3f}<br/>Share High: {row['share_high']:.2%}",
        max_width=300
    )
    folium.CircleMarker(
        location=[lat, lon],
        radius=size,
        color=band_color,
        fill=True,
        fill_opacity=0.6,
        fill_color=band_color,
        popup=popup
    ).add_to(m)

st_folium(m, width=1200, height=650)
