# pages/4_📈_Time_Trends.py
# Purpose: Show time trends of predicted late risk.
# UX:
#   - Option A: Upload a SCORED CSV from the Batch page.
#   - Option B: Generate a quick demo and score it on the fly.
#   - Then show late-rate over time + average probability trend.
#
# Notes:
#   - We assume scored CSV contains 'order_purchase_timestamp' and 'score' (0..1) and 'meets_threshold'/'risk_band'.
#   - If unscored RAW is uploaded, we offer to score it.

from __future__ import annotations
import io
from typing import Any, Dict
import numpy as np
import pandas as pd
import streamlit as st

from utils.model_loader import predict_batch, load_metadata

st.set_page_config(page_title="Time Trends", page_icon="📈", layout="wide")
st.title("📈 Time Trends")
st.caption("Track predicted late risk over time. Use your scored CSV or generate a demo.")

st.markdown("---")

meta = load_metadata()
thr = float(meta.get("optimal_threshold", 0.5))

# ---------------------- Helpers ---------------------- #

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "order_purchase_timestamp" in df.columns:
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce").clip(0, 1)
    if "meets_threshold" in df.columns:
        # normalize to bool
        df["meets_threshold"] = df["meets_threshold"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df

def _make_demo(n=400, seed=42) -> pd.DataFrame:
    np.random.seed(int(seed))
    # Build a tiny synthetic timeline
    base = pd.Timestamp("2017-01-01")
    dates = [base + pd.Timedelta(days=int(d)) for d in np.random.randint(0, 540, size=n)]
    # Create minimal RAW input; score to get probabilities
    raw = pd.DataFrame({
        "order_purchase_timestamp": [d.strftime("%Y-%m-%dT%H:%M:%S") for d in dates],
        "estimated_delivery_date": [(d + pd.Timedelta(days=int(np.random.randint(3, 20)))).strftime("%Y-%m-%dT%H:%M:%S") for d in dates],
        "sum_price": np.round(np.random.gamma(2.0, 60.0, size=n), 2),
        "sum_freight": np.round(np.random.gamma(1.5, 12.0, size=n), 2),
        "n_items": np.clip(np.random.poisson(2, size=n) + 1, 1, 6),
        "n_sellers": np.clip(np.random.poisson(1, size=n) + 1, 1, 3),
        "payment_type": np.random.choice(["credit_card","boleto","debit_card","voucher","not_defined"], p=[0.65,0.2,0.08,0.05,0.02], size=n),
        "max_installments": np.clip((np.random.exponential(1.2, size=n)).astype(int) + 1, 1, 12),
        "mode_category": np.random.choice(["bed_bath_table","health_beauty","sports_leisure","computers_accessories","furniture_decor","watches_gifts","housewares","auto","toys","stationery"], size=n),
        "customer_city": np.random.choice(["sao paulo","rio de janeiro","belo horizonte","curitiba","campinas","porto alegre"], size=n),
        "customer_state": np.random.choice(["SP","RJ","MG","PR","RS","BA","ES","SC","GO","DF"], size=n),
    })
    scored = predict_batch(raw)
    return scored

def _aggregate(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    df = df.dropna(subset=["order_purchase_timestamp"]).copy()
    df = df.set_index("order_purchase_timestamp").sort_index()
    grp = df.resample(freq)
    out = pd.DataFrame({
        "avg_prob": grp["score"].mean(),
        "rate_meets": grp["meets_threshold"].mean() if "meets_threshold" in df.columns else np.nan,
    }).reset_index()
    out["late_rate_estimate"] = out["rate_meets"]  # interpretation depends on threshold setting
    return out

# ---------------------- UI ---------------------- #

left, right = st.columns([2, 1])
with left:
    uploaded = st.file_uploader("Upload a SCORED CSV (from Batch page). If RAW, we will score it.", type=["csv"])
with right:
    gen_demo = st.button("Or Generate & Score a Demo")

df = None
if uploaded:
    try:
        df_up = pd.read_csv(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
    # If not scored (no 'score' col), try to score
    if "score" not in df_up.columns:
        with st.spinner("Scoring your RAW upload…"):
            df = predict_batch(df_up)
    else:
        df = df_up
elif gen_demo:
    with st.spinner("Generating & scoring demo…"):
        df = _make_demo(n=400, seed=42)

if df is None:
    st.info("Upload a scored CSV or click the demo button.")
    st.stop()

df = _clean(df)
st.success(f"Loaded {len(df):,} rows.")
st.dataframe(df.head(25))

# Frequency selector
freq = st.selectbox("Aggregate frequency", options=["D", "W", "M"], index=1, help="Daily, Weekly, or Monthly")
agg = _aggregate(df, freq=freq)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Average Predicted Probability")
    st.line_chart(data=agg.set_index("order_purchase_timestamp")["avg_prob"])
with c2:
    st.subheader("Share ≥ Threshold (late-risk rate proxy)")
    if "late_rate_estimate" in agg.columns:
        st.line_chart(data=agg.set_index("order_purchase_timestamp")["late_rate_estimate"])
    else:
        st.info("Threshold-based rate unavailable; missing 'meets_threshold' in data.")
