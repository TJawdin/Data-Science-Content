# pages/2_📊_Single_Prediction.py
# Purpose: User-friendly single prediction (~10 raw inputs) → FE (32 features) → model → PDF export.
# Notes:
# - Every line is commented for clarity and to avoid indentation mistakes.
# - Uses utils.feature_engineering.calculate_features under the hood via model_loader.predict_single.
# - Adds a "Download PDF" button using utils.pdf_generator.generate_single_report.

from __future__ import annotations  # postpone evaluation of annotations for type hints
from datetime import date, time     # import date/time widgets for Streamlit inputs
from typing import Any, Dict        # typing helpers for dictionaries

import pandas as pd                 # DataFrame handling for engineered features echo
import streamlit as st              # Streamlit UI elements

# App utilities: metadata + prediction functions
from utils.model_loader import load_metadata, predict_single  # load model metadata and single prediction
# Feature engineering: to echo the engineered feature vector in the PDF report
from utils.feature_engineering import calculate_features      # transform raw → engineered (32 features)
# PDF generator: to export a single-order PDF summary
from utils.pdf_generator import generate_single_report        # build a PDF in memory for download


# ----------------------------- Helpers ----------------------------- #

def _to_iso_dt(d: date, t: time) -> str:
    """Combine date and time into an ISO-like string."""
    return f"{d.isoformat()}T{t.strftime('%H:%M')}:00"  # produce YYYY-MM-DDTHH:MM:SS

def _clean_text(x: Any) -> str:
    """Trim whitespace and coerce to string."""
    return "" if x is None else str(x).strip()  # ensure a non-None trimmed string

def _clean_state(x: Any) -> str:
    """Uppercase the Brazilian state abbreviation."""
    return _clean_text(x).upper()  # uppercase for consistency


# ----------------------------- Load metadata ----------------------------- #

meta: Dict[str, Any] = load_metadata()                 # read thresholds, bands, metrics from artifacts
thr: float = float(meta.get("optimal_threshold", 0.5)) # operating threshold (0..1)
rb: Dict[str, Any] = meta.get("risk_bands", {})        # risk bands dict {"low_max": int, "med_max": int}
low_max: int = int(rb.get("low_max", 30))              # low band upper bound in percent
med_max: int = int(rb.get("med_max", 67))              # medium band upper bound in percent


# ----------------------------- Page header ----------------------------- #

st.set_page_config(page_title="Single Prediction — Supply Chain Delay", page_icon="📊", layout="wide")  # page config
st.title("📊 Single Prediction")                                  # main page title
st.caption("Provide a few order details; we’ll compute the full feature vector and predict late-risk.")  # subtitle

# Quick model context tiles
c1, c2, c3, c4 = st.columns(4)                                    # create four metric tiles
c1.metric("Threshold", f"{thr:.6f}", f"{thr*100:.2f}%")           # show both probability and percent
c2.metric("AUC-ROC", f"{float(meta.get('best_model_auc', 0.0)):.4f}")  # show AUC to four decimals
c3.metric("Low Band (≤)", f"{low_max}%")                          # show low band cutoff
c4.metric("Medium Band (≤)", f"{med_max}%")                       # show medium band cutoff
st.markdown("---")                                                # visual divider


# ----------------------------- Input form (~10 inputs) ----------------------------- #

with st.form("single_raw_form", clear_on_submit=False):           # start a form to group inputs
    # --- When section (dates/times) ---
    st.subheader("When")                                          # section label
    d1, d2, d3, d4 = st.columns([1.2, 1.0, 1.2, 1.0])             # four input columns
    with d1:
        purch_date = st.date_input("Purchase date", value=date(2017, 7, 1))  # default Olist-era date
    with d2:
        purch_time = st.time_input("Purchase time", value=time(10, 45))      # default morning time
    with d3:
        est_date = st.date_input("Estimated delivery date", value=date(2017, 7, 11))  # default ~10 days later
    with d4:
        est_time = st.time_input("Estimated delivery time", value=time(9, 0))         # optional, informational

    # --- Money & counts section ---
    st.subheader("What")                                          # section label
    m1, m2, m3, m4 = st.columns(4)                                # four input columns
    with m1:
        sum_price = st.number_input("Total items price (R$)", value=120.00, step=1.0, format="%.2f")   # price sum
    with m2:
        sum_freight = st.number_input("Total freight (R$)", value=25.00, step=1.0, format="%.2f")      # freight sum
    with m3:
        n_items = st.number_input("Number of items", min_value=1, value=2, step=1)                     # item count
    with m4:
        n_sellers = st.number_input("Number of sellers", min_value=1, value=1, step=1)                 # seller count

    # --- Payment section ---
    st.subheader("Payment")                                       # section label
    p1, p2 = st.columns([1.5, 1.0])                               # two input columns
    with p1:
        payment_type = st.selectbox(
            "Payment type",
            options=["credit_card", "boleto", "debit_card", "voucher", "not_defined"],  # known types
            index=0,                                                                    # default to credit_card
        )
    with p2:
        max_installments = st.number_input("Max installments", min_value=1, value=1, step=1)  # installments

    # --- Context section ---
    st.subheader("Context")                                       # section label
    c1, c2, c3 = st.columns([1.4, 1.2, 0.8])                      # three input columns
    with c1:
        mode_category = st.selectbox(
            "Main category",
            options=[
                "bed_bath_table", "health_beauty", "sports_leisure",
                "computers_accessories", "furniture_decor", "watches_gifts",
                "housewares", "auto", "toys", "stationery"
            ],
            index=0,                                              # default category
        )
    with c2:
        customer_city = st.text_input("Customer city", value="sao paulo")    # city input
    with c3:
        customer_state = st.selectbox("Customer state", options=["SP", "RJ", "MG", "PR", "RS", "BA", "ES", "SC", "GO", "DF"], index=0)  # state input

    # --- Submit button ---
    submitted = st.form_submit_button("Predict Risk")             # submit to trigger inference


# ----------------------------- Inference & PDF ----------------------------- #

if submitted:                                                     # only run when submitted
    # Build a raw input dictionary consistent with our FE layer
    raw: Dict[str, Any] = {}                                      # initialize raw dict
    raw["order_purchase_timestamp"] = _to_iso_dt(purch_date, purch_time)  # combine date/time for purchase
    raw["estimated_delivery_date"] = _to_iso_dt(est_date, est_time)       # combine date/time for estimate
    raw["sum_price"] = float(sum_price)                           # numeric sum of prices
    raw["sum_freight"] = float(sum_freight)                       # numeric sum of freight
    raw["total_payment"] = float(sum_price) + float(sum_freight)  # explicit total
    raw["n_items"] = int(n_items)                                 # integer items
    raw["n_sellers"] = int(n_sellers)                             # integer sellers
    raw["payment_type"] = str(payment_type)                       # payment type string
    raw["payment_installments"] = [int(max_installments)]         # list form for FE (uses max())
    raw["mode_category"] = _clean_text(mode_category)             # clean category
    raw["customer_city"] = _clean_text(customer_city)             # clean city
    raw["customer_state"] = _clean_state(customer_state)          # uppercase state

    # Run prediction (predict_single calls FE internally for the model input)
    with st.spinner("Scoring…"):                                   # show spinner while computing
        result = predict_single(raw)                               # run single prediction

    # Extract predicted probability and band info
    score: float = float(result.get("score", result.get("probability", 0.0)))  # tolerate alternate key name
    meets: bool = bool(result.get("meets_threshold", False))        # boolean threshold flag
    band: str = str(result.get("risk_band", "N/A"))                 # Low/Medium/High band

    # Results summary tiles
    st.success("Prediction complete.")                              # success banner
    r1, r2, r3 = st.columns(3)                                      # three metric tiles
    r1.metric("Predicted Probability", f"{score:.6f}", f"{score*100:.2f}%")  # probability as number and percent
    r2.metric("Meets Threshold", "Yes" if meets else "No")          # meets threshold
    r3.metric("Risk Band", band)                                    # band label

    # Build engineered features (1 row) to include in PDF's Top Factors fallback
    engineered = calculate_features(pd.DataFrame([raw]))            # transform raw → engineered

    # PDF generation: assemble and offer download
    pdf_bytes = generate_single_report(                             # build the PDF in memory
        order_raw=raw,                                              # raw inputs dictionary
        prediction=result,                                          # model result dictionary
        engineered_features=engineered,                             # 1-row engineered DataFrame
        shap_contributions=None,                                    # optional dict if using SHAP later
        friendly_feature_names=None,                                # optional mapping for business-friendly names
    )
    st.download_button(                                             # render a download button
        "⬇️ Download PDF Report",                                   # button text
        data=pdf_bytes,                                             # PDF bytes payload
        file_name="risk_report.pdf",                                # filename for download
        mime="application/pdf",                                     # correct MIME type
    )

    # Optional: show raw payload sent to FE for traceability
    with st.expander("Raw inputs sent to the model", expanded=False):  # collapsible block
        st.json(raw)                                                # pretty-print raw dictionary
