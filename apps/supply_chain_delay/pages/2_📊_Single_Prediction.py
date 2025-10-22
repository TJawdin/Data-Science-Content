# pages/2_📊_Single_Prediction.py
# Purpose: User-friendly single-shipment prediction with ~10 raw inputs.
#          We convert these into the full engineered feature vector via utils.feature_engineering.
#
# References:
# - Python (latest): https://docs.python.org/3/
# - Pandas (latest): https://pandas.pydata.org/docs/
#
# Notes:
# - We gather raw inputs: purchase timestamp, estimated delivery date, item totals, counts,
#   payment type, installments, location, and top category.
# - We rely on utils.model_loader.predict_single(), which uses calculate_features()
#   to transform raw → engineered (32 features), then runs model.predict_proba().
# - Thorough cleaning: coercions, trimming, safe defaults.

from __future__ import annotations  # enable postponed annotations for clarity
from datetime import datetime, time, date  # datetime utilities for clean UI inputs
from typing import Any, Dict               # typing helpers
from pathlib import Path                   # filesystem paths when needed

import pandas as pd                        # light cleaning / types
import streamlit as st                     # Streamlit UI

# Central app helpers (metadata + inference)
from utils.model_loader import load_metadata, predict_single  # load metadata and score one example


# ----------------------------- Small helpers ----------------------------- #

def _to_iso_dt(d: date, t: time) -> str:
    """Combine separate date and time widgets into ISO-like string."""
    # Return a string like "YYYY-MM-DDTHH:MM:SS" suitable for pandas.to_datetime
    return f"{d.isoformat()}T{t.strftime('%H:%M')}:00"  # :00 seconds for stability

def _clean_text(x: Any) -> str:
    """Normalize text fields (trim spaces, lower where appropriate for categories)."""
    # Return empty string for None; otherwise trimmed text
    if x is None:
        return ""
    return str(x).strip()

def _clean_state(x: Any) -> str:
    """Brazilian state codes are usually uppercase (e.g., 'SP', 'RJ')."""
    # Uppercase the state code for consistency
    return _clean_text(x).upper()


# ----------------------------- Load metadata ----------------------------- #

meta: Dict[str, Any] = load_metadata()                               # read artifacts/final_metadata.json
thr: float = float(meta.get("optimal_threshold", 0.5))               # operating threshold (0–1)
rb: Dict[str, Any] = meta.get("risk_bands", {})                      # risk bands dict
low_max: int = int(rb.get("low_max", 30))                            # low band cutoff in percent
med_max: int = int(rb.get("med_max", 67))                            # medium band cutoff in percent


# ----------------------------- Page header ----------------------------- #

st.title("📊 Single Prediction")                                      # main page title
st.caption("Provide a few order details; we’ll do the rest under the hood.")  # subtitle

# Quick model context tiles
c1, c2, c3, c4 = st.columns(4)                                       # metrics row
c1.metric("Threshold", f"{thr:.6f}", f"{thr*100:.2f}%")              # threshold shown in prob and percent
c2.metric("AUC-ROC", f"{float(meta.get('best_model_auc', 0.0)):.4f}")# display AUC with 4 decimals
c3.metric("Low Band (≤)", f"{low_max}%")                             # low band cutoff
c4.metric("Medium Band (≤)", f"{med_max}%")                          # medium band cutoff
st.markdown("---")                                                   # divider


# ----------------------------- Input form (≈10 raw fields) ----------------------------- #
# Rationale:
# 1) Purchase timestamp + estimated delivery date → time features + est_lead_days
# 2) sum_price + sum_freight (or total_payment) → monetary drivers
# 3) n_items, n_sellers → order complexity
# 4) payment_type + max_installments → payment signals
# 5) mode_category + customer_city + customer_state → categorical context

with st.form("single_raw_form", clear_on_submit=False):              # start a form to group inputs
    # --- Dates & times ---
    st.subheader("When")                                             # section label
    dcol1, dcol2, dcol3, dcol4 = st.columns([1.2, 1.0, 1.2, 1.0])    # four columns for date/time fields
    with dcol1:
        purch_date = st.date_input(                                  # purchase date widget
            "Purchase date", value=date(2017, 7, 1)                  # sensible default in Olist period
        )
    with dcol2:
        purch_time = st.time_input(                                  # purchase time widget
            "Purchase time", value=time(10, 45)                      # default morning time
        )
    with dcol3:
        est_deliv_date = st.date_input(                              # estimated delivery date widget
            "Estimated delivery date", value=date(2017, 7, 11)       # default ~10 days later
        )
    with dcol4:
        est_deliv_time = st.time_input(                              # optional time; mostly unused for lead-days calc
            "Estimated delivery time", value=time(9, 0)              # morning default
        )

    # --- Money & counts ---
    st.subheader("What")                                             # section label
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)                       # four columns for numeric fields
    with mcol1:
        sum_price = st.number_input(                                  # sum of item prices (without freight)
            "Total items price (R$)", value=120.00, step=1.0, format="%.2f"
        )
    with mcol2:
        sum_freight = st.number_input(                                # total freight charge
            "Total freight (R$)", value=25.00, step=1.0, format="%.2f"
        )
    with mcol3:
        n_items = st.number_input(                                    # number of items on the order
            "Number of items", min_value=1, value=2, step=1
        )
    with mcol4:
        n_sellers = st.number_input(                                  # number of distinct sellers
            "Number of sellers", min_value=1, value=1, step=1
        )

    # --- Payment ---
    st.subheader("Payment")                                          # section label
    pcol1, pcol2 = st.columns([1.5, 1])                              # two columns for payment fields
    with pcol1:
        payment_type = st.selectbox(                                  # payment type (maps to paytype_* one-hots)
            "Payment type",
            options=["credit_card", "boleto", "debit_card", "voucher", "not_defined"],
            index=0
        )
    with pcol2:
        max_installments = st.number_input(                           # maximum installments on the order
            "Max installments", min_value=1, value=1, step=1
        )

    # --- Context / Category / Location ---
    st.subheader("Context")                                          # section label
    ccol1, ccol2, ccol3 = st.columns([1.4, 1, 1])                    # three columns for category + location
    with ccol1:
        mode_category = st.selectbox(                                 # primary category (mode across items)
            "Main category",
            options=[
                "bed_bath_table", "health_beauty", "sports_leisure",
                "computers_accessories", "furniture_decor", "watches_gifts",
                "housewares", "auto", "toys", "stationery"
            ],
            index=0
        )
    with ccol2:
        customer_city = st.text_input(                                # customer city (free text; FE will standardize)
            "Customer city", value="sao paulo"
        )
    with ccol3:
        customer_state = st.selectbox(                                # state code (uppercased)
            "Customer state",
            options=["SP", "RJ", "MG", "PR", "RS", "BA", "ES", "SC", "GO", "DF"],
            index=0
        )

    # --- Submit ---
    submitted = st.form_submit_button("Predict Risk")                # submit button to trigger scoring


# ----------------------------- Inference flow ----------------------------- #

if submitted:                                                        # only run when user clicks submit
    # Assemble raw input dict expected by feature_engineering._engineer_row()
    raw: Dict[str, Any] = {}                                         # container for raw fields

    # Combine separate date + time into ISO strings for robust parsing (pandas.to_datetime)
    raw["order_purchase_timestamp"] = _to_iso_dt(purch_date, purch_time)          # e.g., "2017-07-01T10:45:00"
    raw["estimated_delivery_date"] = _to_iso_dt(est_deliv_date, est_deliv_time)   # e.g., "2017-07-11T09:00:00"

    # Monetary totals (model features sum_price, sum_freight, and total_payment are derived)
    raw["sum_price"] = float(sum_price)                                           # ensure numeric
    raw["sum_freight"] = float(sum_freight)                                       # ensure numeric
    raw["total_payment"] = float(sum_price) + float(sum_freight)                  # explicit to avoid ambiguity

    # Order counts
    raw["n_items"] = int(n_items)                                                 # count of items
    raw["n_sellers"] = int(n_sellers)                                             # count of sellers
    # We let feature_engineering default n_products ~ n_items when not provided

    # Payment info
    raw["payment_type"] = str(payment_type)                                       # single payment type string
    raw["payment_installments"] = [int(max_installments)]                         # list form (engineer uses max())

    # Category and location
    raw["mode_category"] = _clean_text(mode_category)                             # mode category (primary)
    raw["customer_city"] = _clean_text(customer_city)                             # trim and normalize text
    raw["customer_state"] = _clean_state(customer_state)                          # uppercase state code

    # Run model prediction; model_loader.predict_single() will:
    # 1) call calculate_features(raw) to create all 32 engineered features in correct order
    # 2) compute predict_proba and apply threshold + risk bands
    with st.spinner("Scoring…"):                                                  # UI spinner during compute
        result = predict_single(raw)                                              # run inference

    # Extract outputs
    score: float = float(result.get("score", 0.0))                                # probability (0–1)
    meets: bool = bool(result.get("meets_threshold", False))                      # threshold flag
    band: str = str(result.get("risk_band", "N/A"))                               # "Low"/"Medium"/"High"

    # Results summary tiles
    st.success("Prediction complete.")                                            # success banner
    r1, r2, r3 = st.columns(3)                                                    # three metric tiles
    r1.metric("Predicted Probability", f"{score:.6f}", f"{score*100:.2f}%")       # show both scales
    r2.metric("Meets Threshold", "Yes" if meets else "No")                         # boolean result
    r3.metric("Risk Band", band)                                                  # band label

    # Echo inputs for traceability (what we sent to the FE layer)
    with st.expander("Raw inputs sent to the model", expanded=False):             # collapsible panel
        st.json(raw)                                                              # show the raw dict
