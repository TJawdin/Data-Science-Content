# pages/3_📦_Batch_Predictions.py
# Purpose: Batch scoring with RAW schema (same as Single page), demo/template generation,
#          per-row explanation (engineered echo + heuristic drivers), and Batch PDF export.
# Notes:
# - Every line is commented to avoid indentation mistakes.
# - predict_batch internally calls calculate_features (raw → engineered) then scores.
# - Section D shows engineered features and a heuristic ranking using model importances if available.

from __future__ import annotations                    # postpone evaluation of annotations for type hints
import io                                             # in-memory bytes for file uploads/downloads
from typing import Any, Dict, List                    # typing helpers
from pathlib import Path                              # filesystem-safe paths

import numpy as np                                    # numeric utilities for demo data
import pandas as pd                                   # DataFrame operations
import streamlit as st                                # Streamlit UI

# App utils: metadata, batch prediction, model loading
from utils.model_loader import load_metadata, predict_batch, load_model  # load meta/model and score batch
# Feature engineering (for Section D engineered echo)
from utils.feature_engineering import calculate_features                 # raw → engineered
# PDF: batch summary export
from utils.pdf_generator import generate_batch_summary_report            # build a batch PDF report


# ----------------------------- Constants ----------------------------- #

RAW_REQUIRED_COLUMNS: List[str] = [                  # required RAW columns for uploads/templates
    "order_purchase_timestamp",                      # ISO-like purchase datetime
    "estimated_delivery_date",                       # ISO-like estimated delivery datetime
    "sum_price",                                     # R$ items total
    "sum_freight",                                   # R$ freight total
    "n_items",                                       # integer item count
    "n_sellers",                                     # integer seller count
    "payment_type",                                  # string payment method
    "max_installments",                              # integer max installments
    "mode_category",                                 # dominant category
    "customer_city",                                 # city text
    "customer_state",                                # state code (e.g., SP)
]


# ----------------------------- Helpers ----------------------------- #

def _artifacts_dir() -> Path:
    """Return artifacts directory path."""
    return Path(__file__).resolve().parents[1] / "artifacts"  # pages/.. → artifacts folder

def _clean_df_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning for uploaded RAW CSVs."""
    df = df.copy()                                                # work on a copy
    df.columns = [str(c).strip() for c in df.columns]             # strip column name whitespace
    df = df.loc[:, ~df.columns.duplicated(keep="first")]          # drop duplicate headers
    for c in df.select_dtypes(include=["object"]).columns:        # trim text-like values
        df[c] = df[c].astype(str).str.strip()                     # strip spaces
    # Coerce likely numeric columns
    for c in ["sum_price", "sum_freight", "n_items", "n_sellers", "max_installments"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")         # numeric or NaN
    # Fill numeric NaNs with safe defaults
    for c in ["sum_price", "sum_freight"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)                             # monetary fields default to 0.0
    for c in ["n_items", "n_sellers", "max_installments"]:
        if c in df.columns:
            df[c] = df[c].fillna(1).astype(int)                   # counts default to 1
    # Uppercase state codes
    if "customer_state" in df.columns:
        df["customer_state"] = df["customer_state"].str.upper()   # normalize to uppercase
    return df                                                     # return cleaned DataFrame

def _demo_raw_row() -> Dict[str, Any]:
    """Generate one realistic RAW row for demo data."""
    # Purchase datetime in Olist-era years
    year = np.random.choice([2017, 2018])                         # choose year
    month = np.random.randint(1, 13)                              # month 1-12
    day = np.random.randint(1, 28)                                # safe day
    hour = np.random.randint(0, 24)                               # hour 0-23
    minute = np.random.randint(0, 60)                             # minute 0-59
    purch = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)  # purchase ts
    est = purch + pd.Timedelta(days=int(np.random.randint(3, 20)))                   # est delivery ts

    # Sample helper
    def _choice(seq, p=None):                                     # small wrapper for numpy choice
        return np.random.choice(seq, p=p)                         # pick an element

    # Return a dict matching RAW_REQUIRED_COLUMNS
    return {
        "order_purchase_timestamp": purch.strftime("%Y-%m-%dT%H:%M:%S"),          # formatted purchase ts
        "estimated_delivery_date": est.strftime("%Y-%m-%dT%H:%M:%S"),             # formatted estimate ts
        "sum_price": float(np.round(np.random.gamma(2.0, 60.0), 2)),              # right-skewed items total
        "sum_freight": float(np.round(np.random.gamma(1.5, 12.0), 2)),            # right-skewed freight total
        "n_items": int(np.clip(np.random.poisson(2) + 1, 1, 6)),                  # 1–6 items
        "n_sellers": int(np.clip(np.random.poisson(1) + 1, 1, 3)),                # 1–3 sellers
        "payment_type": _choice(["credit_card","boleto","debit_card","voucher","not_defined"], p=[0.65,0.20,0.08,0.05,0.02]),  # realistic skew
        "max_installments": int(np.clip(int(np.random.exponential(1.2)) + 1, 1, 12)),  # mostly 1–3
        "mode_category": _choice([
            "bed_bath_table","health_beauty","sports_leisure","computers_accessories",
            "furniture_decor","watches_gifts","housewares","auto","toys","stationery"
        ]),                                                                         # plausible categories
        "customer_city": _choice(["sao paulo","rio de janeiro","belo horizonte","curitiba","campinas","porto alegre"]),  # cities
        "customer_state": _choice(["SP","RJ","MG","PR","RS","BA","ES","SC","GO","DF"]),  # states
    }

def _demo_raw_frame(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a demo RAW DataFrame of length n_rows."""
    np.random.seed(int(seed))                                   # seed numpy RNG
    rows = [_demo_raw_row() for _ in range(int(n_rows))]        # generate rows
    return pd.DataFrame(rows, columns=RAW_REQUIRED_COLUMNS)     # build DataFrame with enforced order


# ----------------------------- Page setup & metadata tiles ----------------------------- #

st.set_page_config(page_title="Batch Predictions — Supply Chain Delay", page_icon="📦", layout="wide")  # config
st.title("📦 Batch Predictions")                                  # page title
st.caption("Upload a RAW CSV to score shipments, or generate a demo/template below.")  # subtitle
st.markdown("---")                                                # divider

meta = load_metadata()                                            # load metadata for tiles
thr = float(meta.get("optimal_threshold", 0.5))                   # threshold 0..1
rb = meta.get("risk_bands", {})                                   # bands dict
low_max = int(rb.get("low_max", 30))                              # low band cutoff
med_max = int(rb.get("med_max", 67))                              # medium band cutoff
auc = float(meta.get("best_model_auc", 0.0))                      # AUC metric

t1, t2, t3, t4 = st.columns(4)                                    # metric tiles
t1.metric("Threshold", f"{thr:.6f}", f"{thr*100:.2f}%")           # threshold with percent
t2.metric("Low Band (≤)", f"{low_max}%")                          # low band
t3.metric("Medium Band (≤)", f"{med_max}%")                       # medium band
t4.metric("AUC-ROC", f"{auc:.4f}")                                # AUC
st.markdown("---")                                                # divider

st.info(f"**Required RAW columns:** {', '.join(RAW_REQUIRED_COLUMNS)}")  # show expected headers


# ----------------------------- (A) Upload & Score ----------------------------- #

st.subheader("A) Upload CSV and Score")                           # section header
uploaded = st.file_uploader("Choose a CSV with the required RAW columns.", type=["csv"])  # uploader

df_scored: pd.DataFrame | None = None                             # will store scored output
df_clean: pd.DataFrame | None = None                              # will store cleaned RAW input

if uploaded is not None:                                          # run when user uploads a file
    try:
        df_in = pd.read_csv(io.BytesIO(uploaded.read()))          # read uploaded CSV into DataFrame
    except Exception as e:
        st.error(f"Could not read CSV: {e}")                      # show parsing error
        st.stop()                                                 # halt execution

    with st.expander("Preview & Validation", expanded=True):      # collapsible preview box
        st.write("**First 12 rows (raw):**")                      # label
        st.dataframe(df_in.head(12))                              # show head of raw data
        df_clean = _clean_df_raw(df_in)                           # run basic cleaning
        st.write("**Shape after cleaning:**", df_clean.shape)     # show resulting shape

        # Validate required headers
        missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df_clean.columns]  # find missing columns
        if missing:
            st.error(f"Missing required columns: {missing}")      # notify user of missing columns
            st.stop()                                             # halt if invalid

        df_clean = df_clean[RAW_REQUIRED_COLUMNS]                 # reorder to canonical order

    with st.spinner("Scoring shipments…"):                        # spinner during scoring
        try:
            df_scored = predict_batch(df_clean)                   # score the cleaned RAW data
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")             # show error from model/FE
            st.stop()                                             # halt on failure

    st.success(f"Scored {len(df_scored):,} shipments successfully.")  # success banner
    st.dataframe(df_scored.head(25))                              # preview top rows

    # Download the scored CSV
    st.download_button(
        label="⬇️ Download Scored CSV",                           # button label
        data=df_scored.to_csv(index=False).encode("utf-8"),       # CSV bytes payload
        file_name="batch_scored.csv",                             # filename
        mime="text/csv",                                          # MIME type
    )

    # Batch PDF summary export
    batch_pdf = generate_batch_summary_report(scored_df=df_scored, sample_preview_rows=25)  # build batch PDF
    st.download_button(
        label="⬇️ Download Batch Summary (PDF)",                  # button label
        data=batch_pdf,                                           # PDF bytes
        file_name="batch_summary.pdf",                            # filename
        mime="application/pdf",                                   # MIME type
    )

    st.markdown("---")                                            # divider
    st.subheader("Risk Band Distribution")                        # band distribution header
    if "risk_band" in df_scored.columns:                          # check presence of risk_band
        counts = (df_scored["risk_band"]                           # count per band
                  .value_counts()
                  .rename_axis("band")
                  .reset_index(name="count"))
        st.bar_chart(counts.set_index("band"))                    # render bar chart
    else:
        st.info("No `risk_band` column found in output. Check band mapping in the model loader.")  # helpful note

st.markdown("---")                                                # divider


# ----------------------------- (B) Generate Demo Dataset ----------------------------- #

st.subheader("B) Generate Demo Dataset")                          # section header
g1, g2, g3, g4 = st.columns([2, 2, 1, 2])                         # controls layout
with g1:
    n_rows = st.number_input("Rows", min_value=10, max_value=5000, value=200, step=10)  # number of demo rows
with g2:
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)  # RNG seed
with g3:
    gen = st.button("Generate Demo")                              # button to generate
with g4:
    auto_score = st.checkbox("Auto-score", value=True)            # auto-score toggle

if gen:                                                           # on button press
    demo = _demo_raw_frame(int(n_rows), int(seed))                # build a demo RAW DataFrame
    st.success(f"Generated {len(demo):,} demo rows.")             # success banner
    st.dataframe(demo.head(25))                                   # preview top rows

    st.download_button(
        label="⬇️ Download Demo CSV",                             # button label
        data=demo.to_csv(index=False).encode("utf-8"),            # CSV bytes
        file_name="demo_batch_raw.csv",                           # filename
        mime="text/csv",                                          # MIME type
    )

    if auto_score:                                                # if auto-score requested
        with st.spinner("Scoring demo dataset…"):                 # spinner during scoring
            try:
                demo_scored = predict_batch(_clean_df_raw(demo))  # clean + score the demo
            except Exception as e:
                st.error(f"Demo scoring failed: {e}")             # show error
                st.stop()                                         # halt on failure
        st.success("Demo dataset scored.")                        # success banner
        st.dataframe(demo_scored.head(25))                        # preview scored rows

        st.download_button(
            label="⬇️ Download Scored Demo CSV",                  # button label
            data=demo_scored.to_csv(index=False).encode("utf-8"), # CSV bytes
            file_name="demo_batch_scored.csv",                    # filename
            mime="text/csv",                                      # MIME type
        )

st.markdown("---")                                                # divider


# ----------------------------- (C) Generate Blank Template ----------------------------- #

st.subheader("C) Generate Blank Template")                        # section header
template = pd.DataFrame(columns=RAW_REQUIRED_COLUMNS)             # build an empty DF with required headers
st.info("This template contains the exact RAW columns expected by the app. Fill it out and upload in Section A.")  # helper text
st.dataframe(template.head(5))                                    # show the header structure
st.download_button(
    label="⬇️ Download Blank Template CSV",                       # button label
    data=template.to_csv(index=False).encode("utf-8"),            # CSV bytes
    file_name="batch_template_raw.csv",                           # filename
    mime="text/csv",                                              # MIME type
)

st.markdown("---")                                                # divider


# ----------------------------- (D) Why was this classified…? ----------------------------- #

st.subheader("D) Why was this classified this way?")              # section header

if "df_scored" not in locals() or df_scored is None or df_clean is None:  # guard for missing scored data
    st.info("Upload and score a file in Section A to enable per-row explanations.")  # guidance
else:
    # Select top-N rows by score and/or explicit indices
    left, right = st.columns([1.2, 2.0])                          # layout for selectors
    with left:
        top_n = st.number_input("Top N highest scores", min_value=1, max_value=min(50, len(df_scored)), value=min(5, len(df_scored)), step=1)  # choose top-N
        default_indices = df_scored.sort_values("score", ascending=False).head(int(top_n)).index.tolist()  # compute default indices
    with right:
        selected = st.multiselect(
            "Select specific row indices to inspect (defaults to Top N by score)",  # label
            options=list(df_scored.index),                                          # all row indices
            default=default_indices,                                                # default top-N
        )

    if not selected:                                                    # if no rows selected
        st.info("Select at least one row to explain.")                  # prompt user
    else:
        # Compute engineered features for the selected rows to show exact model inputs
        engineered_subset = calculate_features(df_clean.loc[selected])  # raw → engineered for selection
        st.markdown("**Engineered features sent to the model (subset):**")  # label
        st.dataframe(engineered_subset.head(len(selected)))             # show engineered features

        # Try to compute a simple heuristic ranking using LightGBM feature_importances_
        try:
            model, _ = load_model()                                     # load model object
            importances = getattr(model, "feature_importances_", None)  # LightGBM/sklearn attribute
            if importances is not None and len(importances) == engineered_subset.shape[1]:  # shape check
                imp = pd.Series(importances, index=engineered_subset.columns).astype(float)  # to Series
                imp_norm = (imp - imp.min()) / (imp.max() - imp.min() + 1e-9)               # normalize 0..1

                st.markdown("**Heuristic driver ranking (NOT SHAP):**")  # header with caveat
                for idx in selected:                                     # loop each selected row
                    st.markdown(f"- **Row {idx}** — score: {df_scored.loc[idx, 'score']:.4f} • band: {df_scored.loc[idx, 'risk_band']}")  # row header
                    row = engineered_subset.loc[idx]                     # engineered row
                    vals = row.copy()                                    # copy for manipulation
                    # For strings, treat non-empty as 1.0 to contribute to salience
                    for c in vals.index:
                        if isinstance(vals[c], str):
                            vals[c] = 1.0 if vals[c] else 0.0            # booleanize strings
                    salience = vals.abs() * imp_norm                     # magnitude × importance
                    topk = salience.sort_values(ascending=False).head(8) # top 8 features
                    show = pd.DataFrame({                                # assemble table
                        "feature": topk.index,
                        "value": row[topk.index].values,
                        "importance": imp_norm[topk.index].round(4).values,
                        "salience≈": topk.round(4).values,
                    })
                    st.dataframe(show.reset_index(drop=True))            # render table
            else:
                st.info("Model importances unavailable or mismatched; showing engineered features only.")  # fallback notice
        except Exception:
            st.info("Could not compute heuristic drivers from model importances. Showing engineered features only.")  # catch-all

        # Optionally render global SHAP PNGs if present for context
        art = _artifacts_dir()                                           # artifacts path
        shap_imgs = [                                                   # potential SHAP images
            art / "shap_summary_lightgbm.png",
            art / "shap_importance_lightgbm.png",
            art / "shap_dependence_top_feature_lightgbm.png",
        ]
        any_found = False                                               # tracker
        for p in shap_imgs:                                             # iterate possible files
            if p.exists():                                              # if image exists
                if not any_found:
                    st.markdown("**Global model context (static SHAP figures):**")  # header once
                st.image(str(p), use_column_width=True)                 # show image
                any_found = True                                        # mark found
        if not any_found:                                               # if none found
            st.caption("Tip: Add SHAP PNGs to `artifacts/` to show global model context.")  # helpful tip
