# pages/3_📦_Batch_Predictions.py
# Purpose: Batch CSV scoring with thorough data cleaning, threshold flagging, and risk band assignment.
# Notes:
# - Expects a CSV with raw order fields (model will transform via feature_engineering if available)
# - Cleans numeric text, dates (optional), and clips any preexisting "score" column if provided.
# - Uses utils.model_loader to get model, metadata, and consistent band/threshold logic.

import io                                      # for in-memory CSV bytes handling
from pathlib import Path                       # not strictly required but handy
import pandas as pd                            # DataFrame handling
import streamlit as st                         # Streamlit UI
from utils.model_loader import (               # our centralized loader/predictor
    load_metadata,
    predict_batch
)

# ---------- page layout ----------

st.set_page_config(                            # ensure page has a friendly title/icon
    page_title="Batch Predictions — Supply Chain Delay",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Batch Predictions")               # main header
st.caption("Upload a CSV of orders. We’ll score each row, apply the threshold, and assign risk bands.")  # subtitle
st.markdown("---")                             # divider

# ---------- metadata summary ----------

meta = load_metadata()                         # load metadata dict
thr = float(meta.get("optimal_threshold", 0.5))# read operating threshold (0–1)
rb = meta.get("risk_bands", {})               # risk bands dict
low_max = int(rb.get("low_max", 30))          # low band upper % cutoff
med_max = int(rb.get("med_max", 67))          # medium band upper % cutoff

col1, col2, col3 = st.columns(3)               # 3 metric tiles
with col1:                                     # tile 1
    st.metric("Optimal Threshold", f"{thr:.6f}", f"{thr*100:.2f}%")  # show prob and % delta
with col2:                                     # tile 2
    st.metric("Low Band (≤)", f"{low_max}%")   # show low band max
with col3:                                     # tile 3
    st.metric("Medium Band (≤)", f"{med_max}%")# show medium band max

st.markdown("---")                             # divider

# ---------- uploader ----------

st.subheader("Upload CSV")                     # uploader header
uploaded = st.file_uploader(                   # file uploader
    "Choose a CSV file with your raw order fields.",
    type=["csv"]
)

# ---------- data cleaning helpers ----------

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generic cleaning:
    - Strip column names and deduplicate
    - Trim string cells
    - Attempt numeric coercion for obviously numeric-looking columns (heuristic)
    - Clip any existing 'score' column to [0,1] (in case pre-scored files are re-run)
    """
    df = df.copy()                                                     # copy to avoid mutating uploaded object
    df.columns = [str(c).strip() for c in df.columns]                  # strip whitespace from column names
    df = df.loc[:, ~df.columns.duplicated(keep="first")]               # drop duplicate-named columns
    for c in df.select_dtypes(include=["object"]).columns:             # iterate string-like columns
        df[c] = df[c].astype(str).str.strip()                          # trim whitespace
    # Heuristic: try numeric cast on columns with high numeric ratio
    for c in df.columns:
        # skip if already numeric
        if pd.api.types.is_numeric_dtype(df[c]):                       # if numeric, continue
            continue
        # sample up to 500 non-null values to test numeric-ness
        sample = df[c].dropna().astype(str).head(500)                  # take sample
        if len(sample) == 0:                                           # if empty sample
            continue
        numeric_like = sample.str.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", case=False).mean()  # fraction numeric-looking
        if numeric_like >= 0.9:                                        # threshold for numeric-like column
            df[c] = pd.to_numeric(df[c], errors="coerce")              # coerce column to numeric
    if "score" in df.columns:                                          # if a score column exists
        df["score"] = pd.to_numeric(df["score"], errors="coerce").clip(0.0, 1.0)  # coerce & clip to [0,1]
    return df                                                          # return cleaned frame

# ---------- main flow ----------

if uploaded is None:                               # if no file uploaded yet
    st.info("Drag & drop a CSV above. You’ll get back a scored file with threshold flags and risk bands.")  # helper text
else:                                              # when a file is present
    try:
        raw_bytes = uploaded.read()                # read uploaded bytes
        df_in = pd.read_csv(io.BytesIO(raw_bytes)) # parse CSV into DataFrame
    except Exception as e:
        st.error(f"Could not read CSV: {e}")       # show parse error
        st.stop()                                  # halt page

    with st.expander("Preview & Cleaning (pre-prediction)", expanded=True):  # expandable preview
        st.write("**First 10 rows (raw):**")        # label
        st.dataframe(df_in.head(10))               # show preview
        df_clean = _clean_dataframe(df_in)         # run cleaning
        st.write("**Detected shape after cleaning:** ", df_clean.shape)  # shape info

    # Run predictions (model_loader handles FE -> model features)
    with st.spinner("Scoring rows…"):              # spinner during compute
        try:
            df_scored = predict_batch(df_clean)    # run batch prediction
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")  # show error
            st.stop()                              # stop if model/FE error

    st.success(f"Scored {len(df_scored):,} rows successfully.")  # success message
    st.dataframe(df_scored.head(25))              # show top rows of result

    # Download button
    csv_bytes = df_scored.to_csv(index=False).encode("utf-8")  # serialize to CSV bytes
    st.download_button(                               # render download button
        label="⬇️ Download Scored CSV",
        data=csv_bytes,
        file_name="batch_scored.csv",
        mime="text/csv"
    )

    # Simple band distribution viz (optional; avoids extra deps)
    st.markdown("---")                               # divider
    st.subheader("Risk Band Distribution")           # viz header
    if "risk_band" in df_scored.columns:             # check existence
        counts = df_scored["risk_band"].value_counts().rename_axis("band").reset_index(name="count")  # count per band
        st.bar_chart(counts.set_index("band"))       # quick bar chart
    else:
        st.info("No `risk_band` column found in output. Check model output and risk band mapping.")
