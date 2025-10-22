# pages/3_📦_Batch_Predictions.py
# Purpose: Batch scoring with three capabilities:
#   (A) Upload CSV and score every shipment
#   (B) Auto-generate a realistic demo dataset (and score it)
#   (C) Generate a blank CSV template with all required columns
#
# This page is metadata-driven:
# - Reads threshold (0–1) and risk bands (0–100) from artifacts/final_metadata.json
# - Reads required input spec from artifacts/feature_metadata.json when available
# - Uses utils.model_loader.predict_batch for inference (with feature engineering if present)
#
# Notes:
# - Thorough data cleaning occurs prior to prediction
# - All generated CSVs are downloadable from the UI

from __future__ import annotations                   # postpone evaluation of annotations for type hints
import io                                            # in-memory bytes buffer for CSV reading/writing
import json                                          # read JSON metadata files
from pathlib import Path                             # filesystem-safe paths
from typing import Any, Dict, List, Tuple            # type hints
import random                                        # random sampling for demo dataset
import math                                          # numeric helpers
import pandas as pd                                  # dataframe operations
import numpy as np                                   # numeric array ops
import streamlit as st                               # Streamlit UI elements

# Centralized metadata + prediction helpers from utils
from utils.model_loader import load_metadata, predict_batch  # import metadata loader and batch predictor


# -----------------------------------------------------------------------------
# Utility: locate the artifacts directory
# -----------------------------------------------------------------------------
def _artifacts_dir() -> Path:
    """Return absolute path to the local artifacts directory."""
    return Path(__file__).resolve().parents[1] / "artifacts"   # pages/.. → apps/supply_chain_delay/artifacts


# -----------------------------------------------------------------------------
# Load feature metadata (optional), which defines required columns, types, choices
# -----------------------------------------------------------------------------
def _load_feature_metadata() -> Dict[str, Any]:
    """Load artifacts/feature_metadata.json if present; else return {}."""
    fpath = _artifacts_dir() / "feature_metadata.json"         # feature metadata path
    if not fpath.exists():                                     # if file missing
        return {}                                              # return empty dict
    try:
        with fpath.open("r", encoding="utf-8") as f:           # open file
            data = json.load(f)                                # parse JSON
        return data if isinstance(data, dict) else {}          # ensure dict
    except Exception:
        return {}                                              # safe fallback if read fails


# -----------------------------------------------------------------------------
# Derive required columns and spec from feature metadata
# Expected schema example (flexible):
# { "inputs": [ {"name":"payment_value","type":"float","required":true,"default":0.0}, ... ] }
# -----------------------------------------------------------------------------
def _extract_inputs_spec(feat_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of input specs from feature metadata (or [])."""
    items = feat_meta.get("inputs", [])                        # read list under "inputs"
    return items if isinstance(items, list) else []            # ensure list


def _required_columns(inputs_spec: List[Dict[str, Any]]) -> List[str]:
    """Return ordered list of required input names from spec."""
    cols = []                                                  # initialize list
    for spec in inputs_spec:                                   # iterate specs
        if spec.get("required", False):                        # check required flag
            name = str(spec.get("name", "")).strip()           # get clean name
            if name:                                           # non-empty
                cols.append(name)                              # append
    return cols                                                # return list


def _all_declared_columns(inputs_spec: List[Dict[str, Any]]) -> List[str]:
    """Return ordered list of all declared input names (required + optional)."""
    cols = []                                                  # initialize
    for spec in inputs_spec:                                   # iterate
        name = str(spec.get("name", "")).strip()               # read name
        if name and name not in cols:                          # unique
            cols.append(name)                                  # append
    return cols                                                # return list


# -----------------------------------------------------------------------------
# Data cleaning: trim text, dedupe columns, coerce numeric-like columns
# -----------------------------------------------------------------------------
def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Generic, robust cleaning for uploaded data."""
    df = df.copy()                                             # copy to avoid mutating original
    df.columns = [str(c).strip() for c in df.columns]          # strip whitespace from column names
    df = df.loc[:, ~df.columns.duplicated(keep="first")]       # drop duplicate-named columns
    for c in df.select_dtypes(include=["object"]).columns:     # for object/string columns
        df[c] = df[c].astype(str).str.strip()                  # trim whitespace from values

    # Heuristic numeric coercion for columns that look numeric
    for c in df.columns:                                       # iterate columns
        if pd.api.types.is_numeric_dtype(df[c]):               # skip numeric columns
            continue                                           # already numeric
        sample = df[c].dropna().astype(str).head(300)          # sample text values
        if len(sample) == 0:                                   # skip if empty
            continue
        numeric_like = sample.str.fullmatch(                   # fraction of numeric-like strings
            r"[-+]?\d*\.?\d+(e[-+]?\d+)?", case=False
        ).mean()
        if numeric_like >= 0.9:                                # threshold for coercion
            df[c] = pd.to_numeric(df[c], errors="coerce")      # coerce to float (NaN on failure)

    # If user mistakenly included a 'score' column, clip it for safety (won’t be used by model input)
    if "score" in df.columns:                                  # check presence
        df["score"] = pd.to_numeric(df["score"], errors="coerce").clip(0.0, 1.0)  # normalize score
    return df                                                  # return cleaned frame


# -----------------------------------------------------------------------------
# CSV generators: blank template + realistic demo dataset
# -----------------------------------------------------------------------------
def _blank_template_df(inputs_spec: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a zero-row DataFrame with all declared columns (required first)."""
    req = _required_columns(inputs_spec)                       # required column names
    all_cols = _all_declared_columns(inputs_spec)              # all declared column names
    ordered = req + [c for c in all_cols if c not in req]      # required first, then remaining
    return pd.DataFrame(columns=ordered)                       # empty DF with headers only


def _random_value_for_spec(spec: Dict[str, Any]) -> Any:
    """Generate a plausible random value for one input spec."""
    itype = str(spec.get("type", "text")).lower()              # read input type
    default = spec.get("default", None)                        # read default (if any)
    choices = spec.get("choices", None)                        # read categorical choices

    # If categorical with choices, sample from choices (fall back to default for safety)
    if itype in {"category", "categorical", "select"} and isinstance(choices, list) and len(choices) > 0:
        return random.choice(choices)                          # random category

    # Numeric generation with soft bounds if provided in spec (min/max)
    if itype in {"int", "integer"}:                            # integer field
        lo = int(spec.get("min", 0))                           # min bound
        hi = int(spec.get("max", 100))                         # max bound
        if lo > hi:                                            # ensure sane bounds
            lo, hi = hi, lo                                    # swap if reversed
        return random.randint(lo, hi)                          # random int in range
    if itype in {"float", "double", "number", "numeric"}:      # float field
        lo = float(spec.get("min", 0.0))                       # min bound
        hi = float(spec.get("max", 100.0))                     # max bound
        if lo > hi:                                            # sanitize bounds
            lo, hi = hi, lo                                    # swap if reversed
        val = random.random() * (hi - lo) + lo                 # uniform in [lo, hi]
        return round(val, 4)                                   # tidy precision for UI

    # Boolean fields default to False unless specified
    if itype in {"bool", "boolean"}:                           # boolean type
        if default is not None:                                # default provided
            return bool(default)                               # cast default
        return random.choice([True, False])                    # random boolean

    # Datetime-like: simple ISO-ish strings (let FE parse exact format if needed)
    if itype in {"datetime", "date"}:                          # datetime type
        # Generate a date range; here use a plausible 2017–2018 Olist window
        y = random.choice([2017, 2018])                        # pick a year
        m = random.randint(1, 12)                              # month 1-12
        d = random.randint(1, 28)                              # day safe 1-28
        hh = random.randint(0, 23)                             # hour 0-23
        mm = random.randint(0, 59)                             # minute 0-59
        return f"{y:04d}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}:00" # ISO-like string

    # Default: text fields (use default if present, else a placeholder)
    if default not in [None, ""]:                              # non-empty default
        return str(default)                                    # return default
    return "unknown"                                           # fallback text


def _demo_dataset_df(inputs_spec: List[Dict[str, Any]], n_rows: int = 200) -> pd.DataFrame:
    """Generate a realistic demo dataset with n_rows, based on feature metadata."""
    cols = _all_declared_columns(inputs_spec)                  # declared columns
    if not cols:                                               # if no spec available
        # Fallback minimal demo schema when no metadata is present
        cols = ["payment_value", "customer_city", "order_purchase_timestamp"]  # minimal set
        inputs_spec = [                                        # synthetic spec for demo
            {"name": "payment_value", "type": "float", "min": 10.0, "max": 500.0},
            {"name": "customer_city", "type": "category", "choices": ["sao paulo", "rio de janeiro", "belo horizonte", "curitiba"]},
            {"name": "order_purchase_timestamp", "type": "datetime"}
        ]
    rows: List[Dict[str, Any]] = []                            # container for rows
    for _ in range(int(n_rows)):                               # loop n_rows times
        row: Dict[str, Any] = {}                               # init row dict
        for spec in inputs_spec:                               # iterate spec
            name = str(spec.get("name", "")).strip()           # get name
            if not name:                                       # skip if empty
                continue
            row[name] = _random_value_for_spec(spec)           # generate value
        rows.append(row)                                       # append row
    df = pd.DataFrame(rows, columns=cols)                      # build DataFrame with column order
    return df                                                  # return demo DF


# -----------------------------------------------------------------------------
# Page scaffold (metrics + controls)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Batch Predictions — Supply Chain Delay", page_icon="📦", layout="wide")  # configure page
st.title("📦 Batch Predictions")                                 # page title
st.caption("Upload a CSV to score shipments, or generate a demo/template CSV below.")      # short subtitle
st.markdown("---")                                               # divider

# Load model training metadata (threshold + bands + metrics)
meta = load_metadata()                                           # load final_metadata.json
thr = float(meta.get("optimal_threshold", 0.5))                  # threshold 0–1
rb = meta.get("risk_bands", {})                                  # risk bands dict
low_max = int(rb.get("low_max", 30))                             # low band upper cutoff (%)
med_max = int(rb.get("med_max", 67))                             # medium band upper cutoff (%)

# Top metrics summary
col1, col2, col3, col4 = st.columns(4)                           # four metric tiles
with col1:
    st.metric("Optimal Threshold", f"{thr:.6f}", f"{thr*100:.2f}%")  # probability + percent
with col2:
    st.metric("Low Band (≤)", f"{low_max}%")                     # band display
with col3:
    st.metric("Medium Band (≤)", f"{med_max}%")                  # band display
with col4:
    st.metric("AUC-ROC", f"{float(meta.get('best_model_auc', 0.0)):.4f}")  # model AUC

st.markdown("---")                                               # divider

# Load feature metadata for schema-driven behaviors
feat_meta = _load_feature_metadata()                              # feature spec JSON
inputs_spec = _extract_inputs_spec(feat_meta)                     # list of input specs
required_cols = _required_columns(inputs_spec)                    # required columns list
all_cols = _all_declared_columns(inputs_spec)                     # all declared columns list

# Explain required columns (if known)
if required_cols:                                                 # if we have a spec
    st.info(f"**Required columns**: {', '.join(required_cols)}")  # show required column names
else:
    st.info("No `feature_metadata.json` found. Upload any CSV your model’s feature engineering can handle. For convenience, you can generate a demo dataset or a blank template below.")

# -----------------------------------------------------------------------------
# Section A: Upload CSV and Score
# -----------------------------------------------------------------------------
st.subheader("A) Upload CSV and Score")                           # section header
uploaded = st.file_uploader("Choose a CSV file with your shipment fields.", type=["csv"])  # file uploader

if uploaded is not None:                                          # if a file was uploaded
    try:
        raw_bytes = uploaded.read()                               # read file bytes
        df_in = pd.read_csv(io.BytesIO(raw_bytes))                # parse CSV
    except Exception as e:
        st.error(f"Could not read CSV: {e}")                      # show parse error
        st.stop()                                                 # stop further execution

    with st.expander("Preview & Cleaning (pre-prediction)", expanded=True):  # preview panel
        st.write("**First 12 rows (raw):**")                      # label
        st.dataframe(df_in.head(12))                              # show preview
        df_clean = _clean_dataframe(df_in)                        # run cleaning
        st.write("**Detected shape after cleaning:** ", df_clean.shape)  # show shape

        # If we know required columns, check their presence and order
        if required_cols:                                         # if schema known
            missing = [c for c in required_cols if c not in df_clean.columns]  # find missing required
            if missing:                                           # if any missing
                st.error(f"Missing required columns: {missing}. Please fix your file or use the template generator below.")
                st.stop()                                         # halt if missing

    # Score the cleaned data
    with st.spinner("Scoring shipments…"):                        # show spinner
        try:
            df_scored = predict_batch(df_clean)                   # batch predict via utils
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")             # display error
            st.stop()                                             # stop on error

    st.success(f"Scored {len(df_scored):,} shipments successfully.")  # success message
    st.dataframe(df_scored.head(25))                              # show top rows of result

    # Download scored CSV
    scored_bytes = df_scored.to_csv(index=False).encode("utf-8")  # serialize to CSV
    st.download_button(                                           # download button
        label="⬇️ Download Scored CSV",
        data=scored_bytes,
        file_name="batch_scored.csv",
        mime="text/csv"
    )

    # Quick distribution chart by risk band (if present)
    st.markdown("---")                                           # divider
    st.subheader("Risk Band Distribution")                       # chart header
    if "risk_band" in df_scored.columns:                         # check presence
        counts = (df_scored["risk_band"]                         # count rows per band
                  .value_counts()
                  .rename_axis("band")
                  .reset_index(name="count"))
        st.bar_chart(counts.set_index("band"))                   # display bar chart
    else:
        st.info("No `risk_band` column found in output. Ensure band mapping is enabled in your model loader.")

st.markdown("---")                                               # divider

# -----------------------------------------------------------------------------
# Section B: Generate Demo Dataset (and Score It)
# -----------------------------------------------------------------------------
st.subheader("B) Generate Demo Dataset")                         # section header
demo_cols = st.columns([2, 2, 1, 2])                             # columns for controls
with demo_cols[0]:
    n_rows = st.number_input("Number of demo rows", min_value=10, max_value=5000, value=200, step=10)  # row count
with demo_cols[1]:
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)             # RNG seed
with demo_cols[2]:
    gen_btn = st.button("Generate Demo")                         # generate button
with demo_cols[3]:
    auto_score = st.checkbox("Auto-score after generation", value=True)  # auto-score flag

if gen_btn:                                                      # on click generate
    random.seed(int(seed))                                       # seed Python RNG
    np.random.seed(int(seed))                                    # seed NumPy RNG
    demo_df = _demo_dataset_df(inputs_spec, int(n_rows))         # build demo data
    st.success(f"Generated {len(demo_df):,} demo rows.")         # success message
    st.dataframe(demo_df.head(25))                               # show preview

    # Download raw demo CSV
    demo_bytes = demo_df.to_csv(index=False).encode("utf-8")     # bytes for download
    st.download_button(
        label="⬇️ Download Demo CSV",
        data=demo_bytes,
        file_name="demo_batch.csv",
        mime="text/csv"
    )

    # Optionally score the demo DF
    if auto_score:                                               # if checkbox set
        with st.spinner("Scoring demo dataset…"):                # spinner
            try:
                demo_scored = predict_batch(_clean_dataframe(demo_df))  # score cleaned demo
            except Exception as e:
                st.error(f"Demo scoring failed: {e}")            # error display
                st.stop()                                        # stop
        st.success("Demo dataset scored.")                       # success message
        st.dataframe(demo_scored.head(25))                       # show sample

        # Download scored demo CSV
        demo_scored_bytes = demo_scored.to_csv(index=False).encode("utf-8")  # serialize
        st.download_button(
            label="⬇️ Download Scored Demo CSV",
            data=demo_scored_bytes,
            file_name="demo_batch_scored.csv",
            mime="text/csv"
        )

st.markdown("---")                                               # divider

# -----------------------------------------------------------------------------
# Section C: Generate Blank Template CSV
# -----------------------------------------------------------------------------
st.subheader("C) Generate Blank Template")                       # section header
if all_cols:                                                     # if we know the schema
    template_df = _blank_template_df(inputs_spec)                # build zero-row template
else:
    # Minimal fallback when feature metadata is not available
    template_df = pd.DataFrame(columns=["payment_value", "customer_city", "order_purchase_timestamp"])  # minimal headers
st.info("Download this template, fill in the required columns, and upload it in Section A to score your shipments.")  # helper text
st.dataframe(template_df.head(5))                                # show empty structure (will be blank)
template_bytes = template_df.to_csv(index=False).encode("utf-8") # serialize to CSV
st.download_button(
    label="⬇️ Download Blank Template CSV",
    data=template_bytes,
    file_name="batch_template.csv",
    mime="text/csv"
)
