# pages/2_📊_Single_Prediction.py
# Purpose: Single-order prediction with robust input form, dynamic metadata-driven threshold/bands,
#          and clean inference path that leverages feature engineering if available.
# Notes:
# - Attempts to auto-generate inputs from artifacts/feature_metadata.json
# - Falls back to a minimalist manual entry mode if metadata is missing
# - Cleans inputs (trim strings, coerce numerics) before scoring
# - Displays score (0–1), threshold flag, and risk band (0–100 cutoff)

import json                                              # import json to load feature metadata
from pathlib import Path                                 # import Path for OS-agnostic paths
from typing import Any, Dict, List                       # typing helpers for clarity
import pandas as pd                                      # import pandas for light cleaning/structuring
import streamlit as st                                   # import Streamlit for the page UI

# Centralized loader/predictors from our upgraded utils
from utils.model_loader import load_metadata, predict_single  # import metadata loader and single-inference

# -----------------------------------------------------------------------------
# Page Identity (avoid set_page_config duplication across pages)
# -----------------------------------------------------------------------------
st.title("📊 Single Prediction")                         # render page title
st.caption("Enter order details and get a real-time late-delivery risk score.")  # helpful subtitle
st.markdown("---")                                      # visual divider

# -----------------------------------------------------------------------------
# Load app-level metadata (threshold + bands)
# -----------------------------------------------------------------------------
meta: Dict[str, Any] = load_metadata()                   # read metadata dict from artifacts/final_metadata.json
thr: float = float(meta.get("optimal_threshold", 0.5))   # read operating threshold (0–1)
rb: Dict[str, Any] = meta.get("risk_bands", {})          # read risk bands dict
low_max: int = int(rb.get("low_max", 30))                # low band cutoff as percent
med_max: int = int(rb.get("med_max", 67))                # medium band cutoff as percent

# Show quick context metrics
c1, c2, c3, c4 = st.columns(4)                           # create four metric tiles
with c1:                                                 # tile 1: threshold (probability)
    st.metric("Optimal Threshold", f"{thr:.6f}", f"{thr*100:.2f}%")  # display both raw and percent
with c2:                                                 # tile 2: AUC
    st.metric("AUC-ROC", f"{float(meta.get('best_model_auc', 0.0)):.4f}")  # show AUC
with c3:                                                 # tile 3: Low band
    st.metric("Low Band (≤)", f"{low_max}%")            # low band upper bound
with c4:                                                 # tile 4: Medium band
    st.metric("Medium Band (≤)", f"{med_max}%")         # medium band upper bound

st.markdown("---")                                      # divider

# -----------------------------------------------------------------------------
# Feature metadata loader (optional) to auto-build form controls
# -----------------------------------------------------------------------------
def _artifacts_dir() -> Path:
    """Return absolute path to the local artifacts directory (apps/supply_chain_delay/artifacts)."""
    return Path(__file__).resolve().parents[1] / "artifacts"  # pages/.. → repo apps/supply_chain_delay/artifacts

def _load_feature_metadata() -> Dict[str, Any]:
    """
    Load artifacts/feature_metadata.json if present.
    Expected (flexible) structure example:
    {
      "inputs": [
        {"name": "order_purchase_timestamp", "type": "datetime", "required": true},
        {"name": "payment_value", "type": "float", "required": false, "default": 0.0},
        {"name": "customer_city", "type": "category", "required": true, "choices": ["sao paulo", "rio de janeiro", ...]}
      ]
    }
    """
    fpath = _artifacts_dir() / "feature_metadata.json"   # compute feature metadata path
    if not fpath.exists():                               # if metadata file is missing
        return {}                                        # return empty dict to trigger fallback
    try:                                                 # try reading the JSON
        with fpath.open("r", encoding="utf-8") as f:     # open file safely
            data = json.load(f)                          # parse JSON
        if isinstance(data, dict):                       # ensure dictionary
            return data                                  # return parsed metadata
        return {}                                        # non-dict content fallback
    except Exception:
        return {}                                        # any read/parse error → fallback

feat_meta: Dict[str, Any] = _load_feature_metadata()     # attempt to load feature metadata
inputs_spec: List[Dict[str, Any]] = feat_meta.get("inputs", [])  # list of input specs (may be empty)

# -----------------------------------------------------------------------------
# Helper: clean a single input-row dict (trim strings, coerce numerics)
# -----------------------------------------------------------------------------
def _clean_single_input(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean a flat dict of user inputs:
    - Trim whitespace for string-ish values
    - Coerce numeric strings to float where appropriate (heuristic if spec type is numeric)
    """
    cleaned: Dict[str, Any] = {}                         # initialize result dict
    for k, v in raw.items():                             # iterate over key/value pairs
        if isinstance(v, str):                           # if value is a string
            v = v.strip()                                # trim whitespace
        cleaned[k] = v                                   # assign cleaned value
    # Attempt numeric coercion for obvious numeric-like values when spec hints exist
    type_map = {i.get("name"): i.get("type") for i in inputs_spec}  # map name → type from metadata
    for name, val in list(cleaned.items()):              # iterate again for type-based coercion
        t = (type_map.get(name) or "").lower()           # get declared type (if any)
        if t in {"int", "integer"}:                      # integer fields
            try:
                cleaned[name] = int(float(val))          # coerce float-ish strings to int
            except Exception:
                pass                                     # leave as-is if coercion fails
        elif t in {"float", "double", "number", "numeric"}:  # float fields
            try:
                cleaned[name] = float(val)               # coerce to float
            except Exception:
                pass                                     # leave as-is if coercion fails
        # Dates/timestamps are left as strings; FE function should parse if needed
    return cleaned                                       # return cleaned inputs

# -----------------------------------------------------------------------------
# UI: build the input form (auto from metadata or fallback to key/value editor)
# -----------------------------------------------------------------------------
st.subheader("Enter Order Details")                      # section header

with st.form("single_prediction_form", clear_on_submit=False):  # begin form to aggregate inputs
    form_values: Dict[str, Any] = {}                    # container for form inputs

    if inputs_spec:                                     # if we have a spec, build dynamic controls
        # Split into columns to reduce vertical scrolling
        cols = st.columns(3)                            # create three columns
        col_idx = 0                                     # track which column to place the next widget
        for spec in inputs_spec:                        # iterate over declared inputs
            name = spec.get("name", "").strip()         # read input name
            itype = (spec.get("type") or "text").lower()  # read input type, default text
            required = bool(spec.get("required", False))  # read required flag
            default = spec.get("default", "")           # read default value
            choices = spec.get("choices")               # read categorical choices if present

            # Choose the current column for this widget
            with cols[col_idx % 3]:                     # place widget in a rotating column
                # Render appropriate control by type
                if itype in {"category", "categorical", "select"} and isinstance(choices, list):
                    form_values[name] = st.selectbox(   # create dropdown for categorical inputs
                        label=f"{name} {'*' if required else ''}",
                        options=choices,
                        index=choices.index(default) if default in choices else 0
                    )
                elif itype in {"int", "integer"}:
                    form_values[name] = st.number_input(  # integer numeric input
                        label=f"{name} {'*' if required else ''}",
                        value=int(default) if str(default).strip() != "" else 0,
                        step=1
                    )
                elif itype in {"float", "double", "number", "numeric"}:
                    form_values[name] = st.number_input(  # float numeric input
                        label=f"{name} {'*' if required else ''}",
                        value=float(default) if str(default).strip() != "" else 0.0,
                        step=0.01,
                        format="%.6f"
                    )
                elif itype in {"bool", "boolean"}:
                    form_values[name] = st.checkbox(     # boolean input
                        label=f"{name} {'*' if required else ''}",
                        value=bool(default)
                    )
                elif itype in {"datetime", "date"}:
                    # Keep as text input; let feature engineering parse exact format
                    form_values[name] = st.text_input(   # datetime as free-text
                        label=f"{name} (YYYY-MM-DD or ISO8601) {'*' if required else ''}",
                        value=str(default)
                    )
                else:
                    # Generic text input
                    form_values[name] = st.text_input(   # default to text field
                        label=f"{name} {'*' if required else ''}",
                        value=str(default)
                    )

            col_idx += 1                                 # increment column index for next widget

    else:
        # Fallback UI when feature metadata is unavailable:
        st.info("No `feature_metadata.json` found. Enter key/value pairs (JSON) for raw order fields.")
        default_json = "{\n  \"payment_value\": 120.50,\n  \"customer_city\": \"sao paulo\",\n  \"order_purchase_timestamp\": \"2017-07-01T10:45:00\"\n}"
        json_text = st.text_area(                        # text area to paste JSON
            label="Raw input JSON",
            value=default_json,
            height=180
        )
        try:
            form_values = json.loads(json_text)          # parse JSON if possible
            if not isinstance(form_values, dict):        # ensure dict structure
                st.warning("The JSON must represent a single object (key/value pairs).")  # warn user
                form_values = {}
        except Exception:
            st.warning("Invalid JSON. Fix the syntax or use the default example.")       # warn about syntax
            form_values = {}

    # Submit button to run prediction
    submitted = st.form_submit_button("Predict Risk")   # render submit button

# -----------------------------------------------------------------------------
# Inference execution upon submit
# -----------------------------------------------------------------------------
if submitted:                                           # if user pressed the Predict button
    if not form_values:                                 # if the form is empty
        st.error("No inputs were provided. Please complete the form.")  # show error
        st.stop()                                       # halt the page

    # Clean inputs before sending to model
    cleaned_inputs = _clean_single_input(form_values)   # sanitize user inputs

    # Run prediction with robust handling
    with st.spinner("Scoring…"):                        # show spinner while scoring
        try:
            result = predict_single(cleaned_inputs)     # call the central single-predict function
        except Exception as e:
            st.error(f"Prediction failed: {e}")         # show any runtime error
            st.stop()                                   # halt the page

    # Extract results cleanly
    score: float = float(result.get("score", 0.0))      # read probability score
    meets: bool = bool(result.get("meets_threshold", False))  # read threshold flag
    band: str = str(result.get("risk_band", "N/A"))     # read risk band

    # Display a compact result card
    st.success("Prediction complete.")                  # success banner

    r1, r2, r3 = st.columns(3)                          # three result tiles
    with r1:                                            # tile: predicted score
        st.metric("Predicted Probability", f"{score:.6f}", f"{score*100:.2f}%")  # show raw + percent
    with r2:                                            # tile: threshold comparison
        st.metric("Meets Threshold", "Yes" if meets else "No")  # yes/no flag
    with r3:                                            # tile: risk band
        st.metric("Risk Band", band)                    # risk band label

    # Echo the cleaned inputs for traceability
    with st.expander("View cleaned inputs sent to the model", expanded=False):  # collapsible panel
        st.json(cleaned_inputs)                           # show json echo

    st.markdown("---")                                  # divider

    # Optional interpretability section: display SHAP figures if available
    st.subheader("Model Insights (Static Artifacts)")   # section header
    art = _artifacts_dir()                               # artifacts directory
    shap_paths = [
        art / "shap_summary_lightgbm.png",               # SHAP summary image
        art / "shap_importance_lightgbm.png",            # SHAP importance
        art / "shap_dependence_top_feature_lightgbm.png" # SHAP dependence plot
    ]
    found_any = False                                    # track if any images exist
    for p in shap_paths:                                 # iterate over images
        if p.exists():                                   # check file existence
            st.image(str(p), use_column_width=True)      # display image
            found_any = True                             # mark that at least one was found
    if not found_any:                                    # if no images are available
        st.info("No SHAP images found in `artifacts/`. Add PNGs to visualize feature effects.")  # helpful note
