# pages/3_📦_Batch_Predictions.py
# Purpose: Batch scoring using RAW, user-friendly columns (not engineered 32).
# Features:
#   (A) Upload a raw CSV → clean → feature_engineering → model → scored CSV
#   (B) Auto-generate a RAW demo dataset (and score it)
#   (C) Generate a blank RAW template with required columns
#   (D) Explain predictions for selected rows (engineered features + importance-weighted heuristic)
#
# Notes:
# - predict_batch() internally calls calculate_features() to transform RAW → engineered.
# - Section D is an interpretability aid:
#     * Shows the engineered features for selected rows (the exact model inputs).
#     * Computes a heuristic ranking using LightGBM feature_importances_ if available.
#       This is NOT SHAP—just a directional proxy. We label it clearly.
#     * If static SHAP PNGs exist in artifacts/, we render them as global interpretability context.
#
# References:
# - Python:  https://docs.python.org/3/
# - Pandas:  https://pandas.pydata.org/docs/

from __future__ import annotations                        # postpone annotations for clarity
import io                                                # read uploaded CSV bytes
from typing import Any, Dict, List, Tuple                # typing helpers
from pathlib import Path                                 # filesystem-safe paths
from datetime import date, time                          # for demo datetime construction
import random                                            # RNG for demo data
import numpy as np                                       # numeric ops
import pandas as pd                                      # dataframe ops
import streamlit as st                                   # Streamlit UI

# Central helpers
from utils.model_loader import load_metadata, predict_batch, load_model  # metadata, scoring, and model access
from utils.feature_engineering import calculate_features                 # to echo engineered features in Section D

# ------------------------------------------------------------------------------
# RAW schema (match Single page)
# ------------------------------------------------------------------------------
RAW_REQUIRED_COLUMNS: List[str] = [
    "order_purchase_timestamp",        # ISO-like "YYYY-MM-DDTHH:MM:SS"
    "estimated_delivery_date",         # ISO-like "YYYY-MM-DDTHH:MM:SS"
    "sum_price",                       # R$
    "sum_freight",                     # R$
    "n_items",                         # int
    "n_sellers",                       # int
    "payment_type",                    # {"credit_card","boleto","debit_card","voucher","not_defined"}
    "max_installments",                # int
    "mode_category",                   # text (e.g., "bed_bath_table")
    "customer_city",                   # text (e.g., "sao paulo")
    "customer_state",                  # text (e.g., "SP")
]

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _artifacts_dir() -> Path:
    """Path to local artifacts directory next to the app."""
    return Path(__file__).resolve().parents[1] / "artifacts"  # pages/.. → artifacts

def _clean_df_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning for raw uploads: standardize headers, trim, coerce numerics, uppercase state."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]            # strip header whitespace
    df = df.loc[:, ~df.columns.duplicated(keep="first")]         # drop duplicated headers
    for c in df.select_dtypes(include=["object"]).columns:       # trim strings
        df[c] = df[c].astype(str).str.strip()
    # Coerce numerics safely (NaN on failure)
    for c in ["sum_price", "sum_freight", "n_items", "n_sellers", "max_installments"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Fill numeric NA with conservative defaults
    for c in ["sum_price", "sum_freight"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    for c in ["n_items", "n_sellers", "max_installments"]:
        if c in df.columns:
            df[c] = df[c].fillna(1).astype(int)
    # Uppercase state codes
    if "customer_state" in df.columns:
        df["customer_state"] = df["customer_state"].str.upper()
    return df

def _demo_raw_row() -> Dict[str, Any]:
    """Generate a single realistic RAW row."""
    year = np.random.choice([2017, 2018])                   # Olist-era years
    month = np.random.randint(1, 13)
    day = np.random.randint(1, 28)
    hour = np.random.randint(0, 24)
    minute = np.random.randint(0, 60)
    purch = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
    est = purch + pd.Timedelta(days=int(np.random.randint(3, 20)))  # 3–19 days lead time

    def _choice(seq, p=None):
        return np.random.choice(seq, p=p)

    return {
        "order_purchase_timestamp": purch.strftime("%Y-%m-%dT%H:%M:%S"),
        "estimated_delivery_date": est.strftime("%Y-%m-%dT%H:%M:%S"),
        "sum_price": float(np.round(np.random.gamma(2.0, 60.0), 2)),
        "sum_freight": float(np.round(np.random.gamma(1.5, 12.0), 2)),
        "n_items": int(np.clip(np.random.poisson(2) + 1, 1, 6)),
        "n_sellers": int(np.clip(np.random.poisson(1) + 1, 1, 3)),
        "payment_type": _choice(["credit_card","boleto","debit_card","voucher","not_defined"], p=[0.65,0.20,0.08,0.05,0.02]),
        "max_installments": int(np.clip(int(np.random.exponential(1.2)) + 1, 1, 12)),
        "mode_category": _choice([
            "bed_bath_table","health_beauty","sports_leisure","computers_accessories",
            "furniture_decor","watches_gifts","housewares","auto","toys","stationery"
        ]),
        "customer_city": _choice(["sao paulo","rio de janeiro","belo horizonte","curitiba","campinas","porto alegre"]),
        "customer_state": _choice(["SP","RJ","MG","PR","RS","BA","ES","SC","GO","DF"]),
    }

def _demo_raw_frame(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a RAW demo DataFrame with required columns and realistic ranges."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    rows = [_demo_raw_row() for _ in range(int(n_rows))]
    return pd.DataFrame(rows, columns=RAW_REQUIRED_COLUMNS)  # enforce header order

# ------------------------------------------------------------------------------
# Page setup + metadata tiles
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Batch Predictions — Supply Chain Delay", page_icon="📦", layout="wide")
st.title("📦 Batch Predictions")
st.caption("Upload a raw CSV to score shipments, or generate a demo/template CSV below.")
st.markdown("---")

meta = load_metadata()                                   # read final_metadata.json
thr = float(meta.get("optimal_threshold", 0.5))          # 0–1 threshold
rb = meta.get("risk_bands", {})                          # risk bands dict
low_max = int(rb.get("low_max", 30))                     # %
med_max = int(rb.get("med_max", 67))                     # %
auc = float(meta.get("best_model_auc", 0.0))             # AUC metric

t1, t2, t3, t4 = st.columns(4)
t1.metric("Threshold", f"{thr:.6f}", f"{thr*100:.2f}%")
t2.metric("Low Band (≤)", f"{low_max}%")
t3.metric("Medium Band (≤)", f"{med_max}%")
t4.metric("AUC-ROC", f"{auc:.4f}")
st.markdown("---")

st.info(f"**Required RAW columns:** {', '.join(RAW_REQUIRED_COLUMNS)}")

# ------------------------------------------------------------------------------
# (A) Upload RAW CSV and score
# ------------------------------------------------------------------------------
st.subheader("A) Upload CSV and Score")
uploaded = st.file_uploader("Choose a CSV with the required RAW columns.", type=["csv"])

df_scored: pd.DataFrame | None = None      # will hold scored output for reuse in Section D
df_clean: pd.DataFrame | None = None       # cleaned RAW input (for engineered echo later)

if uploaded is not None:
    try:
        df_in = pd.read_csv(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    with st.expander("Preview & Validation", expanded=True):
        st.write("**First 12 rows (raw):**")
        st.dataframe(df_in.head(12))
        df_clean = _clean_df_raw(df_in)
        st.write("**Shape after cleaning:**", df_clean.shape)
        # Strict header check
        missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df_clean.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()
        # Reorder for traceability
        df_clean = df_clean[RAW_REQUIRED_COLUMNS]

    with st.spinner("Scoring shipments…"):
        try:
            df_scored = predict_batch(df_clean)  # adds columns: score, meets_threshold, risk_band
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
            st.stop()

    st.success(f"Scored {len(df_scored):,} shipments successfully.")
    st.dataframe(df_scored.head(25))
    st.download_button(
        label="⬇️ Download Scored CSV",
        data=df_scored.to_csv(index=False).encode("utf-8"),
        file_name="batch_scored.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("Risk Band Distribution")
    if "risk_band" in df_scored.columns:
        counts = df_scored["risk_band"].value_counts().rename_axis("band").reset_index(name="count")
        st.bar_chart(counts.set_index("band"))
    else:
        st.info("No `risk_band` column found in output. Check band mapping in the model loader.")

st.markdown("---")

# ------------------------------------------------------------------------------
# (B) Generate RAW demo dataset (and score it)
# ------------------------------------------------------------------------------
st.subheader("B) Generate Demo Dataset")
cA, cB, cC, cD = st.columns([2, 2, 1, 2])
with cA:
    n_rows = st.number_input("Rows", min_value=10, max_value=5000, value=200, step=10)
with cB:
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
with cC:
    gen = st.button("Generate Demo")
with cD:
    auto_score = st.checkbox("Auto-score", value=True)

if gen:
    demo = _demo_raw_frame(int(n_rows), int(seed))
    st.success(f"Generated {len(demo):,} demo rows.")
    st.dataframe(demo.head(25))
    st.download_button(
        label="⬇️ Download Demo CSV",
        data=demo.to_csv(index=False).encode("utf-8"),
        file_name="demo_batch_raw.csv",
        mime="text/csv"
    )
    if auto_score:
        with st.spinner("Scoring demo dataset…"):
            try:
                demo_scored = predict_batch(_clean_df_raw(demo))
            except Exception as e:
                st.error(f"Demo scoring failed: {e}")
                st.stop()
        st.success("Demo dataset scored.")
        st.dataframe(demo_scored.head(25))
        st.download_button(
            label="⬇️ Download Scored Demo CSV",
            data=demo_scored.to_csv(index=False).encode("utf-8"),
            file_name="demo_batch_scored.csv",
            mime="text/csv"
        )

st.markdown("---")

# ------------------------------------------------------------------------------
# (C) Generate blank RAW template
# ------------------------------------------------------------------------------
st.subheader("C) Generate Blank Template")
template = pd.DataFrame(columns=RAW_REQUIRED_COLUMNS)
st.info("This template contains the exact RAW columns expected by the app. Fill it out and upload in Section A.")
st.dataframe(template.head(5))
st.download_button(
    label="⬇️ Download Blank Template CSV",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="batch_template_raw.csv",
    mime="text/csv"
)

st.markdown("---")

# ------------------------------------------------------------------------------
# (D) Why was this classified High/Medium/Low?
# ------------------------------------------------------------------------------
st.subheader("D) Why was this classified this way?")

if df_scored is None or df_clean is None:
    st.info("Upload and score a file in Section A to enable per-row explanations.")
else:
    # Option to pick top-N by score or manual indices
    left, right = st.columns([1.2, 2])
    with left:
        top_n = st.number_input("Top N highest scores", min_value=1, max_value=min(50, len(df_scored)), value=min(5, len(df_scored)), step=1)
        default_indices = df_scored.sort_values("score", ascending=False).head(int(top_n)).index.tolist()
    with right:
        selected = st.multiselect(
            "Select specific row indices to inspect (defaults to Top N by score)",
            options=list(df_scored.index),
            default=default_indices
        )

    if not selected:
        st.info("Select at least one row to explain.")
    else:
        # Compute engineered features for the selected rows so users see exactly what hit the model
        engineered_subset = calculate_features(df_clean.loc[selected])  # transform RAW → engineered 32
        st.markdown("**Engineered features sent to the model (subset):**")
        st.dataframe(engineered_subset.head(len(selected)))

        # Try to get LightGBM feature_importances_ for a heuristic ranking
        try:
            model, meta_ = load_model()                                 # load model object + metadata
            importances = getattr(model, "feature_importances_", None)  # LightGBM/sklearn style
            if importances is not None and len(importances) == engineered_subset.shape[1]:
                # Normalize importances to [0,1] for weighting
                imp = pd.Series(importances, index=engineered_subset.columns).astype(float)
                imp_norm = (imp - imp.min()) / (imp.max() - imp.min() + 1e-9)

                st.markdown("**Heuristic driver ranking (NOT SHAP):**")
                for idx in selected:
                    st.markdown(f"- **Row {idx}** — score: {df_scored.loc[idx, 'score']:.4f} • band: {df_scored.loc[idx, 'risk_band']}")
                    row = engineered_subset.loc[idx]
                    # Heuristic “salience” = |value| * normalized_importance (numeric + one-hot)
                    # For categoricals (strings), we treat non-empty as 1.0
                    vals = row.copy()
                    for c in vals.index:
                        if isinstance(vals[c], str):
                            vals[c] = 1.0 if vals[c] else 0.0
                    salience = vals.abs() * imp_norm
                    topk = salience.sort_values(ascending=False).head(8)  # top 8 drivers
                    # Show a tidy table of top drivers
                    show = pd.DataFrame({
                        "feature": topk.index,
                        "value": row[topk.index].values,
                        "importance": imp_norm[topk.index].round(4).values,
                        "salience≈": topk.round(4).values
                    })
                    st.dataframe(show.reset_index(drop=True))
            else:
                st.info("Model importances unavailable or mismatched; showing engineered features only.")
        except Exception:
            st.info("Could not compute heuristic drivers from model importances. Showing engineered features only.")

        # Optional: render global SHAP PNGs if present (context, not per-row)
        art = _artifacts_dir()
        shap_imgs = [
            art / "shap_summary_lightgbm.png",
            art / "shap_importance_lightgbm.png",
            art / "shap_dependence_top_feature_lightgbm.png",
        ]
        any_found = False
        for p in shap_imgs:
            if p.exists():
                if not any_found:
                    st.markdown("**Global model context (static SHAP figures):**")
                st.image(str(p), use_column_width=True)
                any_found = True
        if not any_found:
            st.caption("Tip: Add SHAP PNGs to `artifacts/` to show global model context.")
