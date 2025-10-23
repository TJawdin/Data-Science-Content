# pages/0_🧪_Diagnostics.py
# Purpose: One-click health check for env, artifacts, model, FE, and prediction.
# Every line is commented so it’s easy to tweak.

from __future__ import annotations  # allowed at top of file
import importlib                   # to read versions of libs dynamically
from pathlib import Path           # robust path handling
from typing import Any, Dict       # typing helpers

import numpy as np                 # numeric ops
import pandas as pd                # dataframes
import streamlit as st             # UI

# App utilities (we assume your updated utils are in place)
from utils.model_loader import load_metadata, load_model, predict_single, predict_batch
from utils.feature_engineering import calculate_features

# ----------------------------- Page config ----------------------------- #
st.set_page_config(page_title="Diagnostics", page_icon="🧪", layout="wide")
st.title("🧪 App Diagnostics")
st.caption("Environment → Artifacts → Model → Feature Engineering → Inference")

st.markdown("---")

# ----------------------------- Helpers ----------------------------- #

def _app_root() -> Path:
    """apps/supply_chain_delay root."""
    return Path(__file__).resolve().parents[1]

def _art_dir() -> Path:
    """./artifacts directory."""
    return _app_root() / "artifacts"

def _probe_version(pkg: str) -> str:
    """Safely import a package and return its version or a friendly marker."""
    try:
        m = importlib.import_module(pkg)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"not importable: {e}"

def _demo_raw_row() -> Dict[str, Any]:
    """A minimal RAW row matching Single/Batch schema."""
    ts = pd.Timestamp("2017-08-02T10:15:00")
    return {
        "order_purchase_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "estimated_delivery_date": (ts + pd.Timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%S"),
        "sum_price": 120.00,
        "sum_freight": 25.00,
        "n_items": 2,
        "n_sellers": 1,
        "payment_type": "credit_card",
        "max_installments": 1,
        "mode_category": "housewares",
        "customer_city": "sao paulo",
        "customer_state": "SP",
    }

# ----------------------------- Section 1: Environment ----------------------------- #
st.subheader("1) Environment")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    import sys
    st.metric("Python", ".".join(map(str, sys.version_info[:3])))
with c2:
    st.metric("numpy", _probe_version("numpy"))
with c3:
    st.metric("pandas", _probe_version("pandas"))
with c4:
    st.metric("scikit-learn", _probe_version("sklearn"))
with c5:
    st.metric("lightgbm", _probe_version("lightgbm"))

st.caption("Tip: Python should be 3.11.x, sklearn pinned to 1.6.1 to match the pickle.")

st.markdown("---")

# ----------------------------- Section 2: Artifacts & Metadata ----------------------------- #
st.subheader("2) Artifacts & Metadata")

art = _art_dir()
meta = None
err_meta = None

# Show expected files
expected = [
    art / "best_model_lightgbm.pkl",
    art / "final_metadata.json",
    art / "optimal_threshold_lightgbm.txt",
]
grid = st.columns(len(expected))
for col, p in zip(grid, expected):
    with col:
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        st.write(f"**{p.name}**")
        st.write("Exists:", "✅" if exists else "❌")
        st.write("Size:", f"{size:,} bytes")

# Load metadata
try:
    meta = load_metadata()
    st.success("Loaded final_metadata.json")
    show = {
        "best_model": meta.get("best_model"),
        "best_model_auc": meta.get("best_model_auc"),
        "optimal_threshold": meta.get("optimal_threshold"),
        "risk_bands": meta.get("risk_bands"),
        "n_features": meta.get("n_features"),
        "training_date": meta.get("training_date"),
    }
    st.json(show)
except Exception as e:
    err_meta = e
    st.error(f"Failed to load metadata: {e}")

st.markdown("---")

# ----------------------------- Section 3: Model Load ----------------------------- #
st.subheader("3) Model Load")

model = None
err_model = None
try:
    model, _ = load_model()
    st.success(f"Model loaded: {type(model)}")
    # If sklearn pipeline, show step names
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            steps = [name for name, _ in model.steps]
            st.write("Pipeline steps:", " → ".join(steps))
    except Exception:
        pass
except Exception as e:
    err_model = e
    st.exception(e)

st.markdown("---")

# ----------------------------- Section 4: Feature Engineering ----------------------------- #
st.subheader("4) Feature Engineering")

raw = _demo_raw_row()
df_raw = pd.DataFrame([raw])
st.write("RAW row preview:")
st.dataframe(df_raw)

try:
    X = calculate_features(df_raw)
    st.success(f"Engineered features shape: {X.shape}")
    st.dataframe(X.head(1))
    if meta:
        expected_n = int(meta.get("n_features", 32))
        if X.shape[1] != expected_n:
            st.warning(f"Expected {expected_n} features per metadata but got {X.shape[1]}. Check FE → feature order/names.")
except Exception as e:
    st.error(f"Feature engineering failed: {e}")
    st.stop()

st.markdown("---")

# ----------------------------- Section 5: Inference (Single & Batch) ----------------------------- #
st.subheader("5) Inference")

# Single
try:
    res_single = predict_single(raw)  # let model_loader call FE internally
    st.success("Single prediction succeeded.")
    st.json(res_single)
except Exception as e:
    st.error(f"Single prediction failed: {e}")
    st.stop()

# Batch (2 rows)
try:
    demo2 = pd.DataFrame([raw, raw])
    res_batch = predict_batch(demo2)
    st.success(f"Batch prediction succeeded. Shape: {res_batch.shape}")
    st.dataframe(res_batch.head(5))
    # Sanity: required output columns
    for c in ("score", "meets_threshold", "risk_band"):
        if c not in res_batch.columns:
            st.warning(f"Missing expected output column: {c}")
except Exception as e:
    st.error(f"Batch prediction failed: {e}")
    st.stop()

st.markdown("---")

st.info("If any step above shows ❌ or an exception, that’s the first link to fix. Share that section’s error and we’ll patch it.")
