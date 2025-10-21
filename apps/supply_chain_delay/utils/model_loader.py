"""
Model loading, threshold loading, prediction, and feature importance.
Dynamic risk bands are derived from the tuned threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# Paths / discovery
# ----------------------------
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

# File names we may encounter
FINAL_META = ARTIFACTS_DIR / "final_metadata.json"
OPT_THR_TXT_LGB = ARTIFACTS_DIR / "optimal_threshold_lightgbm.txt"
OPT_THR_TXT = ARTIFACTS_DIR / "best_threshold.txt"           # legacy fallback
ANY_MODEL_GLOB = "best_model_*.pkl"


# ----------------------------
# Caching loaders
# ----------------------------
@st.cache_resource
def load_model() -> Any:
    """Load the trained model (best_* .pkl) from artifacts/."""
    # Prefer LightGBM file name if present
    candidates = list(ARTIFACTS_DIR.glob(ANY_MODEL_GLOB))
    if not candidates:
        raise FileNotFoundError(
            "No model file found in artifacts/. Expected pattern 'best_model_*.pkl'."
        )
    # If multiple, prefer lightgbm, else first
    lightgbm_first = [p for p in candidates if "lightgbm" in p.stem.lower()]
    model_path = lightgbm_first[0] if lightgbm_first else candidates[0]
    model = joblib.load(model_path)
    return model


@st.cache_data
def load_metadata() -> Dict[str, Any]:
    """Load final_metadata.json if available; else return a minimal dict."""
    if FINAL_META.exists():
        with open(FINAL_META, "r") as f:
            return json.load(f)
    return {
        "best_model": "LightGBM",
        "best_model_auc": None,
        "n_features": None,
        "n_samples_train": None,
        "training_date": None,
        "decision_threshold": None,
        "notes": "final_metadata.json not found; using defaults."
    }


@st.cache_data
def load_threshold(default: float = 0.50) -> float:
    """
    Load the tuned decision threshold.
    Priority:
      1) final_metadata.json -> 'decision_threshold'
      2) optimal_threshold_lightgbm.txt
      3) best_threshold.txt (legacy)
      4) default (0.50)
    """
    meta = load_metadata()
    thr = meta.get("decision_threshold", None)
    if isinstance(thr, (int, float)) and 0.0 <= float(thr) <= 1.0:
        return float(thr)

    for p in (OPT_THR_TXT_LGB, OPT_THR_TXT):
        if p.exists():
            try:
                return float(p.read_text().strip())
            except Exception:
                pass

    return float(default)


# ----------------------------
# Risk band helper (derived from threshold)
# ----------------------------
def risk_band(prob: float, threshold: float) -> str:
    """
    3-band scheme relative to tuned threshold:
      LOW:     p < 0.50 * threshold
      MEDIUM:  0.50 * threshold ≤ p < threshold
      HIGH:    p ≥ threshold
    """
    low_cut = 0.50 * threshold
    if prob < low_cut:
        return "LOW"
    if prob < threshold:
        return "MEDIUM"
    return "HIGH"


def risk_palette(level: str) -> str:
    return {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(level, "gray")


def risk_band_text(threshold: float) -> str:
    return (
        f"Risk bands are derived from the tuned threshold (τ = {threshold:.3f}):  "
        f"LOW: p < {0.5*threshold:.3f} • "
        f"MEDIUM: {0.5*threshold:.3f} ≤ p < {threshold:.3f} • "
        f"HIGH: p ≥ {threshold:.3f}"
    )


# ----------------------------
# Prediction APIs
# ----------------------------
def _proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    # very rare case
    preds = model.predict(X)
    # if predict already returns prob, ensure in [0,1]
    preds = np.asarray(preds, dtype=float)
    if preds.min() >= 0 and preds.max() <= 1:
        return preds
    # map labels {0,1} to [0,1]
    return (preds > 0).astype(float)


def predict_single(model: Any, features_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Predict for a single order.
    Returns: dict with prediction, label, probability, risk_score, risk_level, risk_color.
    """
    try:
        threshold = load_threshold()
        p = float(_proba(model, features_df)[0])
        pred = int(p >= threshold)
        level = risk_band(p, threshold)
        color = risk_palette(level)
        return {
            "prediction": pred,
            "prediction_label": "Late" if pred == 1 else "On-Time",
            "probability": p,
            "risk_score": int(round(p * 100)),
            "risk_level": level,
            "risk_color": color,
            "threshold": threshold,
        }
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None


def predict_batch(model: Any, features_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Batch predictions. Returns a DataFrame with Prediction, Late_Probability, Risk_Score, risk_level.
    """
    try:
        threshold = load_threshold()
        proba = _proba(model, features_df)
        preds = (proba >= threshold).astype(int)
        levels = [risk_band(p, threshold) for p in proba]

        out = pd.DataFrame({
            "Prediction": np.where(preds == 1, "Late", "On-Time"),
            "Late_Probability": proba,
            "Risk_Score": (proba * 100).round(0).astype(int),
            "risk_level": levels,
        })
        return out
    except Exception as e:
        st.error(f"❌ Batch prediction error: {e}")
        return None


# ----------------------------
# Feature importance (for tree / linear)
# ----------------------------
def get_feature_importance(model: Any, feature_names: list[str]) -> Optional[pd.DataFrame]:
    try:
        actual = model.named_steps.get("clf") if hasattr(model, "named_steps") else model

        if hasattr(actual, "feature_importances_"):
            imp = actual.feature_importances_
        elif hasattr(actual, "coef_"):
            imp = np.abs(actual.coef_[0])
        else:
            st.warning("⚠️ Model does not expose feature importance.")
            return None

        df = pd.DataFrame({"Feature": feature_names, "Importance": imp})
        return df.sort_values("Importance", ascending=False).reset_index(drop=True)
    except Exception as e:
        st.error(f"❌ Feature importance error: {e}")
        return None
