"""
Model Loading and Prediction Functions
- Loads best model from artifacts (prefers LightGBM file if present)
- Uses threshold & risk bands from utils.constants (driven by final_metadata.json)
"""

from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

from utils.constants import (
    ARTIFACTS_DIR,
    OPTIMAL_THRESHOLD,
    RISK_BANDS,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _assign_risk_level(score_percent: int) -> str:
    """Map 0..100 score to LOW/MEDIUM/HIGH using RISK_BANDS."""
    if score_percent <= RISK_BANDS["low_max"]:
        return "LOW"
    if score_percent <= RISK_BANDS["med_max"]:
        return "MEDIUM"
    return "HIGH"

@st.cache_resource
def load_model():
    """
    Load a trained model from artifacts.
    Prefers 'best_model_lightgbm.pkl', otherwise first 'best_model_*.pkl'.
    """
    try:
        # Prefer LightGBM if present
        preferred = ARTIFACTS_DIR / "best_model_lightgbm.pkl"
        if preferred.exists():
            return joblib.load(preferred)

        # Fallback: any best_model_*.pkl
        picks = sorted(ARTIFACTS_DIR.glob("best_model_*.pkl"))
        if picks:
            return joblib.load(picks[0])

        # Last resort: any model_*.pkl
        picks = sorted(ARTIFACTS_DIR.glob("model_*.pkl"))
        if picks:
            return joblib.load(picks[0])

        st.error(
            "⚠️ No model file found in `artifacts/`. "
            "Expected `best_model_lightgbm.pkl` or `best_model_*.pkl`."
        )
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


def predict_single(model, features_df: pd.DataFrame):
    """
    Predict for a single order.
    Returns dict with prediction, probability, risk_score (0..100), and risk_level.
    """
    try:
        if hasattr(model, "predict_proba"):
            prob_late = float(model.predict_proba(features_df)[:, 1][0])
        else:
            # Calibrated or decision_function-less fallback
            pred = float(model.predict(features_df)[0])
            prob_late = max(0.0, min(1.0, pred))

        # Threshold from metadata
        is_late = int(prob_late >= OPTIMAL_THRESHOLD)

        risk_score = int(round(prob_late * 100))
        risk_level = _assign_risk_level(risk_score)

        return {
            "prediction": is_late,
            "prediction_label": "Late" if is_late else "On-Time",
            "probability": prob_late,
            "risk_score": risk_score,
            "risk_level": risk_level,
        }
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None


def predict_batch(model, features_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Batch predictions. Returns DataFrame with:
    [Prediction, Late_Probability, Risk_Score, Risk_Level]
    """
    try:
        if hasattr(model, "predict_proba"):
            prob_late = model.predict_proba(features_df)[:, 1]
        else:
            raw = model.predict(features_df)
            prob_late = np.clip(raw.astype(float), 0.0, 1.0)

        preds = (prob_late >= OPTIMAL_THRESHOLD).astype(int)
        scores = (prob_late * 100).round().astype(int)
        levels = [_assign_risk_level(int(s)) for s in scores]

        return pd.DataFrame(
            {
                "Prediction": np.where(preds == 1, "Late", "On-Time"),
                "Late_Probability": prob_late,
                "Risk_Score": scores,
                "Risk_Level": levels,        # Title case (used by dashboards)
            }
        )
    except Exception as e:
        st.error(f"❌ Batch prediction error: {e}")
        return None


def get_feature_importance(model, feature_names):
    """
    Extract feature importance if available.
    """
    try:
        core = model
        if hasattr(model, "named_steps"):  # pipeline case
            core = model.named_steps.get("clf", model)

        if hasattr(core, "feature_importances_"):
            vals = np.asarray(core.feature_importances_).ravel()
        elif hasattr(core, "coef_"):
            vals = np.abs(np.asarray(core.coef_).ravel())
        else:
            st.warning("⚠️ Model does not expose feature importance.")
            return None

        return (
            pd.DataFrame({"Feature": feature_names, "Importance": vals})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as e:
        st.error(f"❌ Feature importance error: {e}")
        return None
