"""
Model Loading and Prediction Functions
"""

from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

from utils.constants import load_runtime_thresholds, ARTIFACTS_DIR

# Cache thresholds & bands once
_THRESH = load_runtime_thresholds()
_OPT_THR = __THRESH["THRESHOLD"]
_LOW_MAX = __THRESH["LOW_MAX"]
_MED_MAX = __THRESH["MED_MAX"]


@st.cache_resource
def load_model():
    """
    Load the trained model from artifacts folder (cached).
    Priority:
      1) best_model_lightgbm.pkl
      2) best_model_*.pkl
      3) model_*.pkl
    """
    try:
        priority = list(ARTIFACTS_DIR.glob("best_model_lightgbm.pkl"))
        if not priority:
            priority = list(ARTIFACTS_DIR.glob("best_model_*.pkl"))
        if not priority:
            priority = list(ARTIFACTS_DIR.glob("model_*.pkl"))

        if not priority:
            st.error(
                "⚠️ No model file found in artifacts/. "
                "Please copy your trained model to `apps/supply_chain_delay/artifacts/`."
            )
            return None

        model_path = priority[0]
        model = joblib.load(model_path)
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


def _risk_level_from_score(score_int: int) -> str:
    """Map 0–100 score to LOW/MEDIUM/HIGH using global bands."""
    if score_int < _LOW_MAX:
        return "LOW"
    if score_int < _MED_MAX:
        return "MEDIUM"
    return "HIGH"


def predict_single(model, features_df: pd.DataFrame) -> dict | None:
    """
    Predict a single order. Returns dict with:
      - prediction (0/1), prediction_label
      - probability (float 0..1), risk_score (0..100 int)
      - risk_level ('LOW'|'MEDIUM'|'HIGH')
    """
    try:
        if hasattr(model, "predict_proba"):
            prob_late = float(model.predict_proba(features_df)[0, 1])
        else:
            # Fallback (rare): use raw prediction as "probability"
            prob_late = float(model.predict(features_df)[0])

        prediction = int(prob_late >= _OPT_THR)
        risk_score = int(round(prob_late * 100))
        risk_level = _risk_level_from_score(risk_score)

        return {
            "prediction": prediction,
            "prediction_label": "Late" if prediction == 1 else "On-Time",
            "probability": prob_late,
            "risk_score": risk_score,
            "risk_level": risk_level,
        }

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None


def predict_batch(model, features_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Batch predictions. Returns DataFrame with:
      - Prediction ('On-Time'|'Late')
      - Late_Probability (0..1)
      - Risk_Score (0..100)
      - risk_level ('LOW'|'MEDIUM'|'HIGH')
    """
    try:
        if hasattr(model, "predict_proba"):
            prob_late = model.predict_proba(features_df)[:, 1]
        else:
            prob_late = model.predict(features_df).astype(float)

        predictions = (prob_late >= _OPT_THR).astype(int)
        risk_scores = np.rint(prob_late * 100).astype(int)
        risk_levels = np.where(
            risk_scores < _LOW_MAX, "LOW",
            np.where(risk_scores < _MED_MAX, "MEDIUM", "HIGH")
        )

        return pd.DataFrame({
            "Prediction": np.where(predictions == 1, "Late", "On-Time"),
            "Late_Probability": prob_late,
            "Risk_Score": risk_scores,
            "risk_level": risk_levels,  # lowercase col name expected by visuals
        })

    except Exception as e:
        st.error(f"❌ Batch prediction error: {str(e)}")
        return None


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame | None:
    """Feature importance (works for tree & linear models)."""
    try:
        m = model.named_steps.get("clf", model) if hasattr(model, "named_steps") else model

        if hasattr(m, "feature_importances_"):
            imp = m.feature_importances_
        elif hasattr(m, "coef_"):
            imp = np.abs(m.coef_[0])
        else:
            st.warning("⚠️ Model does not expose feature importance.")
            return None

        return (
            pd.DataFrame({"Feature": feature_names, "Importance": imp})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as e:
        st.error(f"❌ Error extracting feature importance: {str(e)}")
        return None
