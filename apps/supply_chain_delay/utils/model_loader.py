# utils/model_loader.py
# Purpose: Centralized, metadata-driven model loading and prediction helpers for the Streamlit app.
# Notes:
# - Reads ./artifacts/final_metadata.json
# - Loads LightGBM (or compatible sklearn) model
# - Applies optimal_threshold (0–1) and risk_bands (0–100)
# - Exposes predict_single() and predict_batch() with strong input validation
# - Gracefully handles absolute Windows paths in metadata by resolving to repo-local ./artifacts

from __future__ import annotations  # enable postponed evaluation of annotations for type hints
import json                         # JSON parsing for metadata file
import joblib                       # model loading (sklearn/lightgbm via joblib)
from pathlib import Path            # OS-agnostic path handling
from typing import Dict, Tuple      # typing for dicts/tuples
import pandas as pd                 # DataFrame inputs for batch predictions

# Try to import feature engineering; handle absence gracefully
try:
    from utils.feature_engineering import calculate_features  # project-specific feature function
    _HAS_FE = True                                            # flag that FE is available
except Exception:
    _HAS_FE = False                                           # set flag false if import fails

# ---------- helpers to load/normalize metadata ----------

def _artifacts_dir() -> Path:
    """Return the absolute path to the local ./artifacts directory next to app.py."""
    return Path(__file__).resolve().parents[1] / "artifacts"  # utils/.. -> repo/apps/supply_chain_delay/artifacts

def _read_metadata() -> Dict:
    """Load final_metadata.json and return as a dict with minimal validation."""
    meta_path: Path = _artifacts_dir() / "final_metadata.json"                              # compute metadata path
    with meta_path.open("r", encoding="utf-8") as f:                                       # open JSON file
        data: Dict = json.load(f)                                                          # parse JSON
    # Back-compat normalize: allow legacy keys if present
    if "precision" in data and "best_model_precision" not in data:                         # map legacy precision
        data["best_model_precision"] = data["precision"]
    if "recall" in data and "best_model_recall" not in data:                               # map legacy recall
        data["best_model_recall"] = data["recall"]
    if "f1" in data and "best_model_f1" not in data:                                       # map legacy f1
        data["best_model_f1"] = data["f1"]
    # Basic presence checks and defaults
    data.setdefault("best_model", "LightGBM")                                              # default model name
    data.setdefault("best_model_auc", 0.0)                                                 # default metrics
    data.setdefault("best_model_precision", 0.0)
    data.setdefault("best_model_recall", 0.0)
    data.setdefault("best_model_f1", 0.0)
    data.setdefault("n_features", 0)                                                       # default counts
    data.setdefault("n_samples_train", 0)
    data.setdefault("n_samples_test", 0)
    data.setdefault("training_date", "YYYY-MM-DD")                                         # default date
    # Threshold + risk bands sanity defaults
    data.setdefault("optimal_threshold", 0.5)                                              # probability threshold
    data.setdefault("risk_bands", {"low_max": 30, "med_max": 67})                          # percent-based bands
    # Ensure nested structure for artifact paths
    data.setdefault("artifact_files", {})                                                  # ensure dict
    return data                                                                            # return normalized dict

def _resolve_artifact_path(path_str: str) -> Path:
    """
    Resolve artifact path from metadata:
    - If metadata contains an absolute Windows path, use its basename inside local ./artifacts.
    - If path is relative, resolve relative to ./artifacts.
    """
    artifacts = _artifacts_dir()                                                           # locate artifacts dir
    p = Path(path_str)                                                                     # create Path object
    if p.is_absolute():                                                                    # absolute path case
        return artifacts / p.name                                                          # use local artifacts with same filename
    return (artifacts / p).resolve()                                                       # join relative to artifacts dir

# ---------- public API ----------

def load_metadata() -> Dict:
    """Public wrapper for reading metadata dict."""
    return _read_metadata()                                                                # return metadata dict

def load_model() -> Tuple[object, Dict]:
    """
    Load the serialized model defined in metadata and return (model, metadata).
    Model must support predict_proba(X) or decision_function(X).
    """
    meta: Dict = _read_metadata()                                                          # load metadata
    model_path_in_meta: str = meta.get("artifact_files", {}).get("model", "best_model_lightgbm.pkl")  # get model path
    model_path: Path = _resolve_artifact_path(model_path_in_meta)                          # resolve to local path
    model = joblib.load(model_path)                                                        # load serialized model
    return model, meta                                                                     # return both model and metadata

def _score_to_band(score: float, low_max: int, med_max: int) -> str:
    """Map probability (0–1) to 'Low'|'Medium'|'High' using percent cut points."""
    pct: float = float(score) * 100.0                                                      # convert probability to percent
    if pct <= int(low_max):                                                                # compare to low max
        return "Low"                                                                       # within low band
    if pct <= int(med_max):                                                                # within medium band
        return "Medium"                                                                    # return medium
    return "High"                                                                          # otherwise high

def _proba_from_model(model: object, X: pd.DataFrame) -> pd.Series:
    """
    Get positive-class probabilities from a scikit-learn style model.
    Supports predict_proba and decision_function.
    """
    if hasattr(model, "predict_proba"):                                                    # if model supports predict_proba
        proba = model.predict_proba(X)[:, -1]                                              # take positive class column
        return pd.Series(proba, index=X.index, name="score")                               # return as Series
    if hasattr(model, "decision_function"):                                                # fallback: decision_function
        df = model.decision_function(X)                                                    # raw scores
        # Min-max scale to 0–1 defensively (monotonic, not calibrated)
        s = pd.Series(df, index=X.index).astype(float)                                     # to Series
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)                                     # safe scaling
        s.name = "score"                                                                    # set name
        return s                                                                           # return scaled
    raise AttributeError("Model lacks predict_proba/decision_function required for probabilities.")  # error

def _clean_scores(s: pd.Series) -> pd.Series:
    """Coerce to numeric, drop NaNs, and clip to [0,1]."""
    s = pd.to_numeric(s, errors="coerce")                                                  # coerce to numeric
    s = s.clip(0.0, 1.0)                                                                    # clip to [0,1]
    return s                                                                               # return cleaned series

def predict_single(raw_features: Dict) -> Dict:
    """
    Predict for a single example represented as a dict of raw inputs.
    Returns a dict with score, meets_threshold, risk_band, and echo of inputs.
    """
    model, meta = load_model()                                                             # load model + metadata
    thr: float = float(meta.get("optimal_threshold", 0.5))                                 # read threshold
    low_max: int = int(meta.get("risk_bands", {}).get("low_max", 30))                      # low band %
    med_max: int = int(meta.get("risk_bands", {}).get("med_max", 67))                      # med band %
    X = pd.DataFrame([raw_features])                                                       # create single-row DataFrame
    if _HAS_FE:                                                                            # if feature engineering exists
        try:
            X = calculate_features(X)                                                      # transform to model features
        except Exception:
            pass                                                                           # fail open: attempt raw if FE fails
    score_series = _proba_from_model(model, X)                                             # get probabilities
    score = float(_clean_scores(score_series).iloc[0])                                     # clean and extract single float
    meets = bool(score >= thr)                                                             # compare to threshold
    band = _score_to_band(score, low_max, med_max)                                         # map to risk band
    return {                                                                               # return rich result
        "inputs": raw_features,
        "score": score,
        "meets_threshold": meets,
        "risk_band": band,
        "threshold": thr,
        "bands": {"low_max": low_max, "med_max": med_max}
    }

def predict_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Batch predict for a DataFrame of raw inputs.
    - Safely applies feature engineering when available.
    - Produces columns: score, meets_threshold, risk_band.
    - Preserves original index and original columns.
    """
    model, meta = load_model()                                                             # load model + metadata
    thr: float = float(meta.get("optimal_threshold", 0.5))                                 # threshold
    low_max: int = int(meta.get("risk_bands", {}).get("low_max", 30))                      # low band %
    med_max: int = int(meta.get("risk_bands", {}).get("med_max", 67))                      # med band %
    X = df_raw.copy()                                                                      # shallow copy for safety
    if _HAS_FE:                                                                            # apply feature engineering if present
        try:
            X = calculate_features(X)                                                      # transform raw → model features
        except Exception:
            # If FE fails, proceed with raw; model may still accept (or fail clearly)
            pass
    scores = _proba_from_model(model, X)                                                   # positive-class probs
    scores = _clean_scores(scores)                                                         # clean and clip
    meets = scores.ge(thr)                                                                 # boolean threshold flag
    bands = scores.apply(lambda s: _score_to_band(float(s), low_max, med_max))             # band mapping
    out = df_raw.copy()                                                                     # start with original columns
    out["score"] = scores.values                                                            # attach score column
    out["meets_threshold"] = meets.values                                                   # attach meets_threshold
    out["risk_band"] = bands.values                                                         # attach risk band
    return out                                                                              # return annotated DataFrame
