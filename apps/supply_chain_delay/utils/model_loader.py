# utils/model_loader.py
# Purpose: Load model + metadata, resolve artifact paths, and expose
#          predict_single / predict_batch that accept RAW or engineered inputs.
#
# References:
# - Python stdlib: https://docs.python.org/3/library/
# - pandas: https://pandas.pydata.org/docs/
# - scikit-learn: https://scikit-learn.org/stable/
#
# Notes:
# - This module never trusts absolute Windows paths in metadata. It always
#   resolves artifacts relative to the repo's ./artifacts directory.
# - It uses utils.feature_engineering.calculate_features to transform RAW
#   inputs into the exact engineered vector expected by the model.

from __future__ import annotations  # postpone evaluation of annotations
from pathlib import Path            # robust filesystem paths
from typing import Any, Dict, Tuple # typing helpers
import json                         # parse metadata json
import joblib                       # load pickled model
import pandas as pd                 # DataFrame handling
import numpy as np                  # numeric ops

# Feature engineering to convert RAW → engineered (32 features)
from utils.feature_engineering import calculate_features  # create exact model features


# ----------------------------- Paths & loaders ----------------------------- #

def _app_root() -> Path:
    """Return the app root directory (apps/supply_chain_delay)."""
    # utils/ → apps/supply_chain_delay/utils → parent() == apps/supply_chain_delay
    return Path(__file__).resolve().parents[1]

def _artifacts_dir() -> Path:
    """Return the artifacts directory path (./artifacts)."""
    return _app_root() / "artifacts"

def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file into a dict with UTF-8 handling and friendly errors."""
    # Always open with UTF-8 to avoid encoding gotchas
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _resolve_artifact(filename_hint: str, default_name: str) -> Path:
    """
    Resolve an artifact path. Ignore absolute paths from metadata and prefer
    ./artifacts/<default_name>.
    """
    # Construct the expected local path in ./artifacts
    local = _artifacts_dir() / default_name
    if local.exists():
        return local  # prefer repo-local file
    # If not found locally but a hint was provided, try to use just the basename under artifacts
    if filename_hint:
        # Extract basename from any hint (e.g., Windows absolute path)
        hint_name = Path(filename_hint).name
        candidate = _artifacts_dir() / hint_name
        if candidate.exists():
            return candidate  # use candidate if present
    # Last resort: return the intended local path (caller can handle not found)
    return local


# ----------------------------- Public metadata/model APIs ----------------------------- #

def load_metadata() -> Dict[str, Any]:
    """
    Load final metadata from ./artifacts/final_metadata.json with safe defaults.
    """
    # Build path to final_metadata.json inside artifacts
    meta_path = _artifacts_dir() / "final_metadata.json"
    # Provide defaults in case the file is missing or malformed
    defaults = {
        "best_model": "LightGBM",
        "best_model_auc": 0.7890,
        "best_model_precision": 0.304,
        "best_model_recall": 0.443,
        "best_model_f1": 0.361,
        "optimal_threshold": 0.50,
        "risk_bands": {"low_max": 30, "med_max": 67},
        "n_features": 32,
        "n_samples_train": 0,
        "n_samples_test": 0,
        "training_date": "N/A",
        "artifact_files": {},
    }
    try:
        # Try reading the JSON metadata from disk
        meta = _read_json(meta_path)
        # Merge with defaults to ensure required keys exist
        for k, v in defaults.items():
            meta.setdefault(k, v)
        return meta  # return the merged metadata
    except Exception:
        # Fallback to defaults if reading fails
        return defaults

def load_model():
    meta = load_metadata()
    hints = meta.get("artifact_files", {})
    model_path = _resolve_artifact(str(hints.get("model", "")), "best_model_lightgbm.pkl")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        import sys
        pyver = ".".join(map(str, sys.version_info[:3]))
        raise RuntimeError(
            "Failed to load model artifact via joblib (likely Python / library mismatch).\n"
            f"- Running Python: {pyver}\n"
            "- The model artifact was probably saved under Python 3.11 with specific library versions.\n"
            "Fix:\n"
            "  • Switch the app runtime to Python 3.11 (see runtime.txt/.python-version + host setting), OR\n"
            "  • Re-save the model under the current Python version with matching LightGBM/sklearn.\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    return model, meta

    # Load the model using joblib (works for LightGBM pickles and sklearn pipelines)
    model = joblib.load(model_path)
    return model, meta  # return both for caller convenience


# ----------------------------- Threshold & band helpers ----------------------------- #

def _risk_band_from_prob(p: float, bands: Dict[str, int]) -> str:
    """
    Map probability (0..1) to a band label using metadata percent cutpoints.
    """
    # Convert probability to percent for comparison with band cut points
    pct = float(p) * 100.0
    # Read low and medium max thresholds with safe defaults
    low_max = int(bands.get("low_max", 30))
    med_max = int(bands.get("med_max", 67))
    # Assign band based on percent thresholds
    if pct <= low_max:
        return "Low"
    if pct <= med_max:
        return "Medium"
    return "High"

def _apply_threshold_and_bands(scores: np.ndarray, meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Given an array of probabilities, return a DataFrame with:
      - score (0..1)
      - meets_threshold (bool)
      - risk_band (str)
    """
    # Ensure scores are 1-D numpy array of floats
    s = np.asarray(scores, dtype=float).reshape(-1)
    # Read threshold and bands from metadata with safe defaults
    thr = float(meta.get("optimal_threshold", 0.5))
    bands = meta.get("risk_bands", {"low_max": 30, "med_max": 67})
    # Build a list of dicts with computed fields
    out_rows = []
    for p in s:
        out_rows.append({
            "score": float(p),                                           # predicted probability
            "meets_threshold": bool(p >= thr),                           # True if p >= threshold
            "risk_band": _risk_band_from_prob(p, bands),                 # band label
        })
    # Convert the list of dicts into a DataFrame
    return pd.DataFrame(out_rows)


# ----------------------------- Public prediction APIs ----------------------------- #

def predict_single(raw_or_engineered: Dict[str, Any] | pd.Series | pd.DataFrame) -> Dict[str, Any]:
    """
    Score a single example, accepting either RAW inputs (dict/Series)
    or a 1-row engineered DataFrame.

    Returns a dict with keys:
      - score (float 0..1)
      - meets_threshold (bool)
      - risk_band (str)
    """
    # Coerce incoming data into a 1-row DataFrame
    if isinstance(raw_or_engineered, pd.DataFrame):
        df_in = raw_or_engineered.copy()                     # already a DataFrame
    elif isinstance(raw_or_engineered, pd.Series):
        df_in = pd.DataFrame([raw_or_engineered.to_dict()])  # wrap Series
    else:
        df_in = pd.DataFrame([dict(raw_or_engineered)])      # wrap dict

    # If the DataFrame doesn't look engineered, transform RAW → engineered
    # We detect this by checking a few known engineered columns
    needs_fe = True                                          # assume we need FE
    engineered_probe_cols = {"n_items", "sum_price", "purch_year"}  # probe features
    if engineered_probe_cols.issubset(set(df_in.columns)):
        needs_fe = False                                     # looks engineered already

    # Apply feature engineering if needed
    X = calculate_features(df_in) if needs_fe else df_in     # produce engineered features

    # Load model and metadata once per call
    model, meta = load_model()                                # model object and metadata
    # Predict probabilities (assume binary model with predict_proba)
    proba = model.predict_proba(X)[:, 1]                      # take the positive class prob
    # Apply threshold and bands to wrap outputs neatly
    out = _apply_threshold_and_bands(proba, meta).iloc[0].to_dict()
    return out  # return the result dict

def predict_batch(raw_or_engineered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a batch DataFrame. Accepts RAW columns (we'll FE them) or already-engineered.
    Returns a new DataFrame = input columns + ['score','meets_threshold','risk_band'].
    """
    # Make a copy to avoid mutating caller's DataFrame
    df_in = raw_or_engineered_df.copy()

    # Decide whether feature engineering is needed (same probe trick as single)
    needs_fe = True
    engineered_probe_cols = {"n_items", "sum_price", "purch_year"}
    if engineered_probe_cols.issubset(set(df_in.columns)):
        needs_fe = False

    # Transform RAW → engineered if required
    X = calculate_features(df_in) if needs_fe else df_in

    # Load model + metadata and predict probabilities
    model, meta = load_model()
    proba = model.predict_proba(X)[:, 1]

    # Wrap into a DataFrame with threshold/bands
    wrapped = _apply_threshold_and_bands(proba, meta)

    # Concatenate original input with the new result columns (align by index)
    out = pd.concat([df_in.reset_index(drop=True), wrapped], axis=1)
    return out  # return the scored DataFrame
