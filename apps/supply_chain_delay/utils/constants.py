"""
Global constants for model cut-points & thresholds.
This is the single source of truth for the Streamlit app.
"""

from __future__ import annotations
from pathlib import Path
import json

# --- DEFAULTS (must match your current notebook Section 6.3) ---
# LightGBM winner @ F1-optimal threshold
THRESHOLD = 0.1966            # 19.66%
LOW_MAX = 12                  # 0–11%  -> LOW
MED_MAX = 30                  # 12–29% -> MEDIUM ; 30%+ -> HIGH

# Locations
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
FINAL_META = ARTIFACTS_DIR / "final_metadata.json"


def _coerce_float(v, fallback):
    try:
        return float(v)
    except Exception:
        return fallback


def load_runtime_thresholds() -> dict:
    """
    Load thresholds from final_metadata.json if present; otherwise use defaults.
    The JSON should contain: "decision_threshold": 0.1966
    """
    t = THRESHOLD
    low = LOW_MAX
    med = MED_MAX

    try:
        if FINAL_META.exists():
            meta = json.loads(FINAL_META.read_text(encoding="utf-8"))
            # Prefer explicit key; fall back to any legacy fields if needed
            t = _coerce_float(meta.get("decision_threshold", t), t)

            # Optional: allow overriding bands (else keep defaults)
            bands = meta.get("risk_bands", {}) or {}
            low = int(bands.get("low_max", low))
            med = int(bands.get("med_max", med))
    except Exception:
        # Silently fall back to defaults
        pass

    return {
        "THRESHOLD": t,
        "THRESHOLD_PCT": t * 100.0,
        "LOW_MAX": int(low),
        "MED_MAX": int(med),
    }
