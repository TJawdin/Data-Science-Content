"""
Global constants & metadata loader
Reads artifacts/final_metadata.json and exposes:
- OPTIMAL_THRESHOLD (float 0..1)
- RISK_BANDS (dict with low/med cutpoints in 0..100 scale)
- MODEL_METADATA (full dict)
- FRIENDLY_FEATURE_NAMES (business-facing labels)
"""

from __future__ import annotations
import json
from pathlib import Path

# ---------------------------------------------------------------------
# Locate artifacts and load final metadata (with safe defaults)
# ---------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
FINAL_META_PATH = ARTIFACTS_DIR / "final_metadata.json"

_DEFAULT_META = {
    "best_model": "LightGBM",
    "best_model_auc": 0.78,
    "best_model_precision": 0.30,
    "best_model_recall": 0.44,
    "best_model_f1": 0.36,
    # if the file is missing, default to a conservative 0.50 threshold
    "optimal_threshold": 0.50,
    # risk bands in percent; will be normalized below
    "risk_bands": {"low_max": 30, "med_max": 67},
    "n_features": 29,
    "n_samples_train": 0,
    "n_samples_test": 0,
    "training_date": "N/A",
}

try:
    MODEL_METADATA = json.loads(FINAL_META_PATH.read_text(encoding="utf-8"))
except Exception:
    MODEL_METADATA = dict(_DEFAULT_META)

# ---------------------------------------------------------------------
# Threshold & risk bands
# ---------------------------------------------------------------------
OPTIMAL_THRESHOLD: float = float(MODEL_METADATA.get("optimal_threshold", 0.50))

# Bands are expressed in percent (0..100). Ensure med_max ~= threshold% so
# the High band starts at the operating threshold.
rb = MODEL_METADATA.get("risk_bands", {}) or {}
_low_max = int(rb.get("low_max", 30))
_med_max = int(rb.get("med_max", round(OPTIMAL_THRESHOLD * 100)))

# Make sure ordering is valid: 0 <= low < med <= 100
_low_max = max(0, min(_low_max, 100))
_med_max = max(_low_max + 1, min(_med_max, 100))

RISK_BANDS = {
    "low_max": _low_max,            # e.g., 30  → LOW: 0..30
    "med_max": _med_max,            # e.g., 67  → MED: 31..67, HIGH: >= 67
}

# ---------------------------------------------------------------------
# Business-friendly feature labels (used across app & plots)
# ---------------------------------------------------------------------
FRIENDLY_FEATURE_NAMES = {
    # Order Complexity
    "num_items": "Number of Items",
    "num_sellers": "Number of Sellers",
    "num_products": "Number of Products",
    "is_multi_seller": "Multi-Seller Order",
    "is_multi_item": "Multi-Item Order",

    # Financial
    "total_order_value": "Total Order Value ($)",
    "avg_item_price": "Average Item Price ($)",
    "max_item_price": "Highest Item Price ($)",
    "total_shipping_cost": "Total Shipping Cost ($)",
    "avg_shipping_cost": "Avg Shipping Cost ($)",
    "weight_to_price_ratio": "Weight / Price Ratio",
    "shipping_cost_per_km": "Shipping Cost per KM ($)",

    # Physical
    "total_weight_g": "Total Weight (g)",
    "avg_weight_g": "Average Weight (g)",
    "max_weight_g": "Heaviest Item (g)",
    "avg_length_cm": "Avg Length (cm)",
    "avg_height_cm": "Avg Height (cm)",
    "avg_width_cm": "Avg Width (cm)",
    "avg_product_volume_cm3": "Avg Product Volume (cm³)",

    # Geographic
    "avg_shipping_distance_km": "Shipping Distance (km)",
    "max_shipping_distance_km": "Max Shipping Distance (km)",
    "is_cross_state": "Cross-State Shipping",

    # Temporal
    "order_weekday": "Order Day of Week",
    "order_month": "Order Month",
    "order_hour": "Order Hour",
    "is_weekend_order": "Weekend Order",
    "is_holiday_season": "Holiday Season Order",
    "is_rush_order": "Rush Order (<7 days)",

    # Time Estimation
    "estimated_days": "Estimated Delivery Days",
}
