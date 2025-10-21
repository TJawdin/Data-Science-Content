"""
Feature Engineering Functions
- Produces EXACT 29 features expected by the trained model
- Attempts to read feature order from artifacts/final_metadata.json
- Falls back to the canonical 29-feature order below if metadata not found
"""

from __future__ import annotations

import json
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, Any, Iterable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Paths / metadata (so app stays aligned with the notebook)
# -----------------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
FINAL_META = ARTIFACTS_DIR / "final_metadata.json"

# Canonical 29-feature order (fallback if metadata not present)
FALLBACK_FEATURE_ORDER: list[str] = [
    # Order Complexity
    'num_items',
    'num_sellers',
    'num_products',
    'is_multi_seller',
    'is_multi_item',

    # Financial
    'total_order_value',
    'avg_item_price',
    'max_item_price',
    'total_shipping_cost',
    'avg_shipping_cost',

    # Derived ratios
    'weight_to_price_ratio',
    'shipping_cost_per_km',   # <- position 12 as trained

    # Physical
    'total_weight_g',         # <- 13
    'avg_weight_g',
    'max_weight_g',
    'avg_length_cm',
    'avg_height_cm',
    'avg_width_cm',
    'avg_product_volume_cm3',

    # Geographic
    'avg_shipping_distance_km',
    'max_shipping_distance_km',
    'is_cross_state',

    # Temporal
    'order_weekday',
    'order_month',
    'order_hour',
    'is_weekend_order',
    'is_holiday_season',

    # Time Estimation
    'is_rush_order',
    'estimated_days'
]


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_feature_order() -> list[str]:
    """
    If final_metadata.json has 'feature_names', use that order.
    Otherwise use FALLBACK_FEATURE_ORDER.
    """
    if FINAL_META.exists():
        try:
            meta = json.loads(FINAL_META.read_text())
            names = meta.get("feature_names", None)
            if isinstance(names, list) and all(isinstance(c, str) for c in names):
                return names
        except Exception:
            pass
    return FALLBACK_FEATURE_ORDER


# -----------------------------------------------------------------------------
# Optional: Haversine distance (kept for future geo features if needed)
# -----------------------------------------------------------------------------
def calculate_distance_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in kilometers (Haversine)."""
    R = 6371.0
    try:
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
        c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
        return float(R * c)
    except Exception:
        return float("nan")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def calculate_features(order_data: Dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """
    Build the EXACT feature vector (29 cols) the model expects.
    - Safe defaults for missing inputs
    - Robust numeric coercion
    - Derived ratios protected against division by zero
    - Column order enforced using final_metadata.json if present
    """
    # Normalize input to single-row DataFrame
    if isinstance(order_data, dict):
        raw = pd.DataFrame([order_data])
    else:
        raw = order_data.copy()

    # Helper to get scalar with default
    def get_scalar(name: str, default: float | int = 0):
        if name in raw:
            val = raw[name]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            return val
        return default

    features: dict[str, float] = {}

    # -------------------- Order Complexity --------------------
    num_items = _safe_float(get_scalar('num_items', 1), 1)
    num_sellers = _safe_float(get_scalar('num_sellers', 1), 1)
    features['num_items'] = num_items
    features['num_sellers'] = num_sellers
    features['num_products'] = _safe_float(get_scalar('num_products', 1), 1)
    features['is_multi_seller'] = float(1 if num_sellers > 1 else 0)
    features['is_multi_item'] = float(1 if num_items > 1 else 0)

    # -------------------- Financial ---------------------------
    total_order_value = _safe_float(get_scalar('total_order_value', 0), 0)
    features['total_order_value'] = total_order_value
    features['avg_item_price'] = _safe_float(get_scalar('avg_item_price', 0), 0)
    features['max_item_price'] = _safe_float(get_scalar('max_item_price', 0), 0)
    total_shipping_cost = _safe_float(get_scalar('total_shipping_cost', 0), 0)
    features['total_shipping_cost'] = total_shipping_cost
    features['avg_shipping_cost'] = _safe_float(get_scalar('avg_shipping_cost', 0), 0)

    # -------------------- Derived ratios ----------------------
    total_weight_g = _safe_float(get_scalar('total_weight_g', 0), 0)
    avg_shipping_distance_km = _safe_float(get_scalar('avg_shipping_distance_km', 500), 500)

    # weight_to_price_ratio = total_weight_g / (total_order_value + 1)
    features['weight_to_price_ratio'] = float(total_weight_g / (total_order_value + 1.0))

    # shipping_cost_per_km = total_shipping_cost / (avg_shipping_distance_km + 1)
    features['shipping_cost_per_km'] = float(total_shipping_cost / (avg_shipping_distance_km + 1.0))

    # -------------------- Physical ----------------------------
    features['total_weight_g'] = total_weight_g
    features['avg_weight_g'] = _safe_float(get_scalar('avg_weight_g', 0), 0)
    features['max_weight_g'] = _safe_float(get_scalar('max_weight_g', 0), 0)

    avg_length_cm = _safe_float(get_scalar('avg_length_cm', 0), 0)
    avg_height_cm = _safe_float(get_scalar('avg_height_cm', 0), 0)
    avg_width_cm = _safe_float(get_scalar('avg_width_cm', 0), 0)
    features['avg_length_cm'] = avg_length_cm
    features['avg_height_cm'] = avg_height_cm
    features['avg_width_cm'] = avg_width_cm
    features['avg_product_volume_cm3'] = float(avg_length_cm * avg_height_cm * avg_width_cm)

    # -------------------- Geographic --------------------------
    features['avg_shipping_distance_km'] = avg_shipping_distance_km
    features['max_shipping_distance_km'] = _safe_float(get_scalar('max_shipping_distance_km', 500), 500)
    features['is_cross_state'] = float(_safe_float(get_scalar('is_cross_state', 0), 0) != 0.0)

    # -------------------- Temporal ----------------------------
    order_weekday = _safe_float(get_scalar('order_weekday', 2), 2)
    order_month = _safe_float(get_scalar('order_month', 6), 6)
    features['order_weekday'] = order_weekday
    features['order_month'] = order_month
    features['order_hour'] = _safe_float(get_scalar('order_hour', 14), 14)
    features['is_weekend_order'] = float(_safe_float(get_scalar('is_weekend_order', 0), 0) != 0.0)

    # If not provided, derive is_holiday_season from month (Nov-Dec)
    if 'is_holiday_season' in raw.columns:
        features['is_holiday_season'] = float(_safe_float(get_scalar('is_holiday_season', 0), 0) != 0.0)
    else:
        features['is_holiday_season'] = float(order_month in (11.0, 12.0))

    # -------------------- Time Estimation ---------------------
    estimated_days = _safe_float(get_scalar('estimated_days', 10), 10)
    features['is_rush_order'] = float(1 if estimated_days < 7 else 0)
    features['estimated_days'] = estimated_days

    # -------------------- Assemble, coerce, order --------------------
    out = pd.DataFrame([features])

    # numeric coercion & NaN safety
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Enforce exact column order using metadata if available
    expected_order = _read_feature_order()

    # If metadata included extra columns not produced here, add them as 0.0
    for col in expected_order:
        if col not in out.columns:
            out[col] = 0.0

    # Keep only expected columns, in order
    out = out[expected_order]

    # Final guard
    assert len(out.columns) == 29, f"Expected 29 features, got {len(out.columns)}"
    return out


def get_feature_descriptions() -> Dict[str, str]:
    """Business-friendly descriptions for UI/tooltips."""
    return {
        'num_items': 'Number of Items in Order',
        'num_sellers': 'Number of Sellers',
        'num_products': 'Number of Unique Products',
        'is_multi_seller': 'Multi-Seller Order (0/1)',
        'is_multi_item': 'Multi-Item Order (0/1)',

        'total_order_value': 'Total Order Value ($)',
        'avg_item_price': 'Average Item Price ($)',
        'max_item_price': 'Highest Item Price ($)',
        'total_shipping_cost': 'Total Shipping Cost ($)',
        'avg_shipping_cost': 'Average Shipping Cost ($)',

        'weight_to_price_ratio': 'Weight / Price Ratio',
        'shipping_cost_per_km': 'Shipping Cost per KM ($)',

        'total_weight_g': 'Total Weight (g)',
        'avg_weight_g': 'Average Weight (g)',
        'max_weight_g': 'Heaviest Item (g)',
        'avg_length_cm': 'Average Length (cm)',
        'avg_height_cm': 'Average Height (cm)',
        'avg_width_cm': 'Average Width (cm)',
        'avg_product_volume_cm3': 'Average Product Volume (cm³)',

        'avg_shipping_distance_km': 'Shipping Distance (km)',
        'max_shipping_distance_km': 'Max Shipping Distance (km)',
        'is_cross_state': 'Cross-State Shipping (0/1)',

        'order_weekday': 'Order Day of Week (0=Mon, 6=Sun)',
        'order_month': 'Order Month (1-12)',
        'order_hour': 'Order Hour (0-23)',
        'is_weekend_order': 'Weekend Order (0/1)',
        'is_holiday_season': 'Holiday Season Order (0/1)',
        'is_rush_order': 'Rush Order (< 7 Estimated Days, 0/1)',
        'estimated_days': 'Estimated Delivery Days'
    }
