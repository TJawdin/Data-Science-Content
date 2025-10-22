# utils/feature_engineering.py
# Purpose: Convert raw shipment/order fields into the engineered features required by the model.
# This module is schema-driven (reads artifacts/feature_metadata.json) and robust to partial inputs.
#
# Key public API:
#   - calculate_features(df_raw: pd.DataFrame) -> pd.DataFrame
#
# Behavior:
#   1) If df_raw already contains ALL engineered columns from feature_metadata.json, we:
#      - reorder columns to the exact expected order
#      - coerce numeric columns
#      - fill missing numeric NaNs with 0.0 and categorical NaNs with ""
#   2) Otherwise, we attempt to derive engineered features from common raw fields:
#      Raw fields (any subset is fine; missing pieces get safe defaults):
#        - order_purchase_timestamp (str/ts)
#        - estimated_delivery_date (str/ts)  → est_lead_days
#        - item_prices, item_freights, item_weights_g, item_lengths_cm, item_heights_cm, item_widths_cm (lists or comma strings)
#        - seller_ids, seller_states, product_ids, categories (lists or comma strings)
#        - payment_type or payment_types (str or list)  → paytype_* one-hots
#        - payment_installments (int or list[int])      → max_installments
#        - payment_records (list / count)               → n_payment_records
#        - totals: total_payment, sum_price, sum_freight (if provided, used directly)
#        - customer_city, customer_state (str)
#      We compute the 32 engineered features and return them in the exact required order.
#
# Notes:
#   - All numeric coercions follow Python 3.12/NumPy/Pandas latest docs practices (astype/to_numeric).
#   - Datetime handling via pandas.to_datetime; see pandas docs for formats.
#   - Trig encoding for hour uses sin/cos on 24-hour circle.
#
# References:
#   - Python 3.12: https://docs.python.org/3/library/
#   - Pandas latest: https://pandas.pydata.org/docs/
#
# Author: (rewritten) 2025-10-22

from __future__ import annotations  # postpone eval of annotations for type hints
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import ast                           # safely parse Python-like list strings
import math                          # trig for hour encoding
import numpy as np                   # numeric ops
import pandas as pd                  # dataframe ops


# ----------------------------- Schema loading helpers ----------------------------- #

def _artifacts_dir() -> Path:
    """Return absolute path to ./artifacts directory (next to app)."""
    # utils/ → apps/supply_chain_delay/utils; go up one and into artifacts
    return Path(__file__).resolve().parents[1] / "artifacts"


def _load_feature_schema() -> Dict[str, Any]:
    """Load artifacts/feature_metadata.json; fall back to embedded defaults if missing."""
    f = _artifacts_dir() / "feature_metadata.json"                      # locate schema file
    if f.exists():                                                      # if present
        try:
            return pd.read_json(f, typ="series").to_dict()              # fast JSON load via pandas
        except Exception:
            try:
                import json
                return json.loads(f.read_text(encoding="utf-8"))        # fallback JSON loader
            except Exception:
                pass                                                    # fall through to defaults

    # Embedded fallback (matches what you shared)
    return {
        "feature_names": [
            "n_items","n_sellers","n_products","sum_price","sum_freight","total_payment",
            "n_payment_records","max_installments","avg_weight_g","avg_length_cm","avg_height_cm",
            "avg_width_cm","n_seller_states","purch_year","purch_month","purch_dayofweek","purch_hour",
            "purch_is_weekend","purch_hour_sin","purch_hour_cos","est_lead_days","n_categories",
            "mode_category_count","paytype_boleto","paytype_credit_card","paytype_debit_card",
            "paytype_not_defined","paytype_voucher","mode_category","seller_state_mode",
            "customer_city","customer_state"
        ],
        "numeric_feats": [
            "n_items","n_sellers","n_products","sum_price","sum_freight","total_payment",
            "n_payment_records","max_installments","avg_weight_g","avg_length_cm","avg_height_cm",
            "avg_width_cm","n_seller_states","purch_year","purch_month","purch_dayofweek","purch_hour",
            "purch_is_weekend","purch_hour_sin","purch_hour_cos","est_lead_days","n_categories",
            "mode_category_count"
        ],
        "paytype_feats": [
            "paytype_boleto","paytype_credit_card","paytype_debit_card","paytype_not_defined","paytype_voucher"
        ],
        "categorical_feats": [
            "mode_category","seller_state_mode","customer_city","customer_state"
        ],
    }


_SCHEMA = _load_feature_schema()                                       # cache schema at import
FEATURE_ORDER: List[str] = list(_SCHEMA.get("feature_names", []))      # expected full column order
NUMERIC_FEATS: List[str] = list(_SCHEMA.get("numeric_feats", []))      # numeric feature names
PAYTYPE_FEATS: List[str] = list(_SCHEMA.get("paytype_feats", []))      # one-hot payment names
CATEGORICAL_FEATS: List[str] = list(_SCHEMA.get("categorical_feats", []))  # categorical names


# ----------------------------- Generic parsing utilities ----------------------------- #

def _to_list(value: Any) -> List[Any]:
    """
    Convert strings like "a,b,c" or "['a','b','c']" or already-lists into a python list.
    Returns [] for None/empty/invalid inputs.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, (np.ndarray, pd.Series)):
        return [v for v in value.tolist() if pd.notna(v)]
    s = str(value).strip()
    if not s:
        return []
    # Try safe eval for python-like lists
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
    except Exception:
        pass
    # Fallback: split by comma
    return [p.strip() for p in s.split(",") if p.strip()]


def _to_float_list(value: Any) -> List[float]:
    """Parse list-like into list[float], coercing each element; drop non-numeric."""
    out: List[float] = []
    for x in _to_list(value):
        try:
            out.append(float(x))
        except Exception:
            continue
    return out


def _to_int_list(value: Any) -> List[int]:
    """Parse list-like into list[int], coercing each element; drop non-integer-ish."""
    out: List[int] = []
    for x in _to_list(value):
        try:
            out.append(int(float(x)))
        except Exception:
            continue
    return out


def _mode_and_count(values: List[Any]) -> Tuple[str, int]:
    """Return (mode_value_as_str, frequency). Empty → ('', 0)."""
    if not values:
        return "", 0
    ser = pd.Series(values)
    vc = ser.value_counts(dropna=True)
    if len(vc) == 0:
        return "", 0
    mode_val = vc.index[0]
    freq = int(vc.iloc[0])
    return str(mode_val), freq


def _to_datetime(value: Any) -> Optional[pd.Timestamp]:
    """Coerce to pandas.Timestamp; return None if invalid."""
    if value is None:
        return None
    try:
        return pd.to_datetime(value, errors="coerce", utc=False)
    except Exception:
        return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Coerce to float with default on failure."""
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    """Coerce to int with default on failure."""
    try:
        return int(float(x))
    except Exception:
        return default


def _clip01(x: float) -> float:
    """Clip to [0,1]."""
    return float(min(1.0, max(0.0, x)))


# ----------------------------- Row-wise engineering core ----------------------------- #

def _engineer_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a single row of engineered features from raw inputs.
    The function is defensive: any missing signal gets a safe default.
    """

    # ---- 1) Parse common raw fields (each of these may be missing) ----
    purchase_ts = _to_datetime(raw.get("order_purchase_timestamp") or raw.get("purchase_timestamp"))
    est_delivery_ts = _to_datetime(raw.get("estimated_delivery_date") or raw.get("customer_estimated_delivery"))

    item_prices = _to_float_list(raw.get("item_prices"))
    item_freights = _to_float_list(raw.get("item_freights"))
    item_weights_g = _to_float_list(raw.get("item_weights_g"))
    item_lengths_cm = _to_float_list(raw.get("item_lengths_cm"))
    item_heights_cm = _to_float_list(raw.get("item_heights_cm"))
    item_widths_cm = _to_float_list(raw.get("item_widths_cm"))

    seller_ids = _to_list(raw.get("seller_ids"))
    seller_states = _to_list(raw.get("seller_states"))
    product_ids = _to_list(raw.get("product_ids"))
    categories = _to_list(raw.get("categories"))

    # Payments can appear as a single string "credit_card" or a list of types
    payment_types = _to_list(raw.get("payment_types") or raw.get("payment_type"))
    payment_installments = _to_int_list(raw.get("payment_installments"))
    n_payment_records_raw = raw.get("n_payment_records")

    total_payment_raw = raw.get("total_payment") or raw.get("payment_value")
    sum_price_raw = raw.get("sum_price") or raw.get("total_items_price")
    sum_freight_raw = raw.get("sum_freight") or raw.get("total_freight")

    customer_city = str(raw.get("customer_city") or "").strip()
    customer_state = str(raw.get("customer_state") or "").strip()

    # ---- 2) Aggregate basics from lists or direct totals ----
    sum_price = _safe_float(sum_price_raw, default=sum(item_prices) if item_prices else 0.0)
    sum_freight = _safe_float(sum_freight_raw, default=sum(item_freights) if item_freights else 0.0)
    total_payment = _safe_float(total_payment_raw, default=(sum_price + sum_freight))

    n_items = _safe_int(raw.get("n_items"), default=(len(item_prices) if item_prices else 1))
    n_products = _safe_int(raw.get("n_products"), default=(len(set(product_ids)) if product_ids else n_items))
    n_sellers = _safe_int(raw.get("n_sellers"), default=(len(set(seller_ids)) if seller_ids else 1))

    # ---- 3) Dimensional averages ----
    def _avg(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    avg_weight_g = _safe_float(raw.get("avg_weight_g"), default=_avg(item_weights_g))
    avg_length_cm = _safe_float(raw.get("avg_length_cm"), default=_avg(item_lengths_cm))
    avg_height_cm = _safe_float(raw.get("avg_height_cm"), default=_avg(item_heights_cm))
    avg_width_cm = _safe_float(raw.get("avg_width_cm"), default=_avg(item_widths_cm))

    # ---- 4) Seller state counts and modes ----
    n_seller_states = _safe_int(raw.get("n_seller_states"), default=len(set([s for s in seller_states if s])))
    seller_state_mode, _seller_state_mode_cnt = _mode_and_count([s for s in seller_states if s])

    # ---- 5) Time features from purchase timestamp ----
    if purchase_ts is not None and pd.notna(purchase_ts):
        purch_year = int(purchase_ts.year)
        purch_month = int(purchase_ts.month)
        purch_dayofweek = int(purchase_ts.dayofweek)  # Monday=0
        purch_hour = int(purchase_ts.hour)
    else:
        purch_year = _safe_int(raw.get("purch_year"), 2018)
        purch_month = _safe_int(raw.get("purch_month"), 6)
        purch_dayofweek = _safe_int(raw.get("purch_dayofweek"), 2)
        purch_hour = _safe_int(raw.get("purch_hour"), 12)

    purch_is_weekend = 1 if purch_dayofweek in (5, 6) else 0           # Sat=5, Sun=6

    # Circular encoding for hour (0..23)
    purch_hour_sin = math.sin(2 * math.pi * (purch_hour % 24) / 24.0)
    purch_hour_cos = math.cos(2 * math.pi * (purch_hour % 24) / 24.0)

    # ---- 6) Lead time in days ----
    if est_delivery_ts is not None and purchase_ts is not None and pd.notna(est_delivery_ts) and pd.notna(purchase_ts):
        est_lead_days = int(max(0, (est_delivery_ts - purchase_ts).days))
    else:
        est_lead_days = _safe_int(raw.get("est_lead_days"), 10)

    # ---- 7) Category stats ----
    n_categories = _safe_int(raw.get("n_categories"), default=len(set([c for c in categories if c])))
    mode_category, mode_category_count = _mode_and_count([c for c in categories if c])
    mode_category = str(raw.get("mode_category", mode_category or "unknown")).strip()

    # ---- 8) Payments: n_records, max_installments, one-hot types ----
    n_payment_records = _safe_int(n_payment_records_raw, default=(len(payment_types) if payment_types else 1))
    max_installments = _safe_int(raw.get("max_installments"), default=(max(payment_installments) if payment_installments else _safe_int(raw.get("installments"), 1)))

    # Initialize all paytype one-hots to 0
    pay_onehots: Dict[str, int] = {k: 0 for k in PAYTYPE_FEATS}

    # Normalize payment types (strings) to expected keys
    # Map common aliases to your one-hot columns
    alias_map = {
        "boleto": "paytype_boleto",
        "credit_card": "paytype_credit_card",
        "debit_card": "paytype_debit_card",
        "not_defined": "paytype_not_defined",
        "voucher": "paytype_voucher",
        # common variants
        "credit card": "paytype_credit_card",
        "debit": "paytype_debit_card",
        "credit": "paytype_credit_card",
        "undefined": "paytype_not_defined",
    }
    if payment_types:
        norm_types = []
        for t in payment_types:
            key = str(t).strip().lower()
            mapped = alias_map.get(key)
            if mapped and mapped in pay_onehots:
                norm_types.append(mapped)
        # If we recognized at least one, turn on those flags; else leave as zeros
        for mt in set(norm_types):
            pay_onehots[mt] = 1
    else:
        # If no info at all, keep all zeros (model can learn "unknown")
        pass

    # ---- 9) Customer city/state fallbacks ----
    customer_city = str(raw.get("customer_city", customer_city)).strip()
    customer_state = str(raw.get("customer_state", customer_state)).strip()
    if not seller_state_mode:  # if we couldn't infer seller_state_mode earlier
        seller_state_mode = str(raw.get("seller_state_mode", "SP")).strip()

    # ---- 10) Assemble engineered row ----
    engineered: Dict[str, Any] = {
        "n_items": n_items,
        "n_sellers": n_sellers,
        "n_products": n_products,
        "sum_price": float(sum_price),
        "sum_freight": float(sum_freight),
        "total_payment": float(total_payment),
        "n_payment_records": n_payment_records,
        "max_installments": max_installments,
        "avg_weight_g": float(avg_weight_g),
        "avg_length_cm": float(avg_length_cm),
        "avg_height_cm": float(avg_height_cm),
        "avg_width_cm": float(avg_width_cm),
        "n_seller_states": n_seller_states,
        "purch_year": purch_year,
        "purch_month": purch_month,
        "purch_dayofweek": purch_dayofweek,
        "purch_hour": purch_hour,
        "purch_is_weekend": purch_is_weekend,
        "purch_hour_sin": float(purch_hour_sin),
        "purch_hour_cos": float(purch_hour_cos),
        "est_lead_days": est_lead_days,
        "n_categories": n_categories,
        "mode_category_count": mode_category_count,
        **pay_onehots,                          # expand the one-hot dict
        "mode_category": mode_category,
        "seller_state_mode": seller_state_mode,
        "customer_city": customer_city,
        "customer_state": customer_state,
    }

    # Ensure all expected keys exist (with safe defaults) and no extras sneak in
    for col in FEATURE_ORDER:
        if col not in engineered:
            # numeric default 0.0 / int 0 for known numeric features; empty string for categoricals
            if col in NUMERIC_FEATS or col in PAYTYPE_FEATS:
                engineered[col] = 0 if col in PAYTYPE_FEATS else 0.0
            elif col in CATEGORICAL_FEATS:
                engineered[col] = ""
            else:
                # unknown bucket → empty string
                engineered[col] = ""

    return engineered


# ----------------------------- Public API ----------------------------- #

def calculate_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw shipment/order rows into the model's engineered features.
    - If df_raw already has all engineered columns, we reorder + coerce types and return.
    - Otherwise, we derive features row-by-row from best-guess raw fields.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Incoming rows, either already engineered or containing raw business fields.

    Returns
    -------
    pd.DataFrame
        DataFrame with EXACT columns FEATURE_ORDER, in order, ready for model.predict_proba().
    """
    df = df_raw.copy()                                                    # work on a copy

    # Case A: already engineered → reorder, coerce, fill
    if set(FEATURE_ORDER).issubset(df.columns):
        # Reorder columns
        df = df.reindex(columns=FEATURE_ORDER)                            # exact order

        # Coerce numeric features
        for f in NUMERIC_FEATS:
            df[f] = pd.to_numeric(df[f], errors="coerce")

        # Payment one-hots → 0/1
        for f in PAYTYPE_FEATS:
            if f in df.columns:
                df[f] = df[f].map(lambda x: 1 if str(x).strip() in ("1", "True", "true", "YES", "yes") else 0)

        # Fill NaNs: numerics to 0.0, categoricals to ""
        if NUMERIC_FEATS:
            df[NUMERIC_FEATS] = df[NUMERIC_FEATS].fillna(0.0)
        for f in CATEGORICAL_FEATS:
            if f in df.columns:
                df[f] = df[f].fillna("")

        return df

    # Case B: derive engineered features from raw
    out_rows: List[Dict[str, Any]] = []
    # Convert each row (Series → dict) and engineer
    for _, row in df.iterrows():
        engineered = _engineer_row(row.to_dict())
        out_rows.append(engineered)

    out = pd.DataFrame(out_rows, columns=FEATURE_ORDER)                   # enforce order

    # Final coercions (safety net)
    for f in NUMERIC_FEATS:
        out[f] = pd.to_numeric(out[f], errors="coerce").fillna(0.0)
    for f in PAYTYPE_FEATS:
        out[f] = out[f].map(lambda x: 1 if str(x).strip() in ("1", "True", "true", "YES", "yes") else 0)
    for f in CATEGORICAL_FEATS:
        out[f] = out[f].fillna("")

    return out
