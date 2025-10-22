# utils/feature_labels.py
# Purpose: Central dictionary of "technical feature name" → "business-friendly label".
# Notes:
# - Mirrors your 32 engineered features from feature_metadata.json.
# - Import get_friendly_feature_map() wherever you want human-readable labels.
# - Keep this file tiny and dependency-free so it’s safe to import anywhere.

from __future__ import annotations  # future annotations for clarity
from typing import Dict             # type hints


# ----------------------------- Public API ----------------------------- #

def get_friendly_feature_map() -> Dict[str, str]:
    """
    Return a mapping from engineered feature names (technical) to
    concise, business-friendly labels used across the app (PDF + charts).

    You can edit any label here to suit stakeholder language.
    """
    return {
        # -------- Counts & aggregates
        "n_items": "Items (count)",
        "n_sellers": "Sellers (count)",
        "n_products": "Products (unique)",
        "sum_price": "Items total (R$)",
        "sum_freight": "Freight total (R$)",
        "total_payment": "Total payment (R$)",
        "n_payment_records": "Payment records (count)",
        "max_installments": "Max installments",
        "n_categories": "Categories (count)",

        # -------- Package geometry & weight (averages across order items)
        "avg_weight_g": "Avg item weight (g)",
        "avg_length_cm": "Avg item length (cm)",
        "avg_height_cm": "Avg item height (cm)",
        "avg_width_cm": "Avg item width (cm)",

        # -------- Seller diversity (route complexity proxy)
        "n_seller_states": "Seller states (unique)",

        # -------- Purchase timestamp breakdown
        "purch_year": "Purchase year",
        "purch_month": "Purchase month",
        "purch_dayofweek": "Purchase day of week (0=Mon)",
        "purch_hour": "Purchase hour (0–23)",
        "purch_is_weekend": "Weekend purchase",
        "purch_hour_sin": "Purchase hour (sin)",
        "purch_hour_cos": "Purchase hour (cos)",

        # -------- Delivery promise / SLA proxy
        "est_lead_days": "Estimated lead time (days)",

        # -------- Category concentration (dominance proxy)
        "mode_category_count": "Mode category count",

        # -------- Payment one-hots
        "paytype_boleto": "Pay: boleto",
        "paytype_credit_card": "Pay: credit card",
        "paytype_debit_card": "Pay: debit card",
        "paytype_not_defined": "Pay: not defined",
        "paytype_voucher": "Pay: voucher",

        # -------- Categorical groupings (kept as text; shown verbatim)
        "mode_category": "Dominant category",
        "seller_state_mode": "Dominant seller state",
        "customer_city": "Customer city",
        "customer_state": "Customer state",
    }
