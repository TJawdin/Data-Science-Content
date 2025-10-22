# pages/1_🎯_Example_Scenarios.py
# Purpose: Provide 3–5 curated RAW examples (Low/Medium/High) and
#          allow users to push one example directly into the Single page form.
#
# Implementation detail:
# - We surface examples as raw dicts that match the Single page input schema.
# - "Copy to clipboard" and "Download JSON/CSV" are provided for convenience.

from __future__ import annotations
from typing import Dict, Any, List
import json
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Example Scenarios", page_icon="🎯", layout="wide")
st.title("🎯 Example Scenarios")
st.caption("Kick the tires with a few realistic orders. You can download or paste into the Single page.")

# --- Curated RAW examples (hand-picked for bands) ---
EXAMPLES: List[Dict[str, Any]] = [
    # Likely Low
    {
        "name": "Low — small local order",
        "payload": {
            "order_purchase_timestamp": "2017-08-02T10:15:00",
            "estimated_delivery_date": "2017-08-08T09:00:00",
            "sum_price": 99.90,
            "sum_freight": 15.00,
            "n_items": 1,
            "n_sellers": 1,
            "payment_type": "credit_card",
            "max_installments": 1,
            "mode_category": "housewares",
            "customer_city": "sao paulo",
            "customer_state": "SP",
        },
        "notes": "Short lead time, one seller, simple payment."
    },
    # Likely Medium
    {
        "name": "Medium — multi-seller, moderate freight",
        "payload": {
            "order_purchase_timestamp": "2018-03-15T21:40:00",
            "estimated_delivery_date": "2018-03-28T12:00:00",
            "sum_price": 280.50,
            "sum_freight": 42.90,
            "n_items": 3,
            "n_sellers": 2,
            "payment_type": "boleto",
            "max_installments": 1,
            "mode_category": "sports_leisure",
            "customer_city": "curitiba",
            "customer_state": "PR",
        },
        "notes": "Boleto + multi-seller; moderate distance/time."
    },
    # Likely High
    {
        "name": "High — long lead, multiple sellers",
        "payload": {
            "order_purchase_timestamp": "2017-11-24T23:10:00",
            "estimated_delivery_date": "2017-12-20T09:00:00",
            "sum_price": 820.00,
            "sum_freight": 120.00,
            "n_items": 5,
            "n_sellers": 3,
            "payment_type": "credit_card",
            "max_installments": 6,
            "mode_category": "computers_accessories",
            "customer_city": "rio de janeiro",
            "customer_state": "RJ",
        },
        "notes": "Black Friday-ish purchase; long estimated lead."
    },
]

st.markdown("---")
for i, ex in enumerate(EXAMPLES, start=1):
    with st.container(border=True):
        left, right = st.columns([2, 1])
        with left:
            st.subheader(f"{i}) {ex['name']}")
            st.json(ex["payload"])
            st.caption(ex["notes"])
        with right:
            raw = ex["payload"]
            # Download as JSON
            st.download_button(
                label="⬇️ Download JSON",
                data=json.dumps(raw, indent=2).encode("utf-8"),
                file_name=f"scenario_{i}.json",
                mime="application/json",
                key=f"download_json_{i}",
            )
            # Download as CSV (single-row)
            df = pd.DataFrame([raw])
            st.download_button(
                label="⬇️ Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"scenario_{i}.csv",
                mime="text/csv",
                key=f"download_csv_{i}",
            )
            st.info("Open the Single page and paste these values directly.")
