"""
Example Scenarios Page
Pre-loaded scenarios for quick testing and demos
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path (so utils imports work when running as a page)
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_engineering import calculate_features, get_feature_descriptions
from utils.model_loader import load_model, predict_single
from utils.visualization import create_risk_gauge
from utils.theme_adaptive import apply_adaptive_theme
from utils.constants import (
    FINAL_METADATA,     # loaded final_metadata.json as dict
    RISK_BANDS,         # {"low_max": int, "med_max": int}
    OPTIMAL_THRESHOLD,  # float, e.g. 0.669271...
)

# ---------------------------------------------------------------------------
# Page config + adaptive theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Example Scenarios",
    page_icon="🎯",
    layout="wide",
)
apply_adaptive_theme()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎯 Example Scenarios")
st.markdown("""
Quick-test the model with pre-configured realistic scenarios.
Perfect for demos, training, and understanding model behavior!
""")
st.markdown("---")

# ---------------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------------
model = load_model()
if model is None:
    st.error("⚠️ Model not found. Please copy your trained model to the artifacts folder.")
    st.stop()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_cost_per_km(total_shipping_cost: float, avg_distance_km: float) -> float:
    if avg_distance_km is None or avg_distance_km <= 0:
        return 0.0
    return float(total_shipping_cost) / float(avg_distance_km)

def format_money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

def yes_no(v):
    return "Yes" if int(v or 0) == 1 else "No"

# ---------------------------------------------------------------------------
# Example Scenarios (calibrated to current model logic)
# Note: We don't hardcode band cutoffs here; risk bands come from FINAL_METADATA.
# ---------------------------------------------------------------------------
scenarios = {
    "🔴 HIGH RISK: Budget Shipping, Long Distance, Tight ETA": {
        "description": (
            "Low order value, budget shipping (very low $/km), long distance, "
            "holiday-weekend timing, and a tight ETA."
        ),
        "data": {
            "num_items": 1,
            "num_sellers": 1,
            "num_products": 1,
            "total_order_value": 35.0,
            "avg_item_price": 35.0,
            "max_item_price": 35.0,
            "total_shipping_cost": 5.0,
            "avg_shipping_cost": 5.0,
            "total_weight_g": 1500,
            "avg_weight_g": 1500,
            "max_weight_g": 1500,
            "avg_length_cm": 40.0,
            "avg_height_cm": 30.0,
            "avg_width_cm": 20.0,
            "avg_shipping_distance_km": 1500,
            "max_shipping_distance_km": 1500,
            "is_cross_state": 1,
            "order_weekday": 6,   # Saturday
            "order_month": 12,    # Dec
            "order_hour": 20,
            "is_weekend_order": 1,
            "is_holiday_season": 1,
            "estimated_days": 3,  # tight timeline
        },
        "color": "red",
    },

    "🟢 LOW RISK: Same-City Express, Comfortable ETA": {
        "description": (
            "Local delivery, premium spend per km, simple order, and a comfortable ETA window."
        ),
        "data": {
            "num_items": 1,
            "num_sellers": 1,
            "num_products": 1,
            "total_order_value": 120.0,
            "avg_item_price": 120.0,
            "max_item_price": 120.0,
            "total_shipping_cost": 15.0,
            "avg_shipping_cost": 15.0,
            "total_weight_g": 600,
            "avg_weight_g": 600,
            "max_weight_g": 600,
            "avg_length_cm": 22.0,
            "avg_height_cm": 16.0,
            "avg_width_cm": 12.0,
            "avg_shipping_distance_km": 80,
            "max_shipping_distance_km": 80,
            "is_cross_state": 0,
            "order_weekday": 3,  # Thursday
            "order_month": 5,
            "order_hour": 11,
            "is_weekend_order": 0,
            "is_holiday_season": 0,
            "estimated_days": 12,
        },
        "color": "green",
    },

    "🟡 MEDIUM RISK: Weekend Holiday, Multi-Seller": {
        "description": (
            "Holiday season, weekend order, moderate distance and complexity. "
            "Balanced value and shipping cost per km."
        ),
        "data": {
            "num_items": 4,
            "num_sellers": 2,
            "num_products": 4,
            "total_order_value": 220.0,
            "avg_item_price": 55.0,
            "max_item_price": 90.0,
            "total_shipping_cost": 24.0,
            "avg_shipping_cost": 6.0,
            "total_weight_g": 3000,
            "avg_weight_g": 750,
            "max_weight_g": 1200,
            "avg_length_cm": 32.0,
            "avg_height_cm": 24.0,
            "avg_width_cm": 18.0,
            "avg_shipping_distance_km": 820,
            "max_shipping_distance_km": 820,
            "is_cross_state": 1,
            "order_weekday": 6,  # Sunday
            "order_month": 11,   # Nov
            "order_hour": 18,
            "is_weekend_order": 1,
            "is_holiday_season": 1,
            "estimated_days": 8,
        },
        "color": "orange",
    },
}

# ---------------------------------------------------------------------------
# Scenario Selection
# ---------------------------------------------------------------------------
st.markdown("## 📋 Select a Scenario to Test")

cols = st.columns(len(scenarios))
selected_scenario = st.session_state.get("selected_scenario")

for idx, (name, _) in enumerate(scenarios.items()):
    with cols[idx]:
        if st.button(name, use_container_width=True, type="secondary"):
            selected_scenario = name
            st.session_state["selected_scenario"] = name

if not selected_scenario:
    selected_scenario = list(scenarios.keys())[0]

st.markdown("---")

# ---------------------------------------------------------------------------
# Display Selected Scenario + Prediction
# ---------------------------------------------------------------------------
st.markdown(f"## {selected_scenario}")
scenario = scenarios[selected_scenario]

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Scenario Description")
    st.info(scenario["description"])

    st.markdown("### 📊 Order Details")

    data = scenario["data"]
    dollars_per_km = safe_cost_per_km(
        data["total_shipping_cost"], data["avg_shipping_distance_km"]
    )

    detail_rows = [
        ("Number of Items",                data["num_items"]),
        ("Number of Sellers",              data["num_sellers"]),
        ("Total Order Value",              format_money(data["total_order_value"])),
        ("Total Shipping Cost",            format_money(data["total_shipping_cost"])),
        ("Shipping $/km",                  f"${dollars_per_km:0.4f}"),
        ("Total Weight",                   f"{data['total_weight_g']} g"),
        ("Shipping Distance",              f"{data['avg_shipping_distance_km']} km"),
        ("Cross-State",                    yes_no(data.get("is_cross_state", 0))),
        ("Weekend Order",                  yes_no(data.get("is_weekend_order", 0))),
        ("Holiday Season",                 yes_no(data.get("is_holiday_season", 0))),
        ("Estimated Delivery (days)",      data["estimated_days"]),
    ]
    st.table(pd.DataFrame(detail_rows, columns=["Attribute", "Value"]))

with col2:
    # Make prediction
    with st.spinner("Calculating risk..."):
        try:
            features_df = calculate_features(scenario["data"])
            result = predict_single(model, features_df)

            if result:
                # Risk gauge (internally uses current bands from constants)
                fig = create_risk_gauge(result["risk_score"], result["risk_level"])
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Prediction", result["prediction_label"])
                with col_b:
                    st.metric("Risk Score", f"{result['risk_score']}/100")
                with col_c:
                    st.metric("Risk Level", result["risk_level"])

                st.caption(
                    f"Operating threshold: **{int(round(OPTIMAL_THRESHOLD*100))}%** "
                    f"(auto-optimized from notebook)."
                )
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            result = None

st.markdown("---")

# ---------------------------------------------------------------------------
# Recommendations (based on current bands from constants)
# ---------------------------------------------------------------------------
if result:
    low_max = int(RISK_BANDS["low_max"])
    med_max = int(RISK_BANDS["med_max"])
    score = int(result["risk_score"])

    st.markdown("### 💡 Recommended Actions")

    if score > med_max:
        st.error("""
**🚨 HIGH RISK – Immediate Action Recommended**
- Upgrade to expedited shipping immediately
- Proactively contact customer with realistic timeline
- Flag order for priority processing in warehouse
- Consider splitting order across warehouses if possible
- Budget for potential refund/compensation
- Daily monitoring until delivery confirmed
""")
    elif score > low_max:
        st.warning("""
**⚠️ MEDIUM RISK – Monitor Closely**
- Add to daily monitoring watchlist
- Send automated tracking updates to customer
- Ensure optimal carrier selection for route
- Review route for potential bottlenecks
- Prepare customer service for possible inquiries
""")
    else:
        st.success("""
**✅ LOW RISK – Standard Processing**
- Proceed with normal workflow
- Standard customer communication
- Include in regular batch monitoring
""")

st.markdown("---")

# ---------------------------------------------------------------------------
# Feature Breakdown (friendly names)
# ---------------------------------------------------------------------------
with st.expander("🔍 View Detailed Feature Values"):
    try:
        feature_descriptions = get_feature_descriptions()
        if 'features_df' in locals():
            display_rows = []
            for col in features_df.columns:
                display_rows.append({
                    "Feature (Business Name)": feature_descriptions.get(col, col),
                    "Technical Name": col,
                    "Value": features_df[col].values[0],
                })

            st.dataframe(
                pd.DataFrame(display_rows),
                use_container_width=True,
                height=420
            )
    except Exception as e:
        st.error(f"Could not display feature breakdown: {str(e)}")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎯 Example Scenarios")
    st.info("""
**Purpose**
- Quick model testing
- Stakeholder demos
- Training new users
- Understanding risk factors
""")

    st.markdown("---")

    st.markdown("## 📊 Scenario Summary")
    st.markdown(f"""
**Total Scenarios:** {len(scenarios)}
- 🟢 Low Risk: 1
- 🟡 Medium Risk: 1
- 🔴 High Risk: 1
""")

    st.markdown("---")

    st.markdown("## 💡 Model Insight")
    st.success(f"""
**Operating threshold:** {int(round(OPTIMAL_THRESHOLD*100))}%  
**Risk bands:** LOW ≤ {RISK_BANDS['low_max']} | MEDIUM ≤ {RISK_BANDS['med_max']} | HIGH > {RISK_BANDS['med_max']}
""")

    st.caption(f"Best model: **{FINAL_METADATA.get('best_model','N/A')}** "
               f"(AUC {FINAL_METADATA.get('best_model_auc',0):.3f}).")
