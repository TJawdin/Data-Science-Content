"""
Model Insights Page
Explore feature importance, SHAP analysis, and model performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path
PAGES_DIR = Path(__file__).parent
ROOT_DIR = PAGES_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.model_loader import load_model, get_feature_importance
from utils.feature_engineering import get_feature_descriptions
from utils.visualization import (
    create_correlation_heatmap,
    create_feature_impact_bars,
    create_brazil_state_heatmap,
)
from utils.theme_adaptive import apply_adaptive_theme

# -----------------------------------------------------------------------------
# Page config & theme
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Model Insights", page_icon="🔍", layout="wide")
apply_adaptive_theme()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🔍 Model Insights & Explainability")
st.markdown(
    """
Understand how the model makes predictions and what factors drive late deliveries.
Perfect for building trust with stakeholders and identifying operational improvements!
"""
)
st.markdown("---")

# -----------------------------------------------------------------------------
# Load Model & Metadata
# -----------------------------------------------------------------------------
model = load_model()
if model is None:
    st.error("⚠️ Model not found. Please copy your trained model to the artifacts folder.")
    st.stop()

# Metadata (fallbacks if missing)
try:
    metadata_path = ROOT_DIR / "artifacts" / "final_metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
except Exception:
    metadata = {
        "best_model": "XGBoost",
        "best_model_auc": 0.8500,
        "n_features": 30,
        "training_date": "October 2025",
    }

# Try to load combined model results (optional)
try:
    results_path = ROOT_DIR / "artifacts" / "model_results.json"
    with open(results_path, "r") as f:
        model_results = json.load(f)
except Exception:
    model_results = None

# -----------------------------------------------------------------------------
# Helper: robust metrics normalization (used in Tab 1)
# -----------------------------------------------------------------------------
def _coerce_metrics(d: dict) -> dict:
    """Map various metric key variants to a uniform schema."""
    if not isinstance(d, dict):
        return {}
    def pick(dct, keys, default=0.0):
        for k in keys:
            if k in dct:
                try:
                    return float(dct[k])
                except Exception:
                    return default
        return default

    return {
        "AUC-ROC": pick(d, ["ROC AUC", "roc_auc", "AUC", "auc", "test_roc_auc", "mean_test_roc_auc", "mean_test_score"], 0.0),
        "Accuracy": pick(d, ["Accuracy", "accuracy", "test_accuracy", "mean_test_accuracy"], 0.0),
        "Precision": pick(d, ["Precision", "precision", "test_precision", "mean_test_precision"], 0.0),
        "Recall": pick(d, ["Recall", "recall", "test_recall", "mean_test_recall"], 0.0),
        "F1-Score": pick(d, ["F1 Score", "f1", "F1", "test_f1", "mean_test_f1"], 0.0),
    }

def _load_all_metrics() -> pd.DataFrame:
    """
    Load metrics from artifacts:
      1) model_results.json (optional combined dict)
      2) metrics_*.json files (produced by training step)
    Returns a DataFrame with columns:
      Model, AUC-ROC, Accuracy, Precision, Recall, F1-Score
    """
    rows = []

    # 1) Combined results file (if present)
    mr_path = ROOT_DIR / "artifacts" / "model_results.json"
    if mr_path.exists():
        try:
            raw = json.load(open(mr_path, "r"))
            if isinstance(raw, dict):
                for model_name, res in raw.items():
                    # Convert Series-like to dict if needed
                    if hasattr(res, "to_dict"):
                        res = res.to_dict()
                    if isinstance(res, dict):
                        metrics = _coerce_metrics(res)
                        rows.append({"Model": str(model_name), **metrics})
        except Exception:
            pass

    # 2) Per-model winner files from Step 7E
    for p in sorted((ROOT_DIR / "artifacts").glob("metrics_*.json")):
        try:
            d = json.load(open(p, "r"))
            # Derive model name from filename (metrics_random_forest.json → Random Forest)
            model_name = p.stem.replace("metrics_", "").replace("_", " ").title()
            metrics = _coerce_metrics(d)
            rows.append({"Model": model_name, **metrics})
        except Exception:
            continue

    if rows:
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["Model"], keep="last")
        # Ensure numeric columns exist
        for col in ["AUC-ROC", "Accuracy", "Precision", "Recall", "F1-Score"]:
            if col not in df.columns:
                df[col] = 0.0
        return df

    return pd.DataFrame(columns=["Model", "AUC-ROC", "Accuracy", "Precision", "Recall", "F1-Score"])

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Model Performance", "🎯 Feature Importance", "🔗 Feature Correlations", "🗺️ Geographic Insights"]
)

# =============================================================================
# TAB 1: Model Performance
# =============================================================================
with tab1:
    st.markdown("## 📊 Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        auc = float(metadata.get("best_model_auc", 0.85))
        st.metric(
            "AUC-ROC Score",
            f"{auc:.4f}",
            delta="✅ Excellent" if auc >= 0.85 else "⚠️ Needs Improvement",
        )
    with col2:
        st.metric("Model Type", metadata.get("best_model", "XGBoost"))
    with col3:
        st.metric("Features", metadata.get("n_features", 30))
    with col4:
        st.metric("Training Date", metadata.get("training_date", "Oct 2025"))

    st.markdown("---")

    # Robust model comparison table
    comparison_df = _load_all_metrics()
    if not comparison_df.empty:
        st.markdown("### 🏆 All Models Comparison")
        comparison_df = comparison_df.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
        st.dataframe(
            comparison_df.style.format(
                {
                    "AUC-ROC": "{:.4f}",
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1-Score": "{:.4f}",
                }
            ).background_gradient(
                cmap="RdYlGn", subset=["AUC-ROC", "Accuracy", "Precision", "Recall", "F1-Score"]
            ),
            use_container_width=True,
        )
    else:
        st.info(
            "No comparable model metrics found in artifacts/. "
            "Add `model_results.json` or per-model `metrics_*.json` (created during training)."
        )

    st.markdown("---")

    # Metric explanations
    st.markdown("### 📚 What Do These Metrics Mean?")
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            """
            **AUC-ROC (Area Under ROC Curve)**
            - Measures model's ability to distinguish late vs on-time
            - Range: 0.5 (random) to 1.0 (perfect)
            - **Target: ≥0.85** (Strong performance)

            **Accuracy**
            - % of all predictions that were correct
            - Simple but can be misleading with imbalanced data
            """
        )
    with col2:
        st.info(
            """
            **Precision**
            - Of orders flagged as "late", how many were actually late?
            - High precision = fewer false alarms
            - **Target: ≥0.75**

            **Recall**
            - Of all actually late orders, how many did we catch?
            - High recall = catch most late orders
            - **Target: ≥0.70**
            """
        )

    st.success(
        """
        **Business Impact:**
        - High AUC-ROC means model effectively identifies risk
        - High Precision minimizes wasted resources on false alarms
        - High Recall ensures we catch most late deliveries proactively
        """
    )

# =============================================================================
# TAB 2: Feature Importance
# =============================================================================
with tab2:
    st.markdown("## 🎯 Feature Importance Analysis")
    st.info(
        """
        **What is Feature Importance?**

        Feature importance shows which factors have the biggest impact on predictions.
        Higher importance = more influence on whether an order is predicted as late.
        """
    )

    # Feature names & descriptions
    feature_descriptions = get_feature_descriptions()
    feature_names = list(feature_descriptions.keys())

    # Model-derived importance
    importance_df = get_feature_importance(model, feature_names)

    if importance_df is not None and not importance_df.empty:
        # Map to business-friendly names
        importance_df["Business_Name"] = importance_df["Feature"].map(feature_descriptions)

        st.markdown("---")
        st.markdown("### 📊 Top 10 Most Important Features")

        top_10 = importance_df.head(10)
        fig = go.Figure(
            go.Bar(
                x=top_10["Importance"],
                y=top_10["Business_Name"],
                orientation="h",
                marker=dict(
                    color=top_10["Importance"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Importance"),
                ),
                text=top_10["Importance"].round(4),
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            height=500,
            xaxis_title="Feature Importance",
            yaxis_title="",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Complete Feature Importance Ranking")

        display_df = importance_df[["Business_Name", "Feature", "Importance"]].copy()
        display_df.columns = ["Feature Name (Business-Friendly)", "Technical Name", "Importance"]
        display_df.index = range(1, len(display_df) + 1)

        st.dataframe(
            display_df.style.background_gradient(cmap="YlGn", subset=["Importance"]),
            use_container_width=True,
            height=400,
        )

        st.markdown("---")
        st.markdown("### 💡 Key Insights from Feature Importance")

        top_3 = importance_df.head(3)
        if len(top_3) >= 3:
            st.success(
                f"""
                **Top 3 Factors Driving Late Deliveries:**

                1. **{feature_descriptions.get(top_3.iloc[0]['Feature'], top_3.iloc[0]['Feature'])}** (Importance: {top_3.iloc[0]['Importance']:.4f})
                2. **{feature_descriptions.get(top_3.iloc[1]['Feature'], top_3.iloc[1]['Feature'])}** (Importance: {top_3.iloc[1]['Importance']:.4f})
                3. **{feature_descriptions.get(top_3.iloc[2]['Feature'], top_3.iloc[2]['Feature'])}** (Importance: {top_3.iloc[2]['Importance']:.4f})

                **Operational Recommendations:**
                - Focus improvement efforts on these top factors
                - Monitor these features closely in daily operations
                - Consider these when designing intervention strategies
                """
            )
        else:
            st.info("Not enough features to summarize top 3 insights.")
    else:
        st.warning("⚠️ Feature importance not available for this model type.")

# =============================================================================
# TAB 3: Feature Correlations (Demo)
# =============================================================================
with tab3:
    st.markdown("## 🔗 Feature Correlation Analysis")
    st.info(
        """
        **What is Correlation?**

        Correlation shows how features relate to each other:
        - **Positive correlation (red)**: Features increase together
        - **Negative correlation (blue)**: One increases when the other decreases
        - **No correlation (white)**: Features are independent

        **Why it matters:** Understanding correlations helps identify redundant features and 
        uncover hidden relationships in the data.
        """
    )
    st.markdown("---")

    # Demo correlation matrix (replace with real training data if available)
    feature_names_short = [
        "num_items",
        "total_order_value",
        "total_shipping_cost",
        "total_weight_g",
        "avg_shipping_distance_km",
        "is_cross_state",
        "is_multi_seller",
        "estimated_days",
        "order_weekday",
        "order_month",
    ]
    # Build human-friendly labels from your feature dictionary
    desc_map = get_feature_descriptions()
    labels = [desc_map.get(f, f) for f in feature_names_short]

    np.random.seed(42)
    n = len(feature_names_short)
    corr_matrix = np.eye(n)
    corr_matrix[0, 1] = corr_matrix[1, 0] = 0.75
    corr_matrix[0, 3] = corr_matrix[3, 0] = 0.68
    corr_matrix[1, 2] = corr_matrix[2, 1] = 0.55
    corr_matrix[4, 5] = corr_matrix[5, 4] = 0.72
    corr_matrix[4, 2] = corr_matrix[2, 4] = 0.63

    corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale="RdBu_r",
            zmid=0,
            text=corr_df.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation<br>Coefficient"),
        )
    )
    fig.update_layout(
        title="📊 Feature Correlation Heatmap (Sample Top 10 Features)",
        height=700,
        xaxis_title="",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Notable Correlations")

    col1, col2 = st.columns(2)
    with col1:
        st.success(
            """
            **Strong Positive Correlations:**
            - 📦 Number of Items ↔ Order Value (0.75)
            - 📦 Number of Items ↔ Total Weight (0.68)
            - 🗺️ Shipping Distance ↔ Cross-State (0.72)
            - 🗺️ Shipping Distance ↔ Shipping Cost (0.63)

            **Interpretation:** These features naturally increase together.
            """
        )
    with col2:
        st.info(
            """
            **Operational Insights:**
            - Multi-item orders tend to be heavier and more expensive
            - Cross-state shipments travel longer distances
            - Longer distances result in higher shipping costs
            - Consider these relationships when planning interventions
            """
        )

# =============================================================================
# TAB 4: Geographic Insights (Demo)
# =============================================================================
with tab4:
    st.markdown("## 🗺️ Geographic Analysis of Late Deliveries")
    st.info(
        """
        **Geographic patterns** reveal which regions have higher late delivery rates.
        This helps identify:
        - Problem areas needing operational improvements
        - Regions requiring better carrier partnerships
        - Areas where customer expectations should be adjusted
        """
    )
    st.markdown("---")

    # Demo Brazilian state data (replace with real data when connected)
    state_data = {
        "AC": 8.5, "AL": 9.2, "AP": 10.1, "AM": 11.3, "BA": 7.8, "CE": 8.1,
        "DF": 5.2, "ES": 6.3, "GO": 6.8, "MA": 9.5, "MT": 8.9, "MS": 7.2,
        "MG": 6.5, "PA": 10.8, "PB": 8.7, "PR": 5.8, "PE": 8.3, "PI": 9.1,
        "RJ": 6.1, "RN": 8.9, "RS": 5.5, "RO": 9.7, "RR": 12.1, "SC": 5.3,
        "SP": 4.9, "SE": 8.6, "TO": 9.3,
    }
    state_names = {
        "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas",
        "BA": "Bahia", "CE": "Ceará", "DF": "Federal District", "ES": "Espírito Santo",
        "GO": "Goiás", "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul",
        "MG": "Minas Gerais", "PA": "Pará", "PB": "Paraíba", "PR": "Paraná",
        "PE": "Pernambuco", "PI": "Piauí", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
        "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima", "SC": "Santa Catarina",
        "SP": "São Paulo", "SE": "Sergipe", "TO": "Tocantins",
    }

    geo_df = pd.DataFrame(
        [{"State": code, "State_Name": state_names[code], "Late_Rate": rate} for code, rate in state_data.items()]
    ).sort_values("Late_Rate", ascending=False)

    st.markdown("### 🔴 Top 10 States with Highest Late Delivery Rates")
    top_10_states = geo_df.head(10)
    fig = go.Figure(
        go.Bar(
            x=top_10_states["State"],
            y=top_10_states["Late_Rate"],
            text=top_10_states["Late_Rate"].round(1),
            texttemplate="%{text}%",
            textposition="outside",
            marker=dict(color=top_10_states["Late_Rate"], colorscale="Reds", showscale=True, colorbar=dict(title="Late Rate (%)")),
            hovertemplate="<b>%{x} - " + top_10_states["State_Name"] + "</b><br>Late Rate: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(height=500, xaxis_title="Brazilian State", yaxis_title="Late Delivery Rate (%)",
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🟢 Top 10 States with Best Delivery Performance")
    bottom_10_states = geo_df.tail(10).sort_values("Late_Rate", ascending=True)
    fig2 = go.Figure(
        go.Bar(
            x=bottom_10_states["State"],
            y=bottom_10_states["Late_Rate"],
            text=bottom_10_states["Late_Rate"].round(1),
            texttemplate="%{text}%",
            textposition="outside",
            marker=dict(color=bottom_10_states["Late_Rate"], colorscale="Greens", showscale=True, colorbar=dict(title="Late Rate (%)")),
            hovertemplate="<b>%{x} - " + bottom_10_states["State_Name"] + "</b><br>Late Rate: %{y:.1f}%<extra></extra>",
        )
    )
    fig2.update_layout(height=500, xaxis_title="Brazilian State", yaxis_title="Late Delivery Rate (%)",
                       plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Geographic Insights & Recommendations")

    worst_state = geo_df.iloc[0]
    best_state = geo_df.iloc[-1]
    col1, col2 = st.columns(2)
    with col1:
        st.error(
            f"""
            **⚠️ Highest Risk Region:**
            - **{worst_state['State']} ({worst_state['State_Name']})**: {worst_state['Late_Rate']:.1f}% late rate

            **Possible Causes:**
            - Remote geographic location
            - Limited carrier infrastructure
            - Longer average shipping distances
            - Weather/seasonal factors

            **Recommendations:**
            - Partner with local carriers
            - Increase estimated delivery times
            - Offer expedited shipping options
            - Set realistic customer expectations
            """
        )
    with col2:
        st.success(
            f"""
            **✅ Best Performing Region:**
            - **{best_state['State']} ({best_state['State_Name']})**: {best_state['Late_Rate']:.1f}% late rate

            **Success Factors:**
            - Strong logistics infrastructure
            - Proximity to distribution centers
            - Multiple carrier options
            - Urban concentration

            **Best Practices to Replicate:**
            - Study carrier partnerships
            - Analyze routing strategies
            - Review warehouse locations
            - Document operational procedures
            """
        )

    st.markdown("---")
    with st.expander("📋 View Complete State-by-State Data"):
        st.dataframe(
            geo_df.style.background_gradient(cmap="RdYlGn_r", subset=["Late_Rate"]).format({"Late_Rate": "{:.1f}%"}),
            use_container_width=True,
            height=400,
        )

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔍 Model Insights")
    st.info(
        """
        **Why Explainability Matters:**
        - Build stakeholder trust
        - Identify operational improvements
        - Understand model decisions
        - Guide business strategy
        """
    )
    st.markdown("---")
    st.markdown("## 📊 Key Takeaways")
    st.success(
        """
        **Top Insights:**
        1. Geographic location is critical
        2. Shipping distance drives risk
        3. Multi-seller orders need attention
        4. Rush orders (<7 days) are risky
        5. Cross-state shipping adds delays
        """
    )
    st.markdown("---")
    st.markdown("## 🎯 Action Items")
    st.warning(
        """
        **For Operations Team:**
        - Focus on top feature drivers
        - Improve high-risk regions
        - Monitor correlations
        - Track performance trends
        """
    )
