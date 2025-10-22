"""
PDF Report Generator
Creates professional PDF reports for order risk predictions
"""

from __future__ import annotations
from fpdf import FPDF
import pandas as pd
from datetime import datetime

# NEW: import current model metadata, thresholds, and friendly names
from utils.constants import (
    FINAL_METADATA,          # dict loaded from artifacts/final_metadata.json
    OPTIMAL_THRESHOLD,       # float, e.g. 0.669271
    RISK_BANDS,              # {"low_max": 12, "med_max": 30}
    FRIENDLY_FEATURE_NAMES,  # technical -> business-friendly mapping
)

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _risk_color(level: str):
    if level.upper() == "LOW":
        return (46, 204, 113)   # green
    if level.upper() == "MEDIUM":
        return (243, 156, 18)   # orange
    return (231, 76, 60)        # red


def _band_for_score(score_0_100: int):
    low_max = int(RISK_BANDS["low_max"])
    med_max = int(RISK_BANDS["med_max"])
    if score_0_100 <= low_max:
        return "LOW"
    if score_0_100 <= med_max:
        return "MEDIUM"
    return "HIGH"


def _friendly(name: str) -> str:
    return FRIENDLY_FEATURE_NAMES.get(name, name)


# --------------------------------------------------------------------------- #
# PDF Class
# --------------------------------------------------------------------------- #

class RiskReportPDF(FPDF):
    """Custom PDF class for risk assessment reports"""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, 'Supply Chain Delay Risk Assessment Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(52, 73, 94)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, body)
        self.ln()

    def add_colored_box(self, text, color_rgb):
        self.set_fill_color(*color_rgb)
        self.set_font('Arial', 'B', 12)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, text, 0, 1, 'C', fill=True)
        self.ln(2)


# --------------------------------------------------------------------------- #
# Main API
# --------------------------------------------------------------------------- #

def generate_risk_report(
    order_data: dict,
    prediction_result: dict,
    features_df: pd.DataFrame,
    shap_contributions: dict | None = None,
) -> bytes:
    """
    Generate a PDF risk assessment report

    Parameters
    ----------
    order_data : dict
        Original order information (what the user entered)
    prediction_result : dict
        Output from predict_single(...) in model_loader (probability, risk_score, risk_level, etc.)
    features_df : pd.DataFrame
        Final model-ready features (1 row). Used to display top feature values if SHAP not provided.
    shap_contributions : dict, optional
        Per-feature SHAP values for the same row, in the form {technical_feature: shap_value}.
        When provided, we present the Top Contributing Factors by absolute SHAP.

    Returns
    -------
    bytes : PDF file bytes
    """

    pdf = RiskReportPDF()
    pdf.add_page()

    # ------------------------------------------------------------------ #
    # EXECUTIVE SUMMARY
    # ------------------------------------------------------------------ #
    pdf.chapter_title('Executive Summary')

    risk_level = prediction_result['risk_level']
    risk_score = int(prediction_result['risk_score'])  # 0..100
    late_prob = float(prediction_result['probability'])  # 0..1

    # Ensure level aligns with current band rules (in case upstream changed)
    risk_level = _band_for_score(risk_score)

    pdf.add_colored_box(f"RISK LEVEL: {risk_level} ({risk_score}/100)", _risk_color(risk_level))

    thr_pct = int(round(OPTIMAL_THRESHOLD * 100))
    auc = FINAL_METADATA.get("best_model_auc", 0.0)
    prec = FINAL_METADATA.get("best_model_precision", 0.0)
    rec = FINAL_METADATA.get("best_model_recall", 0.0)
    f1 = FINAL_METADATA.get("best_model_f1", 0.0)

    pdf.chapter_body(
        f"Prediction: {prediction_result['prediction_label']}\n"
        f"Late Delivery Probability: {late_prob:.1%}\n"
        f"Operating Threshold: {thr_pct}% (auto-optimized)\n"
        f"Model AUC-ROC: {auc:.4f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}\n"
        f"Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    # ------------------------------------------------------------------ #
    # ORDER DETAILS
    # ------------------------------------------------------------------ #
    pdf.chapter_title('Order Details')

    order_summary = f"""
Number of Items: {order_data.get('num_items', 'N/A')}
Number of Sellers: {order_data.get('num_sellers', 'N/A')}
Total Order Value: ${order_data.get('total_order_value', 0):.2f}
Total Shipping Cost: ${order_data.get('total_shipping_cost', 0):.2f}
Total Weight: {order_data.get('total_weight_g', 0)} g

Shipping Distance: {order_data.get('avg_shipping_distance_km', 0)} km
Cross-State Shipping: {'Yes' if order_data.get('is_cross_state', 0) == 1 else 'No'}
Estimated Delivery: {order_data.get('estimated_days', 0)} days
Weekend Order: {'Yes' if order_data.get('is_weekend_order', 0) == 1 else 'No'}
Holiday Season: {'Yes' if order_data.get('is_holiday_season', 0) == 1 else 'No'}
    """
    pdf.chapter_body(order_summary.strip())

    # ------------------------------------------------------------------ #
    # RISK ASSESSMENT (bands from metadata)
    # ------------------------------------------------------------------ #
    pdf.chapter_title('Risk Assessment Breakdown')

    low_max = int(RISK_BANDS["low_max"])
    med_max = int(RISK_BANDS["med_max"])

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Risk Score Interpretation:', 0, 1)
    pdf.set_font('Arial', '', 10)

    if risk_score <= low_max:
        interpretation = f"""
LOW RISK (0–{low_max}%): This order has a low probability of late delivery.
Standard processing procedures are recommended. No special intervention required.
        """
    elif risk_score <= med_max:
        interpretation = f"""
MEDIUM RISK ({low_max+1}–{med_max}%): This order has moderate risk. Monitor
closely and ensure optimal carrier selection. Consider proactive customer communication.
        """
    else:
        interpretation = f"""
HIGH RISK ({med_max+1}–100%): This order has a high probability of late delivery.
Immediate action recommended: upgrade shipping, prioritize processing, and proactively
contact the customer.
        """
    pdf.multi_cell(0, 5, interpretation.strip())
    pdf.ln(3)

    # ------------------------------------------------------------------ #
    # RECOMMENDATIONS (based on band)
    # ------------------------------------------------------------------ #
    pdf.chapter_title('Recommended Actions')

    if risk_score > med_max:
        recommendations = """
IMMEDIATE ACTIONS REQUIRED:
- Upgrade to expedited shipping immediately
- Flag order for priority warehouse processing
- Proactively contact customer with realistic timeline
- Consider splitting order across warehouses if possible
- Budget for potential refund or compensation
- Implement daily monitoring until delivery confirmed

ESCALATION:
- Notify operations manager
- Alert customer service team
- Document all actions taken
        """
    elif risk_score > low_max:
        recommendations = """
MONITORING REQUIRED:
- Add to daily monitoring watchlist
- Send automated tracking updates to customer
- Ensure optimal carrier selection for route
- Review shipping route for potential bottlenecks
- Prepare customer service for potential inquiries

PREVENTIVE MEASURES:
- Consider upgrading shipping if within budget
- Verify warehouse stock availability
- Check carrier performance history for this route
        """
    else:
        recommendations = """
STANDARD PROCESSING:
- Proceed with normal shipping workflow
- Apply standard customer communication
- No special intervention required
- Include in regular batch monitoring

QUALITY ASSURANCE:
- Verify all items in stock
- Confirm carrier pickup schedule
- Standard quality checks apply
        """
    pdf.chapter_body(recommendations.strip())

    # ------------------------------------------------------------------ #
    # TOP CONTRIBUTING FACTORS (prefer SHAP if provided)
    # ------------------------------------------------------------------ #
    pdf.add_page()
    pdf.chapter_title('Top Contributing Factors')

    # Build rows: either SHAP contributions (preferred) or feature values
    rows = []
    if shap_contributions:
        # Sort by absolute SHAP value and show contribution
        items = [( _friendly(k), float(v) ) for k, v in shap_contributions.items()]
        rows = sorted(items, key=lambda x: abs(x[1]), reverse=True)[:10]

        pdf.set_font('Arial', 'B', 10)
        pdf.cell(120, 7, 'Feature (Business Name)', 1, 0, 'L')
        pdf.cell(20, 7, 'Impact', 1, 1, 'C')
        pdf.set_font('Arial', '', 9)
        for fname, val in rows:
            label = fname.encode('ascii', 'ignore').decode('ascii')
            label = label[:60] + '...' if len(label) > 60 else label
            pdf.cell(120, 6, label, 1, 0, 'L')
            pdf.cell(20, 6, f'{val:+.3f}', 1, 1, 'C')
    else:
        # Fall back to showing top absolute feature values (scaled numerically)
        ser = features_df.iloc[0]
        items = [(_friendly(k), float(v)) for k, v in ser.items()]
        rows = sorted(items, key=lambda x: abs(x[1]), reverse=True)[:10]

        pdf.set_font('Arial', 'B', 10)
        pdf.cell(120, 7, 'Feature (Business Name)', 1, 0, 'L')
        pdf.cell(20, 7, 'Value', 1, 1, 'C')
        pdf.set_font('Arial', '', 9)
        for fname, val in rows:
            label = fname.encode('ascii', 'ignore').decode('ascii')
            label = label[:60] + '...' if len(label) > 60 else label
            pdf.cell(120, 6, label, 1, 0, 'L')
            pdf.cell(20, 6, f'{val:.2f}', 1, 1, 'C')

    pdf.ln(5)

    # ------------------------------------------------------------------ #
    # MODEL INFORMATION (from final_metadata.json)
    # ------------------------------------------------------------------ #
    pdf.chapter_title('Model Information')

    model_info = f"""
Model: {FINAL_METADATA.get('best_model', 'Model')}
Features: {FINAL_METADATA.get('n_features', 0)} domain-engineered features
Training Samples: {FINAL_METADATA.get('n_samples_train', 0):,}
Test Samples: {FINAL_METADATA.get('n_samples_test', 0):,}
Operating Threshold: {int(round(OPTIMAL_THRESHOLD*100))}%

Performance (Test):
- AUC-ROC: {FINAL_METADATA.get('best_model_auc', 0):.4f}
- Precision: {FINAL_METADATA.get('best_model_precision', 0):.3f}
- Recall: {FINAL_METADATA.get('best_model_recall', 0):.3f}
- F1-Score: {FINAL_METADATA.get('best_model_f1', 0):.3f}

This prediction is based on a machine learning model trained on historical
delivery patterns and order characteristics. Use alongside operational context.
    """
    pdf.chapter_body(model_info.strip())

    # ------------------------------------------------------------------ #
    # DISCLAIMER
    # ------------------------------------------------------------------ #
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(
        0,
        4,
        "DISCLAIMER: This risk assessment is generated by a machine learning model and should be "
        "used as a decision-support tool. Actual outcomes may vary based on factors not captured "
        "in the model. Apply business judgment when making operational decisions."
    )

    pdf.ln(5)
    pdf.set_font('Arial', 'B', 9)
    pdf.set_text_color(52, 73, 94)
    pdf.cell(0, 5, 'Generated by Supply Chain Delay Prediction System', 0, 1, 'C')
    pdf.set_font('Arial', '', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 4, f'Report ID: {datetime.utcnow().strftime("%Y%m%d%H%M%S")}', 0, 1, 'C')

    # Return as bytes for Streamlit download_button
    return bytes(pdf.output())
