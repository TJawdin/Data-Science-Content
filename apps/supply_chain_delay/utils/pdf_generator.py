"""
PDF Report Generator
Creates professional PDF reports for order risk predictions
"""

from fpdf import FPDF
import pandas as pd
from datetime import datetime

class RiskReportPDF(FPDF):
    """Custom PDF class for risk assessment reports"""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        """Header for each page"""
        self.set_font('Arial', 'B', 16)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, 'Supply Chain Delay Risk Assessment Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        """Footer for each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        """Add a chapter title"""
        self.set_font('Arial', 'B', 14)
        self.set_text_color(52, 73, 94)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
    
    def chapter_body(self, body):
        """Add chapter body text"""
        self.set_font('Arial', '', 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, body)
        self.ln()
    
    def add_colored_box(self, text, color_rgb):
        """Add a colored info box"""
        self.set_fill_color(*color_rgb)
        self.set_font('Arial', 'B', 12)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, text, 0, 1, 'C', fill=True)
        self.ln(2)


def generate_risk_report(order_data, prediction_result, features_df):
    """
    Generate a PDF risk assessment report
    
    Parameters:
    -----------
    order_data : dict
        Original order information
    prediction_result : dict
        Prediction results from model
    features_df : pd.DataFrame
        Calculated features
    
    Returns:
    --------
    bytes : PDF file as bytes
    """
    
    pdf = RiskReportPDF()
    pdf.add_page()
    
    # ========================================================================
    # EXECUTIVE SUMMARY
    # ========================================================================
    
    pdf.chapter_title('Executive Summary')
    
    # Risk level box with color
    risk_level = prediction_result['risk_level']
    risk_score = prediction_result['risk_score']
    
    if risk_level == 'LOW':
        color = (46, 204, 113)  # Green
    elif risk_level == 'MEDIUM':
        color = (243, 156, 18)  # Orange
    else:
        color = (231, 76, 60)  # Red
    
    pdf.add_colored_box(
        f"RISK LEVEL: {risk_level} ({risk_score}/100)",
        color
    )
    
    pdf.chapter_body(
        f"Prediction: {prediction_result['prediction_label']}\n"
        f"Late Delivery Probability: {prediction_result['probability']:.1%}\n"
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    
    # ========================================================================
    # ORDER DETAILS
    # ========================================================================
    
    pdf.chapter_title('Order Details')
    
    order_summary = f"""
Number of Items: {order_data.get('num_items', 'N/A')}
Number of Sellers: {order_data.get('num_sellers', 'N/A')}
Total Order Value: ${order_data.get('total_order_value', 0):.2f}
Total Shipping Cost: ${order_data.get('total_shipping_cost', 0):.2f}
Total Weight: {order_data.get('total_weight_g', 0)}g

Shipping Distance: {order_data.get('avg_shipping_distance_km', 0)}km
Cross-State Shipping: {'Yes' if order_data.get('is_cross_state', 0) == 1 else 'No'}
Estimated Delivery: {order_data.get('estimated_days', 0)} days
Weekend Order: {'Yes' if order_data.get('is_weekend_order', 0) == 1 else 'No'}
Holiday Season: {'Yes' if order_data.get('is_holiday_season', 0) == 1 else 'No'}
    """
    
    pdf.chapter_body(order_summary.strip())
    
    # ========================================================================
    # RISK ASSESSMENT
    # ========================================================================
    
    pdf.chapter_title('Risk Assessment Breakdown')
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Risk Score Interpretation:', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    if risk_score < 30:
        interpretation = """
LOW RISK (0-30): This order has a low probability of late delivery. Standard 
processing procedures are recommended. No special intervention required.
        """
    elif risk_score < 70:
        interpretation = """
MEDIUM RISK (30-70): This order has moderate late delivery risk. Monitor 
closely and ensure optimal carrier selection. Consider proactive customer 
communication.
        """
    else:
        interpretation = """
HIGH RISK (70-100): This order has a high probability of late delivery. 
Immediate action recommended: upgrade shipping, prioritize processing, and 
proactively contact the customer.
        """
    
    pdf.multi_cell(0, 5, interpretation.strip())
    pdf.ln(3)
    
    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    
    pdf.chapter_title('Recommended Actions')
    
    if risk_level == 'HIGH':
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
    elif risk_level == 'MEDIUM':
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
    
    # ========================================================================
    # TOP RISK FACTORS (New Page)
    # ========================================================================
    
    pdf.add_page()
    pdf.chapter_title('Top Contributing Factors')
    
    # Get top 10 features by absolute value
    feature_contributions = features_df.iloc[0].to_dict()
    top_features = sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(100, 7, 'Feature', 1, 0, 'L')
    pdf.cell(40, 7, 'Value', 1, 1, 'C')
    
    pdf.set_font('Arial', '', 9)
    for feature, value in top_features:
        # Truncate long feature names and remove special chars
        feature_name = str(feature).encode('ascii', 'ignore').decode('ascii')
        feature_name = feature_name[:40] + '...' if len(feature_name) > 40 else feature_name
        pdf.cell(100, 6, feature_name, 1, 0, 'L')
        pdf.cell(40, 6, f'{value:.2f}', 1, 1, 'C')
    
    pdf.ln(5)
    
    # ========================================================================
    # MODEL INFORMATION
    # ========================================================================
    
    pdf.chapter_title('Model Information')
    
    model_info = """
Model Type: XGBoost Classifier
Features: 29 domain-engineered features
Training Data: 100,000+ historical e-commerce orders
Performance: AUC-ROC >= 0.85

This prediction is based on machine learning analysis of historical delivery 
patterns and order characteristics. While highly accurate, predictions should 
be used as guidance alongside human judgment and business context.
    """
    
    pdf.chapter_body(model_info.strip())
    
    # ========================================================================
    # DISCLAIMER
    # ========================================================================
    
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 4, 
        "DISCLAIMER: This risk assessment is generated by a machine learning model and should be "
        "used as a decision support tool. Actual delivery outcomes may vary based on factors not "
        "captured in the model. Always apply business judgment when making operational decisions."
    )
    
    # ========================================================================
    # Footer Information
    # ========================================================================
    
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 9)
    pdf.set_text_color(52, 73, 94)
    pdf.cell(0, 5, 'Generated by Supply Chain Delay Prediction System', 0, 1, 'C')
    pdf.set_font('Arial', '', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 4, f'Report ID: {datetime.now().strftime("%Y%m%d%H%M%S")}', 0, 1, 'C')
    
    # Return PDF as bytes
    return pdf.output(dest='S').encode('latin-1')
