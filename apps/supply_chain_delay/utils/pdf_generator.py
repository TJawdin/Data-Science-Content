"""
PDF Report Generation Module
Creates professional PDF reports for predictions
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import io


def generate_prediction_report(prediction_data, output_path=None):
    """
    Generate a PDF report for a single prediction
    
    Args:
        prediction_data: Dictionary with prediction details
        output_path: Path to save PDF (if None, returns BytesIO)
    
    Returns:
        str or BytesIO: Path to saved file or BytesIO object
    """
    # Create output buffer or file
    if output_path is None:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
    else:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
    
    # Container for elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#FF6B6B'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    elements.append(Paragraph("Supply Chain Delay Prediction Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Prediction Summary
    elements.append(Paragraph("Prediction Summary", heading_style))
    
    risk_level = prediction_data.get('risk_level', 'Unknown')
    probability = prediction_data.get('probability', 0)
    prediction = prediction_data.get('prediction', 'Unknown')
    
    # Risk color
    risk_colors = {
        'Low': colors.green,
        'Medium': colors.orange,
        'High': colors.red
    }
    risk_color = risk_colors.get(risk_level, colors.grey)
    
    summary_data = [
        ['Metric', 'Value'],
        ['Delay Probability', f'{probability*100:.1f}%'],
        ['Risk Level', risk_level],
        ['Prediction', 'Delayed' if prediction == 1 else 'On Time']
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF6B6B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    
    elements.append(summary_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Order Details
    if 'order_details' in prediction_data:
        elements.append(Paragraph("Order Details", heading_style))
        
        order_details = prediction_data['order_details']
        details_data = [['Feature', 'Value']]
        
        for key, value in order_details.items():
            if isinstance(value, float):
                details_data.append([key, f'{value:.2f}'])
            else:
                details_data.append([key, str(value)])
        
        details_table = Table(details_data, colWidths=[3*inch, 3*inch])
        details_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        
        elements.append(details_table)
        elements.append(Spacer(1, 0.3*inch))
    
    # Recommendations
    elements.append(Paragraph("Recommendations", heading_style))
    
    if risk_level == 'High':
        recommendations = [
            "• Consider expedited shipping options",
            "• Communicate proactively with customer about potential delays",
            "• Review seller performance and logistics capacity",
            "• Monitor order closely throughout fulfillment process",
            "• Consider splitting multi-seller orders if possible"
        ]
    elif risk_level == 'Medium':
        recommendations = [
            "• Monitor order progress regularly",
            "• Ensure adequate inventory at fulfillment centers",
            "• Verify seller shipping capabilities",
            "• Set realistic customer delivery expectations"
        ]
    else:
        recommendations = [
            "• Proceed with standard fulfillment process",
            "• Continue monitoring for any unexpected issues",
            "• Maintain current shipping practices"
        ]
    
    for rec in recommendations:
        elements.append(Paragraph(rec, styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Model Information
    elements.append(Paragraph("Model Information", heading_style))
    model_info = [
        f"• Model Type: LightGBM Classifier",
        f"• Model AUC-ROC: 0.789",
        f"• Optimal Threshold: 66.9%",
        f"• Training Date: 2025-10-21"
    ]
    
    for info in model_info:
        elements.append(Paragraph(info, styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    footer_text = "This report is generated by an AI/ML system. Results should be reviewed by domain experts."
    elements.append(Paragraph(footer_text, styles['Italic']))
    
    # Build PDF
    doc.build(elements)
    
    if output_path is None:
        buffer.seek(0)
        return buffer
    else:
        return output_path


def generate_batch_report(predictions_df, output_path=None):
    """
    Generate a PDF report for batch predictions
    
    Args:
        predictions_df: DataFrame with prediction results
        output_path: Path to save PDF (if None, returns BytesIO)
    
    Returns:
        str or BytesIO: Path to saved file or BytesIO object
    """
    # Create output buffer or file
    if output_path is None:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
    else:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
    
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#FF6B6B'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    elements.append(Paragraph("Batch Predictions Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
    elements.append(Paragraph(f"<b>Total Predictions:</b> {len(predictions_df)}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Summary Statistics
    elements.append(Paragraph("Summary Statistics", heading_style))
    
    if 'risk_level' in predictions_df.columns:
        risk_counts = predictions_df['risk_level'].value_counts()
        avg_prob = predictions_df['delay_probability'].mean() * 100
        
        stats_data = [
            ['Metric', 'Value'],
            ['Total Orders', str(len(predictions_df))],
            ['Average Delay Probability', f'{avg_prob:.1f}%'],
            ['High Risk Orders', str(risk_counts.get('High', 0))],
            ['Medium Risk Orders', str(risk_counts.get('Medium', 0))],
            ['Low Risk Orders', str(risk_counts.get('Low', 0))]
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF6B6B')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(stats_table)
        elements.append(Spacer(1, 0.3*inch))
    
    # Action Items
    elements.append(Paragraph("Recommended Actions", heading_style))
    
    high_risk = predictions_df[predictions_df['risk_level'] == 'High'] if 'risk_level' in predictions_df.columns else pd.DataFrame()
    
    if len(high_risk) > 0:
        elements.append(Paragraph(f"<b>High Priority:</b> {len(high_risk)} orders require immediate attention", styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
    
    actions = [
        "• Review and prioritize high-risk orders",
        "• Implement proactive customer communication",
        "• Optimize fulfillment center allocation",
        "• Monitor supplier performance metrics"
    ]
    
    for action in actions:
        elements.append(Paragraph(action, styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
    
    # Build PDF
    doc.build(elements)
    
    if output_path is None:
        buffer.seek(0)
        return buffer
    else:
        return output_path
