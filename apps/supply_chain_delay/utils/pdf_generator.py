"""
PDF report generation utilities
"""

import pandas as pd
from datetime import datetime


def generate_prediction_report(prediction_data, output_format='dict'):
    """
    Generate prediction report data
    
    Args:
        prediction_data: Dictionary with prediction information
        output_format: Format of output ('dict', 'markdown', 'html')
    
    Returns:
        Report data in specified format
    """
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prediction': prediction_data
    }
    
    if output_format == 'markdown':
        return format_as_markdown(report)
    elif output_format == 'html':
        return format_as_html(report)
    else:
        return report


def generate_batch_report(results_df, summary_stats):
    """
    Generate batch prediction report
    
    Args:
        results_df: DataFrame with batch predictions
        summary_stats: Dictionary with summary statistics
    
    Returns:
        dict: Report data
    """
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_predictions': len(results_df),
        'summary': summary_stats,
        'results': results_df
    }


def format_as_markdown(report):
    """
    Format report as markdown
    
    Args:
        report: Report dictionary
    
    Returns:
        str: Markdown formatted report
    """
    md = f"""
# Supply Chain Delay Prediction Report

**Generated:** {report['timestamp']}

## Prediction Results

- **Delay Probability:** {report['prediction']['probability']:.2f}%
- **Risk Category:** {report['prediction']['risk_category']}
- **Classification:** {'High Risk' if report['prediction']['is_high_risk'] else 'Low Risk'}

## Key Features

"""
    
    if 'features' in report['prediction']:
        for feature, value in report['prediction']['features'].items():
            md += f"- **{feature}:** {value}\n"
    
    return md


def format_as_html(report):
    """
    Format report as HTML
    
    Args:
        report: Report dictionary
    
    Returns:
        str: HTML formatted report
    """
    risk_colors = {
        'Low': '#00CC96',
        'Medium': '#FFA500',
        'High': '#EF553B'
    }
    
    risk = report['prediction']['risk_category']
    color = risk_colors.get(risk, '#888888')
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #1f77b4; }}
            .risk-badge {{
                background-color: {color};
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                display: inline-block;
                font-weight: bold;
            }}
            .metric {{ margin: 20px 0; }}
            .metric-label {{ font-weight: bold; color: #555; }}
            .metric-value {{ font-size: 1.2em; color: #333; }}
        </style>
    </head>
    <body>
        <h1>Supply Chain Delay Prediction Report</h1>
        <p><strong>Generated:</strong> {report['timestamp']}</p>
        
        <div class="metric">
            <div class="metric-label">Risk Category</div>
            <div class="risk-badge">{risk}</div>
        </div>
        
        <div class="metric">
            <div class="metric-label">Delay Probability</div>
            <div class="metric-value">{report['prediction']['probability']:.2f}%</div>
        </div>
    </body>
    </html>
    """
    
    return html


def create_excel_report(results_df, summary_stats, filename='prediction_report.xlsx'):
    """
    Create Excel report with multiple sheets
    
    Args:
        results_df: DataFrame with predictions
        summary_stats: Dictionary with summary statistics
        filename: Output filename
    
    Returns:
        BytesIO: Excel file as bytes
    """
    from io import BytesIO
    
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write predictions
        results_df.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Write summary
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    output.seek(0)
    return output


def format_batch_summary(summary_stats):
    """
    Format batch summary statistics as text
    
    Args:
        summary_stats: Dictionary with summary statistics
    
    Returns:
        str: Formatted summary text
    """
    text = f"""
Batch Prediction Summary
========================

Total Predictions: {summary_stats.get('total', 0)}

Risk Distribution:
- Low Risk: {summary_stats.get('low_risk', 0)} ({summary_stats.get('low_risk_pct', 0):.1f}%)
- Medium Risk: {summary_stats.get('medium_risk', 0)} ({summary_stats.get('medium_risk_pct', 0):.1f}%)
- High Risk: {summary_stats.get('high_risk', 0)} ({summary_stats.get('high_risk_pct', 0):.1f}%)

Average Delay Probability: {summary_stats.get('avg_probability', 0):.2f}%
"""
    return text
