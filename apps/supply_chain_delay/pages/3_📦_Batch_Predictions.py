"""
Batch Predictions Page
Upload CSV files to predict delay risk for multiple orders
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils import (
    load_model_artifacts,
    predict_delay,
    prepare_features,
    validate_input,
    apply_custom_css,
    show_page_header,
    plot_probability_distribution,
    plot_risk_distribution,
    generate_batch_report
)

# Page config
st.set_page_config(page_title="Batch Predictions", page_icon="📦", layout="wide")
apply_custom_css()

# Load model
model, final_metadata, feature_metadata, threshold = load_model_artifacts()

# Header
show_page_header(
    title="Batch Predictions",
    description="Upload a CSV file with multiple orders to predict delay risk at scale",
    icon="📦"
)

# Instructions
st.markdown("""
### 📋 How to Use Batch Predictions

1. **Prepare Your CSV File**: Ensure your CSV contains all required features (see template below)
2. **Upload File**: Click the upload button and select your CSV file
3. **Review Results**: View predictions, risk distributions, and download results
4. **Export**: Download predictions as CSV or generate a PDF report

""")

# Feature requirements
with st.expander("📄 View Required CSV Format & Download Template"):
    st.markdown("### Required Columns")
    st.markdown("Your CSV file must contain the following columns:")
    
    # Display feature names in a nice format
    col1, col2, col3 = st.columns(3)
    
    features = feature_metadata['feature_names']
    third = len(features) // 3
    
    with col1:
        st.markdown("**Group 1:**")
        for feat in features[:third]:
            st.text(f"• {feat}")
    
    with col2:
        st.markdown("**Group 2:**")
        for feat in features[third:2*third]:
            st.text(f"• {feat}")
    
    with col3:
        st.markdown("**Group 3:**")
        for feat in features[2*third:]:
            st.text(f"• {feat}")
    
    # Create sample template
    st.markdown("---")
    st.markdown("### 📥 Download Template")
    
    sample_data = {
        'n_items': [2, 1, 5],
        'n_sellers': [1, 1, 3],
        'n_products': [2, 1, 4],
        'sum_price': [150.0, 45.0, 600.0],
        'sum_freight': [25.0, 12.0, 80.0],
        'total_payment': [175.0, 57.0, 680.0],
        'n_payment_records': [1, 1, 2],
        'max_installments': [3, 1, 10],
        'avg_weight_g': [2000.0, 500.0, 4000.0],
        'avg_length_cm': [30.0, 15.0, 50.0],
        'avg_height_cm': [15.0, 8.0, 30.0],
        'avg_width_cm': [20.0, 10.0, 40.0],
        'n_seller_states': [1, 1, 2],
        'purch_year': [2024, 2024, 2024],
        'purch_month': [6, 3, 12],
        'purch_dayofweek': [2, 1, 6],
        'purch_hour': [14, 10, 23],
        'purch_is_weekend': [0, 0, 1],
        'purch_hour_sin': [np.sin(2*np.pi*14/24), np.sin(2*np.pi*10/24), np.sin(2*np.pi*23/24)],
        'purch_hour_cos': [np.cos(2*np.pi*14/24), np.cos(2*np.pi*10/24), np.cos(2*np.pi*23/24)],
        'est_lead_days': [7.0, 3.0, 15.0],
        'n_categories': [1, 1, 3],
        'mode_category_count': [2, 1, 2],
        'paytype_boleto': [0, 0, 1],
        'paytype_credit_card': [1, 1, 0],
        'paytype_debit_card': [0, 0, 0],
        'paytype_not_defined': [0, 0, 0],
        'paytype_voucher': [0, 0, 0],
        'mode_category': ['electronics', 'health_beauty', 'furniture_decor'],
        'seller_state_mode': ['SP', 'SP', 'RJ'],
        'customer_city': ['sao paulo', 'sao paulo', 'manaus'],
        'customer_state': ['SP', 'SP', 'AM']
    }
    
    template_df = pd.DataFrame(sample_data)
    
    csv = template_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV Template",
        data=csv,
        file_name="batch_prediction_template.csv",
        mime="text/csv",
        help="Download a sample CSV file with the correct format"
    )

st.markdown("---")

# File upload
st.markdown("### 📤 Upload Your CSV File")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file containing order data with all required features"
)

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! Found {len(df)} orders.")
        
        # Validate data
        st.markdown("### 🔍 Data Validation")
        
        is_valid, error_message = validate_input(df, feature_metadata['feature_names'])
        
        if not is_valid:
            st.error(f"❌ Validation Error: {error_message}")
            st.info("Please ensure your CSV file contains all required columns with correct names.")
            st.stop()
        
        st.success("✅ Data validation passed!")
        
        # Show preview
        with st.expander("👁️ Preview Uploaded Data (First 10 Rows)"):
            st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Make predictions button
        if st.button("🔮 Run Batch Predictions", type="primary", use_container_width=True):
            with st.spinner("Processing predictions..."):
                # Prepare features
                features_df = prepare_features(df, feature_metadata['feature_names'])
                
                # Make predictions
                predictions, probabilities, risk_levels = predict_delay(model, features_df, threshold)
                
                # Add results to dataframe
                df['delay_probability'] = probabilities
                df['delay_probability_pct'] = probabilities * 100
                df['prediction'] = predictions
                df['prediction_label'] = ['Delayed' if p == 1 else 'On Time' for p in predictions]
                df['risk_level'] = risk_levels
                
                # Store in session state
                st.session_state['prediction_results'] = df
                st.session_state['predictions_made'] = True
            
            st.success("✅ Predictions completed successfully!")
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please check that your file format matches the template and try again.")

# Display results if predictions have been made
if 'predictions_made' in st.session_state and st.session_state['predictions_made']:
    df = st.session_state['prediction_results']
    
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Orders",
            len(df),
            help="Total number of orders processed"
        )
    
    with col2:
        avg_prob = df['delay_probability'].mean() * 100
        st.metric(
            "Avg Delay Risk",
            f"{avg_prob:.1f}%",
            help="Average delay probability across all orders"
        )
    
    with col3:
        high_risk_count = (df['risk_level'] == 'High').sum()
        st.metric(
            "High Risk Orders",
            high_risk_count,
            help="Number of orders with high delay risk"
        )
    
    with col4:
        predicted_delays = (df['prediction'] == 1).sum()
        st.metric(
            "Predicted Delays",
            predicted_delays,
            help="Number of orders predicted to be delayed"
        )
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Probability Distribution")
        fig_dist = plot_probability_distribution(df['delay_probability'].values, threshold)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Risk Level Distribution")
        fig_risk = plot_risk_distribution(df['risk_level'].values)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed results table
    st.markdown("### 📋 Detailed Results")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=['Low', 'Medium', 'High'],
            default=['Low', 'Medium', 'High']
        )
    
    with col2:
        prediction_filter = st.multiselect(
            "Filter by Prediction",
            options=['On Time', 'Delayed'],
            default=['On Time', 'Delayed']
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            options=['delay_probability', 'risk_level', 'sum_price', 'est_lead_days'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    # Apply filters
    filtered_df = df[
        (df['risk_level'].isin(risk_filter)) &
        (df['prediction_label'].isin(prediction_filter))
    ]
    
    # Sort
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
    
    # Select important columns to display
    display_cols = [
        'delay_probability_pct', 'risk_level', 'prediction_label',
        'n_items', 'n_sellers', 'total_payment', 'est_lead_days',
        'customer_state', 'seller_state_mode'
    ]
    
    # Display filtered and sorted data
    st.dataframe(
        filtered_df[display_cols].style.format({
            'delay_probability_pct': '{:.1f}%',
            'total_payment': 'R$ {:.2f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.info(f"Showing {len(filtered_df)} of {len(df)} orders")
    
    st.markdown("---")
    
    # High priority orders
    high_risk_df = df[df['risk_level'] == 'High'].sort_values('delay_probability', ascending=False)
    
    if len(high_risk_df) > 0:
        st.markdown("### 🚨 High Priority Orders (Immediate Attention Required)")
        st.warning(f"Found {len(high_risk_df)} high-risk orders that need immediate review")
        
        priority_cols = [
            'delay_probability_pct', 'total_payment', 'n_items', 'n_sellers',
            'est_lead_days', 'customer_state', 'seller_state_mode'
        ]
        
        st.dataframe(
            high_risk_df[priority_cols].head(20).style.format({
                'delay_probability_pct': '{:.1f}%',
                'total_payment': 'R$ {:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # Export options
    st.markdown("### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        csv_export = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=csv_export,
            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # High risk only CSV
        if len(high_risk_df) > 0:
            high_risk_csv = high_risk_df.to_csv(index=False)
            st.download_button(
                label="🚨 Download High Risk Only (CSV)",
                data=high_risk_csv,
                file_name=f"high_risk_orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # PDF report
    if st.button("📄 Generate PDF Report", use_container_width=True):
        try:
            with st.spinner("Generating PDF report..."):
                pdf_buffer = generate_batch_report(df)
                st.download_button(
                    label="💾 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            st.success("✅ PDF report generated successfully!")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
    
    # Action recommendations
    st.markdown("---")
    st.markdown("### 💡 Recommended Actions")
    
    if len(high_risk_df) > 0:
        st.error("**Immediate Actions for High-Risk Orders:**")
        st.markdown("""
        1. 🔍 **Review Priority Orders**: Focus on the highest probability delays first
        2. 📞 **Proactive Communication**: Contact customers for high-risk orders
        3. 🚀 **Expedite Shipping**: Consider faster shipping options where feasible
        4. 📊 **Monitor Closely**: Set up enhanced tracking for all high-risk orders
        5. 💼 **Resource Allocation**: Prioritize warehouse and logistics resources
        """)
    else:
        st.success("✅ **No high-risk orders detected. Maintain standard operations.**")
    
    # Clear results button
    if st.button("🔄 Clear Results and Upload New File"):
        st.session_state['predictions_made'] = False
        if 'prediction_results' in st.session_state:
            del st.session_state['prediction_results']
        st.rerun()

else:
    # Show placeholder when no predictions yet
    st.info("👆 Upload a CSV file to get started with batch predictions")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>💡 Tip: Process large batches efficiently by ensuring your CSV is properly formatted before upload</p>
</div>
""", unsafe_allow_html=True)
