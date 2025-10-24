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

# ============================================================================
# NEW: SAMPLE DATA GENERATOR
# ============================================================================
st.markdown("---")
st.markdown("### 🎲 Generate Sample Data")
st.markdown("Don't have a CSV file? Generate realistic sample data to test batch predictions!")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    sample_size = st.selectbox(
        "Number of sample orders:",
        options=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        index=0,
        help="Select how many sample orders to generate"
    )

with col2:
    risk_distribution = st.selectbox(
        "Risk distribution:",
        options=['Balanced Mix', 'More Low Risk', 'More High Risk', 'Random'],
        help="Choose how risks should be distributed in the sample data"
    )

with col3:
    if st.button("🎲 Generate", type="primary", use_container_width=True):
        with st.spinner(f"Generating {sample_size} sample orders..."):
            
            # Define risk distribution weights
            if risk_distribution == 'Balanced Mix':
                low_weight, med_weight, high_weight = 0.33, 0.34, 0.33
            elif risk_distribution == 'More Low Risk':
                low_weight, med_weight, high_weight = 0.50, 0.30, 0.20
            elif risk_distribution == 'More High Risk':
                low_weight, med_weight, high_weight = 0.20, 0.30, 0.50
            else:  # Random
                low_weight, med_weight, high_weight = 0.33, 0.34, 0.33
            
            # Brazilian states and cities
            brazilian_states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'ES', 'GO', 
                               'PE', 'CE', 'PA', 'AM', 'MA', 'RN', 'PB', 'AL', 'PI', 'SE']
            
            cities_by_state = {
                'SP': ['sao paulo', 'campinas', 'santos', 'sorocaba', 'guarulhos', 'osasco'],
                'RJ': ['rio de janeiro', 'niteroi', 'duque de caxias', 'sao goncalo'],
                'MG': ['belo horizonte', 'uberlandia', 'contagem', 'juiz de fora'],
                'RS': ['porto alegre', 'caxias do sul', 'pelotas'],
                'AM': ['manaus', 'itacoatiara'],
                'BA': ['salvador', 'feira de santana', 'vitoria da conquista'],
                'DF': ['brasilia'],
                'SC': ['florianopolis', 'joinville', 'blumenau'],
                'PR': ['curitiba', 'londrina', 'maringa'],
                'CE': ['fortaleza', 'juazeiro do norte']
            }
            
            categories = ['electronics', 'furniture_decor', 'health_beauty', 'sports_leisure',
                         'bed_bath_table', 'computers_accessories', 'housewares', 'watches_gifts',
                         'telephony', 'auto', 'toys', 'cool_stuff', 'perfumery', 'baby',
                         'fashion_bags_accessories']
            
            sample_orders = []
            
            # Determine how many of each risk level
            n_low = int(sample_size * low_weight)
            n_med = int(sample_size * med_weight)
            n_high = sample_size - n_low - n_med
            
            risk_types = ['low'] * n_low + ['medium'] * n_med + ['high'] * n_high
            np.random.shuffle(risk_types)
            
            for risk_type in risk_types:
                # Set parameters based on desired risk level
                if risk_type == 'low':
                    # Low risk: Big cities, short lead times, standard products
                    state = np.random.choice(['SP', 'RJ', 'MG', 'RS', 'PR'])
                    est_lead_days = np.random.uniform(2, 5)
                    sum_freight = np.random.uniform(8, 25)
                    n_items = np.random.randint(1, 3)
                    n_sellers = 1
                    purch_hour = np.random.randint(9, 18)  # Business hours
                    purch_dayofweek = np.random.randint(0, 5)  # Weekday
                    purch_month = np.random.choice([2, 3, 4, 5, 8, 9, 10])  # Non-holiday
                    
                elif risk_type == 'medium':
                    # Medium risk: Mixed scenarios
                    state = np.random.choice(brazilian_states[:15])
                    est_lead_days = np.random.uniform(5, 10)
                    sum_freight = np.random.uniform(20, 50)
                    n_items = np.random.randint(2, 5)
                    n_sellers = np.random.randint(1, 3)
                    purch_hour = np.random.randint(0, 24)
                    purch_dayofweek = np.random.randint(0, 7)
                    purch_month = np.random.randint(1, 13)
                    
                else:  # high risk
                    # High risk: Remote areas, long lead times, complex orders
                    state = np.random.choice(['AM', 'RR', 'AP', 'AC', 'RO', 'PA', 'MA', 'PI'])
                    est_lead_days = np.random.uniform(12, 25)
                    sum_freight = np.random.uniform(60, 150)
                    n_items = np.random.randint(4, 10)
                    n_sellers = np.random.randint(2, 5)
                    purch_hour = np.random.choice([22, 23, 0, 1, 2, 3])  # Late night
                    purch_dayofweek = np.random.choice([5, 6])  # Weekend
                    purch_month = np.random.choice([11, 12, 1])  # Holiday season
                
                # Get city for state
                city = np.random.choice(cities_by_state.get(state, ['capital']))
                
                # Generate order details
                seller_state = np.random.choice(['SP', 'RJ', 'MG', 'SC', 'PR'])
                category = np.random.choice(categories)
                n_products = n_items
                n_categories = np.random.randint(1, min(n_items + 1, 4))
                
                # Pricing
                item_price = np.random.uniform(30, 500)
                sum_price = item_price * n_items
                total_payment = sum_price + sum_freight
                
                # Payment
                payment_type = np.random.choice(['credit_card', 'boleto', 'debit_card', 'voucher'])
                max_installments = np.random.choice([1, 2, 3, 6, 10, 12]) if payment_type == 'credit_card' else 1
                
                # Physical dimensions
                avg_weight_g = np.random.uniform(300, 5000)
                avg_length_cm = np.random.uniform(10, 60)
                avg_height_cm = np.random.uniform(5, 40)
                avg_width_cm = np.random.uniform(8, 50)
                
                # Temporal features
                purch_is_weekend = 1 if purch_dayofweek >= 5 else 0
                purch_hour_sin = np.sin(2 * np.pi * purch_hour / 24)
                purch_hour_cos = np.cos(2 * np.pi * purch_hour / 24)
                
                order = {
                    'n_items': n_items,
                    'n_sellers': n_sellers,
                    'n_products': n_products,
                    'sum_price': round(sum_price, 2),
                    'sum_freight': round(sum_freight, 2),
                    'total_payment': round(total_payment, 2),
                    'n_payment_records': 1,
                    'max_installments': max_installments,
                    'avg_weight_g': round(avg_weight_g, 1),
                    'avg_length_cm': round(avg_length_cm, 1),
                    'avg_height_cm': round(avg_height_cm, 1),
                    'avg_width_cm': round(avg_width_cm, 1),
                    'n_seller_states': n_sellers,
                    'purch_year': 2024,
                    'purch_month': purch_month,
                    'purch_dayofweek': purch_dayofweek,
                    'purch_hour': purch_hour,
                    'purch_is_weekend': purch_is_weekend,
                    'purch_hour_sin': round(purch_hour_sin, 6),
                    'purch_hour_cos': round(purch_hour_cos, 6),
                    'est_lead_days': round(est_lead_days, 1),
                    'n_categories': n_categories,
                    'mode_category_count': np.random.randint(1, n_items + 1),
                    'paytype_boleto': 1 if payment_type == 'boleto' else 0,
                    'paytype_credit_card': 1 if payment_type == 'credit_card' else 0,
                    'paytype_debit_card': 1 if payment_type == 'debit_card' else 0,
                    'paytype_not_defined': 0,
                    'paytype_voucher': 1 if payment_type == 'voucher' else 0,
                    'mode_category': category,
                    'seller_state_mode': seller_state,
                    'customer_city': city,
                    'customer_state': state
                }
                
                sample_orders.append(order)
            
            # Create DataFrame
            sample_df = pd.DataFrame(sample_orders)
            
            # Generate CSV for download
            csv_data = sample_df.to_csv(index=False)
            
            st.success(f"✅ Generated {sample_size} sample orders!")
            
            # Show preview
            with st.expander("👁️ Preview Generated Data (First 10 Rows)"):
                st.dataframe(sample_df.head(10), use_container_width=True)
            
            # Download button
            st.download_button(
                label=f"📥 Download Sample Data ({sample_size} orders)",
                data=csv_data,
                file_name=f"sample_orders_{sample_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
            
            st.info("💡 Download this file and upload it below to test batch predictions!")

st.markdown("---")

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
    
    # Create sample template (small)
    st.markdown("---")
    st.markdown("### 📥 Download Small Template (3 orders)")
    
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
        label="📥 Download Minimal Template",
        data=csv,
        file_name="batch_prediction_template.csv",
        mime="text/csv",
        help="Download a minimal CSV file with just 3 sample orders"
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
        
        # Check for required columns
        required_features = feature_metadata['feature_names']
        missing_features = [f for f in required_features if f not in df.columns]
        
        if len(missing_features) > 0:
            st.error(f"❌ Missing Required Columns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Columns Found in Your File:**")
                st.code('\n'.join(sorted(df.columns)[:20]))
                if len(df.columns) > 20:
                    st.info(f"...and {len(df.columns) - 20} more columns")
            
            with col2:
                st.markdown("**Missing Required Columns:**")
                st.code('\n'.join(missing_features))
            
            st.info("💡 Tip: Download the template or generate sample data to see the correct format.")
            st.stop()
        
        # Check for empty dataframe
        if len(df) == 0:
            st.error("❌ File is empty - no rows to process")
            st.stop()
        
        # Check for columns with all null values (warning only)
        null_columns = [col for col in required_features if df[col].isna().all()]
        if len(null_columns) > 0:
            st.warning(f"⚠️ Warning: These columns contain all null values: {', '.join(null_columns[:5])}")
            st.info("This may affect prediction quality. Please check your data.")
        
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
    st.info("👆 Generate sample data or upload a CSV file to get started with batch predictions")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>💡 Tip: Use the sample data generator to create test files with different risk distributions</p>
</div>
""", unsafe_allow_html=True)
