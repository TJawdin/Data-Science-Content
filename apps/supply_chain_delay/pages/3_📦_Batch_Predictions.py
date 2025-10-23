import streamlit as st
import pandas as pd
import numpy as np
from utils.model_loader import ModelLoader
from utils.visualization import create_risk_distribution
import io

st.set_page_config(page_title="Batch Predictions", page_icon="📦", layout="wide")

st.title("📦 Batch Prediction System")
st.markdown("Upload a CSV file with multiple orders to predict delays in bulk")

# Initialize model loader
@st.cache_resource
def init_model_loader():
    return ModelLoader(artifacts_path="./artifacts")

model_loader = init_model_loader()
model = model_loader.load_model()
metadata, feature_metadata = model_loader.load_metadata()

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} orders")
        
        # Display sample
        with st.expander("View Data Sample"):
            st.dataframe(df.head(10))
        
        # Check for required columns
        required_columns = feature_metadata['feature_names']
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Required columns: " + ", ".join(required_columns))
        else:
            # Make predictions
            if st.button("🔮 Predict All Orders", type="primary"):
                with st.spinner("Processing predictions..."):
                    
                    # Ensure correct column order
                    df_features = df[required_columns]
                    
                    # Get predictions
                    predictions, probabilities, risk_levels = model_loader.predict_with_probability(df_features)
                    
                    # Add results to dataframe
                    df['delay_prediction'] = ['Delayed' if p == 1 else 'On Time' for p in predictions]
                    df['delay_probability'] = probabilities * 100
                    df['risk_level'] = risk_levels
                    
                    st.success("✅ Predictions complete!")
                    
                    # Display summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        delayed_count = sum(predictions)
                        st.metric("Predicted Delays", f"{delayed_count} ({delayed_count/len(predictions)*100:.1f}%)")
                    
                    with col2:
                        high_risk = sum([1 for r in risk_levels if r == 'High'])
                        st.metric("High Risk Orders", f"{high_risk} ({high_risk/len(risk_levels)*100:.1f}%)")
                    
                    with col3:
                        avg_risk = np.mean(probabilities) * 100
                        st.metric("Average Risk", f"{avg_risk:.1f}%")
                    
                    with col4:
                        on_time = len(predictions) - delayed_count
                        st.metric("On Time Orders", f"{on_time} ({on_time/len(predictions)*100:.1f}%)")
                    
                    # Risk distribution chart
                    st.subheader("Risk Distribution")
                    fig = create_risk_distribution(pd.DataFrame({
                        'risk_probability': probabilities * 100,
                        'risk_level': risk_levels
                    }))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed results
                    st.subheader("Detailed Results")
                    
                    # Filter options
                    col1, col2 = st.columns(2)
                    with col1:
                        filter_risk = st.selectbox("Filter by Risk Level", 
                                                  ["All", "High", "Medium", "Low"])
                    with col2:
                        filter_prediction = st.selectbox("Filter by Prediction", 
                                                        ["All", "Delayed", "On Time"])
                    
                    # Apply filters
                    filtered_df = df.copy()
                    if filter_risk != "All":
                        filtered_df = filtered_df[filtered_df['risk_level'] == filter_risk]
                    if filter_prediction != "All":
                        filtered_df = filtered_df[filtered_df['delay_prediction'] == filter_prediction]
                    
                    # Display filtered results
                    st.dataframe(
                        filtered_df[['delay_prediction', 'delay_probability', 'risk_level'] + 
                                   list(df.columns[:10])],  # Show first 10 original columns
                        use_container_width=True
                    )
                    
                    # Download results
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f'delay_predictions_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv'
                    )
                    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV has all required columns in the correct format")

else:
    # Show template
    st.info("👆 Upload a CSV file to begin batch predictions")
    
    with st.expander("📋 CSV Template"):
        st.markdown("Your CSV should include these columns:")
        
        template_data = {col: ['example_value'] for col in feature_metadata['feature_names']}
        template_df = pd.DataFrame(template_data)
        
        st.dataframe(template_df)
        
        # Download template
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv,
            file_name='batch_prediction_template.csv',
            mime='text/csv'
        )
