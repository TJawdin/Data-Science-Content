# Supply Chain Delay Prediction System

A comprehensive, user-friendly Streamlit application for predicting delivery delays using machine learning.

## 🌟 Features

- **Single Order Predictions**: Make individual predictions with detailed explanations
- **Batch Processing**: Upload CSV files for bulk predictions
- **Example Scenarios**: Pre-configured examples to understand model behavior
- **Time Trend Analysis**: Analyze temporal patterns in delivery delays
- **Geographic Distribution**: Visualize delay risk across different regions
- **PDF Reports**: Generate professional prediction reports
- **Interactive Visualizations**: Modern, responsive charts using Plotly
- **Risk Assessment**: Three-tier risk classification (Low/Medium/High)

## 📋 Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the repository
cd supply_chain_app

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Artifacts

Place your model artifacts in the `artifacts/` directory:
- `best_model_lightgbm.pkl` - Trained LightGBM model
- `optimal_threshold_lightgbm.txt` - Optimal classification threshold
- `final_metadata.json` - Model performance metadata
- `feature_metadata.json` - Feature specifications

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
supply_chain_app/
├── app.py                          # Main application file
├── config.py                       # Configuration and constants
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── artifacts/                      # Model artifacts (you need to add these)
│   ├── best_model_lightgbm.pkl
│   ├── optimal_threshold_lightgbm.txt
│   ├── final_metadata.json
│   └── feature_metadata.json
├── utils/                         # Utility modules
│   ├── __init__.py
│   ├── model_loader.py           # Model loading and prediction
│   ├── feature_engineering.py    # Feature preparation and validation
│   ├── visualization.py          # Chart creation
│   └── pdf_generator.py          # PDF report generation
└── pages/                         # Multi-page app sections
    ├── 1_🎯_Example_Scenarios.py
    ├── 2_📊_Single_Prediction.py
    ├── 3_📦_Batch_Predictions.py
    ├── 4_📈_Time_Trends.py
    └── 5_🗺️_Geographic_Map.py
```

## 🎯 Usage Guide

### Single Prediction

1. Navigate to "📊 Single Prediction" page
2. Fill in order details (items, prices, dimensions)
3. Enter temporal information (purchase date/time)
4. Select payment method and geographic details
5. Click "Predict Delay Risk"
6. Review results and download PDF report

### Batch Predictions

1. Navigate to "📦 Batch Predictions" page
2. Download the CSV template
3. Fill in your order data
4. Upload the CSV file
5. Click "Run Batch Prediction"
6. Review results and download predictions

### Example Scenarios

1. Navigate to "🎯 Example Scenarios" page
2. Select a pre-configured scenario
3. Click "Run Prediction"
4. Compare different scenarios to understand model behavior

### Time Trends

1. Navigate to "📈 Time Trends" page
2. Upload data or use sample data
3. Select time dimension (month, day, hour)
4. Analyze patterns and insights

### Geographic Analysis

1. Navigate to "🗺️ Geographic Map" page
2. Upload data or use sample data
3. Select geographic level (state or city)
4. Identify high-risk regions

## 📊 Model Information

### Performance Metrics
- **AUC**: 0.7888
- **Precision**: 0.3040
- **Recall**: 0.4424
- **F1 Score**: 0.3603

### Risk Bands
- **Low Risk**: 0-30% delay probability
- **Medium Risk**: 30-67% delay probability
- **High Risk**: 67-100% delay probability

### Features (32 total)

#### Order Details (8)
- Number of items, sellers, products
- Prices and freight costs
- Payment records and installments

#### Product Dimensions (4)
- Average weight, length, height, width

#### Temporal Features (8)
- Purchase year, month, day, hour
- Weekend indicator
- Cyclical time features
- Estimated lead days

#### Categories & Geographic (7)
- Number of categories
- Mode category and count
- Seller states
- Customer city and state

#### Payment Types (5 - one-hot encoded)
- Boleto, Credit Card, Debit Card, Voucher, Not Defined

## 🔧 Configuration

### Customization

Edit `config.py` to customize:
- Risk band thresholds
- Feature groups and display names
- Color schemes
- Default values
- Brazilian states and product categories

### Streamlit Theme

Edit `.streamlit/config.toml` to change:
- Color scheme
- Font settings
- Upload size limits

## 📝 CSV File Format

### Required Columns

Your CSV file must include all 32 features:

```
n_items, n_sellers, n_products, sum_price, sum_freight, total_payment,
n_payment_records, max_installments, avg_weight_g, avg_length_cm,
avg_height_cm, avg_width_cm, n_seller_states, purch_year, purch_month,
purch_dayofweek, purch_hour, purch_is_weekend, purch_hour_sin,
purch_hour_cos, est_lead_days, n_categories, mode_category_count,
paytype_boleto, paytype_credit_card, paytype_debit_card,
paytype_not_defined, paytype_voucher, mode_category,
seller_state_mode, customer_city, customer_state
```

### Example Row

```csv
n_items,n_sellers,n_products,sum_price,sum_freight,total_payment,...
1,1,1,100.0,15.0,115.0,1,1,500.0,20.0,10.0,15.0,...
```

Download the template from the Batch Predictions page.

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure all artifact files are in the `artifacts/` directory
   - Check file paths in `config.py`

2. **Import errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Ensure Python version is 3.8 or higher

3. **CSV upload fails**
   - Verify all required columns are present
   - Check for correct data types
   - Remove any null values in required fields

4. **Prediction errors**
   - Validate input ranges (e.g., month 1-12, hour 0-23)
   - Ensure exactly one payment type is selected
   - Check geographic data (valid state codes)

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment

#### Streamlit Community Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Add artifact files to repository or use secrets

#### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t supply-chain-app .
docker run -p 8501:8501 supply-chain-app
```

## 📈 Performance Optimization

### Caching
- Model and metadata are cached using `@st.cache_resource`
- Data transformations are cached for better performance
- Clear cache with "Clear Cache" button in Streamlit menu

### Large Files
- For CSV files >100MB, consider chunking
- Process in batches for very large datasets
- Use database connection for production deployments

## 🔒 Security Considerations

- Model artifacts should not be committed to public repositories
- Use environment variables for sensitive configurations
- Implement authentication for production deployments
- Validate and sanitize all user inputs

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [LightGBM Documentation](https://lightgbm.readthedocs.io)
- [Plotly Documentation](https://plotly.com/python/)

## 🤝 Contributing

To contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and commercial use.

## 📧 Support

For technical support or questions:
- Check the troubleshooting section
- Review the usage guide
- Contact your system administrator

---

**Built with ❤️ using Streamlit**

Version: 1.0.0
Last Updated: October 2024
