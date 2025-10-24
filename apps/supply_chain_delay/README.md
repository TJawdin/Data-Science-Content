# 🚚 Supply Chain Delay Prediction System

A comprehensive, production-ready Streamlit application for predicting delivery delays using machine learning. Features inverted probability correction, interactive visualizations, batch processing, and geographic risk analysis.

[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat&logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-green?style=flat)](https://lightgbm.readthedocs.io)

## 🌟 Key Features

### Core Functionality
- **✅ Inverted Probability Correction**: Automatic correction for backwards-trained models
- **📊 Single Order Predictions**: Individual predictions with detailed risk gauges
- **📦 Batch Processing**: Upload CSV files for bulk predictions (up to 1000+ orders)
- **🎲 Sample Data Generator**: Generate realistic test data with configurable risk distributions
- **🎯 Example Scenarios**: Pre-configured low, medium, and high-risk examples
- **📈 Time Trend Analysis**: Analyze temporal patterns (hourly, daily, monthly)
- **🗺️ Geographic Distribution**: Visualize delay risk by state and city with formatted names
- **📄 PDF Reports**: Generate professional prediction reports
- **🎨 Interactive Visualizations**: Modern, responsive charts using Plotly
- **🔍 Risk Assessment**: Three-tier risk classification with dynamic color coding

### Enhanced Features
- **State/City Formatting**: Brazilian locations displayed as "SP - São Paulo" and "São Paulo"
- **Enhanced Risk Gauge**: Visual gauge with zone highlighting and exact probability markers
- **Session State Management**: Smooth navigation without data loss
- **Data Validation**: Comprehensive input validation with helpful error messages
- **Export Options**: CSV downloads for all analysis results

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for batch processing)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Key Dependencies
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
plotly>=5.17.0
python-dateutil>=2.8.2
openpyxl>=3.1.0
reportlab>=4.0.0
```

See `requirements.txt` for complete list.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd supply_chain_app

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Model Artifacts

Place your model artifacts in the `artifacts/` directory:

```
artifacts/
├── best_model_lightgbm.pkl           # Trained LightGBM model
├── optimal_threshold_lightgbm.txt    # Classification threshold
├── final_metadata.json               # Model performance metrics
└── feature_metadata.json             # Feature specifications
```

**Required Metadata Structure:**

`final_metadata.json`:
```json
{
  "auc": 0.7888,
  "precision": 0.3040,
  "recall": 0.4424,
  "f1_score": 0.3603,
  "risk_bands": {
    "low_max": 30,
    "med_max": 67
  }
}
```

`feature_metadata.json`:
```json
{
  "feature_names": ["n_items", "n_sellers", ...]
}
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open automatically at `http://localhost:8501`

## 📁 Project Structure

```
supply_chain_app/
├── app.py                              # Main application entry point
├── config.py                           # Configuration and constants
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── .streamlit/
│   └── config.toml                    # Streamlit theme configuration
│
├── artifacts/                         # Model files (not in repo)
│   ├── best_model_lightgbm.pkl
│   ├── optimal_threshold_lightgbm.txt
│   ├── final_metadata.json
│   └── feature_metadata.json
│
├── utils/                             # Core utility modules
│   ├── __init__.py                   # Package initialization with exports
│   ├── model_loader.py               # Model loading and predictions
│   ├── feature_engineering.py        # Feature preparation
│   ├── visualization.py              # Chart creation (enhanced gauges)
│   ├── formatting.py                 # State/city name formatting
│   ├── theme_adaptive.py             # Theme and styling
│   └── pdf_generator.py              # PDF report generation
│
└── pages/                             # Multi-page app sections
    ├── 1_🎯_Example_Scenarios.py     # Pre-configured examples
    ├── 2_🔮_Single_Prediction.py     # Individual order predictions
    ├── 3_📦_Batch_Predictions.py     # Bulk CSV processing
    ├── 4_🗺️_Geographic_Map.py        # Regional risk analysis
    └── 5_📈_Time_Analysis.py         # Temporal pattern analysis
```

## 🎯 Page-by-Page Usage Guide

### Home Page
- Overview of system capabilities
- Model performance metrics with visual indicators
- Quick navigation to all features
- System status and health checks

### 1️⃣ Example Scenarios
**Purpose**: Understand model behavior with pre-configured examples

**Features**:
- Three risk level examples (Low, Medium, High)
- Side-by-side scenario comparison
- Feature importance visualization
- Instant predictions without data entry

**Use Case**: Training, demonstrations, model validation

### 2️⃣ Single Prediction
**Purpose**: Predict delay risk for individual orders

**How to Use**:
1. **Order Details**: Enter items, sellers, prices, freight
2. **Product Info**: Add dimensions (weight, length, height, width)
3. **Temporal Data**: Select purchase date and time
4. **Payment**: Choose payment method
5. **Geographic**: Select customer state and city
6. **Predict**: Click to see results with risk gauge
7. **Export**: Download PDF report

**Key Features**:
- Real-time validation
- Interactive risk gauge with zone highlighting
- Detailed risk explanation
- Professional PDF reports
- State/city names properly formatted

### 3️⃣ Batch Predictions
**Purpose**: Process multiple orders efficiently

**How to Use**:
1. **Generate Sample Data** (NEW!):
   - Select quantity (100-1000 orders)
   - Choose risk distribution
   - Download generated CSV
   
2. **Or Download Template**:
   - Small template (3 orders)
   - Shows required format
   
3. **Upload Your File**:
   - Automatic validation
   - Helpful error messages
   - Preview before processing
   
4. **Run Predictions**:
   - Processes all orders
   - Shows distribution charts
   - Filter and sort results
   
5. **Export**:
   - Full results CSV
   - High-risk orders only
   - PDF report

**Performance**:
- Handles 1000+ orders efficiently
- Progress indicators for large files
- Session state preservation

### 4️⃣ Geographic Map
**Purpose**: Analyze delivery risk by location

**How to Use**:
1. **Choose Data Source**:
   - Generate sample data (100-1000 orders)
   - Use batch prediction results
   
2. **Select Geographic Level**:
   - Customer State (formatted: "SP - São Paulo")
   - Customer City (formatted: "São Paulo")
   
3. **Analyze**:
   - Risk distribution bar chart
   - Highest/lowest risk regions
   - Detailed statistics table
   - Strategic recommendations

**Key Features**:
- Session state management (no page resets!)
- Proper aggregation before formatting
- Color-coded risk levels
- CSV export capability

### 5️⃣ Time Analysis
**Purpose**: Identify temporal patterns in delay risk

**How to Use**:
1. Upload data or generate samples
2. Select time dimension:
   - Hour of day
   - Day of week
   - Month of year
3. View insights:
   - Highest/lowest risk periods
   - Trend charts
   - Strategic recommendations

**Insights Provided**:
- Peak risk hours
- Seasonal patterns
- Weekday vs weekend trends
- Holiday season impacts

## 🔧 Technical Details

### Inverted Probability Correction

**The Issue**: Model was trained with inverted labels (delays marked as 0, on-time as 1)

**The Solution**: Automatic probability inversion in `predict_delay()` function:

```python
def predict_delay(model, input_data, threshold):
    """Predict with INVERTED probabilities to fix backwards model"""
    probs = model.predict_proba(input_data)[:, 1]
    inverted_probs = 1 - probs  # FIX: Invert the probabilities
    predictions = (inverted_probs >= threshold).astype(int)
    risk_levels = [get_risk_category(p * 100) for p in inverted_probs]
    return predictions, inverted_probs, risk_levels
```

**Result**: Intuitive behavior where high risk factors = high delay probability

### Enhanced Risk Gauge

Features:
- **Dynamic zones**: Low (green), Medium (orange), High (red)
- **Exact marker**: Bar stops at precise probability
- **Zone highlighting**: Bold borders for active risk zone
- **Automatic formatting**: Handles both decimal (0-1) and percentage (0-100) inputs

### State and City Formatting

```python
# State: "SP" → "SP - São Paulo"
format_state_name('SP')  # Returns: "SP - São Paulo"

# City: "sao paulo" → "São Paulo"
format_city_name('sao paulo')  # Returns: "São Paulo"
```

**Benefits**:
- Professional presentation
- Better user experience
- Maintains data integrity (formatting after aggregation)

### Model Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC** | 0.7888 | Good discriminative ability |
| **Precision** | 0.3040 | 30% of flagged orders actually delay |
| **Recall** | 0.4424 | Catches 44% of actual delays |
| **F1 Score** | 0.3603 | Balanced performance measure |

### Risk Bands

| Level | Probability Range | Color | Action |
|-------|------------------|-------|--------|
| **Low** | 0-30% | 🟢 Green | Standard processing |
| **Medium** | 30-67% | 🟠 Orange | Monitor closely |
| **High** | 67-100% | 🔴 Red | Immediate attention |

## 📊 Features Reference

### All 32 Model Features

#### Order Details (8 features)
- `n_items`: Number of items in order
- `n_sellers`: Number of different sellers
- `n_products`: Number of unique products
- `sum_price`: Total price of items (R$)
- `sum_freight`: Total shipping cost (R$)
- `total_payment`: Total payment amount (R$)
- `n_payment_records`: Number of payment transactions
- `max_installments`: Maximum installment count

#### Product Dimensions (4 features)
- `avg_weight_g`: Average weight in grams
- `avg_length_cm`: Average length in cm
- `avg_height_cm`: Average height in cm
- `avg_width_cm`: Average width in cm

#### Temporal Features (8 features)
- `purch_year`: Year of purchase
- `purch_month`: Month (1-12)
- `purch_dayofweek`: Day of week (0=Monday)
- `purch_hour`: Hour of day (0-23)
- `purch_is_weekend`: Weekend indicator (0/1)
- `purch_hour_sin`: Cyclical hour encoding (sin)
- `purch_hour_cos`: Cyclical hour encoding (cos)
- `est_lead_days`: Estimated delivery lead time

#### Categories & Geographic (7 features)
- `n_categories`: Number of product categories
- `mode_category_count`: Count of most common category
- `mode_category`: Most common product category
- `n_seller_states`: Number of different seller states
- `seller_state_mode`: Primary seller state
- `customer_city`: Customer city (lowercase)
- `customer_state`: Customer state (2-letter code)

#### Payment Types (5 one-hot encoded)
- `paytype_boleto`: Boleto payment (0/1)
- `paytype_credit_card`: Credit card (0/1)
- `paytype_debit_card`: Debit card (0/1)
- `paytype_voucher`: Voucher (0/1)
- `paytype_not_defined`: Not defined (0/1)

### Brazilian State Codes

| Code | State Name | Code | State Name |
|------|-----------|------|-----------|
| AC | Acre | PA | Pará |
| AL | Alagoas | PB | Paraíba |
| AP | Amapá | PE | Pernambuco |
| AM | Amazonas | PI | Piauí |
| BA | Bahia | PR | Paraná |
| CE | Ceará | RJ | Rio de Janeiro |
| DF | Distrito Federal | RN | Rio Grande do Norte |
| ES | Espírito Santo | RO | Rondônia |
| GO | Goiás | RR | Roraima |
| MA | Maranhão | RS | Rio Grande do Sul |
| MG | Minas Gerais | SC | Santa Catarina |
| MS | Mato Grosso do Sul | SE | Sergipe |
| MT | Mato Grosso | SP | São Paulo |
|  |  | TO | Tocantins |

## 📝 CSV File Format

### Required Format

Your CSV must include all 32 features with exact column names:

```csv
n_items,n_sellers,n_products,sum_price,sum_freight,total_payment,n_payment_records,max_installments,avg_weight_g,avg_length_cm,avg_height_cm,avg_width_cm,n_seller_states,purch_year,purch_month,purch_dayofweek,purch_hour,purch_is_weekend,purch_hour_sin,purch_hour_cos,est_lead_days,n_categories,mode_category_count,paytype_boleto,paytype_credit_card,paytype_debit_card,paytype_not_defined,paytype_voucher,mode_category,seller_state_mode,customer_city,customer_state
2,1,2,150.0,25.0,175.0,1,3,2000.0,30.0,15.0,20.0,1,2024,6,2,14,0,0.866025,-0.5,7.0,1,2,0,1,0,0,0,electronics,SP,sao paulo,SP
```

### Data Validation Rules

- **Numeric fields**: Must be numeric, non-negative
- **Date fields**: Valid ranges (month 1-12, hour 0-23)
- **Payment types**: Exactly one must equal 1
- **State codes**: Valid Brazilian state codes
- **Cities**: Lowercase, no special characters

### Sample Data Generation

Use the built-in generator for testing:
1. Select quantity (100-1000)
2. Choose risk distribution:
   - Balanced Mix (33% each)
   - More Low Risk (50/30/20)
   - More High Risk (20/30/50)
3. Download CSV
4. Upload and process

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Import/Module Errors

**Error**: `ModuleNotFoundError: No module named 'formatting'`

**Solution**:
```bash
# Ensure formatting.py exists in utils/
ls utils/formatting.py

# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Restart Streamlit
streamlit run app.py
```

#### 2. Model Loading Errors

**Error**: `FileNotFoundError: artifacts/best_model_lightgbm.pkl`

**Solution**:
- Verify all artifact files are present
- Check file permissions (should be readable)
- Ensure files are not corrupted

#### 3. CSV Upload Validation Errors

**Error**: `Missing Required Columns`

**Solution**:
- Download the template from the app
- Compare your columns with required list
- Check for typos in column names
- Ensure no extra spaces in headers

#### 4. Geographic Page Errors

**Error**: `KeyError: 'risk_category'`

**Solution**:
- Use the latest fixed version of Geographic Map page
- Compatibility layer handles column name differences
- Clear browser cache and refresh

#### 5. Prediction Errors

**Error**: `Invalid probability value`

**Solution**:
- Check input ranges (e.g., month 1-12, not 0-11)
- Ensure payment type selected (exactly one)
- Validate state codes (must be 2-letter Brazilian codes)

### Performance Issues

**Slow batch processing**:
- Reduce batch size (<500 orders)
- Close other browser tabs
- Clear Streamlit cache (Settings → Clear Cache)
- Check system resources (RAM, CPU)

**Memory errors**:
- Process in smaller batches
- Restart application
- Increase system swap space

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Community Cloud

1. **Prepare Repository**:
   ```bash
   # .gitignore should include:
   artifacts/
   *.pkl
   .streamlit/secrets.toml
   ```

2. **Deploy**:
   - Push to GitHub
   - Connect repository to Streamlit Cloud
   - Add artifacts via file upload or secrets

3. **Configuration**:
   - Set Python version: 3.8+
   - Configure secrets if needed
   - Monitor resource usage

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and run**:
```bash
docker build -t supply-chain-app .
docker run -p 8501:8501 supply-chain-app
```

### Production Considerations

- **Authentication**: Implement user authentication
- **Rate Limiting**: Add request throttling
- **Monitoring**: Set up logging and alerts
- **Backups**: Regular artifact backups
- **Scaling**: Use load balancing for high traffic
- **Security**: Sanitize all inputs, use HTTPS

## 📈 Performance Optimization

### Caching Strategy

```python
@st.cache_resource
def load_model_artifacts():
    """Model loaded once, reused across sessions"""
    ...

@st.cache_data
def generate_sample_data(n_samples):
    """Cache generated samples"""
    ...
```

### Best Practices

1. **Session State**: Use for data persistence between pages
2. **Lazy Loading**: Load data only when needed
3. **Batch Processing**: Process in chunks for large files
4. **Memory Management**: Clear unused data from session state

## 🔒 Security Best Practices

### Data Protection
- Validate all user inputs
- Sanitize file uploads
- Limit file sizes (200MB default)
- No sensitive data in logs

### Access Control
- Implement authentication for production
- Use role-based access control
- Audit user actions
- Secure artifact storage

### Model Security
- Don't commit .pkl files to public repos
- Use environment variables for configs
- Encrypt sensitive data
- Regular security updates

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [LightGBM Guide](https://lightgbm.readthedocs.io)
- [Plotly Python](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org)

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test thoroughly
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Update README for major changes

## 🔄 Version History

### v1.2.0 (Current) - October 2024
- ✨ Added sample data generator with configurable risk distributions
- 🐛 Fixed Geographic Map page batch results compatibility
- 🎨 Enhanced risk gauge with zone highlighting
- 🌍 Added Brazilian state/city name formatting
- 🔧 Improved session state management
- 📝 Better error messages and validation
- 🚀 Performance optimizations

### v1.1.0 - October 2024
- 🔧 Implemented inverted probability correction
- 📊 Added Model Diagnostics page
- 🗺️ Enhanced Geographic Map with state filtering
- 📈 Improved Time Analysis visualizations

### v1.0.0 - October 2024
- 🎉 Initial release
- 📊 Single and batch predictions
- 📈 Time trend analysis
- 🗺️ Geographic risk distribution
- 📄 PDF report generation

## 📄 License

This project is provided as-is for educational and commercial use.

## 📧 Support

For questions or issues:
1. Check this README thoroughly
2. Review troubleshooting section
3. Check Streamlit documentation
4. Contact system administrator

---

**Built with ❤️ using Streamlit, LightGBM, and Plotly**

**Version**: 1.2.0  
**Last Updated**: October 2024  
**Python**: 3.8+  
**Status**: Production Ready ✅
