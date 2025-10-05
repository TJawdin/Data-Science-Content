# 📦 Supply-Chain Delay Predictor

Production-ready ML application to predict delivery delays in supply chain operations.

## 🚀 Live Demo
[View App on Streamlit Cloud](https://your-app-url.streamlit.app)

## 📊 Features
- **Batch Predictions**: Upload CSV files for bulk predictions
- **Single Record Prediction**: Manual entry via form
- **Model Explainability**: SHAP analysis and feature importance
- **Adjustable Threshold**: Balance precision vs recall
- **Downloadable Results**: Export predictions as CSV

## 🛠️ Tech Stack
- **ML Framework**: scikit-learn, XGBoost, CatBoost
- **Feature Engineering**: Featuretools
- **Explainability**: SHAP, permutation importance
- **Frontend**: Streamlit
- **Data Processing**: pandas, numpy

## 📁 Project Structure
supply-chain-delay-predictor/
├── .gitignore
├── README.md
├── requirements.txt
├── streamlit_app.py
├── artifacts/
│   ├── model_xgboost.pkl              # or whichever model won
│   ├── feature_names.json
│   ├── metrics_xgboost.json
│   ├── explain_manifest.json
│   ├── global_importance_permutation.png
│   ├── shap_summary_beeswarm.png
│   ├── shap_summary_bar.png
│   └── shap_dependence_*.png          # optional
├── notebooks/
│   └── supply_chain_model_training.ipynb
├── data/                               # optional - sample data
│   └── sample_input.csv
└── .streamlit/                         # optional - for custom config
    └── config.toml

# 🎯 Usage
## Batch Predictions
  1. Navigate to the "Batch Predict (CSV)" tab
  2. Upload a CSV file with order data
  3. Adjust the decision threshold if needed
  4. Download predictions
     
## Single Record
  1. Navigate to the "Single Record Form" tab
  2. Enter feature values manually or via JSON
  3. Click "Predict" to see results

## Explainability
  1. Navigate to the "Explainability" tab
  2. View global feature importance
  3. Explore SHAP analysis (if available)

## 📊 Input Data Format
The model expects features generated from:
  - Order information (timestamps, customer info)
  - Item details (product, seller, pricing)
  - Historical aggregations (via Featuretools)

See data/sample_input.csv for example format.

## 🔬 Model Training
The model training pipeline is documented in notebooks/supply_chain_model_training.ipynb.

Key steps:
  1. Data preprocessing & feature engineering (Featuretools)
  2. Train/test split with stratification
  3. Hyperparameter tuning (RandomizedSearchCV)
  4. Model evaluation & selection
  5. SHAP analysis for explainability

## 📝 License
MIT License

## 👤 Author
TJawdin
  - Github: @TJawdin

## 🙏 Acknowledgments
  - Dataset: Olist Brazilian E-Commerce
  - Built with Streamlit
