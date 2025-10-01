import joblib
import os
import pandas as pd

# --- Step 1: Build the absolute path to the models folder ---
base_dir = os.path.dirname(__file__)               # /nba_dashboard/app
model_path = os.path.join(base_dir, 'pts_predictor_xgb.pkl')
features_path = os.path.join(base_dir, 'model_features.pkl')

# --- Step 2: Load model and features ---
try:
    model = joblib.load(model_path)
    features = joblib.load(features_path)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Could not load model or features: {e}")

# --- Step 3: Define reusable prediction function ---
def predict_pts(input_df: pd.DataFrame) -> float:
    """
    Predicts points (PTS) using a trained model.
    Expects a DataFrame with the same feature columns as the training data.
    """
    return model.predict(input_df[features])[0]