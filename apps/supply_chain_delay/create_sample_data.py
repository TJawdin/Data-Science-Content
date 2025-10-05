# create_sample_data.py
"""
Generate sample data for Streamlit demo using saved artifacts.
Run this after training: python create_sample_data.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
ARTIFACTS = Path("artifacts")
DATA_DIR = Path("data")
N_SAMPLES = 15  # Total samples to generate

def main():
    print("🔧 Creating sample data for Streamlit demo...")
    
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    
    # Load feature names
    features_path = ARTIFACTS / "feature_names.json"
    if not features_path.exists():
        print("❌ feature_names.json not found. Run Step 7E first.")
        return
    
    with open(features_path, "r") as f:
        features = json.load(f)
    
    print(f"✅ Loaded {len(features)} features")
    
    # Generate synthetic data with realistic distributions
    np.random.seed(42)
    sample_data = {}
    
    for feat in features:
        feat_lower = feat.lower()
        
        # Pattern-based generation for realistic values
        if 'count' in feat_lower or 'num_unique' in feat_lower:
            # Counts: Small integers
            sample_data[feat] = np.random.poisson(lam=3, size=N_SAMPLES).astype(float)
            
        elif 'mean' in feat_lower and 'price' in feat_lower:
            # Prices: 20-200 range
            sample_data[feat] = np.random.uniform(20, 200, size=N_SAMPLES)
            
        elif 'sum' in feat_lower and ('price' in feat_lower or 'freight' in feat_lower):
            # Total prices: 50-500 range
            sample_data[feat] = np.random.uniform(50, 500, size=N_SAMPLES)
            
        elif 'month' in feat_lower:
            # Months: 1-12
            sample_data[feat] = np.random.randint(1, 13, size=N_SAMPLES).astype(float)
            
        elif 'weekday' in feat_lower:
            # Weekdays: 0-6
            sample_data[feat] = np.random.randint(0, 7, size=N_SAMPLES).astype(float)
            
        elif 'day' in feat_lower and 'weekday' not in feat_lower:
            # Days: 1-31
            sample_data[feat] = np.random.randint(1, 32, size=N_SAMPLES).astype(float)
            
        elif 'weight' in feat_lower:
            # Weights in grams: 100-5000
            sample_data[feat] = np.random.uniform(100, 5000, size=N_SAMPLES)
            
        elif 'length' in feat_lower or 'height' in feat_lower or 'width' in feat_lower:
            # Dimensions in cm: 5-100
            sample_data[feat] = np.random.uniform(5, 100, size=N_SAMPLES)
            
        elif 'distance' in feat_lower:
            # Distance in km: 10-2000
            sample_data[feat] = np.random.uniform(10, 2000, size=N_SAMPLES)
            
        elif 'std' in feat_lower:
            # Standard deviations: small positive
            sample_data[feat] = np.abs(np.random.normal(5, 2, size=N_SAMPLES))
            
        else:
            # Default: small random values
            sample_data[feat] = np.random.normal(0, 1, size=N_SAMPLES)
    
    # Create DataFrame
    sample = pd.DataFrame(sample_data)
    
    # Ensure non-negative for counts and physical measurements
    for col in sample.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['count', 'num', 'price', 'weight', 'length', 'height', 'width', 'distance']):
            sample[col] = sample[col].abs()
    
    # Add order IDs
    sample.insert(0, 'order_id', [f'demo_order_{i+1:03d}' for i in range(N_SAMPLES)])
    
    # Try to load model and add predictions (optional - makes demo better)
    try:
        model_files = list(ARTIFACTS.glob("model_*.pkl"))
        if model_files:
            model = joblib.load(model_files[0])
            print(f"✅ Loaded model: {model_files[0].name}")
            
            # Make predictions
            X_sample = sample[features]
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_sample)[:, 1]
                sample['predicted_probability'] = proba.round(4)
                sample['predicted_label'] = (proba >= 0.5).astype(int)
                
                # Show distribution
                print(f"\n📊 Prediction distribution:")
                print(f"   High risk (≥0.7): {(proba >= 0.7).sum()}")
                print(f"   Medium risk (0.3-0.7): {((proba >= 0.3) & (proba < 0.7)).sum()}")
                print(f"   Low risk (<0.3): {(proba < 0.3).sum()}")
    except Exception as e:
        print(f"⚠️  Could not load model for predictions: {e}")
        print("   Sample data will be created without prediction columns.")
    
    # Save
    output_path = DATA_DIR / "sample_input.csv"
    sample.to_csv(output_path, index=False)
    
    print(f"\n✅ Sample data created successfully!")
    print(f"   Location: {output_path}")
    print(f"   Rows: {len(sample)}")
    print(f"   Columns: {len(sample.columns)}")
    print(f"\n💡 Upload this file in the Streamlit app 'Batch Predict' tab to test!")

if __name__ == "__main__":
    main()# create_sample_data.py