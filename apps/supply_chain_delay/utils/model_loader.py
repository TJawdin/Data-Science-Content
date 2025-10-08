"""
Model Loading and Prediction Functions
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

# Optimized threshold from model training
OPTIMAL_THRESHOLD = 0.1844  # 18.44%

@st.cache_resource
def load_model():
    """
    Load the trained model from artifacts folder
    Uses caching to load only once
    """
    try:
        # Try to find the best model
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        
        # Look for model files
        model_files = list(artifacts_dir.glob("best_model_*.pkl"))
        
        if not model_files:
            # Fallback: look for any model file
            model_files = list(artifacts_dir.glob("model_*.pkl"))
        
        if not model_files:
            st.error("""
            ⚠️ No model file found in artifacts/ folder.
            
            Please copy your trained model from the notebook to:
            `apps/supply_chain_delay/artifacts/best_model_*.pkl`
            """)
            return None
        
        # Load the first model found
        model_path = model_files[0]
        model = joblib.load(model_path)
        
        return model
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


def predict_single(model, features_df):
    """
    Make prediction for a single order
    
    Parameters:
    -----------
    model : trained model
    features_df : pd.DataFrame with 30 features
    
    Returns:
    --------
    dict with prediction, probability, and risk score
    """
    try:
        # Get probability (this is what we need for custom threshold)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_df)[0]
            prob_late = probabilities[1]
        else:
            # Fallback if no predict_proba
            prob_late = model.predict(features_df)[0]
        
        # Apply custom threshold (18.44%) instead of default 50%
        prediction = 1 if prob_late >= OPTIMAL_THRESHOLD else 0
        
        # Calculate risk score (0-100)
        risk_score = int(prob_late * 100)
        
        # Determine risk level based on optimized threshold (18.44%)
        # Aligned with model's precision-recall optimization
        if risk_score < 10:           # 0-9%: Very low risk
            risk_level = "LOW"
            risk_color = "green"
        elif risk_score < 26:         # 10-25%: Around threshold (18.44%)
            risk_level = "MEDIUM"
            risk_color = "orange"
        else:                         # 26%+: Above threshold = HIGH
            risk_level = "HIGH"
            risk_color = "red"
        
        return {
            'prediction': int(prediction),
            'prediction_label': 'Late' if prediction == 1 else 'On-Time',
            'probability': float(prob_late),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_color': risk_color
        }
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None


def predict_batch(model, features_df):
    """
    Make predictions for multiple orders
    
    Parameters:
    -----------
    model : trained model
    features_df : pd.DataFrame with N rows × 30 features
    
    Returns:
    --------
    pd.DataFrame with predictions and risk scores
    """
    try:
        # Get probabilities (this is what we need for custom threshold)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_df)
            prob_late = probabilities[:, 1]
        else:
            prob_late = model.predict(features_df)
        
        # Apply custom threshold (18.44%) instead of default 50%
        predictions = (prob_late >= OPTIMAL_THRESHOLD).astype(int)
        
        # Calculate risk scores
        risk_scores = (prob_late * 100).astype(int)
        
        # Determine risk levels based on optimized threshold (18.44%)
        def get_risk_level(score):
            if score < 10:
                return 'LOW'
            elif score < 26:
                return 'MEDIUM'
            else:
                return 'HIGH'
        
        risk_levels = [get_risk_level(score) for score in risk_scores]
        
        # Create results dataframe (UPPERCASE to match batch page expectations)
        results = pd.DataFrame({
            'Prediction': ['Late' if p == 1 else 'On-Time' for p in predictions],
            'Late_Probability': prob_late,
            'Risk_Score': risk_scores,
            'risk_level': risk_levels  # lowercase for create_risk_distribution function
        })
        
        return results
    
    except Exception as e:
        st.error(f"❌ Batch prediction error: {str(e)}")
        return None


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from model
    
    Parameters:
    -----------
    model : trained model
    feature_names : list of feature names
    
    Returns:
    --------
    pd.DataFrame with feature importance sorted
    """
    try:
        # Handle pipeline models (Logistic Regression)
        if hasattr(model, 'named_steps'):
            actual_model = model.named_steps.get('clf', model)
        else:
            actual_model = model
        
        # Get feature importance
        if hasattr(actual_model, 'feature_importances_'):
            # Tree-based models
            importances = actual_model.feature_importances_
        elif hasattr(actual_model, 'coef_'):
            # Linear models
            importances = np.abs(actual_model.coef_[0])
        else:
            st.warning("⚠️ Model does not support feature importance extraction")
            return None
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    except Exception as e:
        st.error(f"❌ Error extracting feature importance: {str(e)}")
        return None
