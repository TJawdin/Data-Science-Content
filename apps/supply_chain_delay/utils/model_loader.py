"""
Model Loading and Prediction Functions
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

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
        model_files = list(artifacts_dir.glob("best_model_xgboost.pkl"))
        
        if not model_files:
            # Fallback: look for any model file
            model_files = list(artifacts_dir.glob("model_*.pkl"))
        
        if not model_files:
            st.error("""
            ⚠️ No model file found in artifacts/ folder.
            
            Please copy your trained model from the notebook to:
            `apps/supply_chain_delay/artifacts/best_model_xgboost.pkl`
            """)
            return None
        
        # Load the first model found
        model_path = model_files[0]
        model = joblib.load(model_path)
        
        
        return model
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


def predict_batch(model, features_list):
    """
    Make predictions for multiple orders
    
    Parameters:
    -----------
    model : trained model
    features_list : list of pd.DataFrame (each with 1 row × 30 features)
    
    Returns:
    --------
    pd.DataFrame with predictions and risk scores
    """
    try:
        # Combine all features into one DataFrame
        features_df = pd.concat(features_list, ignore_index=True)
        
        # Get predictions
        predictions = model.predict(features_df)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_df)
            prob_late = probabilities[:, 1]
        else:
            prob_late = predictions
        
        # Calculate risk scores
        risk_scores = (prob_late * 100).astype(int)
        
        # Determine risk levels
        def get_risk_level(score):
            if score < 30:
                return 'LOW'
            elif score < 70:
                return 'MEDIUM'
            else:
                return 'HIGH'
        
        risk_levels = [get_risk_level(score) for score in risk_scores]
        
        # Create results dataframe (lowercase column names for consistency)
        results = pd.DataFrame({
            'prediction': predictions,
            'prediction_label': ['Late' if p == 1 else 'On-Time' for p in predictions],
            'probability': prob_late,
            'risk_score': risk_scores,
            'risk_level': risk_levels
        })
        
        return results
    
    except Exception as e:
        st.error(f"❌ Batch prediction error: {str(e)}")
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
        # Get predictions
        predictions = model.predict(features_df)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_df)
            prob_late = probabilities[:, 1]
        else:
            prob_late = predictions
        
        # Create results dataframe
        results = pd.DataFrame({
            'Prediction': ['Late' if p == 1 else 'On-Time' for p in predictions],
            'Late_Probability': prob_late,
            'Risk_Score': (prob_late * 100).astype(int),
            'risk_level': pd.cut(
                prob_late * 100,
                bins=[0, 30, 70, 100],
                labels=['LOW', 'MEDIUM', 'HIGH']
            )
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
