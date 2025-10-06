"""
Model Loader Utility
Handles loading the trained model and making predictions
"""

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Model path
MODEL_PATH = Path(__file__).parent.parent / 'artifacts' / 'best_model_xgboost.pkl'

@st.cache_resource
def load_model():
    """
    Load the trained XGBoost model from artifacts folder
    
    Returns:
    --------
    model : trained model or None if not found
    """
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        return model
        
    except FileNotFoundError:
        st.warning("Model file not found. Please ensure your trained model is in the artifacts folder.")
        return None
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def predict_single(model, features_df):
    """
    Make a prediction for a single order
    
    Parameters:
    -----------
    model : trained model
        The loaded XGBoost model
    features_df : pd.DataFrame
        Feature DataFrame for one order
    
    Returns:
    --------
    dict : Prediction results with risk score and level
    """
    
    try:
        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0, 1]
        
        # Calculate risk score (0-100)
        risk_score = int(probability * 100)
        
        # Determine risk level
        if risk_score < 30:
            risk_level = 'LOW'
        elif risk_score < 70:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        # Return results
        return {
            'prediction': int(prediction),
            'prediction_label': 'Late' if prediction == 1 else 'On-Time',
            'probability': float(probability),
            'risk_score': risk_score,
            'risk_level': risk_level
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


def predict_batch(model, features_list):
    """
    Make predictions for a batch of orders
    
    Parameters:
    -----------
    model : trained model
        The loaded XGBoost model
    features_list : list of pd.DataFrame
        List of feature DataFrames for each order
    
    Returns:
    --------
    pd.DataFrame : Predictions with risk scores and levels
    """
    
    try:
        # Combine all features into one DataFrame
        all_features = pd.concat(features_list, ignore_index=True)
        
        # Make predictions
        predictions = model.predict(all_features)
        probabilities = model.predict_proba(all_features)[:, 1]
        
        # Calculate risk scores
        risk_scores = (probabilities * 100).astype(int)
        
        # Determine risk levels
        def get_risk_level(score):
            if score < 30:
                return 'LOW'
            elif score < 70:
                return 'MEDIUM'
            else:
                return 'HIGH'
        
        risk_levels = [get_risk_level(score) for score in risk_scores]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': predictions,
            'prediction_label': ['Late' if p == 1 else 'On-Time' for p in predictions],
            'probability': probabilities,
            'risk_score': risk_scores,
            'risk_level': risk_levels
        })
        
        return results
        
    except Exception as e:
        st.error(f"Batch prediction error: {str(e)}")
        return None
