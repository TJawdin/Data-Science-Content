"""
Model Loading and Prediction Module
Handles all ML model operations with caching for performance
"""

import json
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


@st.cache_resource
def load_model_artifacts():
    """
    Load model, metadata, and configurations with caching
    
    Returns:
        tuple: (model, final_metadata, feature_metadata, threshold)
    """
    try:
        artifacts_path = Path("artifacts")
        
        # Load the trained model
        with open(artifacts_path / "best_model_lightgbm.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load final metadata
        with open(artifacts_path / "final_metadata.json", "r") as f:
            final_metadata = json.load(f)
        
        # Load feature metadata
        with open(artifacts_path / "feature_metadata.json", "r") as f:
            feature_metadata = json.load(f)
        
        # Load optimal threshold
        with open(artifacts_path / "optimal_threshold_lightgbm.txt", "r") as f:
            threshold = float(f.read().strip())
        
        return model, final_metadata, feature_metadata, threshold
    
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        st.stop()


def predict_delay(model, features_df, threshold):
    """
    Make predictions with the model
    
    Args:
        model: Trained LightGBM model
        features_df: DataFrame with features
        threshold: Classification threshold
    
    Returns:
        tuple: (predictions, probabilities, risk_level)
    """
    try:
        # Get probability predictions
        probabilities = model.predict_proba(features_df)[:, 1]
        
        # Apply threshold for binary prediction
        predictions = (probabilities >= threshold).astype(int)
        
        # Determine risk level
        risk_levels = []
        for prob in probabilities:
            prob_pct = prob * 100
            if prob_pct <= 30:
                risk_levels.append("Low")
            elif prob_pct <= 67:
                risk_levels.append("Medium")
            else:
                risk_levels.append("High")
        
        return predictions, probabilities, risk_levels
    
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None, None, None


def get_model_performance():
    """
    Get model performance metrics
    
    Returns:
        dict: Performance metrics
    """
    _, final_metadata, _, _ = load_model_artifacts()
    
    return {
        "AUC-ROC": final_metadata["best_model_auc"],
        "Precision": final_metadata["best_model_precision"],
        "Recall": final_metadata["best_model_recall"],
        "F1-Score": final_metadata["best_model_f1"]
    }


def get_feature_names():
    """
    Get list of feature names
    
    Returns:
        list: Feature names
    """
    _, _, feature_metadata, _ = load_model_artifacts()
    return feature_metadata["feature_names"]


def get_feature_types():
    """
    Get feature type mappings
    
    Returns:
        dict: Feature type categories
    """
    _, _, feature_metadata, _ = load_model_artifacts()
    
    return {
        "numeric": feature_metadata["numeric_feats"],
        "payment_types": feature_metadata["paytype_feats"],
        "categorical": feature_metadata["categorical_feats"]
    }
