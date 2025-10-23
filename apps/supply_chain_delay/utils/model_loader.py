"""
Model loading and prediction utilities
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
    Load model, metadata, and threshold with caching for performance
    
    Returns:
        tuple: (model, final_metadata, feature_metadata, threshold)
    """
    try:
        artifacts_path = Path("artifacts")
        
        # Load model
        with open(artifacts_path / "best_model_lightgbm.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load final metadata
        with open(artifacts_path / "final_metadata.json", "r") as f:
            final_metadata = json.load(f)
        
        # Load feature metadata
        with open(artifacts_path / "feature_metadata.json", "r") as f:
            feature_metadata = json.load(f)
        
        # Load threshold
        threshold = final_metadata.get("optimal_threshold", 0.669271)
        
        return model, final_metadata, feature_metadata, threshold
    
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        st.stop()


def load_metadata():
    """
    Load only metadata files (lighter weight for non-prediction pages)
    
    Returns:
        tuple: (final_metadata, feature_metadata)
    """
    try:
        artifacts_path = Path("artifacts")
        
        with open(artifacts_path / "final_metadata.json", "r") as f:
            final_metadata = json.load(f)
        
        with open(artifacts_path / "feature_metadata.json", "r") as f:
            feature_metadata = json.load(f)
        
        return final_metadata, feature_metadata
    
    except Exception as e:
        st.error(f"Error loading metadata: {str(e)}")
        st.stop()


def predict_delay_risk(model, input_data, threshold):
    """
    Make prediction and return probability and risk category
    
    Args:
        model: Trained LightGBM model
        input_data: DataFrame with features
        threshold: Classification threshold
    
    Returns:
        tuple: (probability, risk_category, is_high_risk)
    """
    try:
        # Get probability of delay
        prob = model.predict_proba(input_data)[:, 1]
        
        # Convert to percentage
        prob_pct = prob * 100
        
        # Determine risk category
        risk_category = get_risk_category(prob_pct[0])
        
        # Binary classification based on threshold
        is_high_risk = prob[0] >= threshold
        
        return prob_pct[0], risk_category, is_high_risk
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None


def get_risk_category(probability_pct):
    """
    Categorize risk level based on probability percentage
    
    Args:
        probability_pct: Probability as percentage (0-100)
    
    Returns:
        str: Risk category ('Low', 'Medium', or 'High')
    """
    if probability_pct <= 30:
        return "Low"
    elif probability_pct <= 67:
        return "Medium"
    else:
        return "High"


def get_risk_color(risk_category):
    """
    Get color for risk category
    
    Args:
        risk_category: Risk level ('Low', 'Medium', 'High')
    
    Returns:
        str: Hex color code
    """
    colors = {
        "Low": "#00CC96",      # Green
        "Medium": "#FFA500",   # Orange
        "High": "#EF553B"      # Red
    }
    return colors.get(risk_category, "#888888")


def batch_predict(model, input_data, threshold, final_metadata):
    """
    Make predictions for batch of orders
    
    Args:
        model: Trained model
        input_data: DataFrame with multiple rows
        threshold: Classification threshold
        final_metadata: Metadata dict with risk band info
    
    Returns:
        DataFrame: Input data with predictions added
    """
    try:
        # Get probabilities
        probs = model.predict_proba(input_data)[:, 1]
        probs_pct = probs * 100
        
        # Create results dataframe
        results = input_data.copy()
        results['delay_probability'] = probs_pct
        results['risk_category'] = [get_risk_category(p) for p in probs_pct]
        results['high_risk'] = probs >= threshold
        
        return results
    
    except Exception as e:
        st.error(f"Batch prediction error: {str(e)}")
        return None


def get_model_performance(final_metadata):
    """
    Extract and format model performance metrics
    
    Args:
        final_metadata: Dictionary with model metrics
    
    Returns:
        dict: Formatted performance metrics
    """
    return {
        'AUC-ROC': f"{final_metadata['best_model_auc']:.2%}",
        'Precision': f"{final_metadata['best_model_precision']:.2%}",
        'Recall': f"{final_metadata['best_model_recall']:.2%}",
        'F1-Score': f"{final_metadata['best_model_f1']:.2%}",
        'Threshold': f"{final_metadata['optimal_threshold']:.4f}",
        'Training Samples': f"{final_metadata['n_samples_train']:,}",
        'Test Samples': f"{final_metadata['n_samples_test']:,}",
        'Training Date': final_metadata['training_date']
    }
def get_model_performance(final_metadata):
    """
    Format model performance metrics for display
    
    Args:
        final_metadata: Dictionary containing model metadata
        
    Returns:
        Dictionary with formatted performance metrics
    """
    return {
        'AUC-ROC': f"{final_metadata['best_model_auc']:.1%}",
        'Precision': f"{final_metadata['best_model_precision']:.1%}",
        'Recall': f"{final_metadata['best_model_recall']:.1%}",
        'F1-Score': f"{final_metadata['best_model_f1']:.1%}"
    }
