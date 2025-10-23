import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

class ModelLoader:
    def __init__(self, artifacts_path="./artifacts"):
        self.artifacts_path = Path(artifacts_path)
        self.model = None
        self.metadata = None
        self.feature_metadata = None
        self.optimal_threshold = None
        self.risk_bands = None
        
    @st.cache_resource
    def load_model(_self):
        """Load the LightGBM model"""
        try:
            model_path = _self.artifacts_path / "best_model_lightgbm.pkl"
            with open(model_path, 'rb') as f:
                _self.model = pickle.load(f)
            return _self.model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    @st.cache_resource
    def load_metadata(_self):
        """Load model metadata"""
        try:
            # Load final metadata
            metadata_path = _self.artifacts_path / "final_metadata.json"
            with open(metadata_path, 'r') as f:
                _self.metadata = json.load(f)
            
            # Load feature metadata
            feature_metadata_path = _self.artifacts_path / "feature_metadata.json"
            with open(feature_metadata_path, 'r') as f:
                _self.feature_metadata = json.load(f)
            
            # Load optimal threshold
            threshold_path = _self.artifacts_path / "optimal_threshold_lightgbm.txt"
            with open(threshold_path, 'r') as f:
                _self.optimal_threshold = float(f.read().strip())
            
            # Get risk bands
            _self.risk_bands = _self.metadata.get('risk_bands', {'low_max': 30, 'med_max': 67})
            
            return _self.metadata, _self.feature_metadata
        except Exception as e:
            st.error(f"Error loading metadata: {str(e)}")
            return None, None
    
    def predict_with_probability(self, features_df):
        """Make prediction with probability scores"""
        if self.model is None:
            self.load_model()
        
        try:
            # Get probability of being late (class 1)
            probabilities = self.model.predict_proba(features_df)[:, 1]
            
            # Apply optimal threshold
            predictions = (probabilities >= self.optimal_threshold).astype(int)
            
            # Calculate risk level
            risk_levels = []
            for prob in probabilities * 100:
                if prob <= self.risk_bands['low_max']:
                    risk_levels.append('Low')
                elif prob <= self.risk_bands['med_max']:
                    risk_levels.append('Medium')
                else:
                    risk_levels.append('High')
            
            return predictions, probabilities, risk_levels
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None, None
    
    def get_feature_importance(self):
        """Get feature importance from model"""
        if self.model is None:
            self.load_model()
        
        try:
            importance = self.model.feature_importances_
            feature_names = self.feature_metadata['feature_names']
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        except:
            return None
