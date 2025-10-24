"""
Feature engineering and input preparation utilities
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime


def prepare_single_prediction_input(user_inputs, feature_metadata):
    """
    Prepare single prediction input from user form data
    
    Args:
        user_inputs: Dictionary of user input values
        feature_metadata: Feature metadata dictionary
    
    Returns:
        DataFrame: Single row with all required features
    """
    # Create empty dataframe with all features
    feature_names = feature_metadata['feature_names']
    input_df = pd.DataFrame(columns=feature_names)
    
    # Add user inputs
    for key, value in user_inputs.items():
        if key in feature_names:
            input_df.loc[0, key] = value
    
    # Fill missing values with defaults
    for col in feature_names:
        if col not in input_df.columns or pd.isna(input_df.loc[0, col]):
            input_df.loc[0, col] = get_default_value(col, feature_metadata)
    
    # Ensure correct data types
    input_df = ensure_correct_dtypes(input_df, feature_metadata)
    
    return input_df


def prepare_batch_prediction_input(df, feature_metadata):
    """
    Prepare batch prediction input from uploaded dataframe
    
    Args:
        df: Input dataframe
        feature_metadata: Feature metadata dictionary
    
    Returns:
        DataFrame: Prepared data with all required features
    """
    feature_names = feature_metadata['feature_names']
    
    # Create dataframe with all required features
    prepared_df = pd.DataFrame(columns=feature_names)
    
    # Copy over existing features
    for col in feature_names:
        if col in df.columns:
            prepared_df[col] = df[col]
        else:
            prepared_df[col] = get_default_value(col, feature_metadata)
    
    # Ensure correct data types
    prepared_df = ensure_correct_dtypes(prepared_df, feature_metadata)
    
    return prepared_df


def validate_input_data(df, feature_metadata):
    """
    Validate input data has required features and correct types
    
    Args:
        df: Input dataframe
        feature_metadata: Feature metadata dictionary
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check for required features
    required_features = feature_metadata['feature_names']
    missing_features = set(required_features) - set(df.columns)
    
    if missing_features:
        errors.append(f"Missing features: {', '.join(missing_features)}")
    
    # Check numeric features are numeric
    for col in feature_metadata['numeric_feats']:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Feature '{col}' should be numeric")
    
    # Check for null values in critical features
    critical_features = ['n_items', 'sum_price', 'customer_state']
    for col in critical_features:
        if col in df.columns:
            if df[col].isna().any():
                errors.append(f"Feature '{col}' contains null values")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def get_default_value(feature_name, feature_metadata):
    """
    Get default value for a feature
    
    Args:
        feature_name: Name of feature
        feature_metadata: Feature metadata dictionary
    
    Returns:
        Default value for the feature
    """
    # Numeric features
    if feature_name in feature_metadata['numeric_feats']:
        defaults = {
            'n_items': 1,
            'n_sellers': 1,
            'n_products': 1,
            'sum_price': 100.0,
            'sum_freight': 10.0,
            'total_payment': 110.0,
            'n_payment_records': 1,
            'max_installments': 1,
            'avg_weight_g': 500,
            'avg_length_cm': 20,
            'avg_height_cm': 10,
            'avg_width_cm': 15,
            'n_seller_states': 1,
            'purch_year': 2024,
            'purch_month': 6,
            'purch_dayofweek': 2,
            'purch_hour': 14,
            'purch_is_weekend': 0,
            'purch_hour_sin': 0,
            'purch_hour_cos': 1,
            'est_lead_days': 7,
            'n_categories': 1,
            'mode_category_count': 1
        }
        return defaults.get(feature_name, 0)
    
    # Payment type features (binary)
    elif feature_name in feature_metadata['paytype_feats']:
        return 1 if feature_name == 'paytype_credit_card' else 0
    
    # Categorical features
    elif feature_name in feature_metadata['categorical_feats']:
        defaults = {
            'mode_category': 'bed_bath_table',
            'seller_state_mode': 'SP',
            'customer_city': 'sao paulo',
            'customer_state': 'SP'
        }
        return defaults.get(feature_name, 'unknown')
    
    return 0


def ensure_correct_dtypes(df, feature_metadata):
    """
    Ensure dataframe has correct data types for all features
    
    Args:
        df: Input dataframe
        feature_metadata: Feature metadata dictionary
    
    Returns:
        DataFrame: Dataframe with corrected types
    """
    # Convert numeric features
    for col in feature_metadata['numeric_feats']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert payment type features (binary 0/1)
    for col in feature_metadata['paytype_feats']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Convert categorical features to string
    for col in feature_metadata['categorical_feats']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()
    
    return df


def get_feature_ranges(feature_metadata):
    """
    Get typical ranges for numeric features
    
    Args:
        feature_metadata: Feature metadata dictionary
    
    Returns:
        dict: Feature name to (min, max, default) mapping
    """
    ranges = {
        'n_items': (1, 20, 1),
        'n_sellers': (1, 10, 1),
        'n_products': (1, 20, 1),
        'sum_price': (10.0, 10000.0, 100.0),
        'sum_freight': (5.0, 500.0, 20.0),
        'total_payment': (15.0, 10500.0, 120.0),
        'n_payment_records': (1, 10, 1),
        'max_installments': (1, 24, 1),
        'avg_weight_g': (50, 50000, 1000),
        'avg_length_cm': (5, 100, 30),
        'avg_height_cm': (2, 50, 10),
        'avg_width_cm': (5, 80, 20),
        'n_seller_states': (1, 5, 1),
        'purch_year': (2016, 2025, 2024),
        'purch_month': (1, 12, 6),
        'purch_dayofweek': (0, 6, 2),
        'purch_hour': (0, 23, 14),
        'purch_is_weekend': (0, 1, 0),
        'est_lead_days': (1, 60, 10),
        'n_categories': (1, 10, 1),
        'mode_category_count': (1, 20, 1)
    }
    return ranges


def create_sample_scenarios():
    """
    Create intuitive example scenarios (predictions are inverted in code)
    
    These scenarios now work intuitively:
    - Low risk: Simple order, SP, short lead time, high freight
    - High risk: Complex order, remote state, long lead time, low freight
    
    Returns:
        dict: Scenario name to complete input values mapping
    """
    scenarios = {
        'low_risk': {
            # Order characteristics - simple, single item
            'n_items': 1,
            'n_sellers': 1,
            'n_products': 1,
            'n_categories': 1,
            'mode_category_count': 1,
            'mode_category': 'health_beauty',
            
            # Financial - high freight (premium shipping = reliable)
            'sum_price': 100.0,
            'sum_freight': 50.0,  # High freight = premium = low risk
            'total_payment': 150.0,
            'n_payment_records': 1,
            'max_installments': 1,
            
            # Dimensions - small, lightweight
            'avg_weight_g': 800.0,
            'avg_length_cm': 20.0,
            'avg_height_cm': 10.0,
            'avg_width_cm': 15.0,
            
            # Geographic - SP (São Paulo - major hub, good logistics)
            'customer_city': 'sao paulo',
            'customer_state': 'SP',
            'seller_state_mode': 'SP',
            'n_seller_states': 1,
            
            # Temporal - mid-year, weekday, business hours
            'purch_year': 2024,
            'purch_month': 6,  # June - mid-year
            'purch_dayofweek': 2,  # Wednesday
            'purch_hour': 10,
            'purch_is_weekend': 0,
            'purch_hour_sin': np.sin(2 * np.pi * 10 / 24),
            'purch_hour_cos': np.cos(2 * np.pi * 10 / 24),
            'est_lead_days': 3.0,  # Short lead time = low risk
            
            # Payment type - credit card
            'paytype_boleto': 0,
            'paytype_credit_card': 1,
            'paytype_debit_card': 0,
            'paytype_not_defined': 0,
            'paytype_voucher': 0
        },
        
        'typical': {
            # Order characteristics - moderate
            'n_items': 2,
            'n_sellers': 1,
            'n_products': 2,
            'n_categories': 1,
            'mode_category_count': 2,
            'mode_category': 'electronics',
            
            # Financial - medium freight
            'sum_price': 150.0,
            'sum_freight': 30.0,
            'total_payment': 180.0,
            'n_payment_records': 1,
            'max_installments': 2,
            
            # Dimensions - medium
            'avg_weight_g': 2000.0,
            'avg_length_cm': 30.0,
            'avg_height_cm': 15.0,
            'avg_width_cm': 20.0,
            
            # Geographic - RJ (Rio - moderate)
            'customer_city': 'rio de janeiro',
            'customer_state': 'RJ',
            'seller_state_mode': 'SP',
            'n_seller_states': 1,
            
            # Temporal - moderate
            'purch_year': 2024,
            'purch_month': 8,  # August
            'purch_dayofweek': 3,
            'purch_hour': 15,
            'purch_is_weekend': 0,
            'purch_hour_sin': np.sin(2 * np.pi * 15 / 24),
            'purch_hour_cos': np.cos(2 * np.pi * 15 / 24),
            'est_lead_days': 10.0,  # Medium lead time
            
            # Payment type
            'paytype_boleto': 0,
            'paytype_credit_card': 1,
            'paytype_debit_card': 0,
            'paytype_not_defined': 0,
            'paytype_voucher': 0
        },
        
        'high_risk': {
            # Order characteristics - complex, multiple sellers
            'n_items': 3,
            'n_sellers': 3,
            'n_products': 3,
            'n_categories': 2,
            'mode_category_count': 2,
            'mode_category': 'furniture_decor',
            
            # Financial - LOW freight (cheap shipping = unreliable)
            'sum_price': 400.0,
            'sum_freight': 10.0,  # Low freight = cheap = high risk
            'total_payment': 410.0,
            'n_payment_records': 1,
            'max_installments': 6,
            
            # Dimensions - large, heavy
            'avg_weight_g': 4000.0,
            'avg_length_cm': 50.0,
            'avg_height_cm': 30.0,
            'avg_width_cm': 40.0,
            
            # Geographic - PR (Paraná - farther from main hubs)
            'customer_city': 'curitiba',
            'customer_state': 'PR',
            'seller_state_mode': 'SP',
            'n_seller_states': 2,
            
            # Temporal - early year (busy season), weekend
            'purch_year': 2024,
            'purch_month': 2,  # February - busy period
            'purch_dayofweek': 6,  # Sunday
            'purch_hour': 22,
            'purch_is_weekend': 1,
            'purch_hour_sin': np.sin(2 * np.pi * 22 / 24),
            'purch_hour_cos': np.cos(2 * np.pi * 22 / 24),
            'est_lead_days': 25.0,  # Long lead time = high risk
            
            # Payment type - boleto (slower processing)
            'paytype_boleto': 1,
            'paytype_credit_card': 0,
            'paytype_debit_card': 0,
            'paytype_not_defined': 0,
            'paytype_voucher': 0
        }
    }
    
    return scenarios


def calculate_temporal_features(purchase_datetime):
    """
    Calculate temporal features from purchase datetime
    
    Args:
        purchase_datetime: datetime object
    
    Returns:
        dict: Temporal features
    """
    hour = purchase_datetime.hour
    
    return {
        'purch_year': purchase_datetime.year,
        'purch_month': purchase_datetime.month,
        'purch_dayofweek': purchase_datetime.weekday(),
        'purch_hour': hour,
        'purch_is_weekend': 1 if purchase_datetime.weekday() >= 5 else 0,
        'purch_hour_sin': np.sin(2 * np.pi * hour / 24),
        'purch_hour_cos': np.cos(2 * np.pi * hour / 24)
    }
