"""
Feature Engineering Module
Handles data validation and feature preparation
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime


def validate_input(data, feature_names):
    """
    Validate input data has all required features
    
    Args:
        data: Input data (dict or DataFrame)
        feature_names: List of required feature names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if isinstance(data, dict):
        data_keys = set(data.keys())
    elif isinstance(data, pd.DataFrame):
        data_keys = set(data.columns)
    else:
        return False, "Invalid data format. Must be dict or DataFrame."
    
    required_features = set(feature_names)
    missing_features = required_features - data_keys
    
    if missing_features:
        return False, f"Missing features: {', '.join(missing_features)}"
    
    return True, None


def prepare_features(input_data, feature_names):
    """
    Prepare features in correct order for model prediction
    
    Args:
        input_data: Input data (dict or DataFrame)
        feature_names: Ordered list of feature names
    
    Returns:
        pd.DataFrame: Features ready for prediction
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Ensure all features are present and in correct order
    df = df[feature_names]
    
    return df


def create_example_order(scenario="typical"):
    """
    Create example order data for different scenarios
    
    Args:
        scenario: One of ['typical', 'high_risk', 'low_risk']
    
    Returns:
        dict: Example order features
    """
    examples = {
        "typical": {
            "n_items": 2,
            "n_sellers": 1,
            "n_products": 2,
            "sum_price": 150.0,
            "sum_freight": 25.0,
            "total_payment": 175.0,
            "n_payment_records": 1,
            "max_installments": 3,
            "avg_weight_g": 2000.0,
            "avg_length_cm": 30.0,
            "avg_height_cm": 15.0,
            "avg_width_cm": 20.0,
            "n_seller_states": 1,
            "purch_year": 2024,
            "purch_month": 6,
            "purch_dayofweek": 2,
            "purch_hour": 14,
            "purch_is_weekend": 0,
            "purch_hour_sin": np.sin(2 * np.pi * 14 / 24),
            "purch_hour_cos": np.cos(2 * np.pi * 14 / 24),
            "est_lead_days": 7.0,
            "n_categories": 1,
            "mode_category_count": 2,
            "paytype_boleto": 0,
            "paytype_credit_card": 1,
            "paytype_debit_card": 0,
            "paytype_not_defined": 0,
            "paytype_voucher": 0,
            "mode_category": "electronics",
            "seller_state_mode": "SP",
            "customer_city": "sao paulo",
            "customer_state": "SP"
        },
        "high_risk": {
            "n_items": 8,
            "n_sellers": 4,
            "n_products": 7,
            "sum_price": 850.0,
            "sum_freight": 120.0,
            "total_payment": 970.0,
            "n_payment_records": 2,
            "max_installments": 12,
            "avg_weight_g": 5500.0,
            "avg_length_cm": 65.0,
            "avg_height_cm": 45.0,
            "avg_width_cm": 50.0,
            "n_seller_states": 3,
            "purch_year": 2024,
            "purch_month": 12,
            "purch_dayofweek": 6,
            "purch_hour": 23,
            "purch_is_weekend": 1,
            "purch_hour_sin": np.sin(2 * np.pi * 23 / 24),
            "purch_hour_cos": np.cos(2 * np.pi * 23 / 24),
            "est_lead_days": 18.0,
            "n_categories": 4,
            "mode_category_count": 2,
            "paytype_boleto": 1,
            "paytype_credit_card": 0,
            "paytype_debit_card": 0,
            "paytype_not_defined": 0,
            "paytype_voucher": 0,
            "mode_category": "furniture_decor",
            "seller_state_mode": "RJ",
            "customer_city": "manaus",
            "customer_state": "AM"
        },
        "low_risk": {
            "n_items": 1,
            "n_sellers": 1,
            "n_products": 1,
            "sum_price": 45.0,
            "sum_freight": 12.0,
            "total_payment": 57.0,
            "n_payment_records": 1,
            "max_installments": 1,
            "avg_weight_g": 500.0,
            "avg_length_cm": 15.0,
            "avg_height_cm": 8.0,
            "avg_width_cm": 10.0,
            "n_seller_states": 1,
            "purch_year": 2024,
            "purch_month": 3,
            "purch_dayofweek": 1,
            "purch_hour": 10,
            "purch_is_weekend": 0,
            "purch_hour_sin": np.sin(2 * np.pi * 10 / 24),
            "purch_hour_cos": np.cos(2 * np.pi * 10 / 24),
            "est_lead_days": 3.0,
            "n_categories": 1,
            "mode_category_count": 1,
            "paytype_boleto": 0,
            "paytype_credit_card": 1,
            "paytype_debit_card": 0,
            "paytype_not_defined": 0,
            "paytype_voucher": 0,
            "mode_category": "health_beauty",
            "seller_state_mode": "SP",
            "customer_city": "sao paulo",
            "customer_state": "SP"
        }
    }
    
    return examples.get(scenario, examples["typical"])


def get_feature_descriptions():
    """
    Get human-readable descriptions for all features
    
    Returns:
        dict: Feature descriptions
    """
    return {
        # Order characteristics
        "n_items": "Number of items in the order",
        "n_sellers": "Number of different sellers",
        "n_products": "Number of unique products",
        "n_categories": "Number of product categories",
        "mode_category_count": "Count of most frequent category",
        
        # Financial
        "sum_price": "Total price of all items (R$)",
        "sum_freight": "Total freight/shipping cost (R$)",
        "total_payment": "Total payment amount (R$)",
        "n_payment_records": "Number of payment transactions",
        "max_installments": "Maximum number of installments",
        
        # Product dimensions
        "avg_weight_g": "Average product weight (grams)",
        "avg_length_cm": "Average product length (cm)",
        "avg_height_cm": "Average product height (cm)",
        "avg_width_cm": "Average product width (cm)",
        
        # Geographic
        "n_seller_states": "Number of different seller states",
        "seller_state_mode": "Most common seller state",
        "customer_city": "Customer city",
        "customer_state": "Customer state",
        
        # Temporal
        "purch_year": "Purchase year",
        "purch_month": "Purchase month (1-12)",
        "purch_dayofweek": "Day of week (0=Mon, 6=Sun)",
        "purch_hour": "Hour of purchase (0-23)",
        "purch_is_weekend": "Weekend purchase (0=No, 1=Yes)",
        "purch_hour_sin": "Hour sine encoding",
        "purch_hour_cos": "Hour cosine encoding",
        
        # Logistics
        "est_lead_days": "Estimated delivery lead time (days)",
        "mode_category": "Most common product category",
        
        # Payment type (one-hot encoded)
        "paytype_boleto": "Payment via Boleto",
        "paytype_credit_card": "Payment via Credit Card",
        "paytype_debit_card": "Payment via Debit Card",
        "paytype_not_defined": "Payment type not defined",
        "paytype_voucher": "Payment via Voucher"
    }


def calculate_temporal_features(purchase_datetime):
    """
    Calculate temporal features from datetime
    
    Args:
        purchase_datetime: datetime object
    
    Returns:
        dict: Temporal features
    """
    hour = purchase_datetime.hour
    
    return {
        "purch_year": purchase_datetime.year,
        "purch_month": purchase_datetime.month,
        "purch_dayofweek": purchase_datetime.weekday(),
        "purch_hour": hour,
        "purch_is_weekend": 1 if purchase_datetime.weekday() >= 5 else 0,
        "purch_hour_sin": np.sin(2 * np.pi * hour / 24),
        "purch_hour_cos": np.cos(2 * np.pi * hour / 24)
    }
