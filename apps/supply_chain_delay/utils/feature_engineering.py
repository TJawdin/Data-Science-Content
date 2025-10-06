"""
Feature Engineering Functions
Matches the 30 domain features from the notebook
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def calculate_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points using Haversine formula
    Returns distance in kilometers
    """
    R = 6371.0  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

def calculate_features(order_data):
    """
    Calculate all 30 domain features from raw order data
    
    Parameters:
    -----------
    order_data : dict or pd.DataFrame
        Raw order information
    
    Returns:
    --------
    pd.DataFrame with 30 features matching notebook EXACT order
    """
    
    # Convert to DataFrame if dict
    if isinstance(order_data, dict):
        df = pd.DataFrame([order_data])
    else:
        df = order_data.copy()
    
    # Create features dictionary (ensures no duplicates)
    features_dict = {}
    
    # ========== Order Complexity Features ==========
    features_dict['num_items'] = df.get('num_items', pd.Series([1])).values[0]
    features_dict['num_sellers'] = df.get('num_sellers', pd.Series([1])).values[0]
    features_dict['num_products'] = df.get('num_products', pd.Series([1])).values[0]
    features_dict['is_multi_seller'] = int(features_dict['num_sellers'] > 1)
    features_dict['is_multi_item'] = int(features_dict['num_items'] > 1)
    
    # ========== Financial Features ==========
    features_dict['total_order_value'] = df.get('total_order_value', pd.Series([0])).values[0]
    features_dict['avg_item_price'] = df.get('avg_item_price', pd.Series([0])).values[0]
    features_dict['max_item_price'] = df.get('max_item_price', pd.Series([0])).values[0]
    features_dict['total_shipping_cost'] = df.get('total_shipping_cost', pd.Series([0])).values[0]
    features_dict['avg_shipping_cost'] = df.get('avg_shipping_cost', pd.Series([0])).values[0]
    
    # Weight to price ratio
    features_dict['weight_to_price_ratio'] = (
        df.get('total_weight_g', pd.Series([0])).values[0] / 
        (features_dict['total_order_value'] + 1)
    )
    
    # ========== Physical Features ==========
    features_dict['total_weight_g'] = df.get('total_weight_g', pd.Series([0])).values[0]
    features_dict['avg_weight_g'] = df.get('avg_weight_g', pd.Series([0])).values[0]
    features_dict['max_weight_g'] = df.get('max_weight_g', pd.Series([0])).values[0]
    features_dict['avg_length_cm'] = df.get('avg_length_cm', pd.Series([0])).values[0]
    features_dict['avg_height_cm'] = df.get('avg_height_cm', pd.Series([0])).values[0]
    features_dict['avg_width_cm'] = df.get('avg_width_cm', pd.Series([0])).values[0]
    
    # Product volume
    features_dict['avg_product_volume_cm3'] = (
        features_dict['avg_length_cm'] * 
        features_dict['avg_height_cm'] * 
        features_dict['avg_width_cm']
    )
    
    # ========== Geographic Features ==========
    features_dict['avg_shipping_distance_km'] = df.get('avg_shipping_distance_km', pd.Series([500])).values[0]
    features_dict['max_shipping_distance_km'] = df.get('max_shipping_distance_km', pd.Series([500])).values[0]
    features_dict['is_cross_state'] = df.get('is_cross_state', pd.Series([0])).values[0]
    
    # Shipping cost per km
    features_dict['shipping_cost_per_km'] = (
        features_dict['total_shipping_cost'] / 
        (features_dict['avg_shipping_distance_km'] + 1)
    )
    
    # ========== Temporal Features ==========
    features_dict['order_weekday'] = df.get('order_weekday', pd.Series([2])).values[0]
    features_dict['order_month'] = df.get('order_month', pd.Series([6])).values[0]
    features_dict['order_hour'] = df.get('order_hour', pd.Series([14])).values[0]
    features_dict['is_weekend_order'] = df.get('is_weekend_order', pd.Series([0])).values[0]
    features_dict['is_holiday_season'] = df.get('is_holiday_season', pd.Series([0])).values[0]
    
    # ========== Time Estimation Features ==========
    features_dict['estimated_days'] = df.get('estimated_days', pd.Series([10])).values[0]
    features_dict['is_rush_order'] = int(features_dict['estimated_days'] < 7)
    
    # CRITICAL: Create DataFrame with features in EXACT notebook order
    # This order MUST match the order used during model training
    feature_order = [
        'num_items',
        'num_sellers',
        'num_products',
        'is_multi_seller',
        'is_multi_item',
        'total_order_value',
        'avg_item_price',
        'max_item_price',
        'total_shipping_cost',
        'avg_shipping_cost',
        'weight_to_price_ratio',
        'total_weight_g',
        'avg_weight_g',
        'max_weight_g',
        'avg_length_cm',
        'avg_height_cm',
        'avg_width_cm',
        'avg_product_volume_cm3',
        'avg_shipping_distance_km',
        'max_shipping_distance_km',
        'is_cross_state',
        'shipping_cost_per_km',
        'order_weekday',
        'order_month',
        'order_hour',
        'is_weekend_order',
        'is_holiday_season',
        'is_rush_order',
        'estimated_days'
    ]
    
    # Create DataFrame with exact order
    features_df = pd.DataFrame([features_dict])[feature_order]
    
    # Ensure all values are numeric
    features_df = features_df.astype(float)
    
    # Fill any NaN with 0
    features_df = features_df.fillna(0)
    
    return features_df


def get_feature_descriptions():
    """Return business-friendly descriptions of all features"""
    return {
        'num_items': 'Number of Items in Order',
        'num_sellers': 'Number of Sellers',
        'num_products': 'Number of Unique Products',
        'is_multi_seller': 'Multi-Seller Order (Yes/No)',
        'is_multi_item': 'Multi-Item Order (Yes/No)',
        'total_order_value': 'Total Order Value ($)',
        'avg_item_price': 'Average Item Price ($)',
        'max_item_price': 'Highest Item Price ($)',
        'total_shipping_cost': 'Total Shipping Cost ($)',
        'avg_shipping_cost': 'Average Shipping Cost ($)',
        'weight_to_price_ratio': 'Weight/Price Ratio',
        'total_weight_g': 'Total Weight (grams)',
        'avg_weight_g': 'Average Weight (grams)',
        'max_weight_g': 'Heaviest Item (grams)',
        'avg_length_cm': 'Average Length (cm)',
        'avg_height_cm': 'Average Height (cm)',
        'avg_width_cm': 'Average Width (cm)',
        'avg_product_volume_cm3': 'Average Product Volume (cm³)',
        'avg_shipping_distance_km': 'Shipping Distance (km)',
        'max_shipping_distance_km': 'Max Shipping Distance (km)',
        'is_cross_state': 'Cross-State Shipping (Yes/No)',
        'shipping_cost_per_km': 'Shipping Cost per KM ($)',
        'order_weekday': 'Order Day of Week (0=Mon, 6=Sun)',
        'order_month': 'Order Month (1-12)',
        'order_hour': 'Order Hour (0-23)',
        'is_weekend_order': 'Weekend Order (Yes/No)',
        'is_holiday_season': 'Holiday Season (Nov-Dec)',
        'is_rush_order': 'Rush Order (<7 days)',
        'estimated_days': 'Estimated Delivery Days'
    }
