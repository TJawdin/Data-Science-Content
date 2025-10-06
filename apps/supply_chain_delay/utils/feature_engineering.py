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
    pd.DataFrame with 30 features matching notebook
    """
    
    # Convert to DataFrame if dict
    if isinstance(order_data, dict):
        df = pd.DataFrame([order_data])
    else:
        df = order_data.copy()
    
    features = pd.DataFrame()
    
    # ========== Order Complexity Features ==========
    features['num_items'] = df.get('num_items', 1)
    features['num_sellers'] = df.get('num_sellers', 1)
    features['num_products'] = df.get('num_products', 1)
    features['is_multi_seller'] = (features['num_sellers'] > 1).astype(int)
    features['is_multi_item'] = (features['num_items'] > 1).astype(int)
    
    # ========== Financial Features ==========
    features['total_order_value'] = df.get('total_order_value', 0)
    features['avg_item_price'] = df.get('avg_item_price', 0)
    features['max_item_price'] = df.get('max_item_price', 0)
    features['total_shipping_cost'] = df.get('total_shipping_cost', 0)
    features['avg_shipping_cost'] = df.get('avg_shipping_cost', 0)
    
    # Weight to price ratio
    features['weight_to_price_ratio'] = (
        df.get('total_weight_g', 0) / (features['total_order_value'] + 1)
    )
    
    # ========== Physical Features ==========
    features['total_weight_g'] = df.get('total_weight_g', 0)
    features['avg_weight_g'] = df.get('avg_weight_g', 0)
    features['max_weight_g'] = df.get('max_weight_g', 0)
    features['avg_length_cm'] = df.get('avg_length_cm', 0)
    features['avg_height_cm'] = df.get('avg_height_cm', 0)
    features['avg_width_cm'] = df.get('avg_width_cm', 0)
    
    # Product volume
    features['avg_product_volume_cm3'] = (
        features['avg_length_cm'] * 
        features['avg_height_cm'] * 
        features['avg_width_cm']
    )
    
    # ========== Geographic Features ==========
    # Calculate shipping distance if coordinates provided
    if all(k in df.columns for k in ['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng']):
        features['avg_shipping_distance_km'] = df.apply(
            lambda row: calculate_distance_km(
                row['seller_lat'], row['seller_lng'],
                row['customer_lat'], row['customer_lng']
            ), axis=1
        )
        features['max_shipping_distance_km'] = features['avg_shipping_distance_km']
    else:
        features['avg_shipping_distance_km'] = df.get('avg_shipping_distance_km', 500)
        features['max_shipping_distance_km'] = df.get('max_shipping_distance_km', 500)
    
    features['is_cross_state'] = df.get('is_cross_state', 0)
    
    # Shipping cost per km
    features['shipping_cost_per_km'] = (
        features['total_shipping_cost'] / 
        (features['avg_shipping_distance_km'] + 1)
    )
    
    # ========== Temporal Features ==========
    features['order_weekday'] = df.get('order_weekday', 2)  # Default: Tuesday
    features['order_month'] = df.get('order_month', 6)  # Default: June
    features['order_hour'] = df.get('order_hour', 14)  # Default: 2 PM
    features['is_weekend_order'] = df.get('is_weekend_order', 0)
    features['is_holiday_season'] = df.get('is_holiday_season', 0)
    
    # ========== Time Estimation Features ==========
    features['estimated_days'] = df.get('estimated_days', 10)
    features['is_rush_order'] = (features['estimated_days'] < 7).astype(int)
    
    # Fill any remaining NaN with 0
    features = features.fillna(0)
    
    # Ensure correct order (must match notebook training order)
    feature_order = [
        'num_items', 'num_sellers', 'num_products', 'is_multi_seller', 'is_multi_item',
        'total_order_value', 'avg_item_price', 'max_item_price', 'total_shipping_cost',
        'avg_shipping_cost', 'weight_to_price_ratio', 'total_weight_g', 'avg_weight_g',
        'max_weight_g', 'avg_length_cm', 'avg_height_cm', 'avg_width_cm',
        'avg_product_volume_cm3', 'avg_shipping_distance_km', 'max_shipping_distance_km',
        'is_cross_state', 'shipping_cost_per_km', 'order_weekday', 'order_month',
        'order_hour', 'is_weekend_order', 'is_holiday_season', 'is_rush_order',
        'estimated_days'
    ]
    
    # Reorder to match training
    features = features[feature_order]
    
    return features


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
