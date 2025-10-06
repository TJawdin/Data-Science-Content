def generate_synthetic_orders(n_orders=100):
    """
    Generate synthetic order data with realistic risk distributions
    OPTIMIZED for your XGBoost model's learned patterns
    
    Parameters:
    -----------
    n_orders : int
        Number of orders to generate
    
    Returns:
    --------
    pd.DataFrame : Synthetic order data with proper HIGH/MEDIUM/LOW risk mix
    """
    
    np.random.seed(None)  # Random seed for variety
    
    orders = []
    
    # Target distribution: 40% low, 40% medium, 20% high risk
    risk_categories = np.random.choice(
        ['low', 'medium', 'high'],
        size=n_orders,
        p=[0.4, 0.4, 0.2]
    )
    
    for i, risk_cat in enumerate(risk_categories):
        
        if risk_cat == 'low':
            # LOW RISK: Premium shipping, short/medium distance, high value
            num_items = np.random.randint(1, 4)
            num_sellers = np.random.choice([1, 2], p=[0.8, 0.2])
            total_order_value = np.random.uniform(150, 500)  # Higher value
            avg_shipping_distance_km = np.random.randint(50, 400)  # Short-medium distance
            
            # PREMIUM SHIPPING: $0.05-0.10/km
            total_shipping_cost = avg_shipping_distance_km * np.random.uniform(0.05, 0.10)
            
            total_weight_g = np.random.randint(300, 2000)
            avg_length_cm = np.random.uniform(15, 30)
            avg_height_cm = np.random.uniform(10, 20)
            avg_width_cm = np.random.uniform(8, 15)
            is_cross_state = np.random.choice([0, 1], p=[0.7, 0.3])  # Mostly local
            estimated_days = np.random.randint(7, 18)
            order_weekday = np.random.choice([0, 1, 2, 3, 4])  # Weekdays
            order_month = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Non-holiday
            order_hour = np.random.randint(8, 18)  # Business hours
            
        elif risk_cat == 'medium':
            # MEDIUM RISK: Standard shipping, moderate distance/complexity
            num_items = np.random.randint(2, 5)
            num_sellers = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
            total_order_value = np.random.uniform(100, 300)  # Medium value
            avg_shipping_distance_km = np.random.randint(300, 900)  # Medium-long distance
            
            # STANDARD SHIPPING: $0.02-0.04/km
            total_shipping_cost = avg_shipping_distance_km * np.random.uniform(0.02, 0.04)
            
            total_weight_g = np.random.randint(1000, 4000)
            avg_length_cm = np.random.uniform(25, 40)
            avg_height_cm = np.random.uniform(18, 30)
            avg_width_cm = np.random.uniform(12, 25)
            is_cross_state = np.random.choice([0, 1], p=[0.3, 0.7])  # Mostly cross-state
            estimated_days = np.random.randint(6, 12)
            order_weekday = np.random.randint(0, 7)  # Any day
            order_month = np.random.randint(1, 13)  # Any month
            order_hour = np.random.randint(6, 22)
            
        else:  # high risk
            # HIGH RISK: BUDGET shipping, long distance, low value
            num_items = np.random.randint(1, 3)  # SIMPLE orders!
            num_sellers = 1  # Single seller (less priority)
            total_order_value = np.random.uniform(30, 120)  # LOW VALUE
            avg_shipping_distance_km = np.random.randint(800, 2000)  # LONG distance
            
            # BUDGET SHIPPING: $0.003-0.012/km (KEY!)
            total_shipping_cost = avg_shipping_distance_km * np.random.uniform(0.003, 0.012)
            
            total_weight_g = np.random.randint(800, 3000)
            avg_length_cm = np.random.uniform(25, 50)
            avg_height_cm = np.random.uniform(20, 40)
            avg_width_cm = np.random.uniform(15, 30)
            is_cross_state = 1  # Always cross-state
            estimated_days = np.random.randint(3, 7)  # Rush timeline
            order_weekday = np.random.choice([4, 5, 6])  # Thursday/Friday/Weekend
            order_month = np.random.choice([11, 12])  # Holiday season
            order_hour = np.random.choice(list(range(0, 6)) + list(range(18, 24)))  # Off hours
        
        # Calculate derived fields
        is_weekend_order = 1 if order_weekday >= 5 else 0
        is_holiday_season = 1 if order_month in [11, 12] else 0
        
        order = {
            'order_id': f'ORDER_{i+1:05d}',
            'num_items': num_items,
            'num_sellers': num_sellers,
            'num_products': num_items,
            'total_order_value': round(total_order_value, 2),
            'avg_item_price': round(total_order_value / num_items, 2),
            'max_item_price': round(total_order_value / num_items * 1.2, 2),
            'total_shipping_cost': round(total_shipping_cost, 2),
            'avg_shipping_cost': round(total_shipping_cost / num_items, 2),
            'total_weight_g': int(total_weight_g),
            'avg_weight_g': int(total_weight_g / num_items),
            'max_weight_g': int(total_weight_g / num_items * 1.3),
            'avg_length_cm': round(avg_length_cm, 1),
            'avg_height_cm': round(avg_height_cm, 1),
            'avg_width_cm': round(avg_width_cm, 1),
            'avg_shipping_distance_km': int(avg_shipping_distance_km),
            'max_shipping_distance_km': int(avg_shipping_distance_km * 1.1),
            'is_cross_state': is_cross_state,
            'order_weekday': order_weekday,
            'order_month': order_month,
            'order_hour': order_hour,
            'is_weekend_order': is_weekend_order,
            'is_holiday_season': is_holiday_season,
            'estimated_days': estimated_days
        }
        
        orders.append(order)
    
    df = pd.DataFrame(orders)
    return df
