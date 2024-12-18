# utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def generate_sample_data(days=30, base_traffic=3000000, add_anomalies=True):
    """
    Generate synthetic network traffic data with realistic patterns
    
    Parameters:
    -----------
    days : int
        Number of days of data to generate
    base_traffic : int
        Base traffic level in bytes/second
    add_anomalies : bool
        Whether to add random traffic spikes/drops
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with timestamp and network_traffic columns
    """
    timestamps = []
    traffic = []
    
    start_date = datetime.now() - timedelta(days=days)
    
    # Define business hours (9 AM to 5 PM)
    business_hours = range(9, 17)
    
    # Generate hourly data points
    for hour in range(days * 24):
        current_time = start_date + timedelta(hours=hour)
        current_hour = current_time.hour
        current_weekday = current_time.weekday()
        
        # Base traffic pattern components
        daily_pattern = (np.sin(2 * np.pi * (hour % 24) / 24) + 1) / 2
        if current_hour in business_hours:
            daily_pattern *= 1.5
            

        weekend_factor = 0.7 if current_weekday >= 5 else 1.0 
        

        trend = 1 + (hour / (days * 24)) * 0.1 
        
        noise = np.random.normal(0, 0.1)
        
        # Combine all patterns
        traffic_value = (base_traffic * 
                        daily_pattern * 
                        weekend_factor * 
                        trend +
                        base_traffic * noise)
        
        #Add occasional traffic spikes or drops 
        if add_anomalies and np.random.random() < 0.02:
            if np.random.random() < 0.5:
                traffic_value *= np.random.uniform(1.5, 3.0)
            else:
                traffic_value *= np.random.uniform(0.2, 0.5)
        
        timestamps.append(current_time)
        traffic.append(max(0, traffic_value))  # Ensure non-negative values
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'network_traffic': traffic
    })
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Plot the generated data
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Overall traffic pattern
    plt.subplot(3, 1, 1)
    plt.plot(df['timestamp'], df['network_traffic'])
    plt.title('Generated Network Traffic Pattern')
    plt.xlabel('Time')
    plt.ylabel('Traffic (bytes/s)')
    
    # Plot 2: Daily pattern (average by hour)
    plt.subplot(3, 1, 2)
    hourly_avg = df.groupby(df['timestamp'].dt.hour)['network_traffic'].mean()
    plt.bar(hourly_avg.index, hourly_avg.values)
    plt.title('Average Traffic by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Avg Traffic (bytes/s)')
    
    # Plot 3: Weekly pattern
    plt.subplot(3, 1, 3)
    daily_avg = df.groupby(df['timestamp'].dt.dayofweek)['network_traffic'].mean()
    plt.bar(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], daily_avg.values)
    plt.title('Average Traffic by Day of Week')
    plt.xlabel('Day')
    plt.ylabel('Avg Traffic (bytes/s)')
    
    plt.tight_layout()
    plt.savefig('visualizations/generated_data_patterns.png')
    plt.close()
    
    return df