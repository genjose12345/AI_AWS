import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
import json
from AWSTrafficPredictor import AWSTrafficPredictor

def generate_sample_data(days=30):
    """Generate sample network traffic data for testing"""
    timestamps = []
    traffic = []
    
    start_date = datetime.now() - timedelta(days=days)
    
    for hour in range(days * 24):
        current_time = start_date + timedelta(hours=hour)
        
        # Generate traffic with daily and weekly patterns
        daily_pattern = np.sin(2 * np.pi * (hour % 24) / 24)
        weekly_pattern = np.sin(2 * np.pi * (hour % (24 * 7)) / (24 * 7))
        
        # Add some random noise
        noise = np.random.normal(0, 0.1)
        
        # Combine patterns 
        traffic_value = (3000000 + 
                        1000000 * daily_pattern + 
                        500000 * weekly_pattern +
                        200000 * noise)
        
        timestamps.append(current_time)
        traffic.append(max(0, traffic_value))  # Ensure non-negative values
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'network_traffic': traffic
    })

def plot_training_history(history):
    """Plot model training history"""
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

def plot_predictions(actual, predicted, title='Traffic Predictions'):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(15, 6))
    plt.plot(actual, label='Actual Traffic', alpha=0.7)
    plt.plot(predicted, label='Predicted Traffic', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Network Traffic (bytes/s)')
    plt.legend()
    plt.savefig('predictions.png')
    plt.close()

def main():
    # Initialize predictor
    predictor = AWSTrafficPredictor()
    
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data(days=30)
    
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = predictor.prepare_data(data)
    
    # Build and train model
    print("Building and training model...")
    predictor.build_model()
    history = predictor.train_model(X_train, y_train, epochs=50)
    
    # Plot training history
    plot_training_history(history)
    
    # Make predictions
    print("Making predictions...")
    predictions = []
    actual = []
    
    for i in range(len(X_test)):
        pred = predictor.predict_traffic(X_test[i])
        actual_val = predictor.scaler.inverse_transform(y_test[i].reshape(-1, 1))[0][0]
        predictions.append(pred)
        actual.append(actual_val)
    
    # Plot predictions
    plot_predictions(actual, predictions)
    
    # Get resource recommendations
    print("\nGetting resource recommendations...")
    recommendation = predictor.recommend_resources(predictions[-1])
    print("\nResource Recommendation:")
    print(json.dumps(recommendation, indent=2))
    
    # Save model
    predictor.save_model()
    print("\nModel saved successfully!")

if __name__ == "__main__":
    main()