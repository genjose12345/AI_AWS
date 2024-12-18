# AWSTrafficPredictor.py

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import boto3
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import json
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import generate_sample_data

class AWSTrafficPredictor:
    def __init__(self):
        """Initialize the Enhanced AWS Traffic Predictor with visualization capabilities"""
        # Load environment variables
        load_dotenv()
        
        self.logger = self._setup_logging()
        
        # connect to  AWS session
        self.session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        # create components
        self.cloudwatch = self.session.client('cloudwatch')
        self.ec2 = self.session.client('ec2')
        self.scaler = MinMaxScaler()
        self.model = None
        self.sequence_length = 24
        
        # Create directories for outputs
        os.makedirs('visualizations', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
    def _setup_logging(self):
        """Configure logging for the application"""
        logger = logging.getLogger('AWSTrafficPredictor')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler('logs/predictor.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def prepare_data(self, df):
        """Prepare data for model training"""
        self.logger.info("Preparing data for training...")
        
        # Sort by timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(df[['network_traffic']])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test

    def build_model(self):
        """Build the LSTM model"""
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.sequence_length, 1), 
                 return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        return self.model

    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return history

    def predict_traffic(self, data):
        """Make traffic predictions"""
        if len(data) != self.sequence_length:
            raise ValueError(f"Input data must have length {self.sequence_length}")
        
        # Ensure data is in the right shape
        sequence = data.reshape(1, self.sequence_length, 1)
    
        # Make prediction
        prediction = self.model.predict(sequence, verbose=0)
    
        # Convert prediction back to original scale
        prediction_scaled = prediction.reshape(-1, 1)
        prediction_original = self.scaler.inverse_transform(prediction_scaled)
    
        return prediction_original[0][0]

    def recommend_resources(self, predicted_traffic):
        """Recommend AWS resources based on predicted traffic"""
        predicted_traffic = float(predicted_traffic)
        
        THRESHOLDS = {
            'low': 1000000,    # 1 MB/s
            'medium': 5000000, # 5 MB/s
            'high': 10000000   # 10 MB/s
        }
        
        INSTANCE_RECOMMENDATIONS = {
            'low': {
                'type': 't3.micro',
                'vcpus': 2,
                'memory': '1 GiB',
                'network': 'Up to 5 Gigabit'
            },
            'medium': {
                'type': 't3.small',
                'vcpus': 2,
                'memory': '2 GiB',
                'network': 'Up to 5 Gigabit'
            },
            'high': {
                'type': 't3.medium',
                'vcpus': 2,
                'memory': '4 GiB',
                'network': 'Up to 5 Gigabit'
            }
        }
        
        if predicted_traffic < THRESHOLDS['low']:
            traffic_level = 'low'
        elif predicted_traffic < THRESHOLDS['medium']:
            traffic_level = 'medium'
        else:
            traffic_level = 'high'
        
        recommendation = {
            'predicted_traffic': predicted_traffic,
            'traffic_level': traffic_level,
            'recommended_instance': INSTANCE_RECOMMENDATIONS[traffic_level],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return recommendation
        
    def plot_raw_traffic_data(self, df):
        """Plot raw traffic data with daily and weekly patterns"""
        plt.figure(figsize=(15, 8))
        
        # Plot raw traffic
        plt.subplot(2, 1, 1)
        plt.plot(df['timestamp'], df['network_traffic'], label='Raw Traffic')
        plt.title('Raw Network Traffic Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Traffic (bytes/s)')
        plt.legend()
        
        # Plot daily averages
        plt.subplot(2, 1, 2)
        daily_avg = df.groupby(df['timestamp'].dt.date)['network_traffic'].mean()
        plt.plot(daily_avg.index, daily_avg.values, label='Daily Average')
        plt.title('Daily Average Network Traffic')
        plt.xlabel('Date')
        plt.ylabel('Average Traffic (bytes/s)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/raw_traffic_analysis.png')
        plt.close()

    def plot_traffic_patterns(self, df):
        """Plot traffic patterns by hour and day of week"""
        plt.figure(figsize=(15, 6))
        
        # Hourly patterns
        plt.subplot(1, 2, 1)
        hourly_avg = df.groupby(df['timestamp'].dt.hour)['network_traffic'].mean()
        sns.barplot(x=hourly_avg.index, y=hourly_avg.values)
        plt.title('Average Traffic by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Traffic')
        
        # Daily patterns
        plt.subplot(1, 2, 2)
        daily_avg = df.groupby(df['timestamp'].dt.dayofweek)['network_traffic'].mean()
        sns.barplot(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], y=daily_avg.values)
        plt.title('Average Traffic by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Traffic')
        
        plt.tight_layout()
        plt.savefig('visualizations/traffic_patterns.png')
        plt.close()

    def plot_model_architecture(self):
        """Visualize the LSTM model architecture"""
        tf.keras.utils.plot_model(
            self.model,
            to_file='visualizations/model_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            dpi=96
        )

    def plot_training_metrics(self, history):
        """Plot detailed training metrics"""
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Learning curve
        plt.subplot(2, 2, 2)
        plt.plot(history.history['loss'])
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('visualizations/training_metrics.png')
        plt.close()

    def evaluate_predictions(self, y_true, y_pred):
        """Calculate and plot prediction accuracy metrics"""
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        # Residual plot
        plt.subplot(2, 2, 2)
        residuals = y_pred - y_true
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        
        # Metrics text
        plt.subplot(2, 2, 3)
        plt.text(0.1, 0.8, f'MSE: {mse:.2f}')
        plt.text(0.1, 0.6, f'RMSE: {rmse:.2f}')
        plt.text(0.1, 0.4, f'MAE: {mae:.2f}')
        plt.text(0.1, 0.2, f'R²: {r2:.2f}')
        plt.title('Model Metrics')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/prediction_evaluation.png')
        plt.close()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def plot_resource_recommendations(self, recommendations_history):
        """Visualize resource allocation recommendations over time"""
        plt.figure(figsize=(15, 6))
        
        # Extract data
        timestamps = [r['timestamp'] for r in recommendations_history]
        traffic_levels = [r['traffic_level'] for r in recommendations_history]
        instance_types = [r['recommended_instance']['type'] for r in recommendations_history]
        
        # Plot traffic levels
        plt.subplot(1, 2, 1)
        traffic_level_map = {'low': 1, 'medium': 2, 'high': 3}
        traffic_levels_numeric = [traffic_level_map[level] for level in traffic_levels]
        plt.plot(timestamps, traffic_levels_numeric, marker='o')
        plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
        plt.title('Traffic Level Changes')
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        
        # Plot instance type distribution
        plt.subplot(1, 2, 2)
        instance_counts = pd.Series(instance_types).value_counts()
        plt.pie(instance_counts.values, labels=instance_counts.index, autopct='%1.1f%%')
        plt.title('Instance Type Distribution')
        
        plt.tight_layout()
        plt.savefig('visualizations/resource_recommendations.png')
        plt.close()

    def save_model(self, filepath='models/predictor_model'):
        """Save the model"""
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath='models/predictor_model'):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath)
        self.logger.info(f"Model loaded from {filepath}")


def plot_predictions(self, actual_values, predicted_values, window_size=100):
        """
        Enhanced visualization of predictions vs actual values
        
        Parameters:
        -----------
        actual_values : array-like
            The actual traffic values
        predicted_values : array-like
            The predicted traffic values
        window_size : int
            Number of points to display for clearer visualization
        """
        plt.figure(figsize=(15, 8))
        
        # Convert predictions back to original scale if needed
        if hasattr(self, 'scaler'):
            predicted_scaled = np.array(predicted_values).reshape(-1, 1)
            actual_scaled = np.array(actual_values).reshape(-1, 1)
            
            predicted_values = self.scaler.inverse_transform(predicted_scaled).flatten()
            actual_values = self.scaler.inverse_transform(actual_scaled).flatten()
        
        # Plot both lines
        plt.plot(actual_values[-window_size:], label='Actual Traffic', alpha=0.7)
        plt.plot(predicted_values[-window_size:], label='Predicted Traffic', alpha=0.7)
        
        plt.title('Network Traffic Predictions')
        plt.xlabel('Time')
        plt.ylabel('Network Traffic (bytes/s)')
        plt.legend()
        plt.grid(True)
        
        # Add prediction metrics
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, predicted_values)
        
        plt.text(0.02, 0.95, f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}', 
                 transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('visualizations/predictions_comparison.png')
        plt.close()

# In the main section, add these debug prints:
if __name__ == "__main__":
    predictor = AWSTrafficPredictor()
    
    # Generate sample data
    print("Generating and visualizing sample data...")
    data = generate_sample_data(days=30)
    
    # Debug print
    print("\nSample Data Shape:", data.shape)
    print("Sample Data Head:\n", data.head())
    
    # Prepare and train model
    X_train, X_test, y_train, y_test = predictor.prepare_data(data)
    
    # Debug print
    print("\nData Shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    predictor.build_model()
    history = predictor.train_model(X_train, y_train, epochs=50)
    
    # Make predictions with debugging
    predictions = []
    actuals = []
    recommendations_history = []
    
    print("\nMaking predictions...")
    for i in range(len(X_test)):
        pred = predictor.predict_traffic(X_test[i])
        actual = predictor.scaler.inverse_transform(y_test[i].reshape(-1, 1))[0][0]
        predictions.append(pred)
        actuals.append(actual)
        
        # Debug print every 50 predictions
        if i % 50 == 0:
            print(f"Prediction {i}: Actual = {actual:.2f}, Predicted = {pred:.2f}")
        
        recommendation = predictor.recommend_resources(pred)
        recommendations_history.append(recommendation)
    
    # Debug print before plotting
    print("\nPrediction Statistics:")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Prediction range: [{min(predictions):.2f}, {max(predictions):.2f}]")
    print(f"Actual range: [{min(actuals):.2f}, {max(actuals):.2f}]")
    
    # Plot predictions
    predictor.plot_predictions(actuals, predictions)
    
    # Evaluate and plot metrics
    metrics = predictor.evaluate_predictions(actuals, predictions)
    predictor.plot_resource_recommendations(recommendations_history)
    
    print("\nEvaluation Metrics:")
    print(json.dumps(metrics, indent=2))