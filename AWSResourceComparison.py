import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, List, Tuple
import json
from AWSTrafficPredictor import AWSTrafficPredictor
from utils import generate_sample_data

class AWSResourceComparison:
    def __init__(self):
        """Initialize AWS client and cost parameters"""
        # AWS instance pricing (USD per hour)
        self.instance_pricing = {
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            't3.large': 0.0832
        }
        
        # Instance capacity (requests/second)
        self.instance_capacity = {
            't3.micro': 1000000,  # 1 MB/s
            't3.small': 2500000,  # 2.5 MB/s
            't3.medium': 5000000,  # 5 MB/s
            't3.large': 10000000  # 10 MB/s
        }
        
        # Initialize AI predictor
        self.predictor = AWSTrafficPredictor()
        
    def simulate_static_allocation(self, traffic_data: pd.DataFrame, 
                                 instance_type: str) -> Dict:
        """Simulate static resource allocation with fixed instance type"""
        total_hours = len(traffic_data)
        instance_capacity = self.instance_capacity[instance_type]
        hourly_cost = self.instance_pricing[instance_type]
        
        # Calculate metrics
        overprovisioned_hours = sum(traffic_data['network_traffic'] < 
                                  (instance_capacity * 0.5))
        underprovisioned_hours = sum(traffic_data['network_traffic'] > 
                                   instance_capacity)
        optimal_hours = total_hours - overprovisioned_hours - underprovisioned_hours
        
        total_cost = total_hours * hourly_cost
        
        return {
            'instance_type': instance_type,
            'total_cost': total_cost,
            'overprovisioned_hours': overprovisioned_hours,
            'underprovisioned_hours': underprovisioned_hours,
            'optimal_hours': optimal_hours,
            'efficiency': (optimal_hours / total_hours) * 100
        }
    
    def simulate_ai_allocation(self, traffic_data: pd.DataFrame) -> Dict:
        """Simulate AI-based dynamic resource allocation"""
        costs = []
        instances = []
        efficiency_metrics = []
        
        # Train AI model
        X_train, X_test, y_train, y_test = self.predictor.prepare_data(traffic_data)
        self.predictor.build_model()
        self.predictor.train_model(X_train, y_train, epochs=50)
        
        # Process each hour
        for i in range(len(traffic_data)):
            if i >= 24:
                sequence = traffic_data['network_traffic'].iloc[i-24:i].values
                predicted_traffic = self.predictor.predict_traffic(sequence)
                recommendation = self.predictor.recommend_resources(predicted_traffic)
                instance_type = recommendation['recommended_instance']['type']
            else:
                instance_type = 't3.small' 
            
            actual_traffic = traffic_data['network_traffic'].iloc[i]
            instance_capacity = self.instance_capacity[instance_type]
            hourly_cost = self.instance_pricing[instance_type]
            
            costs.append(hourly_cost)
            instances.append(instance_type)
            
            # Calculate efficiency
            if actual_traffic < (instance_capacity * 0.5):
                efficiency_metrics.append('overprovisioned')
            elif actual_traffic > instance_capacity:
                efficiency_metrics.append('underprovisioned')
            else:
                efficiency_metrics.append('optimal')
        
        return {
            'hourly_costs': costs,
            'total_cost': sum(costs),
            'instance_types': instances,
            'efficiency_metrics': efficiency_metrics,
            'overprovisioned_hours': efficiency_metrics.count('overprovisioned'),
            'underprovisioned_hours': efficiency_metrics.count('underprovisioned'),
            'optimal_hours': efficiency_metrics.count('optimal'),
            'efficiency': (efficiency_metrics.count('optimal') / len(efficiency_metrics)) * 100
        }
    
    def plot_comparison(self, traffic_data: pd.DataFrame, 
                       static_results: Dict, ai_results: Dict,
                       save_path: str = 'comparison_results.png'):
        """Create comprehensive comparison visualizations"""
        fig = plt.figure(figsize=(20, 15))
        
        #Traffic and Resource Allocation
        ax1 = plt.subplot(3, 1, 1)
        traffic_line = ax1.plot(traffic_data.index, traffic_data['network_traffic'], 
                              label='Actual Traffic', color='blue', alpha=0.5)
        ax1.set_ylabel('Traffic (bytes/s)')
        
        # Add instance capacity lines
        static_capacity = self.instance_capacity[static_results['instance_type']]
        ax1.axhline(y=static_capacity, color='red', linestyle='--', 
                   label=f"Static Capacity ({static_results['instance_type']})")
        
        # Add AI instance changes
        for idx, instance in enumerate(ai_results['instance_types']):
            if idx == 0 or instance != ai_results['instance_types'][idx-1]:
                ax1.axvline(x=traffic_data.index[idx], color='green', alpha=0.2)
                
        ax1.set_title('Traffic vs Resource Allocation')
        ax1.legend()
        
        #Cost Comparison
        ax2 = plt.subplot(3, 1, 2)
        labels = ['Static Allocation', 'AI-Based Allocation']
        costs = [static_results['total_cost'], ai_results['total_cost']]
        ax2.bar(labels, costs)
        ax2.set_title('Total Cost Comparison')
        ax2.set_ylabel('Cost (USD)')
        
        # Add cost savings percentage
        cost_saving = ((static_results['total_cost'] - ai_results['total_cost']) / 
                      static_results['total_cost'] * 100)
        ax2.text(0.5, 0.95, f'Cost Savings: {cost_saving:.1f}%', 
                transform=ax2.transAxes, ha='center')
        
        #Efficiency Comparison
        ax3 = plt.subplot(3, 1, 3)
        metrics = ['Optimal', 'Overprovisioned', 'Underprovisioned']
        static_values = [static_results['optimal_hours'], 
                        static_results['overprovisioned_hours'],
                        static_results['underprovisioned_hours']]
        ai_values = [ai_results['optimal_hours'],
                    ai_results['overprovisioned_hours'],
                    ai_results['underprovisioned_hours']]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax3.bar(x - width/2, static_values, width, label='Static')
        ax3.bar(x + width/2, ai_values, width, label='AI-Based')
        
        ax3.set_ylabel('Hours')
        ax3.set_title('Resource Efficiency Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def main():
    # Generate traffic data
    days = 30
    traffic_data = generate_sample_data(days)
    
    # Initialize comparison
    comparison = AWSResourceComparison()
    
    # Run simulations
    static_results = comparison.simulate_static_allocation(traffic_data, 't3.medium')
    ai_results = comparison.simulate_ai_allocation(traffic_data)
    
    # Create visualization
    comparison.plot_comparison(traffic_data, static_results, ai_results)
    
    # Print results
    print("\nStatic Allocation Results:")
    print(json.dumps(static_results, indent=2))
    print("\nAI-Based Allocation Results:")
    print(json.dumps(ai_results, indent=2))

if __name__ == "__main__":
    main()