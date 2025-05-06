import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def analyze_predictions():
    # Get all prediction log files
    log_dir = "prediction_logs"
    log_files = glob.glob(os.path.join(log_dir, "predictions_*.csv"))
    
    if not log_files:
        print("No prediction logs found.")
        return
    
    # Combine all logs
    all_logs = []
    for file in log_files:
        try:
            df = pd.read_csv(file)
            all_logs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_logs:
        print("No valid logs found.")
        return
    
    combined_logs = pd.concat(all_logs, ignore_index=True)
    
    # Convert timestamp to datetime
    combined_logs['timestamp'] = pd.to_datetime(combined_logs['timestamp'])
    
    # Basic statistics
    print(f"Total predictions: {len(combined_logs)}")
    print(f"Positive predictions: {combined_logs['prediction'].sum()} ({combined_logs['prediction'].mean()*100:.1f}%)")
    print(f"Average probability: {combined_logs['probability'].mean():.3f}")
    
    # Time-based analysis
    combined_logs['date'] = combined_logs['timestamp'].dt.date
    daily_counts = combined_logs.groupby('date').size()
    
    plt.figure(figsize=(12, 6))
    daily_counts.plot(kind='bar')
    plt.title('Daily Prediction Counts')
    plt.ylabel('Number of Predictions')
    plt.tight_layout()
    plt.savefig('daily_prediction_counts.png')
    
    # Feature distribution
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(['Glucose', 'BMI', 'Age', 'BloodPressure']):
        plt.subplot(2, 2, i+1)
        sns.histplot(combined_logs[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    
    # Correlation with prediction
    correlation = combined_logs.corr()['prediction'].sort_values(ascending=False)
    print("\nCorrelation with prediction:")
    print(correlation)
    
    # Save analysis results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_logs.to_csv(f'analysis_results/all_predictions_{timestamp}.csv', index=False)
    correlation.to_csv(f'analysis_results/correlation_analysis_{timestamp}.csv')
    
    print(f"\nAnalysis completed. Results saved to 'analysis_results' directory.")

if __name__ == "__main__":
    # Create analysis results directory
    os.makedirs("analysis_results", exist_ok=True)
    analyze_predictions()
