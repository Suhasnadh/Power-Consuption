import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_preprocess(path=None, seq_len=24, forecast_horizon=24):
    # Default path with better error handling
    if path is None:
        path = os.path.join("..", "Data", "pjm_hourly_est.csv")
    
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    
    print(f"Loading data from: {path}")
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df[["Datetime", "PJM_Load"]].dropna().sort_values("Datetime")
    
    print(f"Data loaded: {len(df)} records")
    print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")

    # Scale the data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["PJM_Load"]])
    
    print(f"Data scaled. Min: {scaled.min():.4f}, Max: {scaled.max():.4f}")

    def create_sequences(data, seq_len, forecast_horizon):
        X, y = [], []
        # Fixed the range to ensure we don't go out of bounds
        for i in range(len(data) - seq_len - forecast_horizon + 1):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len:i + seq_len + forecast_horizon])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled, seq_len, forecast_horizon)
    
    print(f"Sequences created: X shape {X.shape}, y shape {y.shape}")
    
    # Ensure we have enough data
    if len(X) == 0:
        raise ValueError(f"Not enough data to create sequences. Need at least {seq_len + forecast_horizon} records.")

    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    print(f"Train set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")

    return X_train, y_train, X_test, y_test, scaler


def inverse_transform_predictions(predictions, scaler):
    """Helper function to inverse transform predictions back to original scale"""
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    return scaler.inverse_transform(predictions)


if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test, scaler = load_and_preprocess()
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        print("X_train:", X_train.shape, "y_train:", y_train.shape)
        print("X_test:", X_test.shape, "y_test:", y_test.shape)
        print(f"Feature dimensions: {X_train.shape[2] if len(X_train.shape) > 2 else 1}")
        print(f"Sequence length: {X_train.shape[1] if len(X_train.shape) > 1 else 'N/A'}")
        print(f"Forecast horizon: {y_train.shape[1] if len(y_train.shape) > 1 else 'N/A'}")
        
        # Show sample data
        print(f"\nSample input sequence shape: {X_train[0].shape}")
        print(f"Sample target sequence shape: {y_train[0].shape}")
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        print("Please check your data file path and format.")