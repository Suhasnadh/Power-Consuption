# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent paths for custom imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Preprocess"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Models"))

from Preprocessing import load_and_preprocess
from models import LSTMModel, BiLSTMModel, TransformerModel

# ARIMA imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    print("Warning: statsmodels not available. ARIMA model will be skipped.")
    print("Install with: pip install statsmodels")
    ARIMA_AVAILABLE = False

# Hyperparameters
epochs = 20
batch_size = 64
lr = 0.001
forecast_horizon = 24  # multi-step forecasting

# Load data
X_train, y_train, X_test, y_test, scaler = load_and_preprocess(seq_len=24, forecast_horizon=forecast_horizon)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Ensure y_train_tensor has the right shape for multi-step forecasting
if len(y_train_tensor.shape) == 3 and y_train_tensor.shape[-1] == 1:
    y_train_tensor = y_train_tensor.squeeze(-1)  # Remove last dimension if it's 1
elif len(y_train_tensor.shape) == 1:
    y_train_tensor = y_train_tensor.unsqueeze(-1)  # Add dimension if needed

print(f"X_train shape: {X_train_tensor.shape}")
print(f"y_train shape: {y_train_tensor.shape}")

# DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)


class SimpleARIMA:
    """Simple ARIMA Model wrapper"""
    
    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.fitted_model = None
        self.is_fitted = False
    
    def fit(self, train_data):
        """Fit ARIMA model to training data"""
        # Convert to 1D array if needed
        if len(train_data.shape) > 1:
            if train_data.shape[1] == 1:
                train_data = train_data.flatten()
            else:
                train_data = train_data[:, 0]
        
        print(f"Fitting ARIMA{self.order} model...")
        print(f"Training data shape: {train_data.shape}")
        
        try:
            # Fit ARIMA model
            model = ARIMA(train_data, order=self.order)
            self.fitted_model = model.fit()
            self.is_fitted = True
            print(f"ARIMA model fitted successfully!")
            print(f"AIC: {self.fitted_model.aic:.2f}")
            
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            self.is_fitted = False
    
    def predict(self, steps):
        """Make predictions for specified number of steps"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            if hasattr(forecast, 'values'):
                forecast = forecast.values
            return np.array(forecast).reshape(-1, 1)
            
        except Exception as e:
            print(f"Error making ARIMA predictions: {e}")
            return np.zeros((steps, 1))
    
    def save_model(self, filepath):
        """Save fitted ARIMA model"""
        if self.is_fitted:
            try:
                self.fitted_model.save(filepath)
                print(f"ARIMA model saved to {filepath}")
                return True
            except Exception as e:
                print(f"Error saving ARIMA model: {e}")
                return False
        else:
            print("Cannot save unfitted ARIMA model")
            return False


def prepare_arima_data(X_train, y_train):
    """Prepare data for ARIMA model"""
    # Reconstruct time series from sequences
    train_series = []
    
    # Add initial context from first sequence
    if len(X_train.shape) == 3:
        train_series.extend(X_train[0, :, 0].tolist())
    else:
        train_series.extend(X_train[0, :].tolist())
    
    # Add all target values
    if len(y_train.shape) == 2:
        for i in range(len(y_train)):
            if y_train.shape[1] > 1:
                train_series.extend(y_train[i, :].tolist())
            else:
                train_series.append(y_train[i, 0])
    else:
        train_series.extend(y_train.tolist())
    
    return np.array(train_series)


# Train function
def train(model, name):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"Starting training for {name}...")
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            
            # Ensure pred and yb have compatible shapes
            if pred.shape != yb.shape:
                if len(yb.shape) == 1 and len(pred.shape) == 2:
                    yb = yb.unsqueeze(-1)
                elif len(yb.shape) == 2 and len(pred.shape) == 1:
                    pred = pred.unsqueeze(-1)
            
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"{name} - Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")

    # Save model
    model_path = f"{name}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved {name} model to {model_path}\n")
    
    return model


def evaluate_model(model, X_test, y_test, scaler, name):
    """Evaluate model on test set"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        if len(y_test_tensor.shape) == 3 and y_test_tensor.shape[-1] == 1:
            y_test_tensor = y_test_tensor.squeeze(-1)
        elif len(y_test_tensor.shape) == 1:
            y_test_tensor = y_test_tensor.unsqueeze(-1)
        
        predictions = model(X_test_tensor)
        
        # Ensure compatible shapes for loss calculation
        if predictions.shape != y_test_tensor.shape:
            if len(y_test_tensor.shape) == 1 and len(predictions.shape) == 2:
                y_test_tensor = y_test_tensor.unsqueeze(-1)
            elif len(y_test_tensor.shape) == 2 and len(predictions.shape) == 1:
                predictions = predictions.unsqueeze(-1)
        
        test_loss = nn.MSELoss()(predictions, y_test_tensor)
        print(f"{name} Test Loss: {test_loss.item():.6f}")
        
    return predictions.numpy()


if __name__ == "__main__":
    # Determine input size from data
    input_size = X_train.shape[2] if len(X_train.shape) == 3 else 1
    
    print(f"Input size: {input_size}")
    print(f"Forecast horizon: {forecast_horizon}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print("-" * 50)
    
    # Train LSTM
    print("Training LSTM...")
    lstm_model = LSTMModel(input_size=input_size, 
                          hidden_size=64, 
                          num_layers=2, 
                          forecast_horizon=forecast_horizon)
    lstm_model = train(lstm_model, "lstm")
    
    # Evaluate LSTM
    print("Evaluating LSTM...")
    lstm_predictions = evaluate_model(lstm_model, X_test, y_test, scaler, "LSTM")
    
    # Train BiLSTM
    print("Training BiLSTM...")
    bilstm_model = BiLSTMModel(input_size=input_size, 
                              hidden_size=64, 
                              num_layers=2, 
                              forecast_horizon=forecast_horizon)
    bilstm_model = train(bilstm_model, "bilstm")
    
    # Evaluate BiLSTM
    print("Evaluating BiLSTM...")
    bilstm_predictions = evaluate_model(bilstm_model, X_test, y_test, scaler, "BiLSTM")
    
    # Train Transformer
    print("Training Transformer...")
    transformer_model = TransformerModel(input_size=input_size, 
                                       d_model=64, 
                                       nhead=4, 
                                       num_layers=2, 
                                       forecast_horizon=forecast_horizon)
    transformer_model = train(transformer_model, "transformer")
    
    # Evaluate Transformer
    print("Evaluating Transformer...")
    transformer_predictions = evaluate_model(transformer_model, X_test, y_test, scaler, "Transformer")
    
    # Train and Evaluate cd.. Model
    if ARIMA_AVAILABLE:
        print("\n" + "="*20 + " ARIMA MODEL " + "="*20)
        try:
            # Prepare data for ARIMA
            arima_train_data = prepare_arima_data(X_train, y_train)
            
            print(f"ARIMA training data shape: {arima_train_data.shape}")
            
            # Create and train ARIMA model
            arima_model = SimpleARIMA(order=(2, 1, 2))
            arima_model.fit(arima_train_data)
            
            if arima_model.is_fitted:
                # Save ARIMA model
                arima_model.save_model("arima_model.pkl")
                
                # Make test predictions
                print("Making ARIMA test predictions...")
                test_length = len(y_test) if len(y_test.shape) == 1 else len(y_test)
                arima_predictions = arima_model.predict(test_length)
                
                # Calculate test loss for comparison
                if len(y_test.shape) == 2:
                    y_test_flat = y_test[:, 0] if y_test.shape[1] > 1 else y_test.flatten()
                else:
                    y_test_flat = y_test
                
                arima_test_loss = np.mean((arima_predictions.flatten()[:len(y_test_flat)] - y_test_flat)**2)
                print(f"ARIMA Test Loss: {arima_test_loss:.6f}")
                
                print("ARIMA model training and evaluation completed!")
            else:
                print("ARIMA model fitting failed!")
                
        except Exception as e:
            print(f"Error with ARIMA model: {e}")
    else:
        print("\nARIMA model skipped - statsmodels not available")
    
    print("\n" + "="*20 + " SAVED MODEL FILES " + "="*20)
    print("The following model files have been created:")
    print("• lstm_model.pth - LSTM model weights")
    print("• bilstm_model.pth - BiLSTM model weights") 
    print("• transformer_model.pth - Transformer model weights")
    if ARIMA_AVAILABLE:
        print("• arima_model.pkl - ARIMA statistical model")
    
    print("\nTraining completed for all models!")