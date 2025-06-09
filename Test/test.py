# test.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Preprocess"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Models"))

from Preprocessing import load_and_preprocess
from models import LSTMModel, BiLSTMModel

# Forecast horizon
forecast_horizon = 24

# Load data
X_train, y_train, X_test, y_test, scaler = load_and_preprocess(seq_len=24, forecast_horizon=forecast_horizon)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Load models
input_size = X_test.shape[2]

lstm_model = LSTMModel(input_size=input_size, forecast_horizon=forecast_horizon)
lstm_model.load_state_dict(torch.load("../Train/lstm_model.pth"))
lstm_model.eval()

bilstm_model = BiLSTMModel(input_size=input_size, forecast_horizon=forecast_horizon)
bilstm_model.load_state_dict(torch.load("../Train/bilstm_model.pth"))
bilstm_model.eval()

# Predict
def predict(model, X):
    with torch.no_grad():
        preds = model(X).numpy()
    return preds

lstm_preds = predict(lstm_model, X_test_tensor)
bilstm_preds = predict(bilstm_model, X_test_tensor)

# Inverse scale (flatten for evaluation)
lstm_preds_inv = scaler.inverse_transform(lstm_preds.reshape(-1, 1))
bilstm_preds_inv = scaler.inverse_transform(bilstm_preds.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test_tensor.numpy().reshape(-1, 1))

# Metrics
def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mae, rmse

mae_lstm, rmse_lstm = evaluate(y_test_inv, lstm_preds_inv)
mae_bilstm, rmse_bilstm = evaluate(y_test_inv, bilstm_preds_inv)

print("LSTM:   MAE = {:.2f}, RMSE = {:.2f}".format(mae_lstm, rmse_lstm))
print("BiLSTM: MAE = {:.2f}, RMSE = {:.2f}".format(mae_bilstm, rmse_bilstm))

# Multi-step Visualization: Plot every 24th sample as a horizon of 24 steps
num_plots = 5
plt.figure(figsize=(15, 10))
for i in range(num_plots):
    idx = i * 50  # sample interval for plotting
    actual_seq = scaler.inverse_transform(y_test_tensor[idx].numpy().reshape(-1, 1)).flatten()
    lstm_seq = scaler.inverse_transform(lstm_preds[idx].reshape(-1, 1)).flatten()
    bilstm_seq = scaler.inverse_transform(bilstm_preds[idx].reshape(-1, 1)).flatten()

    plt.subplot(num_plots, 1, i + 1)
    plt.plot(actual_seq, label='Actual', linewidth=2)
    plt.plot(lstm_seq, label='LSTM', linestyle='--')
    plt.plot(bilstm_seq, label='BiLSTM', linestyle=':')
    plt.title(f"Forecast Horizon Sample {i+1}")
    plt.xlabel("Hour")
    plt.ylabel("Load (MW)")
    plt.grid(True)
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()
