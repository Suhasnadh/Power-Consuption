import torch
import torch.nn as nn
import math
import numpy as np
from statsmodels.tsa.arima.model import ARIMA as ARIMA_model


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, forecast_horizon=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden)
        out = self.fc(out[:, -1, :])  # only last time step
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, forecast_horizon=1):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, forecast_horizon)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.fc(out[:, -1, :])
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, forecast_horizon=1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, forecast_horizon)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        out = self.decoder(out[:, -1, :])
        return out


class ARIMAModel:
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model_fit = None

    def fit(self, train_data):
        self.model = ARIMA_model(train_data, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, steps):
        forecast = self.model_fit.forecast(steps=steps)
        return np.array(forecast)