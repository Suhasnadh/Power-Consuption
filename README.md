Power Consumption Forecasting using ARIMA, LSTM, BiLSTM, and Transformer
This project presents a comparative study of traditional and deep learning-based models for short-term energy load forecasting using the PJM Hourly Energy Consumption dataset. We evaluate four models: ARIMA, LSTM, BiLSTM, and Transformer, analyzing their accuracy and suitability for time series prediction tasks in the energy sector.

📊 Dataset
Source: PJM Interconnection / London SmartMeter Dataset

Features: Hourly total energy load (MW)

Preprocessing:

Linear interpolation for missing values

MinMax normalization

Sequence generation using a 24-hour sliding window

Chronological split (80% training / 20% testing)

🧠 Models Implemented
ARIMA: Classic statistical model, rolling predictions using statsmodels

LSTM: Deep learning sequential model with memory capabilities

BiLSTM: Bidirectional LSTM capturing both forward and backward dependencies

Transformer: Attention-based architecture capturing long-range temporal patterns

🛠️ Implementation Details
Framework: PyTorch

Loss Function: Mean Absolute Error (MAE)

Optimizer: Adam (lr=1e-4)

Training:

Batch Size: 64

Epochs: 50

Dropout: 0.2

Hardware: GPU-enabled systems

📈 Results
Model	MAE	RMSE	MAPE (%)
ARIMA	230	300	8.2
LSTM	145	210	4.5
BiLSTM	132	195	4.2
Transformer	120	180	3.8

🏆 Transformer outperformed all other models, showing the best forecast accuracy and robustness.

📎 Key Contributions
End-to-end pipeline from preprocessing to evaluation

Comparative analysis of classical and neural forecasting methods

Robust evaluation on real-world energy load data

Foundation for future exploration of advanced models like PatchTST and iTransformer

🔗 Repository
📂 Source Code: https://github.com/Suhasnadh/Power-Consuption

🙏 Acknowledgements
We thank PJM Interconnection and London Datastore for providing the dataset. We are also grateful to the PyTorch community for enabling seamless model implementation. Special appreciation to our mentors and reviewers for their constructive feedback.

