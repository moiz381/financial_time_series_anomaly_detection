# financial_time_series_anomaly_detection

📈 Financial Time Series Anomaly Detection

This project identifies anomalies in historical stock price data using a combination of unsupervised learning and time-series forecasting methods. It offers a Streamlit-based web interface for interactive exploration of anomalies across multiple detection techniques.

🚀 Features

Anomaly Detection Models:

Isolation Forest – Unsupervised model to detect outliers based on data isolation.

DBSCAN – Density-based clustering algorithm that flags noise points as anomalies.

Prophet – Time-series forecasting model used to detect deviations from expected price trends.

Technical Indicators:

Simple and Exponential Moving Averages (SMA, EMA)

Relative Strength Index (RSI)

Bollinger Bands

Interactive Web Interface (Streamlit):

Select from multiple stocks (e.g., AAPL, MSFT, S)

Toggle between anomaly detection methods

Visualize price trends, anomalies, and technical indicators

Display raw processed data

🧠 Tech Stack

Frontend: Streamlit

Backend: Python (Pandas, NumPy, Matplotlib)

ML Models: Isolation Forest, DBSCAN, Prophet (Facebook)

Indicators: ta (technical analysis library)

📂 Folder Structure

.
├── app.py                         # Streamlit application
├── *.csv                          # Historical stock data files
├── models/                        # (Optional) Saved models (joblib)
└── README.md                      # Project documentation


📌 Future Enhancements

Deep learning-based anomaly detection (e.g., LSTM, Autoencoders)

Upload your own dataset via the app

Real-time data ingestion

Advanced charting with volume and candlestick overlays
