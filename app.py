import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import ta
from prophet import Prophet
import os
import joblib

# Set page config
st.set_page_config(page_title="Stock Anomaly Detection", layout="wide")

# Title
st.title("Financial Time Series Anomaly Detection")

# Sidebar controls
st.sidebar.header("Settings")
selected_stock = st.sidebar.selectbox("Select Stock", ['AAPL', 'S', 'MSFT'])
detection_method = st.sidebar.selectbox("Anomaly Detection Method", 
                                      ['Isolation Forest', 'DBSCAN', 'Prophet'])
show_indicators = st.sidebar.checkbox("Show Technical Indicators", True)

# Load data function
@st.cache_data
def load_data(stock):
    try:
        df = pd.read_csv(f'{stock}_historical_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        return df
    except FileNotFoundError:
        st.error(f"Data file not found for {stock}")
        return None

# Preprocess data
def preprocess_data(df):
    if 'Price' not in df.columns:
        return None
    price_series = df['Price'].copy().dropna()
    return price_series

# Calculate indicators
def calculate_indicators(series):
    df = pd.DataFrame(series)
    df['Price'] = df['Price'].astype(str).str.replace(',', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'])
    
    # Technical indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['Price'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Price'], window=50)
    df['EMA_20'] = ta.trend.ema_indicator(df['Price'], window=20)
    df['EMA_50'] = ta.trend.ema_indicator(df['Price'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Price'], window=14)
    bb = ta.volatility.BollingerBands(df['Price'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_mid'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    return df

# Anomaly detection functions
def detect_anomalies_isolation_forest(df, contamination=0.05):
    features = ['Price', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI', 'BB_upper', 'BB_lower']
    df_processed = df[features].dropna()
    
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(df_processed)
    
    scores = model.decision_function(df_processed)
    predictions = model.predict(df_processed)
    df.loc[df_processed.index, 'anomaly_score_if'] = scores
    df.loc[df_processed.index, 'anomaly_if'] = predictions
    return df

def detect_anomalies_dbscan(df, eps=0.5, min_samples=5):
    features = ['Price', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI', 'BB_upper', 'BB_lower']
    df_processed = df[features].dropna()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_processed)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_features)
    df.loc[df_processed.index, 'dbscan_cluster'] = clusters
    df.loc[df_processed.index, 'anomaly_dbscan'] = np.where(clusters == -1, -1, 1)
    return df

def detect_anomalies_prophet(df):
    df_prophet = df[['Price']].reset_index().rename(columns={'Date': 'ds', 'Price': 'y'})
    
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=30, freq='D')
    forecast = model.predict(future)
    forecast_relevant = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')
    df_merged = df.merge(forecast_relevant, left_index=True, right_index=True, how='left')
    df_merged['anomaly_prophet'] = 0
    df_merged.loc[df_merged['Price'] > df_merged['yhat_upper'], 'anomaly_prophet'] = 1
    df_merged.loc[df_merged['Price'] < df_merged['yhat_lower'], 'anomaly_prophet'] = -1
    return df_merged

# Visualization functions
def plot_price_with_anomalies(df, method):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot price
    ax.plot(df.index, df['Price'], label='Price', color='blue')
    
    # Plot anomalies based on selected method
    if method == 'Isolation Forest':
        anomalies = df[df['anomaly_if'] == -1]
        ax.scatter(anomalies.index, anomalies['Price'], color='red', label='Anomaly')
    elif method == 'DBSCAN':
        anomalies = df[df['anomaly_dbscan'] == -1]
        ax.scatter(anomalies.index, anomalies['Price'], color='red', label='Anomaly')
    elif method == 'Prophet':
        upper_anomalies = df[df['anomaly_prophet'] == 1]
        lower_anomalies = df[df['anomaly_prophet'] == -1]
        ax.scatter(upper_anomalies.index, upper_anomalies['Price'], color='red', label='Upper Bound Anomaly')
        ax.scatter(lower_anomalies.index, lower_anomalies['Price'], color='green', label='Lower Bound Anomaly')
        
        # Plot forecast bounds
        ax.fill_between(df.index, df['yhat_lower'], df['yhat_upper'], color='lightblue', alpha=0.3, label='Confidence Interval')
        ax.plot(df.index, df['yhat'], color='orange', linestyle='--', label='Forecast')
    
    ax.set_title(f'{selected_stock} Price with {method} Anomalies')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_technical_indicators(df):
    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    
    # Moving Averages
    ax[0].plot(df.index, df['Price'], label='Price', color='blue')
    ax[0].plot(df.index, df['SMA_20'], label='SMA 20', color='orange')
    ax[0].plot(df.index, df['SMA_50'], label='SMA 50', color='green')
    ax[0].plot(df.index, df['EMA_20'], label='EMA 20', color='red', linestyle='--')
    ax[0].plot(df.index, df['EMA_50'], label='EMA 50', color='purple', linestyle='--')
    ax[0].set_title('Moving Averages')
    ax[0].legend()
    ax[0].grid(True)
    
    # RSI
    ax[1].plot(df.index, df['RSI'], label='RSI', color='blue')
    ax[1].axhline(30, color='red', linestyle='--')
    ax[1].axhline(70, color='red', linestyle='--')
    ax[1].set_title('Relative Strength Index (RSI)')
    ax[1].legend()
    ax[1].grid(True)
    
    # Bollinger Bands
    ax[2].plot(df.index, df['Price'], label='Price', color='blue')
    ax[2].plot(df.index, df['BB_upper'], label='Upper Band', color='red', linestyle='--')
    ax[2].plot(df.index, df['BB_mid'], label='Middle Band', color='green', linestyle='--')
    ax[2].plot(df.index, df['BB_lower'], label='Lower Band', color='red', linestyle='--')
    ax[2].set_title('Bollinger Bands')
    ax[2].legend()
    ax[2].grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

# Main app logic
def main():
    # Load data
    stock_data = load_data(selected_stock)
    if stock_data is None:
        return
    
    # Preprocess
    price_data = preprocess_data(stock_data)
    if price_data is None:
        st.error("Price data not found")
        return
    
    # Calculate indicators
    data_with_indicators = calculate_indicators(price_data)
    
    # Detect anomalies based on selected method
    if detection_method == 'Isolation Forest':
        data_with_anomalies = detect_anomalies_isolation_forest(data_with_indicators.copy())
    elif detection_method == 'DBSCAN':
        data_with_anomalies = detect_anomalies_dbscan(data_with_indicators.copy())
    elif detection_method == 'Prophet':
        data_with_anomalies = detect_anomalies_prophet(data_with_indicators.copy())
    
    # Display results
    st.subheader(f"{selected_stock} Stock Analysis")
    
    # Price and anomalies plot
    plot_price_with_anomalies(data_with_anomalies, detection_method)
    
    # Technical indicators if selected
    if show_indicators:
        st.subheader("Technical Indicators")
        plot_technical_indicators(data_with_indicators)
    
    # Show raw data if requested
    if st.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.write(data_with_indicators)

if __name__ == "__main__":
    main()