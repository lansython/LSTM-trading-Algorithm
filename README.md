# 📊 LSTM-Based Trading Strategy for Stock Market Prediction

This project implements a full walk-forward backtesting system using an **LSTM (Long Short-Term Memory)** deep learning model to predict future returns and build trading strategies on historical stock data.

It includes a full pipeline: from **data preparation** to **feature engineering**, **LSTM sequence modeling**, **signal generation**, and **portfolio management** — all integrated with **walk-forward retraining** and realistic backtesting logic.

---

## 🔍 Overview

- **Model**: Deep learning-based time series forecasting using LSTM (via `keras` + `TensorFlow`)
- **Use case**: Predict next-period returns and allocate capital across multiple stocks
- **Backtesting**: Walk-forward validation with periodic retraining (e.g. every 21 days)
- **Features**:
  - Lagged returns, volume, volatility
  - Symbol embeddings
  - Time-based engineered features

---

## 🛠️ Key Modules

| File | Description |
|------|-------------|
| `main.R` | Main script that runs the end-to-end training and walk-forward testing pipeline |
| `utils/feature_engineering.R` | Creates features from price and volume history |
| `model/lstm_model.R` | Builds and trains the LSTM model using Keras |
| `backtest/walkforward.R` | Handles rolling retraining, testing, and portfolio updates |
| `data/` | Input historical price data (e.g. Open, High, Low, Close, Volume) |

---

## ⚙️ How It Works

1. **Data Loading**  
   Load and preprocess historical stock data from CSVs.

2. **Feature Engineering**  
   Create a matrix of lag features, log returns, volatility, etc.

3. **Sequence Generation**  
   Turn feature matrices into 3D LSTM input sequences.

4. **Model Training**  
   Train the LSTM to predict 5-day forward returns for each stock.

5. **Walk-Forward Loop**  
   - Every N days: retrain model on latest data
   - Predict future returns
   - Generate position weights
   - Update portfolio and compute performance

---

## 📈 Performance Metrics

At the end of the walk-forward test, the script outputs:

- 🏁 Final portfolio NAV
- 📉 Daily returns
- ⚖️ Annualized volatility
- 📈 Sharpe ratio
- 📊 Cumulative return plot

