# Advanced Stock Price Prediction

An end-to-end machine learning project that uses an LSTM neural network to predict the next trading day's stock closing price using historical market data and technical indicators.

## Features

- Historical stock data collection using Yahoo Finance
- Data cleaning and preprocessing
- Technical indicator feature engineering
- LSTM-based stock price prediction
- Look-back window and LSTM architecture experiments
- Feature ablation experiments
- Model evaluation using MAE, RMSE, MAPE and R²
- Backtesting utilities
- FastAPI backend
- Interactive HTML/CSS/JavaScript dashboard
- Historical price and prediction visualization

## Tech Stack

**Machine Learning:** Python, TensorFlow/Keras, NumPy, Pandas, Scikit-learn  
**Data:** yfinance  
**Backend:** FastAPI  
**Frontend:** HTML, CSS, JavaScript, Chart.js

## Project Structure

- `app/` — FastAPI backend and prediction logic
- `frontend/` — Dashboard interface
- `src/` — Data processing, indicators, sequences, model and training utilities
- `train.py` — Model training
- `predict.py` — Prediction
- `backtest.py` — Backtesting
- `config.py` — Project configuration

## Model Evaluation

The model is evaluated on unseen test data using:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- R²: Coefficient of Determination

## Disclaimer

This project is intended for educational and research purposes only. Stock market predictions are uncertain and the output should not be considered financial advice.
