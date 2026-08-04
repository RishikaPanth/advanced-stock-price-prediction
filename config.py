"""
Configuration file for the Stock Price Prediction Project.
This file contains all project settings in one place.
"""


# Stock Information


TICKER = "AMZN"

START_DATE = "2019-01-01"

END_DATE = "2025-02-01"


# Model Parameters


LOOK_BACK = 30

EPOCHS = 100

BATCH_SIZE = 32

LEARNING_RATE = 0.001

TARGET_COLUMN = "Close"


# Prediction


FUTURE_DAYS = 30


# Paths


RAW_DATA_PATH = "data/raw/stock_data.csv"

MODEL_PATH = "models/lstm_model.keras"

SCALER_PATH = "models/scaler.pkl"



# Features


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_20",
    "EMA_20",
    "RSI"
]

TARGET_COLUMN = "Close"

LSTM_UNITS = [100, 50, 50]

DROPOUT_RATE = 0.2

LEARNING_RATE = 0.001