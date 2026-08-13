import joblib
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

from config import (
    TICKER,
    LOOK_BACK,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    MODEL_PATH,
    SCALER_PATH
)

from src.data_loader import download_stock_data
from src.preprocessing import clean_data
from src.indicators import add_indicators


# --------------------------------
# Load model and scaler once
# --------------------------------

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

target_index = FEATURE_COLUMNS.index(
    TARGET_COLUMN
)


def predict_stock():

    # --------------------------------
    # Download latest data
    # --------------------------------

    end_date = datetime.now()

    start_date = end_date - timedelta(days=100)

    stock_data = download_stock_data(
        TICKER,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )

    # --------------------------------
    # Clean data
    # --------------------------------

    stock_data = clean_data(
        stock_data
    )

    # --------------------------------
    # Add technical indicators
    # --------------------------------

    stock_data = add_indicators(
        stock_data
    )

    # --------------------------------
    # Select features
    # --------------------------------

    feature_data = stock_data[
        FEATURE_COLUMNS
    ].copy()

    # --------------------------------
    # Get latest LOOK_BACK days
    # --------------------------------

    latest_data = feature_data.tail(
        LOOK_BACK
    )

    # --------------------------------
    # Scale using trained scaler
    # --------------------------------

    scaled_data = scaler.transform(
        latest_data
    )

    # --------------------------------
    # Prepare LSTM input
    # --------------------------------

    X = np.array(
        [scaled_data]
    )

    # --------------------------------
    # Predict
    # --------------------------------

    prediction_scaled = model.predict(
        X,
        verbose=0
    )[0, 0]

    # --------------------------------
    # Inverse scaling
    # --------------------------------

    temp = np.zeros(
        (1, len(FEATURE_COLUMNS))
    )

    temp[
        0,
        target_index
    ] = prediction_scaled

    prediction_actual = scaler.inverse_transform(
        temp
    )[0, target_index]

    # --------------------------------
    # Latest actual price
    # --------------------------------

    latest_price = stock_data[
        TARGET_COLUMN
    ].iloc[-1]

    # --------------------------------
    # Expected change
    # --------------------------------

    expected_change = (
        (prediction_actual - latest_price)
        / latest_price
    ) * 100

    return {
        "ticker": TICKER,

        "latest_price": round(
            float(latest_price),
            2
        ),

        "predicted_price": round(
            float(prediction_actual),
            2
        ),

        "expected_change_percent": round(
            float(expected_change),
            2
        )
    }