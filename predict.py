import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from config import (
    TICKER,
    START_DATE,
    END_DATE,
    LOOK_BACK,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    MODEL_PATH,
    SCALER_PATH
)

from src.data_loader import download_stock_data
from src.preprocessing import clean_data
from src.indicators import add_indicators


def main():

    print("=" * 60)
    print("Stock Price Prediction")
    print("=" * 60)

    # --------------------------------
    # 1. Load trained model
    # --------------------------------

    model = load_model(MODEL_PATH)

    print("✓ Model loaded")

    # --------------------------------
    # 2. Load trained scaler
    # --------------------------------

    scaler = joblib.load(SCALER_PATH)

    print("✓ Scaler loaded")

    # --------------------------------
    # 3. Download stock data
    # --------------------------------

    print(f"Downloading data for {TICKER}...")

    stock_data = download_stock_data(
        TICKER,
        START_DATE,
        END_DATE
    )

    print("✓ Data downloaded")

    # --------------------------------
    # 4. Clean data
    # --------------------------------

    stock_data = clean_data(stock_data)

    # --------------------------------
    # 5. Add technical indicators
    # --------------------------------

    stock_data = add_indicators(stock_data)

    print("✓ Technical indicators calculated")

    # --------------------------------
    # 6. Select required features
    # --------------------------------

    feature_data = stock_data[
        FEATURE_COLUMNS
    ].copy()

    # --------------------------------
    # 7. Take latest LOOK_BACK days
    # --------------------------------

    latest_data = feature_data.tail(
        LOOK_BACK
    )

    print(
        f"✓ Using latest {LOOK_BACK} trading days"
    )

    # --------------------------------
    # 8. Scale using trained scaler
    # --------------------------------

    scaled_data = scaler.transform(
        latest_data
    )

    # --------------------------------
    # 9. Create LSTM input
    # --------------------------------

    X = np.array(
        [scaled_data]
    )

    print("Input shape:", X.shape)

    # --------------------------------
    # 10. Make prediction
    # --------------------------------

    prediction_scaled = model.predict(
        X,
        verbose=0
    )

    # --------------------------------
    # 11. Convert prediction back
    #     to actual price
    # --------------------------------

    temp = np.zeros(
        (1, len(FEATURE_COLUMNS))
    )

    target_index = FEATURE_COLUMNS.index(
        TARGET_COLUMN
    )

    temp[
        0,
        target_index
    ] = prediction_scaled[0, 0]

    prediction_actual = scaler.inverse_transform(
        temp
    )[0, target_index]

    # --------------------------------
    # 12. Display prediction
    # --------------------------------

    last_price = stock_data[
        TARGET_COLUMN
    ].iloc[-1]

    print("\n" + "=" * 50)
    print("Prediction Result")
    print("=" * 50)

    print(
        f"Latest Close : ${last_price:.2f}"
    )

    print(
        f"Predicted Close : ${prediction_actual:.2f}"
    )

    change = (
        prediction_actual - last_price
    ) / last_price * 100

    print(
        f"Expected Change : {change:+.2f}%"
    )

    # --------------------------------
    # 13. Prediction visualization
    # --------------------------------

    recent_prices = stock_data[
    TARGET_COLUMN
    ].tail(60)

    plt.figure(figsize=(14, 6))

    plt.plot(
    recent_prices.index,
    recent_prices.values,
    label="Historical Close",
    linewidth=2
    )

    plt.scatter(
    stock_data.index[-1],
    last_price,
    label="Latest Close",
    s=80
    )

    plt.scatter(
    stock_data.index[-1],
    prediction_actual,
    label="Next-Day Prediction",
    s=100
    )

    plt.plot(
    [stock_data.index[-1], stock_data.index[-1]],
    [last_price, prediction_actual],
    linestyle="--"
    )

    plt.title(
    f"{TICKER} Next-Day Stock Price Prediction"
    )

    plt.xlabel("Date")
    plt.ylabel("Price ($)")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
    "outputs/graphs/next_day_prediction.png"
    )

    plt.close()

    print(
    "✓ Prediction graph saved"
    )


if __name__ == "__main__":
    main()