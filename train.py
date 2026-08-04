from config import *

from src.data_loader import download_stock_data
from src.preprocessing import clean_data, scale_data
from src.indicators import add_indicators
from config import FEATURE_COLUMNS
## from src.utils import save_scaler
from config import SCALER_PATH
import numpy as np
from src.sequence import create_sequences
from src.model import build_lstm_model


def main():

    print("=" * 60)
    print("Advanced Stock Price Prediction")
    print("=" * 60)

    print(f"Downloading data for {TICKER}...")

    stock_data = download_stock_data(
        TICKER,
        START_DATE,
        END_DATE
    )
    print("\nRaw Data")
    print(stock_data.head())

    stock_data = clean_data(stock_data)
    stock_data = add_indicators(stock_data)

    scaled_data, scaler =  scale_data(
    stock_data,
    FEATURE_COLUMNS
  )

    feature_data = scaled_data[FEATURE_COLUMNS].values

    target_index = FEATURE_COLUMNS.index(TARGET_COLUMN)

    X, y = create_sequences(
    feature_data,
    target_index,
    LOOK_BACK
)

    model = build_lstm_model(
    LOOK_BACK,
    len(FEATURE_COLUMNS)
    
)

    ## save_scaler(scaler, SCALER_PATH)
    ## print("\nScaler saved successfully.")
    print("\nScaled Data")

    print(scaled_data.head())

    print("\nSequence Shape")

    print("X:", X.shape)

    print("y:", y.shape)

    model.summary()
    ## print(y[0])
    ## print(stock_data.head())
    ## print(stock_data.columns)
    ## print(type(stock_data["Close"]))


if __name__ == "__main__":
    main()