from config import *

from src.data_loader import download_stock_data
from src.preprocessing import clean_data, scale_data
from src.indicators import add_indicators


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

    scaled_data, scaler = scale_data(
        stock_data,
        ["Close"]
    )

    print("\nScaled Data")

    print(scaled_data.head())
    ## print(stock_data.head())
    ## print(stock_data.columns)
    ## print(type(stock_data["Close"]))


if __name__ == "__main__":
    main()