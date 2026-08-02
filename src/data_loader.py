import yfinance as yf
import pandas as pd


def download_stock_data(ticker, start_date, end_date):

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data