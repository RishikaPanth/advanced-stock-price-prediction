import pandas as pd
from ta.trend import SMAIndicator
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

def add_sma(df, window=20):

    df = df.copy()

    sma = SMAIndicator(close=df["Close"], window=window)

    df[f"SMA_{window}"] = sma.sma_indicator()

    return df


def add_ema(df, window=20):

    df = df.copy()

    ema = EMAIndicator(close=df["Close"], window=window)

    df[f"EMA_{window}"] = ema.ema_indicator()

    return df

def add_rsi(df, window=14):

    df = df.copy()

    rsi = RSIIndicator(close=df["Close"], window=window)

    df["RSI"] = rsi.rsi()

    return df


def add_indicators(df):

    df = add_sma(df)

    df = add_ema(df)

    df = add_rsi(df)

    df.dropna(inplace=True)

    return df