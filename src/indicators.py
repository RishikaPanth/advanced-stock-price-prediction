import pandas as pd
from ta.trend import SMAIndicator
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange
from ta.momentum import StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator


# Trend Indicators
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

def add_macd(df):

    macd = MACD(
        close=df["Close"]
    )

    df["MACD"] = macd.macd()

    return df


# Momentum Indicators
def add_rsi(df, window=14):

    df = df.copy()

    rsi = RSIIndicator(close=df["Close"], window=window)

    df["RSI"] = rsi.rsi()

    return df

def add_stochastic(df):

    stoch = StochasticOscillator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )

    df["STOCH"] = stoch.stoch()

    return df

# Volatility Indicators

def add_atr(df):

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )

    df["ATR"] = atr.average_true_range()

    return df


def add_bollinger(df):

    bb = BollingerBands(
        close=df["Close"]
    )

    df["BB_HIGH"] = bb.bollinger_hband()

    df["BB_LOW"] = bb.bollinger_lband()

    return df

# Volume Indicators

def add_obv(df):

    obv = OnBalanceVolumeIndicator(
        close=df["Close"],
        volume=df["Volume"]
    )

    df["OBV"] = obv.on_balance_volume()

    return df



def add_indicators(df):

    df = add_sma(df)
    df = add_ema(df)
    df = add_macd(df)

    df = add_rsi(df)
    df = add_stochastic(df)

    df = add_atr(df)
    df = add_bollinger(df)

    df = add_obv(df)

    df.dropna(inplace=True)

    return df