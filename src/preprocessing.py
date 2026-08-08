import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df):
    

    df = df.copy()

    df.dropna(inplace=True)

    return df

def fit_scaler(df, feature_columns):
    """
    Fit the scaler ONLY on training data.
    """

    scaler = MinMaxScaler()

    scaler.fit(df[feature_columns])

    return scaler


def transform_data(df, feature_columns, scaler):
    """
    Transform data using an already-fitted scaler.
    """

    scaled_df = df.copy()

    scaled_df[feature_columns] = scaler.transform(
        scaled_df[feature_columns]
    )

    return scaled_df