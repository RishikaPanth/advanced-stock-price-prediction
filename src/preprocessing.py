import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df):
    

    df = df.copy()

    df.dropna(inplace=True)

    return df

def scale_data(df, feature_columns):
    

    scaler = MinMaxScaler()

    scaled_df = df.copy()

    scaled_df[feature_columns] = scaler.fit_transform(scaled_df[feature_columns])

    return scaled_df, scaler