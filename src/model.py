from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from config import LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE


def build_lstm_model(look_back, num_features):

    first_units, second_units, third_units = LSTM_UNITS

    model = Sequential([
        
        Input(
            shape=(look_back, num_features)
        ),

        LSTM(
            first_units,
            return_sequences=True
        ),

        Dropout(
            DROPOUT_RATE
        ),

        LSTM(
            second_units,
            return_sequences=True
        ),

        Dropout(
            DROPOUT_RATE
        ),

        LSTM(
            third_units
        ),

        Dropout(
            DROPOUT_RATE
        ),

        Dense(
            25,
            activation="relu"
        ),

        Dense(1)
    ])

    optimizer = Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error"
    )

    return model