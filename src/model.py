from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from config import LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE


def build_lstm_model(look_back, num_features):

    model = Sequential()

    model.add(
        Input(shape=(look_back, num_features))
    )

    for i, units in enumerate(LSTM_UNITS):

        return_sequences = i < len(LSTM_UNITS) - 1

        model.add(
            LSTM(
                units=units,
                return_sequences=return_sequences
            )
        )

        model.add(
            Dropout(DROPOUT_RATE)
        )

    model.add(
        Dense(
            25,
            activation="relu"
        )
    )

    model.add(
        Dense(1)
    )

    optimizer = Adam(
        learning_rate=LEARNING_RATE
    )

    print("Learning Rate:", LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error"
    )

    return model