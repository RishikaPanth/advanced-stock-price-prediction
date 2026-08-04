from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from config import LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE


def build_lstm_model(look_back, num_features):
    
    model = Sequential()

    model.add(
        LSTM(
            units=LSTM_UNITS[0],
            return_sequences=True,
            input_shape=(look_back, num_features)
        )
    )

    model.add(Dropout(DROPOUT_RATE))

    model.add(
        LSTM(
            units=LSTM_UNITS[1],
            return_sequences=True
        )
    )

    model.add(Dropout(DROPOUT_RATE))

    model.add(
        LSTM(
            units=LSTM_UNITS[2]
        )
    )

    model.add(Dropout(DROPOUT_RATE))

    model.add(
        Dense(
            25,
            activation="relu"
        )
    )

    model.add(Dense(1))

    optimizer = Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error"
    )

    return model