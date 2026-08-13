import numpy as np


def create_sequences(data, target_column_index, look_back):
    

    X = []
    y = []

    for i in range(len(data) - look_back):

        X.append(
            data[i:i + look_back]
        )

        y.append(
            data[i + look_back, target_column_index]
        )

    return np.array(X), np.array(y)


def create_sequences_with_context(
    previous_data,
    current_data,
    target_column_index,
    look_back
):

    combined = np.concatenate(
        [previous_data, current_data],
        axis=0
    )

    X = []
    y = []

    start = len(previous_data)

    for i in range(
        start,
        len(combined)
    ):

        X.append(
            combined[i - look_back:i]
        )

        y.append(
            combined[i, target_column_index]
        )

    return np.array(X), np.array(y)