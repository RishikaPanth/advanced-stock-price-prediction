import numpy as np


def create_sequences(data, target_column_index, look_back):
    """
    Creates input-output sequences for LSTM.

    Parameters
    ----------
    data : numpy.ndarray
        Scaled feature matrix.

    target_column_index : int
        Index of target column.

    look_back : int
        Number of previous days.

    Returns
    -------
    X, y
    """

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