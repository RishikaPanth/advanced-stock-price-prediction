import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score



def inverse_target_scaling(values, scaler, target_column_index):
    """
    Inverse transform only the target column.
    """

    temp = np.zeros((len(values), scaler.n_features_in_))

    temp[:, target_column_index] = values.flatten()

    temp = scaler.inverse_transform(temp)

    return temp[:, target_column_index]


def evaluate_model(
    model,
    X_test,
    y_test,
    scaler,
    target_column_index
):

    # Make predictions
    y_pred = model.predict(X_test, verbose=0)

    # Convert back to actual prices
    y_test_actual = inverse_target_scaling(
        y_test.reshape(-1, 1),
        scaler,
        target_column_index
    )

    y_pred_actual = inverse_target_scaling(
        y_pred,
        scaler,
        target_column_index
    )

    # Calculate metrics
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)

    metrics = {
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "r2": r2,
}


    # Print results
    print("\n" + "=" * 50)
    print("Model Evaluation")
    print("=" * 50)

    print(f"MAE   : {mae:.2f}")
    print(f"RMSE  : {rmse:.2f}")
    print(f"MAPE  : {mape * 100:.2f}%")
    print(f"R²    : {r2:.4f}")

    # Return values for plotting later
    return y_test_actual, y_pred_actual, metrics