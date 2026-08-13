import joblib
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from config import (
    TICKER,
    START_DATE,
    END_DATE,
    LOOK_BACK,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    MODEL_PATH,
    SCALER_PATH
)

from src.data_loader import download_stock_data
from src.preprocessing import clean_data
from src.indicators import add_indicators


def inverse_target_scaling(
    value,
    scaler,
    target_index
):
    """
    Convert a scaled target value
    back to the original price scale.
    """

    temp = np.zeros(
        (1, len(FEATURE_COLUMNS))
    )

    temp[0, target_index] = value

    return scaler.inverse_transform(
        temp
    )[0, target_index]


def main():

    print("=" * 60)
    print("Test-Period Walk-Forward Backtest")
    print("=" * 60)

    # --------------------------------
    # 1. Load trained model
    # --------------------------------

    model = load_model(
        MODEL_PATH
    )

    print("✓ Model loaded")

    # --------------------------------
    # 2. Load training scaler
    # --------------------------------

    scaler = joblib.load(
        SCALER_PATH
    )

    print("✓ Scaler loaded")

    # --------------------------------
    # 3. Download historical data
    # --------------------------------

    print(
        f"Downloading data for {TICKER}..."
    )

    stock_data = download_stock_data(
        TICKER,
        START_DATE,
        END_DATE
    )

    print("✓ Data downloaded")

    # --------------------------------
    # 4. Clean data
    # --------------------------------

    stock_data = clean_data(
        stock_data
    )

    # --------------------------------
    # 5. Add indicators
    # --------------------------------

    stock_data = add_indicators(
        stock_data
    )

    print(
        "✓ Technical indicators calculated"
    )

    # --------------------------------
    # 6. Chronological split
    # --------------------------------

    n = len(stock_data)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_data = stock_data.iloc[
        :train_end
    ].copy()

    val_data = stock_data.iloc[
        train_end:val_end
    ].copy()

    test_data = stock_data.iloc[
        val_end:
    ].copy()

    print(
        f"✓ Train rows: {len(train_data)}"
    )

    print(
        f"✓ Validation rows: {len(val_data)}"
    )

    print(
        f"✓ Test rows: {len(test_data)}"
    )

    # --------------------------------
    # 7. Prepare all feature data
    # --------------------------------

    all_features = stock_data[
        FEATURE_COLUMNS
    ].copy()

    # IMPORTANT:
    # The scaler was fitted ONLY on training data
    scaled_all_data = scaler.transform(
        all_features
    )

    # --------------------------------
    # 8. Test period boundaries
    # --------------------------------

    test_start_index = val_end

    test_end_index = len(stock_data)

    print(
        f"\n✓ Backtesting only test period"
    )

    print(
        f"✓ Test start index: {test_start_index}"
    )

    print(
        f"✓ Test end index: {test_end_index}"
    )

    # --------------------------------
    # 9. Target index
    # --------------------------------

    target_index = FEATURE_COLUMNS.index(
        TARGET_COLUMN
    )

    predictions = []
    actual_values = []

    prediction_dates = []

    # --------------------------------
    # 10. Walk-forward test
    # --------------------------------

    print(
        "\nRunning test-period backtest..."
    )

    for i in range(
        test_start_index,
        test_end_index
    ):

        # Previous LOOK_BACK trading days
        # are used as input.

        sequence = scaled_all_data[
            i - LOOK_BACK:i
        ]

        X = np.expand_dims(
            sequence,
            axis=0
        )

        # Predict next day's scaled Close
        prediction_scaled = model.predict(
            X,
            verbose=0
        )[0, 0]

        # Convert prediction to actual price
        prediction_actual = (
            inverse_target_scaling(
                prediction_scaled,
                scaler,
                target_index
            )
        )

        # Actual Close
        actual_price = stock_data[
            TARGET_COLUMN
        ].iloc[i]

        predictions.append(
            prediction_actual
        )

        actual_values.append(
            actual_price
        )

        prediction_dates.append(
            stock_data.index[i]
        )

    predictions = np.array(
        predictions
    )

    actual_values = np.array(
        actual_values
    )

    # --------------------------------
    # 11. Calculate metrics
    # --------------------------------

    mae = mean_absolute_error(
        actual_values,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predictions
        )
    )

    mape = mean_absolute_percentage_error(
        actual_values,
        predictions
    )

    r2 = r2_score(
        actual_values,
        predictions
    )

    # --------------------------------
    # 12. Print results
    # --------------------------------

    print(
        "\n" + "=" * 50
    )

    print(
        "Test-Period Backtest Results"
    )

    print(
        "=" * 50
    )

    print(
        f"MAE   : {mae:.2f}"
    )

    print(
        f"RMSE  : {rmse:.2f}"
    )

    print(
        f"MAPE  : {mape * 100:.2f}%"
    )

    print(
        f"R²    : {r2:.4f}"
    )

    # --------------------------------
    # 13. Save graph
    # --------------------------------

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        prediction_dates,
        actual_values,
        label="Actual Price",
        linewidth=2
    )

    plt.plot(
        prediction_dates,
        predictions,
        label="Predicted Price",
        linestyle="--",
        linewidth=2
    )

    plt.title(
        "Test-Period Walk-Forward Backtest"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Price ($)"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/graphs/backtest_test_period.png"
    )

    plt.close()

    print(
        "✓ Backtest graph saved"
    )


if __name__ == "__main__":
    main()