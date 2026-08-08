from config import *
import pandas as pd
from src.data_loader import download_stock_data
from src.preprocessing import clean_data,  fit_scaler,transform_data
from src.indicators import add_indicators
from src.sequence import create_sequences
from src.model import build_lstm_model
from sklearn.model_selection import train_test_split
from src.trainer import train_model
from src.utils import save_model, log_step
from src.evaluate import evaluate_model
from src.visualization import plot_training_history, plot_predictions
from src.reproducibility import set_seed


def main():

    set_seed(42)
    print("=" * 60)
    print("Advanced Stock Price Prediction")
    print("=" * 60)

    print(f"Downloading data for {TICKER}...")

    # --------------------------------
    # 1. Download data
    # --------------------------------

    stock_data = download_stock_data(
        TICKER,
        START_DATE,
        END_DATE
    )

    print("✓ Data downloaded successfully")

    # --------------------------------
    # 2. Clean data
    # --------------------------------

    stock_data = clean_data(stock_data)

    print("✓ Data cleaned")

    # --------------------------------
    # 3. Feature engineering
    # --------------------------------

    stock_data = add_indicators(stock_data)

    print("✓ Technical indicators added")

    # --------------------------------
    # 4. Train/Test Split
    # --------------------------------

    split_index = int(len(stock_data) * 0.8)

    train_data = stock_data.iloc[:split_index].copy()
    test_data = stock_data.iloc[split_index:].copy()

    print(f"✓ Train rows: {len(train_data)}")
    print(f"✓ Test rows: {len(test_data)}")

    # --------------------------------
    # 5. Fit scaler ONLY on training data
    # --------------------------------

    scaler = fit_scaler(
        train_data,
        FEATURE_COLUMNS
    )

    print("✓ Scaler fitted on training data")

    # --------------------------------
    # 6. Transform train and test
    # --------------------------------

    train_scaled = transform_data(
        train_data,
        FEATURE_COLUMNS,
        scaler
    )

    test_scaled = transform_data(
        test_data,
        FEATURE_COLUMNS,
        scaler
    )

    print("✓ Train and test data scaled")

    # --------------------------------
    # 7. Prepare features
    # --------------------------------

    target_index = FEATURE_COLUMNS.index(
        TARGET_COLUMN
    )

    train_features = train_scaled[
        FEATURE_COLUMNS
    ].values

    test_features = test_scaled[
        FEATURE_COLUMNS
    ].values

    # --------------------------------
    # 8. Create training sequences
    # --------------------------------

    X_train, y_train = create_sequences(
        train_features,
        target_index,
        LOOK_BACK
    )

    # --------------------------------
    # 9. Add training context to test data
    # --------------------------------

    test_features_with_context = pd.concat(
        [
            train_scaled[FEATURE_COLUMNS].tail(LOOK_BACK),
            test_scaled[FEATURE_COLUMNS]
        ]
    ).values

    # --------------------------------
    # 10. Create test sequences
    # --------------------------------

    X_test, y_test = create_sequences(
        test_features_with_context,
        target_index,
        LOOK_BACK
    )

    print("\nSequence Shape")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    # --------------------------------
    # 11. Build model
    # --------------------------------

    model = build_lstm_model(
        LOOK_BACK,
        len(FEATURE_COLUMNS)
    )

    print("✓ LSTM model created")

    # --------------------------------
    # 12. Train model
    # --------------------------------

    print("\nTraining model...")

    history = train_model(
        model,
        X_train,
        y_train,
        EPOCHS,
        BATCH_SIZE
    )

    print("✓ Training completed")

    # --------------------------------
    # 13. Training graph
    # --------------------------------

    plot_training_history(history)

    log_step("Training curve saved")

    # --------------------------------
    # 14. Save model
    # --------------------------------

    save_model(
        model,
        MODEL_PATH
    )

    print(f"✓ Model saved to {MODEL_PATH}")

    # --------------------------------
    # 15. Evaluate model
    # --------------------------------

    y_test_actual, y_pred_actual, metrics = evaluate_model(
        model,
        X_test,
        y_test,
        scaler,
        target_index
    )

    # --------------------------------
    # 16. Prediction graph
    # --------------------------------

    plot_predictions(
        y_test_actual,
        y_pred_actual
    )

    log_step("Prediction graph saved")


if __name__ == "__main__":
    main()