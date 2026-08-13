from config import *
import pandas as pd
from src.data_loader import download_stock_data
from src.preprocessing import clean_data,  fit_scaler,transform_data
from src.indicators import add_indicators
from src.sequence import create_sequences,create_sequences_with_context
from src.model import build_lstm_model
from sklearn.model_selection import train_test_split
from src.trainer import train_model
from src.utils import save_model, log_step,save_scaler, save_metrics
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
# Chronological Train/Val/Test Split
# --------------------------------

    n = len(stock_data)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_data = stock_data.iloc[:train_end].copy()
    val_data = stock_data.iloc[train_end:val_end].copy()
    test_data = stock_data.iloc[val_end:].copy()

    print(f"✓ Train rows: {len(train_data)}")
    print(f"✓ Validation rows: {len(val_data)}")
    print(f"✓ Test rows: {len(test_data)}")

    # --------------------------------
    # 5. Fit scaler ONLY on training data
    # --------------------------------

    scaler = fit_scaler(
        train_data,
        FEATURE_COLUMNS
    )

    print("✓ Scaler fitted on training data")

    save_scaler(
    scaler,
    SCALER_PATH
)

    print(f"✓ Scaler saved to {SCALER_PATH}")

    # --------------------------------
    # 6. Transform train and test
    # --------------------------------

    train_scaled = transform_data(
        train_data,
        FEATURE_COLUMNS,
        scaler
    )

    val_scaled = transform_data(
        val_data,
        FEATURE_COLUMNS,
        scaler
    )
    
    test_scaled = transform_data(
        test_data,
        FEATURE_COLUMNS,
        scaler
    )

    print("✓ Train, validation and test data scaled")

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

    train_features = train_scaled[
    FEATURE_COLUMNS
    ].values

    X_train, y_train = create_sequences(
    train_features,
    target_index,
    LOOK_BACK
    )


# --------------------------------
# 9. Create validation sequences
# --------------------------------

    train_context = train_scaled[
    FEATURE_COLUMNS
    ].values[-LOOK_BACK:]

    val_features = val_scaled[
    FEATURE_COLUMNS
    ].values

    X_val, y_val = create_sequences_with_context(
    train_context,
    val_features,
    target_index,
    LOOK_BACK
    )


# --------------------------------
# 10. Create test sequences
# --------------------------------

    val_context = val_scaled[
    FEATURE_COLUMNS
    ].values[-LOOK_BACK:]

    test_features = test_scaled[
    FEATURE_COLUMNS
    ].values

    X_test, y_test = create_sequences_with_context(
    val_context,
    test_features,
    target_index,
    LOOK_BACK
    )


    print("\nSequence Shape")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val  :", X_val.shape)
    print("y_val  :", y_val.shape)
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
        X_val,
        y_val,
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

    save_metrics(
    metrics,
    METRICS_PATH
   )

    print(f"✓ Metrics saved to {METRICS_PATH}")

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