from config import *

from src.data_loader import download_stock_data
from src.preprocessing import clean_data, scale_data
from src.indicators import add_indicators
from src.sequence import create_sequences
from src.model import build_lstm_model
from sklearn.model_selection import train_test_split
from src.trainer import train_model
from src.utils import save_model, log_step
from src.evaluate import evaluate_model
from src.visualization import plot_training_history, plot_predictions


def main():

    print("=" * 60)
    print("Advanced Stock Price Prediction")
    print("=" * 60)

    print(f"Downloading data for {TICKER}...")

    
    # 1. Download data
    
    stock_data = download_stock_data(
        TICKER,
        START_DATE,
        END_DATE
    )

    print("✓ Data downloaded successfully")

    
    # 2. Clean data
   
    stock_data = clean_data(stock_data)

    print("✓ Data cleaned")

    
    # 3. Feature engineering
    
    stock_data = add_indicators(stock_data)

    print("✓ Technical indicators added")

    print("\nColumns:")
    print(stock_data.columns)

    print("\nShape:")
    print(stock_data.shape)

   
    # 4. Scale features
    
    scaled_data, scaler = scale_data(
        stock_data,
        FEATURE_COLUMNS
    )

    print("✓ Features scaled")

    
    # 5. Create sequences
    
    feature_data = scaled_data[FEATURE_COLUMNS].values

    target_index = FEATURE_COLUMNS.index(
        TARGET_COLUMN
    )

    X, y = create_sequences(
        feature_data,
        target_index,
        LOOK_BACK
    )

    print(f"✓ Created {len(X)} training sequences")

    print("\nSequence Shape")
    print("X:", X.shape)
    print("y:", y.shape)

    
    # 6. Build model
    
    model = build_lstm_model(
        LOOK_BACK,
        len(FEATURE_COLUMNS)
    )

    print("✓ LSTM model created")

    
    # 7. Train/test split
   
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    print(
        f"✓ Train samples: {len(X_train)} | "
        f"Test samples: {len(X_test)}"
    )

    
    # 8. Train model
   
    print("\nTraining model...")

    history = train_model(
        model,
        X_train,
        y_train,
        EPOCHS,
        BATCH_SIZE
    )

    print("✓ Training completed")

    
    # 9. Training graph
    
    plot_training_history(history)

    log_step("Training curve saved")

    
    # 10. Save model
    
    save_model(
        model,
        MODEL_PATH
    )

    print(f"✓ Model saved to {MODEL_PATH}")

   
    # 11. Evaluate model
    
    y_test_actual, y_pred_actual, metrics = evaluate_model(
        model,
        X_test,
        y_test,
        scaler,
        target_index
    )

    
    # 12. Prediction graph
    
    plot_predictions(
        y_test_actual,
        y_pred_actual
    )

    log_step("Prediction graph saved")


if __name__ == "__main__":
    main()