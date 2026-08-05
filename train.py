from config import *

from src.data_loader import download_stock_data
from src.preprocessing import clean_data, scale_data
from src.indicators import add_indicators
from config import FEATURE_COLUMNS
## from src.utils import save_scaler
from config import SCALER_PATH
import numpy as np
from src.sequence import create_sequences
from src.model import build_lstm_model
from sklearn.model_selection import train_test_split
from src.trainer import train_model
from src.utils import save_model, log_step
from src.evaluate import evaluate_model
from src.visualization import plot_training_history


def main():

    print("=" * 60)
    print("Advanced Stock Price Prediction")
    print("=" * 60)

    print(f"Downloading data for {TICKER}...")

    stock_data = download_stock_data( TICKER, START_DATE, END_DATE )
    print("✓ Data downloaded successfully")

    stock_data = clean_data(stock_data)
    print("✓ Data cleaned")

    stock_data = add_indicators(stock_data)
    print("✓ Technical indicators added")

    scaled_data, scaler =  scale_data( stock_data, FEATURE_COLUMNS)
    print("✓ Features scaled")

    feature_data = scaled_data[FEATURE_COLUMNS].values
    target_index = FEATURE_COLUMNS.index(TARGET_COLUMN)
    X, y = create_sequences( feature_data, target_index, LOOK_BACK)
    print(f"✓ Created {len(X)} training sequences")

    model = build_lstm_model( LOOK_BACK, len(FEATURE_COLUMNS))
    print("✓ LSTM model created")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"✓ Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    print("\nTraining model...")
    history = train_model( model, X_train, y_train, EPOCHS, BATCH_SIZE)
    print("✓ Training completed")

    plot_training_history(history)

    log_step("Training curve saved")
    

    save_model( model, MODEL_PATH )
    print(f"✓ Model saved to {MODEL_PATH}")

    y_test_actual, y_pred_actual, metrics= evaluate_model(
    model,
    X_test,
    y_test,
    scaler,
    target_index
)


if SHOW_DATA_PREVIEW:
    print(stock_data.head())

if SHOW_MODEL_SUMMARY:
    model.summary()


if __name__ == "__main__":
    main()