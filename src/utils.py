import joblib
import os 
import json

def save_scaler(scaler, path):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ## Save trained scaler to disk.
    joblib.dump(scaler, path)



def save_model(model, path):

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ## Save trained model.
    model.save(path)

def log_step(message):
    """
    Print a formatted progress message.
    """
    print(f"✓ {message}")

def save_metrics(metrics, path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(path, "w") as file:
        json.dump(
            {
                "mae": round(float(metrics["mae"]), 4),
                "rmse": round(float(metrics["rmse"]), 4),
                "mape": round(float(metrics["mape"]), 4),
                "r2": round(float(metrics["r2"]), 4)
            },
            file,
            indent=4
        )