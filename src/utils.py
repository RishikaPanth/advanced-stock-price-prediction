import joblib
import os 


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