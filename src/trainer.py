from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


def train_model(model, X_train, y_train, epochs, batch_size):
    
    ## Train the neural network.
    
    early_stopping = EarlyStopping( monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    return history