import os
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):

    os.makedirs(
        "outputs/graphs",
        exist_ok=True
    )

    plt.figure(figsize=(10,6))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("Training vs Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        "outputs/graphs/loss_curve.png"
    )

    plt.close()


def plot_predictions(y_true, y_pred):
    """
    Plot actual vs predicted stock prices.
    """
    os.makedirs(
        "outputs/graphs",
        exist_ok=True
    )
    plt.figure(figsize=(14, 6))

    plt.plot(
        y_true,
        label="Actual Price",
        linewidth=2
    )

    plt.plot(
        y_pred,
        label="Predicted Price",
        linestyle="--",
        linewidth=2
    )

    plt.title("Actual vs Predicted Stock Prices")

    plt.xlabel("Trading Days")

    plt.ylabel("Price ($)")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "outputs/graphs/prediction_plot.png"
    )

    plt.close()