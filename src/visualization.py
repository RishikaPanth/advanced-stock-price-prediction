import os
import matplotlib.pyplot as plt

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