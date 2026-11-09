"""Autoencoder for unsupervised credit card fraud detection.

The network never sees a fraud label during training. It is trained only on the
legitimate transactions of the training split to reconstruct its input
(30 -> 16 -> 8 -> 16 -> 30, ReLU hidden layers, linear output, MSE loss, Adam).
Early stopping uses the reconstruction loss on the legitimate validation
transactions.

At test time, a transaction is flagged as fraud when its reconstruction error is
above the 95th percentile of the reconstruction errors on the training data.
The reconstruction error itself is the score used for the ROC curve.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

THRESHOLD_PERCENTILE = 95


def build_model():
    return tf.keras.Sequential([
        layers.Input(shape=(30,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),   # 8-dimensional bottleneck
        layers.Dense(16, activation="relu"),
        layers.Dense(30, activation="linear"),
    ])


def reconstruction_error(model, X):
    X_hat = model.predict(X, batch_size=4096, verbose=0)
    return np.mean((X - X_hat) ** 2, axis=1)


def main():
    common.set_seed()
    X_train, X_val, X_test, y_train, y_val, y_test = common.neural_split()
    X_train_legit = X_train[y_train == 0]
    X_val_legit = X_val[y_val == 0]

    model = build_model()
    model.compile(optimizer="adam", loss="mse")
    with common.Timer() as t:
        history = model.fit(
            X_train_legit, X_train_legit,
            validation_data=(X_val_legit, X_val_legit),
            epochs=common.EPOCHS,
            batch_size=common.BATCH_SIZE,
            callbacks=[common.keras_early_stopping(monitor="val_loss", mode="min")],
            verbose=2,
        )

    threshold = np.percentile(reconstruction_error(model, X_train_legit), THRESHOLD_PERCENTILE)
    errors = reconstruction_error(model, X_test)
    print(f"Anomaly threshold (95th percentile of training error): {threshold:.4f}")

    return common.evaluate(
        "Autoencoder", "autoencoder", y_test, (errors > threshold).astype(int), errors, t.seconds,
        extra={"epochs_run": len(history.history["loss"]), "threshold": round(float(threshold), 4)},
    )


if __name__ == "__main__":
    main()
