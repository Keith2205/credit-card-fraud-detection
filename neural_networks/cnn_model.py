"""1D Convolutional Neural Network (CNN) for credit card fraud detection.

The 30 features are reshaped to a 6 x 5 grid (6 steps of 5 channels). One
Conv1D layer (32 filters, kernel size 3) is followed by max pooling, flattening,
a 32-unit dense layer with dropout 0.3, and a sigmoid output. Trained with the
shared recipe in common.train_keras_classifier.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers  # noqa: E402


def reshape(X):
    return X.reshape(-1, 6, 5)


def build_model():
    return tf.keras.Sequential([
        layers.Input(shape=(6, 5)),
        layers.Conv1D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    common.set_seed()
    X_train, X_val, X_test, y_train, y_val, y_test = common.neural_split()
    X_train, X_val, X_test = (reshape(X_train), reshape(X_val), reshape(X_test))

    model = build_model()
    model.summary()
    seconds, epochs = common.train_keras_classifier(model, X_train, y_train, X_val, y_val)

    scores = model.predict(X_test, batch_size=4096, verbose=0).ravel()
    return common.evaluate(
        "CNN", "cnn", y_test, (scores >= 0.5).astype(int), scores, seconds,
        extra={"epochs_run": epochs},
    )


if __name__ == "__main__":
    main()
