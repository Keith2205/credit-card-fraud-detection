"""Multilayer Perceptron (MLP) for credit card fraud detection.

A shallow baseline network: one hidden layer of 64 ReLU units and a sigmoid
output. Trained with the shared recipe in common.train_keras_classifier
(Adam, binary cross-entropy, balanced class weights, early stopping).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers  # noqa: E402


def build_model():
    return tf.keras.Sequential([
        layers.Input(shape=(30,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    common.set_seed()
    X_train, X_val, X_test, y_train, y_val, y_test = common.neural_split()

    model = build_model()
    model.summary()
    seconds, epochs = common.train_keras_classifier(model, X_train, y_train, X_val, y_val)

    scores = model.predict(X_test, batch_size=4096, verbose=0).ravel()
    return common.evaluate(
        "MLP", "mlp", y_test, (scores >= 0.5).astype(int), scores, seconds,
        extra={"epochs_run": epochs},
    )


if __name__ == "__main__":
    main()
