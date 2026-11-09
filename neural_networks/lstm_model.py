"""Long Short-Term Memory (LSTM) network for credit card fraud detection.

The data has no real time order, so the 30 features are fed as 30 time steps of
one value each. One LSTM layer (32 units) is followed by dropout 0.2, a 16-unit
dense layer and a sigmoid output. Trained with the shared recipe in
common.train_keras_classifier.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers  # noqa: E402


def reshape(X):
    return X.reshape(-1, 30, 1)


def build_model():
    return tf.keras.Sequential([
        layers.Input(shape=(30, 1)),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
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
        "LSTM", "lstm", y_test, (scores >= 0.5).astype(int), scores, seconds,
        extra={"epochs_run": epochs},
    )


if __name__ == "__main__":
    main()
