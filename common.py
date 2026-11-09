"""Shared pipeline for every model in the project.

All ten model scripts import this module so that they load the same data,
use the same splits, scale features the same way, and report the same metrics.

Splits (stratified, fixed seed, no resampling):
    * One 30% test set is held out first and is identical for all ten models.
    * Classical models train on the remaining 70%.
    * Neural networks split that 70% into 40% train / 30% validation
      (of the full dataset); the validation set drives early stopping.

The StandardScaler is fitted on the training portion only, so no information
from the validation or test sets leaks into training.
"""

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow start-up logs

import matplotlib

matplotlib.use("Agg")  # save figures to disk; never block on a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_SIZE = 0.30
VAL_SIZE = 0.30  # fraction of the FULL dataset used for validation (neural networks)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
SCORES_DIR = RESULTS_DIR / "scores"
ROC_DIR = ROOT / "images" / "roc_curves"
CM_DIR = ROOT / "images" / "confusion_matrices"


def set_seed(seed=SEED):
    """Fix every source of randomness so a re-run gives identical results."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass


def load_data():
    """Return features X (30 columns) and labels y (1 = fraud)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the data/ folder."
        )
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns="Class")
    y = df["Class"].astype(int)
    return X, y


def _scale(fit_on, *others):
    scaler = StandardScaler().fit(fit_on)
    return [scaler.transform(fit_on)] + [scaler.transform(o) for o in others]


def classical_split():
    """70% train / 30% test, scaled on the training set."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_test = _scale(X_train, X_test)
    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()


def neural_split():
    """40% train / 30% validation / 30% test, scaled on the training set.

    The test set is exactly the same rows as in classical_split().
    """
    X, y = load_data()
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=y_rest, random_state=SEED
    )
    X_train, X_val, X_test = _scale(X_train, X_val, X_test)
    return (
        X_train, X_val, X_test,
        y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy(),
    )


def balanced_class_weight(y):
    """Same weighting as scikit-learn's class_weight='balanced', for Keras."""
    counts = np.bincount(y)
    return {c: len(y) / (2 * n) for c, n in enumerate(counts)}


def keras_early_stopping(monitor="val_auc", mode="max", patience=3):
    import tensorflow as tf

    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor, mode=mode, patience=patience, restore_best_weights=True
    )


EPOCHS = 20
BATCH_SIZE = 32


def train_keras_classifier(model, X_train, y_train, X_val, y_val):
    """Shared training recipe for the supervised networks (MLP, DNN, CNN, LSTM).

    Adam + binary cross-entropy, balanced class weights, batch size 32, up to
    20 epochs with early stopping on validation ROC AUC (best weights kept).
    Returns (training seconds, epochs actually run).
    """
    import tensorflow as tf

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    with Timer() as t:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=balanced_class_weight(y_train),
            callbacks=[keras_early_stopping()],
            verbose=2,
        )
    return t.seconds, len(history.history["loss"])


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def evaluate(name, slug, y_true, y_pred, y_score, train_seconds, extra=None):
    """Compute metrics, print them, and save metrics, scores and plots.

    y_pred  : hard 0/1 predictions
    y_score : continuous fraud score used for ROC / PR curves
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "train_seconds": round(train_seconds, 1),
    }
    if extra:
        metrics.update(extra)

    print(f"\n=== {name} (test set: {len(y_true):,} transactions, {int(y_true.sum())} fraud) ===")
    print(f"Confusion matrix  [[TN FP] [FN TP]] = [[{tn} {fp}] [{fn} {tp}]]")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        print(f"{k:>10}: {metrics[k]:.4f}")
    print(f"{'train time':>10}: {metrics['train_seconds']} s")

    for d in (METRICS_DIR, SCORES_DIR, ROC_DIR, CM_DIR):
        d.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{slug}.json").write_text(json.dumps(metrics, indent=2))
    np.save(SCORES_DIR / f"{slug}.npy", y_score)
    if not (SCORES_DIR / "y_test.npy").exists():
        np.save(SCORES_DIR / "y_test.npy", y_true)

    _plot_roc(name, slug, y_true, y_score, metrics["roc_auc"])
    _plot_confusion(name, slug, tn, fp, fn, tp)
    return metrics


def _plot_roc(name, slug, y_true, y_score, auc):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, color="#1f5fa8", lw=2, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve - {name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROC_DIR / f"{slug}.png", dpi=150)
    plt.close(fig)


def _plot_confusion(name, slug, tn, fp, fn, tp):
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(cm.max(), 2)))
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}", ha="center", va="center",
                    fontsize=12, color="white" if cm[i, j] > 1000 else "black")
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(CM_DIR / f"{slug}.png", dpi=150)
    plt.close(fig)
