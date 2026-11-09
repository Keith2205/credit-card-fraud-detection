"""Run all ten models and build the combined results.

Usage:
    python run_all.py              # train and evaluate every model, then summarise
    python run_all.py --summary    # only rebuild the summary from saved results

Outputs:
    results/metrics.csv               one row per model
    results/metrics/<model>.json      full metrics for each model
    images/roc_curves/<model>.png     ROC curve for each model
    images/confusion_matrices/*.png   confusion matrix for each model
    images/roc_comparison.png         all ROC curves on one chart
"""
import importlib.util
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import common

MODELS = [
    # (display name, slug, script)
    ("Logistic Regression", "logistic_regression", "classical_models/logistic_regression_model.py"),
    ("Decision Tree", "decision_tree", "classical_models/decision_tree_model.py"),
    ("Random Forest", "random_forest", "classical_models/random_forest_model.py"),
    ("SVM", "svm", "classical_models/svm_model.py"),
    ("XGBoost", "xgboost", "classical_models/xgboost_model.py"),
    ("MLP", "mlp", "neural_networks/mlp_model.py"),
    ("DNN", "dnn", "neural_networks/dnn_model.py"),
    ("CNN", "cnn", "neural_networks/cnn_model.py"),
    ("LSTM", "lstm", "neural_networks/lstm_model.py"),
    ("Autoencoder", "autoencoder", "neural_networks/autoencoder_model.py"),
]


def run_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def summarise():
    rows = [json.loads((common.METRICS_DIR / f"{slug}.json").read_text()) for _, slug, _ in MODELS]
    table = pd.DataFrame(rows)
    table.insert(1, "family", ["Classical"] * 5 + ["Neural"] * 5)
    table.to_csv(common.RESULTS_DIR / "metrics.csv", index=False, float_format="%.4f")

    y_test = np.load(common.SCORES_DIR / "y_test.npy")
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, slug, _), row in zip(MODELS, rows):
        fpr, tpr, _ = roc_curve(y_test, np.load(common.SCORES_DIR / f"{slug}.npy"))
        style = "-" if slug in {s for _, s, _ in MODELS[:5]} else "--"
        ax.plot(fpr, tpr, style, lw=1.5, label=f"{name} ({row['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves - All Models (solid = classical, dashed = neural)")
    ax.legend(title="Model (ROC AUC)", loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(common.ROOT / "images" / "roc_comparison.png", dpi=150)
    plt.close(fig)

    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "train_seconds"]
    print("\n" + table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved {common.RESULTS_DIR / 'metrics.csv'}")


def main():
    if "--summary" not in sys.argv:
        for name, _, script in MODELS:
            print(f"\n{'=' * 70}\nRunning {name}\n{'=' * 70}", flush=True)
            run_script(common.ROOT / script)
    summarise()


if __name__ == "__main__":
    main()
