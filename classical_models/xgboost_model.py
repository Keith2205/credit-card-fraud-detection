"""XGBoost for credit card fraud detection.

Gradient-boosted trees: 100 estimators, maximum depth 3, learning rate 0.1.
scale_pos_weight is set to (legitimate / fraud) in the training set to offset
the class imbalance.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import numpy as np  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402


def main():
    common.set_seed()
    X_train, X_test, y_train, y_test = common.classical_split()

    neg, pos = np.bincount(y_train)
    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        scale_pos_weight=neg / pos, eval_metric="logloss",
        random_state=common.SEED, n_jobs=-1,
    )
    with common.Timer() as t:
        model.fit(X_train, y_train)

    return common.evaluate(
        "XGBoost", "xgboost", y_test,
        model.predict(X_test), model.predict_proba(X_test)[:, 1], t.seconds,
        extra={"scale_pos_weight": round(neg / pos, 1)},
    )


if __name__ == "__main__":
    main()
