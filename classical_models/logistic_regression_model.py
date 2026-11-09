"""Logistic Regression for credit card fraud detection.

Linear classifier with class_weight='balanced' to offset the 0.17% fraud rate.
No regularisation is applied (C=inf, i.e. an unpenalised fit).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


def main():
    common.set_seed()
    X_train, X_test, y_train, y_test = common.classical_split()

    model = LogisticRegression(C=np.inf, class_weight="balanced", max_iter=1000)
    with common.Timer() as t:
        model.fit(X_train, y_train)

    return common.evaluate(
        "Logistic Regression", "logistic_regression", y_test,
        model.predict(X_test), model.predict_proba(X_test)[:, 1], t.seconds,
    )


if __name__ == "__main__":
    main()
