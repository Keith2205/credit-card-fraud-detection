"""Support Vector Machine for credit card fraud detection.

RBF kernel with class_weight='balanced'. The ROC curve uses the SVM's
decision-function score; Platt-scaled probabilities (probability=True) give the
same ranking, so the same AUC, but cost five extra fits on ~200k rows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from sklearn.svm import SVC  # noqa: E402


def main():
    common.set_seed()
    X_train, X_test, y_train, y_test = common.classical_split()

    model = SVC(kernel="rbf", class_weight="balanced", cache_size=2000, random_state=common.SEED)
    with common.Timer() as t:
        model.fit(X_train, y_train)

    return common.evaluate(
        "SVM", "svm", y_test,
        model.predict(X_test), model.decision_function(X_test), t.seconds,
    )


if __name__ == "__main__":
    main()
