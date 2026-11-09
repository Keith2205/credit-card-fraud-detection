"""Random Forest for credit card fraud detection.

Bagged ensemble of 50 trees, maximum depth 10, class_weight='balanced'.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402


def main():
    common.set_seed()
    X_train, X_test, y_train, y_test = common.classical_split()

    model = RandomForestClassifier(
        n_estimators=50, max_depth=10, class_weight="balanced",
        random_state=common.SEED, n_jobs=-1,
    )
    with common.Timer() as t:
        model.fit(X_train, y_train)

    return common.evaluate(
        "Random Forest", "random_forest", y_test,
        model.predict(X_test), model.predict_proba(X_test)[:, 1], t.seconds,
    )


if __name__ == "__main__":
    main()
