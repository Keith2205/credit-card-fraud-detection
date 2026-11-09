"""Decision Tree for credit card fraud detection.

Gini criterion with class_weight='balanced'. The maximum depth is tuned with a
3-fold stratified grid search on the training set only (scored by F1).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

DEPTH_GRID = [4, 6, 8, 10, 12, None]


def main():
    common.set_seed()
    X_train, X_test, y_train, y_test = common.classical_split()

    search = GridSearchCV(
        DecisionTreeClassifier(criterion="gini", class_weight="balanced", random_state=common.SEED),
        {"max_depth": DEPTH_GRID},
        scoring="f1",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=common.SEED),
        n_jobs=-1,
    )
    with common.Timer() as t:
        search.fit(X_train, y_train)
    model = search.best_estimator_
    best_depth = search.best_params_["max_depth"]
    print(f"Best max_depth from grid search: {best_depth}")

    return common.evaluate(
        "Decision Tree", "decision_tree", y_test,
        model.predict(X_test), model.predict_proba(X_test)[:, 1], t.seconds,
        extra={"max_depth": best_depth},
    )


if __name__ == "__main__":
    main()
