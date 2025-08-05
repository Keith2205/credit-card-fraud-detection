# Random Forest for Credit Card Fraud Detection

# === Import Libraries ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
from pprint import pprint

# === Load Dataset ===
df = pd.read_csv("creditcard.csv")

# === Feature Scaling: Normalize 'Amount' and 'Time' ===
scaler = StandardScaler()
df['normAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['normTime'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# === Split Features and Target ===
X = df.drop('Class', axis=1)
y = df['Class']

# === Train-Test Split (Full Dataset, No Subsampling) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# === Train Random Forest Classifier with Imbalance Handling ===
rf_model = RandomForestClassifier(
    n_estimators=50,      # Number of trees
    max_depth=10,         # Limit tree depth to reduce overfitting
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)
rf_model.fit(X_train, y_train)

# === Make Predictions ===
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# === Evaluation Metrics ===
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)

print("=== Confusion Matrix (Random Forest) ===")
print(conf_matrix_rf)

print("\n=== Classification Report (Random Forest) ===")
pprint(class_report_rf)

print(f"\nROC AUC Score (Random Forest): {roc_auc_rf:.4f}")

# === Plot ROC Curve ===
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(8, 5))
plt.plot(fpr_rf, tpr_rf, label=f"AUC = {roc_auc_rf:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
