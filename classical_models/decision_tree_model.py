# Decision Tree Classifier for Credit Card Fraud Detection

# === Import Libraries ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

# === Train Decision Tree Model (Handles Class Imbalance) ===
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_model.fit(X_train, y_train)

# === Make Predictions ===
y_pred_dt = dt_model.predict(X_test)
y_proba_dt = dt_model.predict_proba(X_test)[:, 1]

# === Evaluation Metrics ===
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
class_report_dt = classification_report(y_test, y_pred_dt, output_dict=True)
roc_auc_dt = roc_auc_score(y_test, y_proba_dt)

print("=== Confusion Matrix (Decision Tree) ===")
print(conf_matrix_dt)

print("\n=== Classification Report (Decision Tree) ===")
pprint(class_report_dt)

print(f"\nROC AUC Score (Decision Tree): {roc_auc_dt:.4f}")

# === Plot ROC Curve ===
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
plt.figure(figsize=(8, 5))
plt.plot(fpr_dt, tpr_dt, label=f"AUC = {roc_auc_dt:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
