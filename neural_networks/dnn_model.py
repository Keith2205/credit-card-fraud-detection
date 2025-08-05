# Deep Neural Network for Credit Card Fraud Detection

# === Import Libraries ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
from pprint import pprint

# === Load Dataset ===
df = pd.read_csv("creditcard.csv")

# === Normalize 'Amount' and 'Time' ===
scaler = StandardScaler()
df['normAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['normTime'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# === Split into Features and Target ===
X = df.drop('Class', axis=1)
y = df['Class']

# === Train-Test Split on Entire Dataset (No Subsampling) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# === Define and Train Deep Neural Network ===
dnn_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    max_iter=300,
    early_stopping=True,
    random_state=42
)
dnn_model.fit(X_train, y_train)

# === Make Predictions ===
y_pred_dnn = dnn_model.predict(X_test)
y_proba_dnn = dnn_model.predict_proba(X_test)[:, 1]

# === Evaluation Metrics ===
conf_matrix_dnn = confusion_matrix(y_test, y_pred_dnn)
class_report_dnn = classification_report(y_test, y_pred_dnn, output_dict=True)
roc_auc_dnn = roc_auc_score(y_test, y_proba_dnn)

# === Display Results ===
print("=== Confusion Matrix (DNN) ===")
print(conf_matrix_dnn)

print("\n=== Classification Report (DNN) ===")
pprint(class_report_dnn)

print(f"\nROC AUC Score (DNN): {roc_auc_dnn:.4f}")

# === Plot ROC Curve ===
fpr_dnn, tpr_dnn, _ = roc_curve(y_test, y_proba_dnn)

plt.figure(figsize=(10, 5))
plt.plot(fpr_dnn, tpr_dnn, label=f"AUC = {roc_auc_dnn:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Deep Neural Network")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
