# MLP Classifier for Credit Card Fraud Detection

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

# === Define and Train MLP Neural Network ===
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=300,
    early_stopping=True,
    random_state=42
)
mlp_model.fit(X_train, y_train)

# === Make Predictions ===
y_pred = mlp_model.predict(X_test)
y_proba = mlp_model.predict_proba(X_test)[:, 1]

# === Evaluation Metrics ===
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
roc_auc = roc_auc_score(y_test, y_proba)

# === Display Results ===
print("=== Confusion Matrix (MLP) ===")
print(conf_matrix)

print("\n=== Classification Report (MLP) ===")
pprint(class_report)

print(f"\nROC AUC Score (MLP): {roc_auc:.4f}")

# === Plot ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - MLP Neural Network")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()