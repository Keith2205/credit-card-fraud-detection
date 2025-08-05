# 1D Convolutional Neural Network for Credit Card Fraud Detection

# === Import Libraries ===
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
import matplotlib.pyplot as plt
from pprint import pprint

# === Load Dataset ===
df = pd.read_csv("creditcard.csv")

# === Normalize 'Amount' and 'Time' ===
scaler = StandardScaler()
df['normAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['normTime'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# === Feature/Target Split ===
X = df.drop('Class', axis=1).values
y = df['Class'].values

# === Full Train/Test Split (No Subsampling) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# === Reshape for 1D CNN Input (e.g., 30 → 6x5) ===
X_train_cnn = X_train.reshape(-1, 6, 5)
X_test_cnn = X_test.reshape(-1, 6, 5)

# === One-Hot Encode Labels for Keras ===
y_train_cnn = to_categorical(y_train, num_classes=2)
y_test_cnn = to_categorical(y_test, num_classes=2)

# === Build CNN Model ===
cnn_model = Sequential([
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(6, 5)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Train CNN ===
cnn_model.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=32, verbose=0)

# === Predict ===
y_pred_proba_cnn = cnn_model.predict(X_test_cnn)
y_pred_cnn = np.argmax(y_pred_proba_cnn, axis=1)
y_true_cnn = np.argmax(y_test_cnn, axis=1)

# === Evaluation ===
conf_matrix_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)
class_report_cnn = classification_report(y_true_cnn, y_pred_cnn, output_dict=True)
roc_auc_cnn = roc_auc_score(y_true_cnn, y_pred_proba_cnn[:, 1])

# === Display Results ===
print("=== Confusion Matrix (CNN) ===")
print(conf_matrix_cnn)

print("\n=== Classification Report (CNN) ===")
pprint(class_report_cnn)

print(f"\nROC AUC Score (CNN): {roc_auc_cnn:.4f}")

# === Plot ROC Curve ===
fpr_cnn, tpr_cnn, _ = roc_curve(y_true_cnn, y_pred_proba_cnn[:, 1])

plt.figure(figsize=(8, 5))
plt.plot(fpr_cnn, tpr_cnn, label=f"AUC = {roc_auc_cnn:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - CNN")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
