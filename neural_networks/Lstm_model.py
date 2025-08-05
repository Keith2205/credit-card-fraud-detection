# LSTM Model for Credit Card Fraud Detection

# === Import Libraries ===
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from pprint import pprint

# === Load Dataset ===
df = pd.read_csv("creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# === Normalize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Full Train-Test Split (No Subsampling) ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# === Reshape for LSTM: [samples, timesteps, features] ===
X_train_lstm = X_train.reshape(X_train.shape[0], 30, 1)
X_test_lstm = X_test.reshape(X_test.shape[0], 30, 1)

# === One-hot Encode Labels ===
y_train_lstm = to_categorical(y_train, num_classes=2)
y_test_lstm = to_categorical(y_test, num_classes=2)

# === Build LSTM Model ===
model = Sequential()
model.add(LSTM(32, input_shape=(30, 1), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# === Train Model ===
model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=1)

# === Make Predictions ===
y_proba_lstm = model.predict(X_test_lstm)
y_pred_lstm = np.argmax(y_proba_lstm, axis=1)
y_true_lstm = np.argmax(y_test_lstm, axis=1)

# === Evaluation ===
conf_matrix_lstm = confusion_matrix(y_true_lstm, y_pred_lstm)
class_report_lstm = classification_report(y_true_lstm, y_pred_lstm, output_dict=True)
roc_auc_lstm = roc_auc_score(y_true_lstm, y_proba_lstm[:, 1])

# === Display Results ===
print("=== Confusion Matrix (LSTM) ===")
print(conf_matrix_lstm)

print("\n=== Classification Report (LSTM) ===")
pprint(class_report_lstm)

print(f"\nROC AUC Score (LSTM): {roc_auc_lstm:.4f}")

# === Plot ROC Curve ===
fpr_lstm, tpr_lstm, _ = roc_curve(y_true_lstm, y_proba_lstm[:, 1])

plt.figure(figsize=(8, 5))
plt.plot(fpr_lstm, tpr_lstm, label=f"AUC = {roc_auc_lstm:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LSTM")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
