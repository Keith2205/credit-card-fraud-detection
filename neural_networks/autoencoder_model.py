# Autoencoder for Unsupervised Fraud Detection

# === Import Libraries ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Dense  # type: ignore
import matplotlib.pyplot as plt
from pprint import pprint

# === Load Dataset ===
df = pd.read_csv("creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# === Normalize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Separate Normal and Fraud Data ===
X_normal = X_scaled[y == 0]
X_fraud = X_scaled[y == 1]

# === Use all normal transactions for training (Unsupervised Learning) ===
X_train = X_normal  # No subsampling

# === Use full dataset for testing ===
X_test = X_scaled
y_test = y.values

# === Build Autoencoder ===
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# === Train Autoencoder ===
autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, shuffle=True, verbose=1)

# === Predict and Compute Reconstruction Error ===
X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# === Set Threshold for Anomaly Detection (95th percentile of normal reconstruction errors) ===
threshold = np.percentile(mse[y == 0], 95)

# === Classify: 1 if error > threshold ===
y_pred = (mse > threshold).astype(int)

# === Evaluation ===
conf_matrix_ae = confusion_matrix(y_test, y_pred)
class_report_ae = classification_report(y_test, y_pred, output_dict=True)
roc_auc_ae = roc_auc_score(y_test, mse)

# === Display Results ===
print("=== Confusion Matrix (Autoencoder) ===")
print(conf_matrix_ae)

print("\n=== Classification Report (Autoencoder) ===")
pprint(class_report_ae)

print(f"\nROC AUC Score (Autoencoder): {roc_auc_ae:.4f}")

# === Plot ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, mse)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_ae:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Autoencoder")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
