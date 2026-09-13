import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

FEATURES_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\data\features.npy"
LABELS_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\data\labels.npy"

MODEL_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\models\voice_model.pkl"
SCALER_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\models\scaler.pkl"


print("Loading features and labels...")

X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)


print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


print("\nScaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("\nTraining Logistic Regression model...")

model = LogisticRegression(max_iter=1000, random_state=42)

model.fit(X_train, y_train)


print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL TRAINING COMPLETED")
print("===================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Bonafide", "Spoof"]))


os.makedirs(r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\models", exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nScaler saved to:")
print(SCALER_PATH)
