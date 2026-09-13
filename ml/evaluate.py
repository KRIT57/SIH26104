import os
import numpy as np
import librosa
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ==============================
# PATHS
# ==============================

AUDIO_DIR = r"E:\LA\LA\ASVspoof2019_LA_dev\flac"

PROTOCOL_PATH = r"E:\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"

MODEL_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\models\voice_model.pkl"

SCALER_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\models\scaler.pkl"


# ==============================
# LOAD MODEL
# ==============================

print("Loading model...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model loaded successfully!")


# ==============================
# READ PROTOCOL
# ==============================

file_labels = {}

with open(PROTOCOL_PATH, "r") as file:
    for line in file:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        file_id = parts[1]
        label = parts[-1]

        if label == "bonafide":
            file_labels[file_id] = 0
        elif label == "spoof":
            file_labels[file_id] = 1


print("Protocol loaded!")
print("Total files:", len(file_labels))


# ==============================
# FEATURE EXTRACTION
# ==============================


def extract_features(audio_path):

    audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)

    features = np.mean(mfcc, axis=1)

    return features


# ==============================
# PROCESS DEV DATASET
# ==============================

y_true = []
y_pred = []
y_scores = []

total = len(file_labels)
count = 0

print("\nProcessing development dataset...")
print("Total files:", total)

for file_id, true_label in file_labels.items():

    audio_path = os.path.join(AUDIO_DIR, file_id + ".flac")

    if not os.path.exists(audio_path):
        continue

    try:

        features = extract_features(audio_path)

        features = features.reshape(1, -1)

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        probabilities = model.predict_proba(features_scaled)[0]

        spoof_probability = probabilities[1]

        y_true.append(true_label)
        y_pred.append(prediction)
        y_scores.append(spoof_probability)

        count += 1

        if count % 1000 == 0:
            print("Processed:", count, "/", total)

    except Exception as error:

        print("Error processing", file_id, ":", error)


# ==============================
# CONVERT TO NUMPY
# ==============================

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_scores = np.array(y_scores)


# ==============================
# ACCURACY
# ==============================

accuracy = accuracy_score(y_true, y_pred)


# ==============================
# ROC-AUC
# ==============================

roc_auc = roc_auc_score(y_true, y_scores)


# ==============================
# EER
# ==============================

fpr, tpr, thresholds = roc_curve(y_true, y_scores)

fnr = 1 - tpr

eer_index = np.nanargmin(np.abs(fpr - fnr))

eer = (fpr[eer_index] + fnr[eer_index]) / 2


# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_true, y_pred)


# ==============================
# RESULTS
# ==============================

print("\n========================================")
print("VOICE SPOOF DETECTION EVALUATION")
print("========================================")

print("\nTotal evaluated samples:", len(y_true))

print("Accuracy:", round(accuracy * 100, 2), "%")

print("ROC-AUC:", round(roc_auc, 4))

print("EER:", round(eer * 100, 2), "%")

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(classification_report(y_true, y_pred, target_names=["Bonafide", "Spoof"]))

print("========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)

print("\nMatrix format:")
print("[[True Bonafide, False Spoof]]")
print("[[False Bonafide, True Spoof]]")

print("\n========================================")
print("EVALUATION COMPLETED SUCCESSFULLY")
print("========================================")
