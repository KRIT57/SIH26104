import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

AUDIO_DIR = r"E:\LA\LA\ASVspoof2019_LA_dev\flac"

PROTOCOL_PATH = r"E:\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"

MODEL_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\models\cnn_voice_model.pth"

SAMPLE_RATE = 16000
DURATION = 4
N_MELS = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================
# CNN MODEL
# ========================================


class VoiceCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x):

        x = self.network(x)

        x = self.classifier(x)

        return x


# ========================================
# START
# ========================================

print("========================================")
print("CNN DEV SET EVALUATION")
print("========================================")

print("Device:", DEVICE)

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))


# ========================================
# LOAD MODEL
# ========================================

model = VoiceCNN().to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

model.eval()

print("\nCNN model loaded successfully.")


# ========================================
# READ PROTOCOL
# ========================================

file_list = []

with open(PROTOCOL_PATH, "r") as file:

    for line in file:

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        file_id = parts[1]

        label = parts[-1]

        if label == "bonafide":

            file_list.append((file_id, 0))

        elif label == "spoof":

            file_list.append((file_id, 1))


print("\nDev dataset:")

print("Total files:", len(file_list))

bonafide_count = sum(1 for _, label in file_list if label == 0)

spoof_count = sum(1 for _, label in file_list if label == 1)

print("Bonafide:", bonafide_count)

print("Spoof:", spoof_count)


# ========================================
# EVALUATION
# ========================================

y_true = []

y_pred = []

y_scores = []

processed = 0

print("\nStarting evaluation...")
print("This may take some time because every audio file is processed.")


with torch.inference_mode():

    for file_id, label in file_list:

        audio_path = os.path.join(AUDIO_DIR, file_id + ".flac")

        if not os.path.exists(audio_path):

            continue

        try:

            # Load audio

            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

            # 4 second fixed length

            target_length = SAMPLE_RATE * DURATION

            if len(audio) < target_length:

                audio = np.pad(audio, (0, target_length - len(audio)))

            else:

                audio = audio[:target_length]

            # Mel spectrogram

            mel = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=N_MELS
            )

            mel = librosa.power_to_db(mel, ref=np.max)

            # Normalize

            mel = (mel - mel.mean()) / (mel.std() + 1e-8)

            # Tensor

            mel = torch.tensor(mel, dtype=torch.float32)

            mel = mel.unsqueeze(0)

            mel = mel.unsqueeze(0)

            mel = mel.to(DEVICE)

            # Prediction

            output = model(mel)

            probabilities = torch.softmax(output, dim=1)

            spoof_probability = probabilities[0][1].item()

            prediction = torch.argmax(probabilities, dim=1).item()

            # Store results

            y_true.append(label)

            y_pred.append(prediction)

            y_scores.append(spoof_probability)

            processed += 1

            if processed % 500 == 0:

                print("Processed:", processed, "/", len(file_list))

        except Exception as error:

            print("\nError:", file_id, error)


# ========================================
# METRICS
# ========================================

print("\n========================================")
print("EVALUATION RESULTS")
print("========================================")


accuracy = accuracy_score(y_true, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")


# Classification report

print("\nClassification Report:")

print(classification_report(y_true, y_pred, target_names=["Bonafide", "Spoof"]))


# Confusion matrix

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")

print(cm)


# ROC-AUC

try:

    roc_auc = roc_auc_score(y_true, y_scores)

    print("\nROC-AUC:", round(roc_auc, 4))

except Exception:

    print("\nROC-AUC could not be calculated.")


# ========================================
# EER
# ========================================

try:

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    fnr = 1 - tpr

    difference = np.abs(fpr - fnr)

    eer_index = np.nanargmin(difference)

    eer = (fpr[eer_index] + fnr[eer_index]) / 2

    print("EER:", round(eer * 100, 2), "%")

except Exception:

    print("EER could not be calculated.")


print("\n========================================")
print("CNN EVALUATION COMPLETED")
print("========================================")
