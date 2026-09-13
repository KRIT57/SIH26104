import os
import numpy as np
import librosa
import torch
import torch.nn as nn

MODEL_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\models\cnn_voice_model.pth"

SAMPLE_RATE = 16000
DURATION = 4
N_MELS = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# CNN MODEL
# ==============================


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


# ==============================
# LOAD MODEL
# ==============================

print("========================================")
print("CNN VOICE SPOOF DETECTION")
print("========================================")

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = VoiceCNN().to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

model.eval()

print("\nCNN model loaded successfully.")


# ==============================
# AUDIO INPUT
# ==============================

audio_path = input("\nEnter audio file path: ").strip()

audio_path = audio_path.strip('"')

if not os.path.exists(audio_path):
    print("\nERROR: Audio file not found.")
    exit()


# ==============================
# LOAD AUDIO
# ==============================

audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

target_length = SAMPLE_RATE * DURATION

if len(audio) < target_length:
    audio = np.pad(audio, (0, target_length - len(audio)))
else:
    audio = audio[:target_length]


# ==============================
# MEL SPECTROGRAM
# ==============================

mel = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=N_MELS
)

mel = librosa.power_to_db(mel, ref=np.max)

mel = (mel - mel.mean()) / (mel.std() + 1e-8)

mel = torch.tensor(mel, dtype=torch.float32)

mel = mel.unsqueeze(0)
mel = mel.unsqueeze(0)

mel = mel.to(DEVICE)


# ==============================
# PREDICTION
# ==============================

with torch.no_grad():

    output = model(mel)

    probabilities = torch.softmax(output, dim=1)

    bonafide_probability = probabilities[0][0].item()
    spoof_probability = probabilities[0][1].item()

    prediction = torch.argmax(probabilities, dim=1).item()


# ==============================
# RESULT
# ==============================

risk_score = spoof_probability * 100

if risk_score < 40:
    risk_level = "LOW"
elif risk_score < 70:
    risk_level = "MEDIUM"
elif risk_score < 90:
    risk_level = "HIGH"
else:
    risk_level = "CRITICAL"


print("\n========================================")
print("VOICE ANALYSIS RESULT")
print("========================================")

print("Audio:", audio_path)

if prediction == 0:
    print("\nPrediction: BONAFIDE / GENUINE")
else:
    print("\nPrediction: SPOOF / POSSIBLE VOICE CLONE")

print("Bonafide Probability:", round(bonafide_probability * 100, 2), "%")

print("Spoof Probability:", round(spoof_probability * 100, 2), "%")

print("Risk Score:", round(risk_score, 2), "/ 100")

print("Risk Level:", risk_level)

print("========================================")
