import os
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# SETTINGS
# ==========================================

AUDIO_DIR = r"E:\LA\LA\ASVspoof2019_LA_train\flac"

PROTOCOL_PATH = (
    r"E:\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
)

MODEL_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\models\cnn_voice_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 16000
DURATION = 4
N_MELS = 64
MAX_FILES_PER_CLASS = 2000

BATCH_SIZE = 16
EPOCHS = 5


print("========================================")
print("CNN VOICE SPOOF DETECTION")
print("========================================")

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# READ PROTOCOL
# ==========================================

bonafide_files = []
spoof_files = []

with open(PROTOCOL_PATH, "r") as file:

    for line in file:

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        file_id = parts[1]
        label = parts[-1]

        if label == "bonafide":
            bonafide_files.append(file_id)

        elif label == "spoof":
            spoof_files.append(file_id)


print("\nOriginal dataset:")
print("Bonafide:", len(bonafide_files))
print("Spoof:", len(spoof_files))


# ==========================================
# BALANCE DATASET
# ==========================================

random.seed(42)

random.shuffle(bonafide_files)
random.shuffle(spoof_files)

bonafide_files = bonafide_files[:MAX_FILES_PER_CLASS]
spoof_files = spoof_files[:MAX_FILES_PER_CLASS]

file_list = []

for file_id in bonafide_files:
    file_list.append((file_id, 0))

for file_id in spoof_files:
    file_list.append((file_id, 1))

random.shuffle(file_list)

print("\nTraining dataset:")
print("Bonafide:", len(bonafide_files))
print("Spoof:", len(spoof_files))
print("Total:", len(file_list))


# ==========================================
# DATASET CLASS
# ==========================================


class VoiceDataset(Dataset):

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file_id, label = self.files[index]

        audio_path = os.path.join(AUDIO_DIR, file_id + ".flac")

        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        target_length = SAMPLE_RATE * DURATION

        if len(audio) < target_length:

            audio = np.pad(audio, (0, target_length - len(audio)))

        else:

            audio = audio[:target_length]

        # ==================================
        # MEL SPECTROGRAM
        # ==================================

        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=N_MELS
        )

        mel = librosa.power_to_db(mel, ref=np.max)

        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        mel = torch.tensor(mel, dtype=torch.float32)

        # CNN expects:
        # Channel x Height x Width

        mel = mel.unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return mel, label


# ==========================================
# DATALOADER
# ==========================================

dataset = VoiceDataset(file_list)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False,
)


# ==========================================
# CNN MODEL
# ==========================================


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


# ==========================================
# CREATE MODEL
# ==========================================

model = VoiceCNN()

model = model.to(DEVICE)

print("\nCNN model created successfully.")


# ==========================================
# LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ==========================================
# TRAINING
# ==========================================

print("\n========================================")
print("STARTING GPU TRAINING")
print("========================================")


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (inputs, labels) in enumerate(loader):

        inputs = inputs.to(DEVICE, non_blocking=True)

        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

        if (batch_index + 1) % 50 == 0:

            print(
                "Epoch",
                epoch + 1,
                "| Batch",
                batch_index + 1,
                "| Loss:",
                round(loss.item(), 4),
            )

    accuracy = (correct / total) * 100

    average_loss = running_loss / len(loader)

    print("\n----------------------------------------")
    print("Epoch:", epoch + 1, "/", EPOCHS)

    print("Loss:", round(average_loss, 4))

    print("Training Accuracy:", round(accuracy, 2), "%")

    print("----------------------------------------")


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

torch.save(model.state_dict(), MODEL_PATH)


print("\n========================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("========================================")

print("Model saved at:")

print(MODEL_PATH)
