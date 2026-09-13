import os
import librosa
import numpy as np

AUDIO_DIR = r"E:\LA\LA\ASVspoof2019_LA_train\flac"

PROTOCOL_PATH = (
    r"E:\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
)

FEATURES_OUTPUT = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\data\features.npy"

LABELS_OUTPUT = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\data\labels.npy"


def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)

    features = np.mean(mfcc, axis=1)

    return features


with open(PROTOCOL_PATH, "r") as file:
    lines = file.readlines()


print("Total files to process:", len(lines))
print("\nStarting feature extraction...\n")


features_list = []
labels_list = []

for i, line in enumerate(lines):

    parts = line.strip().split()

    file_id = parts[1]
    label = parts[-1]

    audio_path = os.path.join(AUDIO_DIR, file_id + ".flac")

    if not os.path.exists(audio_path):
        print("File not found:", file_id)
        continue

    try:
        features = extract_features(audio_path)

        features_list.append(features)

        if label == "bonafide":
            labels_list.append(0)
        else:
            labels_list.append(1)

    except Exception as e:
        print("Error processing:", file_id)
        print(e)
        continue

    if (i + 1) % 100 == 0:
        print("Processed:", i + 1, "/", len(lines))


X = np.array(features_list)
y = np.array(labels_list)


np.save(FEATURES_OUTPUT, X)
np.save(LABELS_OUTPUT, y)


print("\n===================================")
print("Feature extraction completed!")
print("===================================")

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

print("\nFeatures saved to:")
print(FEATURES_OUTPUT)

print("\nLabels saved to:")
print(LABELS_OUTPUT)
