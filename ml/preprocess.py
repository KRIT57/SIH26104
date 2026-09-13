import os
import librosa

PROTOCOL_PATH = (
    r"E:\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
)
AUDIO_DIR = r"E:\LA\LA\ASVspoof2019_LA_train\flac"

with open(PROTOCOL_PATH, "r") as file:
    lines = file.readlines()

print("Total entries:", len(lines))
print("\nChecking first 10 audio files:\n")

for line in lines[:10]:
    parts = line.strip().split()
    file_id = parts[1]
    label = parts[-1]
    audio_path = os.path.join(AUDIO_DIR, file_id + ".flac")

    print(file_id, "->", label, "-> File exists:", os.path.exists(audio_path))

test_file = os.path.join(AUDIO_DIR, "LA_T_1138215.flac")

audio, sample_rate = librosa.load(test_file, sr=None)

print("\nAudio loaded successfully!")
print("Sample rate:", sample_rate)
print("Audio samples:", len(audio))
print("Duration:", round(len(audio) / sample_rate, 2), "seconds")
