import os
import sys
import librosa
import numpy as np
import joblib

MODEL_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\models\voice_model.pkl"
SCALER_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\models\scaler.pkl"


print("Loading trained model...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model loaded successfully!")


def extract_features(file_path):

    audio, sample_rate = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)

    features = np.mean(mfcc, axis=1)

    return features


print("\n===================================")
print("AI VOICE CLONE DETECTOR")
print("===================================")

file_path = input("\nEnter audio file path: ").strip().strip('"')


if not os.path.exists(file_path):

    print("\nERROR: Audio file not found!")
    sys.exit()


print("\nProcessing audio...")

features = extract_features(file_path)

features = features.reshape(1, -1)

features = scaler.transform(features)


prediction = model.predict(features)[0]

probabilities = model.predict_proba(features)[0]

spoof_probability = probabilities[1]

bonafide_probability = probabilities[0]

risk_score = spoof_probability * 100


if prediction == 1:
    result = "SPOOF / POSSIBLE VOICE CLONE"
else:
    result = "BONAFIDE / LIKELY GENUINE"


if risk_score < 40:
    risk_level = "LOW"
elif risk_score < 70:
    risk_level = "MEDIUM"
elif risk_score < 90:
    risk_level = "HIGH"
else:
    risk_level = "CRITICAL"


print("\n===================================")
print("DETECTION RESULT")
print("===================================")

print("\nPrediction:", result)

print("Bonafide Probability:", round(bonafide_probability * 100, 2), "%")

print("Spoof Probability:", round(spoof_probability * 100, 2), "%")

print("Risk Score:", round(risk_score, 2), "/ 100")

print("Risk Level:", risk_level)

print("\n===================================")
print("Analysis completed successfully!")
print("===================================")
