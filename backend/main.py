import os
import json
import hashlib
import tempfile

import numpy as np
import librosa
import torch
import torch.nn as nn

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from web3 import Web3

# ========================================
# PATHS
# ========================================

MODEL_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\models\cnn_voice_model.pth"

ABI_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\blockchain\artifacts\contracts\VoiceDetectionAudit.sol\VoiceDetectionAudit.json"

CONTRACT_ADDRESS = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"

BLOCKCHAIN_RPC = "http://127.0.0.1:8545"

SAMPLE_RATE = 16000
DURATION = 4
N_MELS = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================
# FASTAPI APP
# ========================================

app = FastAPI(
    title="AI Voice Clone Detection API",
    description="AI-powered voice spoof detection with blockchain audit",
    version="2.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
# LOAD MODEL
# ========================================

model = VoiceCNN().to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

model.eval()


# ========================================
# BLOCKCHAIN SETUP
# ========================================

w3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_RPC))

blockchain_connected = w3.is_connected()

contract = None
blockchain_account = None

if blockchain_connected:

    with open(ABI_PATH, "r") as f:
        contract_data = json.load(f)

    contract_abi = contract_data["abi"]

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=contract_abi
    )

    accounts = w3.eth.accounts

    if len(accounts) > 0:
        blockchain_account = accounts[0]


# ========================================
# RISK ENGINE
# ========================================


def calculate_risk(spoof_probability):

    risk_score = spoof_probability * 100

    if risk_score < 40:
        risk_level = "LOW"

    elif risk_score < 70:
        risk_level = "MEDIUM"

    elif risk_score < 90:
        risk_level = "HIGH"

    else:
        risk_level = "CRITICAL"

    return risk_score, risk_level


# ========================================
# AUDIO PROCESSING
# ========================================


def process_audio(audio_path):

    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    target_length = SAMPLE_RATE * DURATION

    if len(audio) < target_length:

        audio = np.pad(audio, (0, target_length - len(audio)))

    else:

        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=N_MELS
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    mel = (mel - mel.mean()) / (mel.std() + 1e-8)

    mel = torch.tensor(mel, dtype=torch.float32)

    mel = mel.unsqueeze(0)
    mel = mel.unsqueeze(0)

    return mel.to(DEVICE)


# ========================================
# BLOCKCHAIN AUDIT
# ========================================


def save_to_blockchain(file_data, prediction, risk_score, risk_level):

    if not blockchain_connected:
        return {
            "blockchain_status": "NOT_CONNECTED",
            "audit_id": None,
            "transaction_hash": None,
            "audio_hash": None,
        }

    if blockchain_account is None:
        return {
            "blockchain_status": "NO_ACCOUNT",
            "audit_id": None,
            "transaction_hash": None,
            "audio_hash": None,
        }

    # SHA-256 hash of audio
    audio_hash_hex = hashlib.sha256(file_data).hexdigest()

    audio_hash_bytes = bytes.fromhex(audio_hash_hex)

    try:

        transaction = contract.functions.recordAudit(
            audio_hash_bytes, prediction, int(round(risk_score)), risk_level
        ).transact({"from": blockchain_account})

        receipt = w3.eth.wait_for_transaction_receipt(transaction)

        audit_count = contract.functions.auditCount().call()

        return {
            "blockchain_status": "RECORDED",
            "audit_id": audit_count,
            "transaction_hash": receipt["transactionHash"].hex(),
            "audio_hash": audio_hash_hex,
        }

    except Exception as e:

        return {
            "blockchain_status": "ERROR",
            "audit_id": None,
            "transaction_hash": None,
            "audio_hash": audio_hash_hex,
            "blockchain_error": str(e),
        }


# ========================================
# HOME API
# ========================================


@app.get("/")
def home():

    return {
        "project": "AI Voice Clone Detection",
        "status": "running",
        "model": "CNN",
        "device": str(DEVICE),
        "blockchain_connected": blockchain_connected,
        "contract_address": CONTRACT_ADDRESS,
    }


# ========================================
# HEALTH CHECK
# ========================================


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(DEVICE),
        "blockchain_connected": blockchain_connected,
        "contract_address": CONTRACT_ADDRESS,
    }


# ========================================
# VOICE DETECTION API
# ========================================


@app.post("/detect")
async def detect_voice(file: UploadFile = File(...)):

    file_extension = os.path.splitext(file.filename)[1]

    if not file_extension:
        file_extension = ".wav"

    temp_path = None

    try:

        # Read uploaded audio
        file_data = await file.read()

        # Save temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:

            temp_file.write(file_data)

            temp_path = temp_file.name

        # ========================================
        # AI PROCESSING
        # ========================================

        audio_tensor = process_audio(temp_path)

        # ========================================
        # CNN PREDICTION
        # ========================================

        with torch.inference_mode():

            output = model(audio_tensor)

            probabilities = torch.softmax(output, dim=1)

            bonafide_probability = probabilities[0][0].item()

            spoof_probability = probabilities[0][1].item()

            prediction = torch.argmax(probabilities, dim=1).item()

        # ========================================
        # RISK SCORE
        # ========================================

        risk_score, risk_level = calculate_risk(spoof_probability)

        if prediction == 0:

            result = "BONAFIDE"

            message = "Voice appears genuine."

        else:

            result = "SPOOF"

            message = "Possible voice cloning or " "synthetic speech detected."

        # ========================================
        # BLOCKCHAIN AUDIT
        # ========================================

        blockchain_result = save_to_blockchain(
            file_data, result, risk_score, risk_level
        )

        # ========================================
        # FINAL RESPONSE
        # ========================================

        return {
            "filename": file.filename,
            "prediction": result,
            "message": message,
            "bonafide_probability": round(bonafide_probability * 100, 2),
            "spoof_probability": round(spoof_probability * 100, 2),
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "device": str(DEVICE),
            "blockchain": blockchain_result,
        }

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)
